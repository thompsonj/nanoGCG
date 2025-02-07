import copy
import gc
import logging
import os

from dataclasses import dataclass
from tqdm import tqdm
from typing import List, Optional, Union

import torch
import transformers
from torch import Tensor
from transformers import set_seed

from nanogcg.utils import INIT_CHARS, find_executable_batch_size, get_nonascii_toks, unravel_index

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logger = logging.getLogger("nanogcg")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

@dataclass
class GCGConfig:
    num_steps: int = 250
    optim_str_init: Union[str, List[str]] = "x x x x x x x x x x x x x x x x x x x x"
    search_width: int = 512
    batch_size: int = None
    topk: int = 256  # how many candidates to sample from
    n_replace: int = 1
    buffer_size: int = 0
    use_mellowmax: bool = False
    mellowmax_alpha: float = 1.0
    early_stop: bool = False
    use_prefix_cache: bool = True
    allow_non_ascii: bool = False
    filter_ids: bool = True
    add_space_before_target: bool = False
    seed: int = None
    verbosity: str = "INFO"
    opt_or_pes: str = 'opt'
    faster: bool = True

@dataclass
class GCGResult:
    best_loss: float
    best_string: str
    best_reward: float
    losses: List[float]
    strings: List[str]
    rewards: List[float]
    ids: List[int]


class GCGHistory:
    def __init__(self):
        self.history = set()

    def add(self, optim_ids) -> None:
        for id in optim_ids.cpu().numpy(): 
            self.history.add(str(id))
        
        # if self.history == []:
        #     self.history = optim_ids
        # else:    
        #     self.history = torch.concat((self.history, optim_ids), dim=0)        
        # if isinstance(optim_ids, list):
        #     self.history.extend(optim_ids)
        # else:
        #     self.history.append(optim_ids)
    
    def get_history(self) -> Tensor:
        return self.history
        # if self.history:
        #     return torch.concat(self.history, dim=0).squeeze()
        # else:
        #     return self.history
    
        
class AttackBuffer:
    def __init__(self, size: int):
        self.buffer = [] # elements are (loss: float, reward: float, optim_ids: Tensor)
        self.size = size

    def add(self, loss: float, reward: float, optim_ids: Tensor) -> None:
        if self.size == 0:
            self.buffer = [(loss, reward, optim_ids)]
            return

        if len(self.buffer) < self.size:
            self.buffer.append((loss, reward, optim_ids))
        else:
            self.buffer[-1] = (loss, reward, optim_ids)

        self.buffer.sort(key=lambda x: x[0])

    def get_best_ids(self) -> Tensor:
        return self.buffer[0][2]

    def get_lowest_loss(self) -> float:
        return self.buffer[0][0]
    
    def get_highest_loss(self) -> float:
        return self.buffer[-1][0]
    
    def log_buffer(self, tokenizer):
        message = "buffer:"
        for loss, reward, ids in self.buffer:
            optim_str = tokenizer.batch_decode(ids)[0]
            optim_str = optim_str.replace("\\", "\\\\")
            optim_str = optim_str.replace("\n", "\\n")
            message += f"\nloss: {loss}" + f" | string: {optim_str} | reward: {reward}"
        logger.info(message)

    def get_ids(self):
        return [ids for (_, _, ids) in self.buffer]

def sample_ids_from_grad(
    ids: Tensor, 
    grad: Tensor, 
    search_width: int, 
    topk: int = 256,
    n_replace: int = 1,
    not_allowed_ids: Tensor = False,
    start_idx: int = 0
):
    """Returns `search_width` combinations of token ids based on the token gradient.

    Args:
        ids : Tensor, shape = (n_optim_ids)
            the sequence of token ids that are being optimized 
        grad : Tensor, shape = (n_optim_ids, vocab_size)
            the gradient of the GCG loss computed with respect to the one-hot token embeddings
        search_width : int
            the number of candidate sequences to return
        topk : int
            the topk to be used when sampling from the gradient
        n_replace : int
            the number of token positions to update per sequence
        not_allowed_ids : Tensor, shape = (n_ids)
            the token ids that should not be used in optimization
    
    Returns:
        sampled_ids : Tensor, shape = (search_width, n_optim_ids)
            sampled token ids
    """
    n_optim_tokens = len(ids)
    original_ids = ids.repeat(search_width, 1)

    if not_allowed_ids is not None:
        grad[:, not_allowed_ids.to(grad.device)] = float("inf")

    sampled_ids_pos,sampled_ids_val = unravel_index((-grad.view((-1,))).topk(start_idx + search_width).indices, grad.shape)
    pos = sampled_ids_pos[start_idx:start_idx+search_width].unsqueeze(dim=-1)
    val = sampled_ids_val[start_idx:start_idx+search_width].unsqueeze(dim=-1)
    new_ids = original_ids.scatter_(1, pos, val)

    # topk_ids = (-grad).topk(topk, dim=1).indices

    # sampled_ids_pos = torch.argsort(torch.rand((search_width, n_optim_tokens), device=grad.device))[..., :n_replace] # which position(s) to swap in each candidate sequence
    # sampled_ids_val = torch.gather(
    #     topk_ids[sampled_ids_pos],
    #     2,
    #     torch.randint(0, topk, (search_width, n_replace, 1), device=grad.device)
    # ).squeeze(2)

    # new_ids = original_ids.scatter_(1, sampled_ids_pos, sampled_ids_val)

    return new_ids

def filter_ids(ids: Tensor, tokenizer: transformers.PreTrainedTokenizer, history: set):
    """Filters out sequeneces of token ids that change after retokenization.

    Args:
        ids : Tensor, shape = (search_width, n_optim_ids) 
            token ids 
        tokenizer : ~transformers.PreTrainedTokenizer
            the model's tokenizer
    
    Returns:
        filtered_ids : Tensor, shape = (new_search_width, n_optim_ids)
            all token ids that are the same after retokenization
    """
    # ids_decoded = tokenizer.batch_decode(ids)
    
    # filtered_ids = []

    # for i in range(len(ids_decoded)):
    #     # Retokenize the decoded token ids
    #     ids_encoded = tokenizer(ids_decoded[i], return_tensors="pt", add_special_tokens=False).to(ids.device)["input_ids"][0]
    #     if torch.equal(ids[i], ids_encoded):
    #        filtered_ids.append(ids[i]) 
    
    # if not filtered_ids:
    #     # This occurs in some cases, e.g. using the Llama-3 tokenizer with a bad initialization
    #     raise RuntimeError(
    #         "No token sequences are the same after decoding and re-encoding. "
    #         "Consider setting `filter_ids=False` or trying a different `optim_str_init`"
    #     )
        
    filtered_ids = ids
    # also remove any sequences that have already been tested
    n_ids = len(filtered_ids)
    # filtered_ids = [id for id in filtered_ids if not any([(id == c_).all() for c_ in history])]
    filtered_ids = [id for id in filtered_ids if not str(id.cpu().numpy()) in history]
    
    # print(f'{len(filtered_ids)} sequences to test after {n_ids - len(filtered_ids)} repeats removed')
    if filtered_ids:
        return torch.stack(filtered_ids)
    else:
        return []

class GCG:
    def __init__(
        self, 
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        config: GCGConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        self.embedding_layer = model.get_input_embeddings()
        self.not_allowed_ids = None if config.allow_non_ascii else get_nonascii_toks(tokenizer, device=model.device)
        self.prefix_cache = None
        self.maximize = True if self.config.opt_or_pes == 'opt' else False
        self.w = 4.5#4 # FasterGCG paper uses 4 or 5 depending on model
        self.faster = config.faster

        self.stop_flag = False

        if model.dtype in (torch.float32, torch.float64):
            logger.warning(f"Model is in {model.dtype}. Use a lower precision data type, if possible, for much faster optimization.")

        if model.device == torch.device("cpu"):
            logger.warning("Model is on the CPU. Use a hardware accelerator for faster optimization.")

        if not tokenizer.chat_template:
            logger.warning("Tokenizer does not have a chat template. Assuming base model and setting chat template to empty.")
            tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    
    def run(
        self,
        prompts: Union[str, List[dict]],
    ) -> GCGResult:
        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        if config.seed is not None:
            set_seed(config.seed)
            torch.use_deterministic_algorithms(True, warn_only=True)
    
        if isinstance(prompts, str):
            messages = [{"role": "user", "content": prompts}, {'role': 'assistant', 'content':''}]
        else:
            messages = copy.deepcopy(prompts)
    
        # Append the GCG string at the end of the prompt if location not specified
        if not any(["{optim_str}" in d["content"] for d in messages]):
            messages[-1]["content"] = messages[-1]["content"] + "{optim_str}"

        template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False) 
        # Remove the BOS token -- this will get added when tokenizing, if necessary
        if tokenizer.bos_token and template.startswith(tokenizer.bos_token):
            template = template.replace(tokenizer.bos_token, "")
        before_str, after_str = template.split("{optim_str}")
        # target = " " + target if config.add_space_before_target else target

        # Tokenize everything that doesn't get optimized
        before = tokenizer([before_str], padding=False, return_tensors="pt")
        before_ids = before["input_ids"].to(model.device, torch.int64)
        after = tokenizer([after_str], add_special_tokens=False, return_tensors="pt")
        after_ids = after["input_ids"].to(model.device, torch.int64)
        # target_ids = tokenizer([target], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)

        # Embed everything that doesn't get optimized
        embedding_layer = self.embedding_layer
        # before_embeds, after_embeds, target_embeds = [embedding_layer(ids) for ids in (before_ids, after_ids, target_ids)]
        before_embeds, after_embeds = [embedding_layer(ids) for ids in (before_ids, after_ids)]
        # Compute the KV Cache for tokens that appear before the optimized tokens
        # if config.use_prefix_cache:
        #     with torch.no_grad():
        #         output = model(inputs_embeds=before_embeds, use_cache=True)
        #         self.prefix_cache = output.past_key_values
        
        # self.target_ids = target_ids
        self.before_embeds = before_embeds
        self.after_embeds = after_embeds
        # self.target_embeds = target_embeds

        # Initialize the response buffer
        buffer = self.init_buffer()
        history = GCGHistory()
        optim_ids = buffer.get_best_ids()

        losses = []
        optim_strings = []
        rewards = []
        ids = []
        
        for _ in tqdm(range(config.num_steps)):
            # Compute the token gradient
            optim_ids_onehot_grad = self.compute_token_gradient(optim_ids)

            with torch.no_grad():

                # Sample candidate token sequences based on the token gradient
                start_idx = 0
                sampled_ids = sample_ids_from_grad(
                    optim_ids.squeeze(0),
                    optim_ids_onehot_grad.squeeze(0),
                    config.search_width,
                    config.topk,
                    config.n_replace,
                    not_allowed_ids=self.not_allowed_ids,
                    start_idx=start_idx
                )

                if config.filter_ids:
                    
                    sampled_ids = filter_ids(sampled_ids, tokenizer, history.get_history())
                    carry_on = False
                    missing = len(sampled_ids) < config.search_width
                    start_idx += config.search_width
                    i = 0
                    print('\n')
                    while missing > 0 and not carry_on:
                        # Sample further candidate token sequences until search width is filled
                        # try:
                        print(f'how far to search the gradient for new sequences: {start_idx}', end="\r")
                        more_sampled_ids = sample_ids_from_grad(
                            optim_ids.squeeze(0),
                            optim_ids_onehot_grad.squeeze(0),
                            (config.search_width - len(sampled_ids)),
                            config.topk + i,
                            config.n_replace,
                            not_allowed_ids=self.not_allowed_ids,
                            start_idx=start_idx
                        )
                        start_idx += (config.search_width - len(sampled_ids))
                        more_sampled_ids = filter_ids(more_sampled_ids, tokenizer, history.get_history())
                        if len(more_sampled_ids) > missing:
                            if sampled_ids == []:
                                sampled_ids = more_sampled_ids[:missing]
                            else:
                                sampled_ids = torch.concat([sampled_ids, more_sampled_ids[:missing]])
                        if len(more_sampled_ids) > 0:
                            if sampled_ids == []:
                                sampled_ids = more_sampled_ids
                            else:
                                sampled_ids = torch.concat([sampled_ids, more_sampled_ids])
                        # except:
                        #     # give up on filling the search width exactly
                        #     if len(sampled_ids) > 0:
                        #         carry_on = True
                        missing = len(sampled_ids) < config.search_width
                        i += 1
                print('\n')
                    
                new_search_width = sampled_ids.shape[0]

                # Compute loss on all candidate sequences 
                batch_size = new_search_width if config.batch_size is None else config.batch_size
                if self.prefix_cache:
                    input_embeds = torch.cat([
                        embedding_layer(sampled_ids),
                        after_embeds.repeat(new_search_width, 1, 1),
                    ], dim=1)
                else:
                    input_embeds = torch.cat([
                        before_embeds.repeat(new_search_width, 1, 1),
                        embedding_layer(sampled_ids),
                        after_embeds.repeat(new_search_width, 1, 1),
                    ], dim=1)

                loss, reward = find_executable_batch_size(self.compute_candidates_loss, batch_size)(input_embeds)
                history.add(sampled_ids)

                current_loss = loss.min().item()
                optim_ids = sampled_ids[loss.argmin()].unsqueeze(0)
                current_reward = reward[loss.argmin()].item()

                # Update the buffer based on the loss
                print(f'Current loss: {current_loss}')
                losses.append(current_loss)
                rewards.append(current_reward)
                ids.append(optim_ids.cpu().numpy()[0])

                if buffer.size == 0 or current_loss < buffer.get_highest_loss():
                    buffer.add(current_loss, current_reward, optim_ids)
                

            # optim_ids = buffer.get_best_ids()
            optim_str = tokenizer.batch_decode(optim_ids)[0]
            optim_strings.append(optim_str)

            buffer.log_buffer(tokenizer)                

            if self.stop_flag:
                logger.info("Early stopping due to finding a perfect match.") 
                break
              
        min_loss_index = losses.index(min(losses)) 

        result = GCGResult(
            best_loss=losses[min_loss_index],
            best_string=optim_strings[min_loss_index],
            best_reward=rewards[min_loss_index],
            losses=losses,
            rewards=rewards,
            strings=optim_strings,
            ids=ids
        )

        return result
    
    def init_buffer(self) -> AttackBuffer:
        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        logger.info(f"Initializing attack buffer of size {config.buffer_size}...")

        # Create the attack buffer and initialize the buffer ids
        buffer = AttackBuffer(config.buffer_size)

        if isinstance(config.optim_str_init, str):
            init_optim_ids = tokenizer(config.optim_str_init, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
            if config.buffer_size > 1:
                init_buffer_ids = tokenizer(INIT_CHARS, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze().to(model.device)
                init_indices = torch.randint(0, init_buffer_ids.shape[0], (config.buffer_size - 1, init_optim_ids.shape[1]))
                init_buffer_ids = torch.cat([init_optim_ids, init_buffer_ids[init_indices]], dim=0)
            else:
                init_buffer_ids = init_optim_ids
                
        else: # assume list
            if (len(config.optim_str_init) != config.buffer_size):
                logger.warning(f"Using {len(config.optim_str_init)} initializations but buffer size is set to {config.buffer_size}")
            try:
                init_buffer_ids = tokenizer(config.optim_str_init, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
            except ValueError:
                logger.error("Unable to create buffer. Ensure that all initializations tokenize to the same length.")

        true_buffer_size = max(1, config.buffer_size) 

        # Compute the loss on the initial buffer entries
        if self.prefix_cache:
            init_buffer_embeds = torch.cat([
                self.embedding_layer(init_buffer_ids),
                self.after_embeds.repeat(true_buffer_size, 1, 1),
                # self.target_embeds.repeat(true_buffer_size, 1, 1),
            ], dim=1)
        else:
            init_buffer_embeds = torch.cat([
                self.before_embeds.repeat(true_buffer_size, 1, 1),
                self.embedding_layer(init_buffer_ids),
                self.after_embeds.repeat(true_buffer_size, 1, 1),
                # self.target_embeds.repeat(true_buffer_size, 1, 1),
            ], dim=1)


        init_buffer_losses, init_buffer_reward = find_executable_batch_size(self.compute_candidates_loss, true_buffer_size)(init_buffer_embeds)

        # Populate the buffer
        for i in range(true_buffer_size):
            buffer.add(init_buffer_losses[i], init_buffer_reward[i], init_buffer_ids[[i]])
        
        buffer.log_buffer(tokenizer)

        logger.info("Initialized attack buffer.")
        
        return buffer
    
    def compute_token_gradient(
        self,
        optim_ids: Tensor,
    ) -> Tensor:
        """Computes the gradient of the GCG loss w.r.t the one-hot token matrix.

        Args:
            optim_ids : Tensor, shape = (1, n_optim_ids)
                the sequence of token ids that are being optimized 
        """
        model = self.model
        embedding_layer = self.embedding_layer
        cap = 30

        # Create the one-hot encoding matrix of our optimized token ids
        optim_ids_onehot = torch.nn.functional.one_hot(optim_ids, num_classes=embedding_layer.num_embeddings)
        optim_ids_onehot = optim_ids_onehot.to(model.device, model.dtype)
        optim_ids_onehot.requires_grad_()

        # (1, num_optim_tokens, vocab_size) @ (vocab_size, embed_dim) -> (1, num_optim_tokens, embed_dim)
        optim_embeds = optim_ids_onehot @ embedding_layer.weight

        if self.prefix_cache:
            input_embeds = torch.cat([optim_embeds, self.after_embeds], dim=1)
            output = model(inputs_embeds=input_embeds, past_key_values=self.prefix_cache)
        else:
            input_embeds = torch.cat([self.before_embeds, optim_embeds, self.after_embeds], dim=1)
            output = model(inputs_embeds=input_embeds)
            
        logits = output.logits
        
        if self.maximize:
            loss = torch.nn.functional.mse_loss(logits, torch.tensor(cap, dtype=model.dtype).view(1, 1).to(self.model.device))
            # loss = torch.nn.functional.l1_loss(logits, torch.tensor(cap, dtype=model.dtype).view(1, 1).to(self.model.device))
        else:
            loss = torch.nn.functional.mse_loss(logits, torch.tensor(cap, dtype=model.dtype).view(1, 1).to(self.model.device))
            # loss = torch.nn.functional.l1_loss(logits, torch.tensor(cap, dtype=model.dtype).view(1, 1).to(self.model.device))
  
        optim_ids_onehot_grad = torch.autograd.grad(outputs=[loss], inputs=[optim_ids_onehot])[0] # (1, response_length, vocabulary size)

        if self.faster:  # Regularize as in FasterGCG paper
            distance = self.compute_token_distance(optim_embeds)
            # print(f'{torch.min(optim_ids_onehot_grad)} {torch.max(distance)}')
            return optim_ids_onehot_grad + self.w * distance # going to look at negative gradient in next step, 
        else:
            return optim_ids_onehot_grad
    
    def compute_token_distance(
        self,
        optim_embeds: Tensor,
    ) -> Tensor:
        """Computes the distance in embedding space for all possible token swaps."""
        embedding_layer = self.embedding_layer
        all_tokens_embeds = embedding_layer(torch.arange(0, embedding_layer.num_embeddings).long().to(self.model.device))
        distance = torch.cdist(optim_embeds, all_tokens_embeds.unsqueeze(0))
        return distance

    def compute_candidates_loss(
        self,
        search_batch_size: int, 
        input_embeds: Tensor, 
    ) -> Tensor:
        """Computes the GCG loss on all candidate token id sequences.

        Args:
            search_batch_size : int
                the number of candidate sequences to evaluate in a given batch
            input_embeds : Tensor, shape = (search_width, seq_len, embd_dim)
                the embeddings of the `search_width` candidate sequences to evaluate
        """
        all_loss = []
        all_reward = []
        prefix_cache_batch = []
        cap = 30 #self.model.config.final_logit_softcapping

        for i in range(0, input_embeds.shape[0], search_batch_size):
            with torch.no_grad():
                input_embeds_batch = input_embeds[i:i+search_batch_size]
                current_batch_size = input_embeds_batch.shape[0]

                if self.prefix_cache:
                    if not prefix_cache_batch or current_batch_size != search_batch_size:
                        prefix_cache_batch = [[x.expand(current_batch_size, -1, -1, -1) for x in self.prefix_cache[i]] for i in range(len(self.prefix_cache))]

                    rewards = self.model(inputs_embeds=input_embeds_batch, past_key_values=prefix_cache_batch).logits
                else:
                    rewards = self.model(inputs_embeds=input_embeds_batch).logits

                if self.maximize:
                    target = torch.tensor(cap).expand(current_batch_size, 1).to(self.model.device)
                else:
                    target = torch.tensor(-cap).expand(current_batch_size, 1).to(self.model.device)
                loss = torch.nn.functional.mse_loss(rewards, target, reduction="none")
                # loss = torch.nn.functional.l1_loss(rewards, target, reduction="none")

                loss = loss.view(current_batch_size, -1).mean(dim=-1)
                rewards = rewards.view(current_batch_size, -1).mean(dim=-1)
                all_loss.append(loss)
                all_reward.append(rewards)

                # if self.config.early_stop:
                #     if torch.any(torch.all(torch.argmax(shift_logits, dim=-1) == shift_labels, dim=-1)).item():
                #         self.stop_flag = True

                # del rewards
                # gc.collect()
                torch.cuda.empty_cache()

        return torch.cat(all_loss, dim=0), torch.cat(all_reward, dim=0)

# A wrapper around the GCG `run` method that provides a simple API
def run(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    prompts: Union[str, List[dict]],
    config: Optional[GCGConfig] = None,
) -> GCGResult:
    """Generates a single optimized string using GCG. 

    Args:
        model: The model to use for optimization.
        tokenizer: The model's tokenizer.
        messages: The conversation to use for optimization.
        target: The target generation.
        config: The GCG configuration to use.
    
    Returns:
        A GCGResult object that contains losses and the optimized strings.
    """
    if config is None:
        config = GCGConfig()

    logger.setLevel(getattr(logging, config.verbosity))

    gcg = GCG(model, tokenizer, config)
    result = gcg.run(prompts)
    return result
    