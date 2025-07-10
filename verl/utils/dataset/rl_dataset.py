# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import logging
import os
import re
from collections import defaultdict
from typing import List, Optional, Union

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)


def collate_fn(batch_data):
    """
    Collate function for RL dataset.
    
    Handles both online RL data (RLHFDataset) and offline RL data (OfflineRLDataset).
    """
    batch_tensor = {}
    batch_non_tensor = defaultdict(list)
    
    # Determine if this is offline RL data by checking for behavior_log_probs or trajectory_id
    is_offline = any("behavior_log_probs" in item or "trajectory_id" in item for item in batch_data)
    
    for item in batch_data:
        for key, value in item.items():
            if isinstance(value, torch.Tensor):
                if key not in batch_tensor:
                    batch_tensor[key] = []
                batch_tensor[key].append(value)
            else:
                batch_non_tensor[key].append(value)
    
    # Stack or pad tensor fields
    for key, value_list in batch_tensor.items():
        if key in ["input_ids", "attention_mask", "responses", "response_attention_mask", "loss_mask", "behavior_log_probs"]:
            # Pad sequences to same length
            max_len = max(v.size(0) for v in value_list)
            padded_tensors = []
            
            for tensor in value_list:
                if tensor.size(0) < max_len:
                    # Pad with zeros (or appropriate pad tokens for input_ids)
                    pad_value = 0
                    if key == "input_ids":
                        # Use tokenizer's pad_token_id if available
                        pad_value = getattr(batch_data[0].get('tokenizer', None), 'pad_token_id', 0) or 0
                    
                    padding = torch.full((max_len - tensor.size(0),), pad_value, dtype=tensor.dtype)
                    padded_tensor = torch.cat([tensor, padding])
                else:
                    padded_tensor = tensor
                
                padded_tensors.append(padded_tensor)
            
            batch_tensor[key] = torch.stack(padded_tensors)
        else:
            # For other tensor fields (like rewards), just stack
            batch_tensor[key] = torch.stack(value_list)
    
    # Handle offline RL specific fields
    if is_offline:
        # Ensure behavior_log_probs is handled correctly
        if "behavior_log_probs" not in batch_tensor:
            # If not present, set to None (will be handled in training loop)
            batch_tensor["behavior_log_probs"] = None
        
        # For offline RL, we often need to combine input_ids and responses
        if "input_ids" in batch_tensor and "responses" in batch_tensor:
            input_ids = batch_tensor["input_ids"]
            responses = batch_tensor["responses"]
            input_attention_mask = batch_tensor.get("attention_mask")
            response_attention_mask = batch_tensor.get("response_attention_mask")
            
            # Create combined sequences for full context
            # Combined format: [input_ids] + [responses]
            batch_size = input_ids.size(0)
            input_len = input_ids.size(1)
            response_len = responses.size(1)
            
            # Create combined input_ids and attention_mask
            combined_input_ids = torch.cat([input_ids, responses], dim=1)
            
            if input_attention_mask is not None and response_attention_mask is not None:
                combined_attention_mask = torch.cat([input_attention_mask, response_attention_mask], dim=1)
            else:
                # Fallback: create attention mask based on non-zero tokens
                combined_attention_mask = (combined_input_ids != 0).long()
            
            # Create loss_mask: no loss on input tokens, loss on response tokens
            loss_mask = torch.cat([
                torch.zeros(batch_size, input_len, dtype=torch.long),  # No loss on input
                torch.ones(batch_size, response_len, dtype=torch.long)  # Loss on response
            ], dim=1)
            
            # Update batch with combined sequences
            batch_tensor["input_ids"] = combined_input_ids
            batch_tensor["attention_mask"] = combined_attention_mask
            batch_tensor["loss_mask"] = loss_mask
            
            # Keep original responses for response_mask computation
            # batch_tensor["responses"] remains as is
    
    # Return in the format expected by DataProto.from_single_dict
    result = {}
    result.update(batch_tensor)
    result.update(dict(batch_non_tensor))
    return result


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, (List, ListConfig)):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            prompt_key = self.prompt_key
            self.dataframe = self.dataframe.filter(
                lambda doc: len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True)) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(self.dataframe)}")

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                for segment in re.split("(<image>|<video>)", content):
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        messages = self._build_messages(row_dict)
        model_inputs = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            multi_modal_data = {}

            images = None
            if self.image_key in row_dict:
                images = [process_image(image) for image in row_dict.pop(self.image_key)]
                multi_modal_data["image"] = images

            videos = None
            if self.video_key in row_dict:
                videos = [process_video(video) for video in row_dict.pop(self.video_key)]
                multi_modal_data["video"] = [video.numpy() for video in videos]

            model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data
            row_dict["multi_modal_inputs"] = dict(model_inputs)

            # second_per_grid_ts isn't used for training, just for mrope
            row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and self.processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)

        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages
        
        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt # array of strings

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()
    

import json
class OfflineRLDataset_(Dataset):
    def __init__(self, data_files, tokenizer, processor, config, is_multi_turn, max_prompt_length, max_response_length):
        self.tokenizer = tokenizer
        self.processor = processor # if multimodal
        self.config = config
        self.is_multi_turn = is_multi_turn
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.data = []

        if isinstance(data_files, str):
            data_files = [data_files]

        for file_path in data_files:
            with open(file_path, 'r') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        # Example: Parse 'runs.jsonl' structure
                        # trajectory is in record['output']['result']['history']
                        trajectory = record.get("output", {}).get("result", {}).get("history", [])
                        if not trajectory:
                            # Fallback for other potential structures or skip if no trajectory
                            # For example, if the top-level record is a single turn
                            # trajectory = [record] # if each line is one turn
                            continue

                        # Process each turn in the trajectory
                        # For multi-turn, you need to accumulate context for the 'prompt'
                        # and identify agent's 'action' and 'reward'.
                        # This parsing logic will be complex and specific to your data format.
                        #
                        # Simplified example for a single agent turn from trajectory:
                        # current_prompt_str = ""
                        # for turn_idx, turn_data in enumerate(trajectory):
                        #    observation = turn_data.get("observation")
                        #    agent_action_str = turn_data.get("action") # The actual action taken
                        #    reward = turn_data.get("reward")
                        #    done = turn_data.get("done", False)
                        #
                        #    if agent_action_str is None or reward is None:
                        #        continue # Skip turns without agent action or reward
                        #
                        #    # Construct prompt for this turn (e.g., current observation + prior context)
                        #    # This needs careful handling for multi-turn.
                        #    # For simplicity, let's assume observation is the full prompt for the agent's action
                        #    prompt_tokens = self.tokenizer(observation, truncation=True, max_length=self.max_prompt_length)
                        #    action_tokens = self.tokenizer(agent_action_str, truncation=True, max_length=self.max_response_length)
                        #
                        #    # *** CRITICAL for multi-turn: Generate loss_mask ***
                        #    # loss_mask should only cover the agent's action tokens.
                        #    # If your 'responses' tensor in the batch combines prompt and action,
                        #    # loss_mask needs to mask out the prompt part.
                        #    # If 'responses' is just the action, loss_mask can be all ones for action tokens.
                        #    # This example assumes 'responses' will be just the action.
                        #    loss_mask = [1] * len(action_tokens['input_ids'])
                        #
                        #    self.data.append({
                        #        "input_ids": prompt_tokens["input_ids"],
                        #        "attention_mask": prompt_tokens["attention_mask"], # For prompt
                        #        "responses": action_tokens["input_ids"], # Agent's action from data
                        #        "response_attention_mask": action_tokens["attention_mask"], # For action
                        #        "rewards": float(reward), # Should be shaped appropriately later
                        #        "loss_mask": loss_mask, # Crucial for multi-turn
                        #        # "old_log_probs": load if available, else will be computed by current policy
                        #    })
                        #    if done: break 
                    except json.JSONDecodeError:
                        print(f"Warning: Could not decode JSON from line: {line.strip()}")
                        continue
        print(f"Loaded {len(self.data)} samples from offline data.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return a dictionary that collate_fn can process
        # Ensure all tensors are padded/truncated to consistent lengths if not handled by collate_fn
        item = self.data[idx]
        # Example:
        # return {
        #     "input_ids": torch.tensor(item["input_ids"]),
        #     "attention_mask": torch.tensor(item["attention_mask"]),
        #     "responses": torch.tensor(item["responses"]), # This is action from dataset
        #     # "rewards" might be a scalar, or token-level if your reward_fn provides that
        #     "rewards": torch.tensor([item["rewards"]]), # Example scalar reward
        #     "loss_mask": torch.tensor(item["loss_mask"]),
        # }
        # The actual content of what you return and how it's collated
        # needs to match what RayPPOTrainer's fit() loop expects.
        # This part requires careful implementation based on your exact data structure
        # and how you structure the batch in RayPPOTrainer.
        # For now, returning the dict as is, assuming collate_fn handles padding.
        return self.data[idx]


# =======  替换原 OfflineRLDataset =======
from torch.utils.data import Dataset
from typing import List, Union, Optional
import json, torch, copy, itertools
from transformers import PreTrainedTokenizer, ProcessorMixin
import os

# class OfflineRLDataset(Dataset):
#     """
#     专供离线 RL（PPO / ILQL / DT 等）使用的数据集。
#     每个样本 = (prompt_tokens + response_tokens)，并携带
#         - loss_mask      : 仅对 response 部分 =1
#         - responses      : response_tokens（单独裁剪，供 compute_response_mask）
#         - rewards        : scalar，trainer 会再分配到 token_level_scores
#         - uid            : 行号，方便 debug 对齐
#     适配 qwq32b.jsonl：
#         record['output']['history']        -> trajectory list
#         record['output']['result']['result'] == True/0 -> 终局成功标志
#     """
#     def __init__(
#         self,
#         data_files: Union[str, List[str]],
#         tokenizer: PreTrainedTokenizer,
#         config,
#         processor: Optional[ProcessorMixin] = None,
#         is_multi_turn: bool = True,
#         max_prompt_length: int = 4096,
#         max_response_length: int = 1024,
#     ):
#         super().__init__()
#         if isinstance(data_files, str):
#             data_files = [data_files]

#         self.tokenizer = tokenizer
#         self.processor = processor
#         self.config = config
#         self.is_multi_turn = is_multi_turn
#         self.max_prompt_length = max_prompt_length
#         self.max_response_length = max_response_length
#         self.data = []

#         uid = 0
#         for fp in data_files:
#             with open(fp) as f:
#                 for line in f:
#                     record = json.loads(line)
#                     # ---------- 1️⃣ 抽 trajectory ----------
#                     traj = (record.get("output", {})
#                                    .get("result", {})
#                                    .get("history", []))
#                     if not traj:               # qwq32b 存在于这里
#                         traj = record.get("output", {}).get("history", [])
#                     if not traj:
#                         continue               # 跳过异常行

#                     # ---------- 2️⃣ reward ----------
#                     scalar_reward = float(bool(
#                         record.get("output", {})
#                               .get("result", {})
#                               .get("result", 0)
#                     ))

#                     # ---------- 3️⃣ 遍历 agent turn ----------
#                     prompt_stack: list[str] = []
#                     for turn in traj:
#                         role, content = turn["role"], turn["content"]
#                         if role == "user":
#                             prompt_stack.append(content)
#                         elif role == "agent":
#                             obs_text  = "\n".join(prompt_stack)
#                             act_text  = content
#                             sample    = self._build_sample(
#                                 obs_text, act_text, scalar_reward, uid)
#                             self.data.append(sample)
#                             uid += 1
#                             prompt_stack.append(act_text)  # 更新上下文
#                         # 其余角色忽略

#     # -------------------------------------------------------
#     def _build_sample(self, obs: str, act: str, rew: float, uid: int):
#         """
#         从 (obs, act) 文本对构造张量 dict.
#         """
#         # 1. tokenization
#         obs_tok  = self.tokenizer(
#             obs,
#             add_special_tokens=False,
#             truncation=True,
#             max_length=self.max_prompt_length,
#         )
#         act_tok  = self.tokenizer(
#             act,
#             add_special_tokens=False,
#             truncation=True,
#             max_length=self.max_response_length,
#         )

#         input_ids       = obs_tok["input_ids"] + act_tok["input_ids"]
#         attention_mask  = [1] * len(input_ids)
#         loss_mask       = [0] * len(obs_tok["input_ids"]) + [1] * len(act_tok["input_ids"])

#         return {
#             "input_ids"  : torch.tensor(input_ids, dtype=torch.long),
#             "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
#             "loss_mask"  : torch.tensor(loss_mask, dtype=torch.long),
#             "responses"  : torch.tensor(act_tok["input_ids"], dtype=torch.long),
#             "rewards"    : torch.tensor([rew], dtype=torch.float),
#             "uid"        : uid,                         # 非 tensor，collate_fn 会自动转成 np.array
#         }

#     # -------------------------------------------------------
#     def __len__(self):  return len(self.data)
#     def __getitem__(self, idx): return self.data[idx]


class OfflineRLDataset(Dataset):
    """
    Offline RL Dataset for loading pre-collected trajectories with rewards and actions.
    
    Supports two modes for behavior policy log probabilities:
    1. Load pre-computed log probabilities from dataset (behavior_log_probs field)
    2. Compute log probabilities using a fixed behavior policy model during loading
    
    Expected data format for JSONL files:
    {
        "trajectory_id": "unique_id",
        "output": {
            "history": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "agent", "content": "2+2 equals 4."}
            ],
            "result": {
                "reward": 1.0,
                "success": true
            },
            "behavior_log_probs": [0.1, 0.2, 0.05, ...] // Optional: pre-computed behavior policy log probs
        }
    }
    """
    
    def __init__(
        self, 
        data_files: list, 
        tokenizer, 
        max_prompt_length: int, 
        max_response_length: int,
        behavior_policy_mode: str = "fixed_model",  # "dataset" or "fixed_model" 
        behavior_model_path: str = None,  # Path to behavior policy model (for fixed_model mode)
        device: str = "cuda"
    ):
        """
        Args:
            data_files: List of JSONL file paths containing offline trajectories
            tokenizer: Tokenizer for processing text
            max_prompt_length: Maximum length for input prompts
            max_response_length: Maximum length for responses
            behavior_policy_mode: How to obtain behavior policy log probabilities
                - "dataset": Load pre-computed log probs from dataset
                - "fixed_model": Compute using a fixed behavior policy model
            behavior_model_path: Path to behavior policy model (required for fixed_model mode)
            device: Device for behavior model computation
        """
        self.data_files = data_files
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.behavior_policy_mode = behavior_policy_mode
        self.behavior_model_path = behavior_model_path
        self.device = device
        
        # Initialize behavior policy model if needed
        self.behavior_model = None
        if behavior_policy_mode == "fixed_model":
            self._init_behavior_model()
        
        # Load and process all data
        self.samples = []
        self._load_all_data()
        
        print(f"Loaded {len(self.samples)} offline RL samples")
        print(f"Behavior policy mode: {behavior_policy_mode}")
        
    def _init_behavior_model(self):
        """Initialize behavior policy model for computing log probabilities"""
        if not self.behavior_model_path:
            raise ValueError("behavior_model_path must be provided when behavior_policy_mode='fixed_model'")
            
        try:
            from transformers import AutoModelForCausalLM
            
            print(f"Loading behavior policy model from {self.behavior_model_path}")
            self.behavior_model = AutoModelForCausalLM.from_pretrained(
                self.behavior_model_path,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True
            )
            self.behavior_model.eval()
            print("Behavior policy model loaded successfully")
            
        except Exception as e:
            print(f"Failed to load behavior model: {e}")
            raise
    
    def _compute_behavior_log_probs(self, input_ids: torch.Tensor, response_ids: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
        """Compute behavior policy log probabilities for given inputs and responses"""
        if self.behavior_model is None:
            raise ValueError("Behavior model not initialized")
            
        # Concatenate input and response
        full_input_ids = torch.cat([input_ids, response_ids], dim=-1)
        
        with torch.no_grad():
            # Get logits from behavior model
            outputs = self.behavior_model(full_input_ids)
            logits = outputs.logits
            
            # Extract response logits (shift by 1 for next-token prediction)
            response_logits = logits[:, input_ids.shape[-1]-1:-1]  # Shape: (batch_size, response_length, vocab_size)
            
            # Compute log probabilities
            log_probs = torch.log_softmax(response_logits, dim=-1)
            
            # Extract log probabilities for actual response tokens
            response_log_probs = torch.gather(
                log_probs, 
                dim=-1, 
                index=response_ids.unsqueeze(-1)
            ).squeeze(-1)  # Shape: (batch_size, response_length)
            
            # Apply response mask
            response_log_probs = response_log_probs * response_mask
            
        return response_log_probs

    def _load_all_data(self):
        """Load and process all data files."""
        for fp in self.data_files:
            self._load_file(fp)

    def _load_file(self, filepath):
        """Load and parse trajectory data from a single file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_idx, line in enumerate(f):
                    try:
                        record = json.loads(line.strip())
                        sample = self._process_record(record)
                        if sample:
                            self.samples.append(sample)
                    except json.JSONDecodeError as e:
                        print(f"[WARNING] Skipping invalid JSON line {filepath}:{line_idx}: {e}")
                        continue
        except FileNotFoundError:
            print(f"[ERROR] File not found: {filepath}")

    def _process_record(self, record):
        """Process a single record and extract required fields"""
        try:
            # Extract conversation history
            output = record.get("output", {})
            history = output.get("history", [])
            
            if not history:
                return None
                
            # Extract reward/score
            reward = self._extract_reward(output)
            
            # Process conversation pairs
            conversation_pairs = self._extract_conversation_pairs(history)
            if not conversation_pairs:
                return None
            
            # Extract behavior log probabilities if available in dataset
            behavior_log_probs_raw = None
            if self.behavior_policy_mode == "dataset":
                behavior_log_probs_raw = output.get("behavior_log_probs", None)
                if behavior_log_probs_raw is None:
                    print(f"Warning: behavior_log_probs not found in dataset record, but behavior_policy_mode='dataset'")
                    return None
            
            # Build the sample
            sample = self._build_sample(conversation_pairs, reward, behavior_log_probs_raw)
            return sample
            
        except Exception as e:
            print(f"Error processing record: {e}")
            return None

    def _extract_reward(self, record):
        """Extract reward value, supporting multiple reward formats."""
        try:
            # Try different reward field sources
            reward_sources = [
                record.get("output", {}).get("result", {}).get("reward"),
                record.get("output", {}).get("result", {}).get("result"),
                record.get("reward"),
                record.get("score")
            ]
            
            for reward in reward_sources:
                if reward is not None:
                    return float(reward)
            
            # If no explicit reward, use success flag
            success = record.get("output", {}).get("result", {}).get("success", False)
            return 1.0 if success else 0.0
            
        except (TypeError, ValueError):
            print(f"[WARNING] Invalid reward value, using default 0.0")
            return 0.0

    def _extract_conversation_pairs(self, trajectory):
        """Extract user-agent conversation pairs from trajectory."""
        pairs = []
        current_user_msg = None
        
        for turn in trajectory:
            if not isinstance(turn, dict) or "role" not in turn or "content" not in turn:
                continue
                
            role = turn["role"]
            content = turn["content"]
            
            if role == "user":
                current_user_msg = content
            elif role == "agent" and current_user_msg is not None:
                pairs.append((current_user_msg, content))
                current_user_msg = None  # Reset after pairing
        
        return pairs

    def _build_sample(self, conversation_pairs, reward, behavior_log_probs_raw=None):
        """Build a training sample from conversation pairs and reward"""
        
        # Combine all user inputs and agent responses for multi-turn
        full_prompt = ""
        full_response = ""
        
        for user_msg, agent_msg in conversation_pairs:
            if full_prompt:
                full_prompt += f"{self.tokenizer.sep_token}{user_msg}"
            else:
                full_prompt = user_msg
            
            if full_response:
                full_response += f"{self.tokenizer.sep_token}{agent_msg}"
            else:
                full_response = agent_msg
        
        # Tokenize prompt and response
        prompt_inputs = self.tokenizer(
            full_prompt,
            max_length=self.max_prompt_length,
            truncation=True,
            padding=False,
            return_tensors="pt"
        )
        
        response_inputs = self.tokenizer(
            full_response,
            max_length=self.max_response_length,
            truncation=True,
            padding=False,
            return_tensors="pt"
        )
        
        prompt_ids = prompt_inputs["input_ids"].squeeze(0)
        response_ids = response_inputs["input_ids"].squeeze(0)
        
        # Create masks
        prompt_attention_mask = prompt_inputs["attention_mask"].squeeze(0)
        response_attention_mask = response_inputs["attention_mask"].squeeze(0)
        
        # Handle behavior log probabilities
        behavior_log_probs = None
        
        if self.behavior_policy_mode == "dataset" and behavior_log_probs_raw is not None:
            # Load pre-computed behavior log probabilities from dataset
            behavior_log_probs = torch.tensor(behavior_log_probs_raw, dtype=torch.float32)
            
            # Ensure correct length (truncate or pad to response length)
            if len(behavior_log_probs) > len(response_ids):
                behavior_log_probs = behavior_log_probs[:len(response_ids)]
            elif len(behavior_log_probs) < len(response_ids):
                # Pad with zeros
                padding_length = len(response_ids) - len(behavior_log_probs)
                behavior_log_probs = torch.cat([
                    behavior_log_probs, 
                    torch.zeros(padding_length, dtype=torch.float32)
                ])
                
        elif self.behavior_policy_mode == "fixed_model":
            # Compute behavior log probabilities using the fixed model
            behavior_log_probs = self._compute_behavior_log_probs(
                input_ids=prompt_ids.unsqueeze(0),
                response_ids=response_ids.unsqueeze(0), 
                response_mask=response_attention_mask.unsqueeze(0).float()
            ).squeeze(0)
        
        return {
            "input_ids": prompt_ids,
            "attention_mask": prompt_attention_mask,
            "responses": response_ids,
            "response_attention_mask": response_attention_mask,
            "rewards": torch.tensor(reward, dtype=torch.float32),
            "behavior_log_probs": behavior_log_probs,  # Behavior policy log probabilities
            "trajectory_id": f"offline_{len(self.samples)}"  # Unique identifier
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# =======  替换结束 =======


# Important Considerations for OfflineRLDataset Implementation:

# Parsing JSONL: You need robust parsing for your specific runs.jsonl structure, especially the nested history array to extract individual (state, action, reward) tuples.
# State Construction: For each agent action in a trajectory, the "state" (which becomes input_ids for the model) needs to be correctly constructed. This might involve concatenating the current "observation" with some context from previous turns in multi_turn scenarios.
# loss_mask Generation: This is critical for multi_turn. When you tokenize the agent's action (from the "action" field in your data), the loss_mask must precisely cover these action tokens. If your model input combines prompt and action, the loss_mask must zero out the prompt tokens.
# Tokenization and Padding: Ensure all sequences are tokenized and padded/truncated to consistent lengths suitable for batching. max_prompt_length and max_response_length from the config will be important here.
# Reward Handling: Your offline data has scalar rewards per turn. The PPO pipeline often expects token_level_rewards. You'll need to decide how to map this scalar reward (e.g., assign it to the last token of the action, or distribute it). The example in RayPPOTrainer.fit() under is_offline_mode shows one way to handle scalar rewards.