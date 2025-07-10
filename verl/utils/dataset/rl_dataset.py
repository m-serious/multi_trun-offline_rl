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


def collate_fn(data_list: list[dict]) -> dict:
    """Collate a batch of data."""
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


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
    Offline Reinforcement Learning Dataset for processing pre-collected trajectory data.
    
    This dataset is specifically designed for offline RL training where we learn from
    fixed trajectory data without online environment interaction.
    
    Data Format Requirements:
    - JSONL format with one JSON object per line
    - Each record contains trajectory history and reward information
    - Supports multi-turn conversation format
    - Automatically generates loss_mask for multi-turn training
    
    Args:
        data_files (str or List[str]): Path(s) to offline trajectory data files
        tokenizer: HuggingFace tokenizer instance
        max_prompt_length (int): Maximum length for prompt/observation tokens
        max_response_length (int): Maximum length for response/action tokens
    """
    def __init__(self, data_files, tokenizer, max_prompt_length=4096, max_response_length=1024):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.data = []
        
        # Ensure sep_token exists for separating prompt and response
        if not hasattr(tokenizer, 'sep_token') or tokenizer.sep_token is None:
            tokenizer.sep_token = "[SEP]"
            tokenizer.add_special_tokens({'sep_token': '[SEP]'})
            print(f"[INFO] Set sep_token to: {tokenizer.sep_token}")
        
        if isinstance(data_files, str):
            data_files = [data_files]
        
        print(f"[INFO] Loading {len(data_files)} offline data files: {data_files}")
        
        for fp in data_files:
            if not os.path.exists(fp):
                print(f"[ERROR] File not found: {fp}")
                continue
                
            self._load_file(fp)
        
        print(f"[INFO] Successfully loaded {len(self.data)} training samples")
        
        if not self.data:
            raise ValueError(f"No valid samples loaded from {data_files}. Please check data format and file content.")

    def _load_file(self, filepath):
        """Load and parse trajectory data from a single file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_idx, line in enumerate(f):
                    try:
                        record = json.loads(line.strip())
                        self._process_record(record, filepath, line_idx)
                    except json.JSONDecodeError as e:
                        print(f"[WARNING] Skipping invalid JSON line {filepath}:{line_idx}: {e}")
                        continue
        except FileNotFoundError:
            print(f"[ERROR] File not found: {filepath}")

    def _process_record(self, record, filepath, line_idx):
        """Process a single record to extract trajectory and reward information."""
        # Extract trajectory data - support multiple data formats
        trajectory = (
            record.get("output", {}).get("result", {}).get("history", []) or
            record.get("output", {}).get("history", []) or
            record.get("history", []) or
            []
        )
        
        if not trajectory:
            print(f"[WARNING] Empty trajectory {filepath}:{line_idx}")
            return
        
        # Extract reward value - support multiple reward fields
        reward = self._extract_reward(record)
        
        # Parse user-agent conversation pairs from trajectory
        conversation_pairs = self._extract_conversation_pairs(trajectory)
        
        # Create training samples for each conversation pair
        for pair_idx, (user_msg, agent_msg) in enumerate(conversation_pairs):
            uid = f"{os.path.basename(filepath)}_{line_idx}_{pair_idx}"
            sample = self._build_sample(user_msg, agent_msg, reward, uid)
            self.data.append(sample)

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

    def _build_sample(self, user_msg, agent_msg, reward, uid):
        """Build a single training sample from user message and agent response."""
        # Tokenize user message and agent response separately
        user_tokens = self.tokenizer(
            user_msg, 
            truncation=True, 
            add_special_tokens=False, 
            max_length=self.max_prompt_length
        )
        
        agent_tokens = self.tokenizer(
            agent_msg, 
            truncation=True, 
            add_special_tokens=False, 
            max_length=self.max_response_length
        )
        
        # Get separator tokens
        sep_tokens = self.tokenizer(
            self.tokenizer.sep_token, 
            add_special_tokens=False
        )["input_ids"]
        
        # Combine input_ids: [user_tokens] + [SEP] + [agent_tokens]
        input_ids = user_tokens["input_ids"] + sep_tokens + agent_tokens["input_ids"]
        attention_mask = [1] * len(input_ids)
        
        # loss_mask: compute loss only on agent response tokens
        loss_mask = (
            [0] * len(user_tokens["input_ids"]) +     # No loss on user tokens
            [0] * len(sep_tokens) +                   # No loss on separator tokens  
            [1] * len(agent_tokens["input_ids"])      # Compute loss on agent tokens
        )
        
        # Ensure total length doesn't exceed maximum
        max_total_length = self.max_prompt_length + self.max_response_length
        if len(input_ids) > max_total_length:
            input_ids = input_ids[:max_total_length]
            attention_mask = attention_mask[:max_total_length]
            loss_mask = loss_mask[:max_total_length]
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.long),
            "responses": torch.tensor(agent_tokens["input_ids"], dtype=torch.long),
            "rewards": torch.tensor([reward], dtype=torch.float),
            "uid": uid,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# =======  替换结束 =======


# Important Considerations for OfflineRLDataset Implementation:

# Parsing JSONL: You need robust parsing for your specific runs.jsonl structure, especially the nested history array to extract individual (state, action, reward) tuples.
# State Construction: For each agent action in a trajectory, the "state" (which becomes input_ids for the model) needs to be correctly constructed. This might involve concatenating the current "observation" with some context from previous turns in multi_turn scenarios.
# loss_mask Generation: This is critical for multi_turn. When you tokenize the agent's action (from the "action" field in your data), the loss_mask must precisely cover these action tokens. If your model input combines prompt and action, the loss_mask must zero out the prompt tokens.
# Tokenization and Padding: Ensure all sequences are tokenized and padded/truncated to consistent lengths suitable for batching. max_prompt_length and max_response_length from the config will be important here.
# Reward Handling: Your offline data has scalar rewards per turn. The PPO pipeline often expects token_level_rewards. You'll need to decide how to map this scalar reward (e.g., assign it to the last token of the action, or distribute it). The example in RayPPOTrainer.fit() under is_offline_mode shows one way to handle scalar rewards.