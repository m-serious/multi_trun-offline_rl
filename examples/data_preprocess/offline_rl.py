from torch.utils.data import Dataset
import json
class OfflineRLDataset(Dataset):
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