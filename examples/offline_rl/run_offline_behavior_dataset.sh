#!/bin/bash

# =============================================================================
# Offline RL Training Script - Scenario 1: Dataset with Behavior Log Probabilities
# =============================================================================
#
# This script demonstrates offline RL training when your dataset already contains 
# pre-computed behavior policy log probabilities. This is the most efficient approach
# when you have access to the original behavior policy that generated the data.
#
# Data Format Expected:
# {
#   "trajectory_id": "math_001",
#   "output": {
#     "history": [
#       {"role": "user", "content": "What is 15 * 23?"},
#       {"role": "agent", "content": "15 * 23 = 345"}
#     ],
#     "result": {"reward": 1.0, "success": true},
#     "behavior_log_probs": [-0.5, -0.3, -0.2, -0.8, -0.1, -0.4]  # Pre-computed behavior policy log probs
#   }
# }

set -x

# Configuration paths
TRAIN_DATA_PATH="/path/to/your/offline_data_with_logprobs.jsonl"
VAL_DATA_PATH="/path/to/your/offline_val_data_with_logprobs.jsonl"

# Model configuration
MODEL_PATH="microsoft/DialoGPT-medium"  # Your target policy model
EXPERIMENT_NAME="offline_rl_dataset_logprobs"

echo "🚀 Starting Offline RL Training with Dataset Behavior Log Probabilities"
echo "📊 Training data: $TRAIN_DATA_PATH"
echo "📝 Validation data: $VAL_DATA_PATH"
echo "🤖 Model: $MODEL_PATH"

python3 -m verl.trainer.main_ppo \
    --config-path verl/trainer/config \
    --config-name offline_ppo_trainer \
    \
    trainer.offline_mode=true \
    trainer.project_name="offline_rl_experiments" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.total_epochs=5 \
    trainer.test_freq=50 \
    trainer.save_freq=100 \
    \
    data.offline_train_files=["$TRAIN_DATA_PATH"] \
    data.offline_val_files=["$VAL_DATA_PATH"] \
    data.behavior_policy_mode="dataset" \
    data.train_batch_size=16 \
    data.val_batch_size=32 \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
    \
    algorithm.adv_estimator="grpo" \
    algorithm.use_kl_in_reward=true \
    algorithm.kl_ctrl.kl_coef=0.02 \
    algorithm.kl_ctrl.target_kl=0.03 \
    \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=5e-8 \
    actor_rollout_ref.actor.clip_ratio=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    \
    critic.optim.lr=1e-6 \
    critic.model.path="$MODEL_PATH" \
    critic.ppo_mini_batch_size=8 \
    critic.ppo_micro_batch_size_per_gpu=2 \
    \
    reward_model.enable=false \
    \
    trainer.default_local_dir="./checkpoints/offline_rl_dataset_mode" \
    trainer.validation_data_dir="./validation_outputs/dataset_mode" \
    $@

echo "✅ Training completed! Check results in:"
echo "   - Checkpoints: ./checkpoints/offline_rl_dataset_mode"
echo "   - Validation: ./validation_outputs/dataset_mode"

# =============================================================================
# Key Features of this approach:
# =============================================================================
# 
# ✅ Advantages:
# - Most efficient: No need to load additional behavior model
# - Exact behavior policy log probabilities from original data generation
# - Faster training startup and execution
# - Lower memory requirements
#
# 📋 Requirements:
# - Dataset must contain "behavior_log_probs" field for each trajectory
# - Log probabilities must correspond exactly to the response tokens
# - Behavior log probs should be from the policy that originally generated the data
#
# 🎯 Use this approach when:
# - You generated the offline data yourself and saved the log probabilities
# - You have access to the original behavior policy and can compute log probs
# - You want maximum efficiency and accuracy in offline RL training 