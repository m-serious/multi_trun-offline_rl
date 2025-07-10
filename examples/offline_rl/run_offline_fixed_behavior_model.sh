#!/bin/bash

# =============================================================================
# Offline RL Training Script - Scenario 2: Fixed Behavior Policy Model
# =============================================================================
#
# This script demonstrates offline RL training when you use a fixed SFT model
# as the behavior policy to compute log probabilities. This approach is useful
# when you don't have access to the original behavior policy log probabilities
# but have a representative model (usually the SFT model used to initialize training).
#
# Data Format Expected (standard format without behavior_log_probs):
# {
#   "trajectory_id": "math_002",
#   "output": {
#     "history": [
#       {"role": "user", "content": "Solve: 2x + 5 = 13"},
#       {"role": "agent", "content": "2x + 5 = 13\n2x = 8\nx = 4"}
#     ],
#     "result": {"reward": 1.0, "success": true}
#   }
# }

set -x

# Configuration paths
TRAIN_DATA_PATH="/path/to/your/offline_data_standard.jsonl"
VAL_DATA_PATH="/path/to/your/offline_val_data_standard.jsonl"

# Model configuration
MODEL_PATH="microsoft/DialoGPT-medium"         # Your target policy model
BEHAVIOR_MODEL_PATH="microsoft/DialoGPT-small"  # Fixed SFT model as behavior policy
EXPERIMENT_NAME="offline_rl_fixed_behavior"

echo "🚀 Starting Offline RL Training with Fixed Behavior Policy Model"
echo "📊 Training data: $TRAIN_DATA_PATH"
echo "📝 Validation data: $VAL_DATA_PATH"
echo "🤖 Target model: $MODEL_PATH"
echo "🎯 Behavior model: $BEHAVIOR_MODEL_PATH"

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
    data.behavior_policy_mode="fixed_model" \
    data.behavior_model_path="$BEHAVIOR_MODEL_PATH" \
    data.device="cuda" \
    data.train_batch_size=16 \
    data.val_batch_size=32 \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
    \
    algorithm.adv_estimator="grpo" \
    algorithm.use_kl_in_reward=true \
    algorithm.kl_ctrl.kl_coef=0.03 \
    algorithm.kl_ctrl.target_kl=0.05 \
    \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-7 \
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
    trainer.default_local_dir="./checkpoints/offline_rl_fixed_behavior" \
    trainer.validation_data_dir="./validation_outputs/fixed_behavior" \
    $@

echo "✅ Training completed! Check results in:"
echo "   - Checkpoints: ./checkpoints/offline_rl_fixed_behavior"
echo "   - Validation: ./validation_outputs/fixed_behavior"

# =============================================================================
# Key Features of this approach:
# =============================================================================
# 
# ✅ Advantages:
# - Works with standard offline datasets (no special preprocessing needed)
# - Good approximation when behavior model is representative of data generation
# - Flexible: can experiment with different behavior models
# - More principled than using current policy as behavior policy
#
# ⚠️ Considerations:
# - Requires loading an additional model (higher memory usage)
# - Behavior log probabilities are approximations, not exact
# - Slower data loading due to on-the-fly computation
# - Choice of behavior model affects training quality
#
# 🎯 Use this approach when:
# - You have standard offline data without behavior log probabilities
# - You have access to an SFT model representative of the data generation process
# - You want more principled offline RL than using current policy as behavior policy
# - You're willing to trade some efficiency for better approximation of behavior policy
#
# 💡 Tips for choosing behavior_model_path:
# - Use the SFT model that was used to initialize the fine-tuning process
# - If unknown, use a model of similar size/architecture trained on similar data
# - Smaller models can work as behavior policies and use less memory
# - Consider using quantized versions (int8/fp16) to reduce memory usage

# =============================================================================
# Advanced Configuration Examples:
# =============================================================================

# Example 1: Using a quantized behavior model to save memory
# data.behavior_model_path="microsoft/DialoGPT-small" \
# data.behavior_model_quantization="int8" \

# Example 2: Conservative offline training with higher KL penalties
# algorithm.kl_ctrl.kl_coef=0.05 \
# algorithm.kl_ctrl.target_kl=0.02 \
# actor_rollout_ref.actor.optim.lr=5e-8 \

# Example 3: Using behavior cloning regularization
# algorithm.offline_rl.behavior_cloning_coef=0.2 \ 