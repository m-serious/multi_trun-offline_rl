#!/bin/bash
# ==============================================================================
# Offline RL Training Script for Gemma 2B Model
# ==============================================================================
# This script is specifically configured for training Gemma-2-2b-it model
# using offline reinforcement learning with pre-collected trajectory data.
# 
# This script is adapted from the original run_gemma.sh for offline RL mode.
# ==============================================================================

set -e  # Exit on any error
set -x  # Print commands as they are executed

# ==============================================================================
# GEMMA OFFLINE RL TRAINING CONFIGURATION
# ==============================================================================

python3 -m verl.trainer.main_ppo \
    `# Core offline RL settings` \
    trainer.offline_mode=true \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=true \
    \
    `# Dataset configuration - MODIFY THESE PATHS FOR YOUR DATA` \
    data.offline_train_files=[/path/to/your/offline_gsm8k_train.jsonl] \
    data.offline_val_files=[/path/to/your/offline_gsm8k_test.jsonl] \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    \
    `# Gemma model configuration` \
    actor_rollout_ref.model.path=google/gemma-2-2b-it \
    actor_rollout_ref.model.use_remove_padding=False \
    critic.model.path=google/gemma-2-2b-it \
    critic.model.use_remove_padding=False \
    \
    `# Conservative learning rates for offline RL` \
    actor_rollout_ref.actor.optim.lr=1e-7 \
    critic.optim.lr=1e-6 \
    \
    `# PPO configuration optimized for offline RL` \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio=0.1 \
    \
    `# Rollout configuration (primarily for validation)` \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    \
    `# Critic configuration` \
    critic.model.enable_gradient_checkpointing=False \
    critic.ppo_micro_batch_size_per_gpu=2 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    \
    `# KL divergence control for conservative updates` \
    algorithm.kl_ctrl.type=adaptive \
    algorithm.kl_ctrl.kl_coef=0.01 \
    algorithm.kl_ctrl.target_kl=0.05 \
    \
    `# Offline RL specific parameters` \
    algorithm.offline_rl.behavior_cloning_coef=0.1 \
    algorithm.offline_rl.importance_sampling_clip=1.5 \
    \
    `# Training schedule for offline RL` \
    trainer.critic_warmup=0 \
    trainer.total_epochs=10 \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    \
    `# Infrastructure settings` \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    \
    `# Logging and experiment tracking` \
    trainer.logger=['console','wandb'] \
    trainer.project_name='offline_rl_gemma' \
    trainer.experiment_name='gemma2b_offline_rl' \
    trainer.log_val_generations=5 \
    trainer.validation_data_dir=./gemma_validation_outputs \
    \
    `# Checkpoint management` \
    trainer.default_local_dir=./checkpoints/gemma_offline_rl \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.max_critic_ckpt_to_keep=3 \
    \
    `# Validation generation settings` \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.1 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    \
    "$@"  # Pass any additional command line arguments

# ==============================================================================
# GEMMA-SPECIFIC OFFLINE RL NOTES
# ==============================================================================
echo "
===============================================================================
GEMMA OFFLINE RL TRAINING CONFIGURATION
===============================================================================

This script is optimized for training Google's Gemma-2-2b-it model using
offline reinforcement learning on mathematical reasoning tasks.

KEY DIFFERENCES FROM ONLINE TRAINING:
- Much lower learning rates (1e-7 for actor, 1e-6 for critic)
- Conservative clip ratio (0.1 instead of 0.2)
- Enabled KL penalties to prevent policy drift
- Smaller batch sizes for stable training
- GRPO advantage estimator (better for offline RL than GAE)

EXPECTED DATA FORMAT:
Your offline data should be in JSONL format with GSM8K-style mathematical
reasoning problems and solutions:

{
  \"trajectory_id\": \"gsm8k_train_001\",
  \"output\": {
    \"history\": [
      {\"role\": \"user\", \"content\": \"What is 25% of 80?\"},
      {\"role\": \"agent\", \"content\": \"To find 25% of 80, I multiply 80 by 0.25. 80 × 0.25 = 20. Therefore, 25% of 80 is 20.\"}
    ],
    \"result\": {
      \"reward\": 1.0,
      \"success\": true
    }
  }
}

HYPERPARAMETER RECOMMENDATIONS:
- For more conservative training: increase behavior_cloning_coef to 0.2-0.5
- For more exploratory training: decrease kl_coef to 0.005
- If training is unstable: reduce learning rates to 1e-8 (actor) and 1e-7 (critic)
- If convergence is slow: increase batch size to 256

MONITORING TIPS:
- Watch 'offline/advantage_mean' and 'offline/advantage_std' metrics
- Monitor 'actor/reward_kl_penalty' to ensure policy doesn't drift too far
- Check validation accuracy on mathematical reasoning tasks
- Log validation generations to inspect solution quality

===============================================================================
" 