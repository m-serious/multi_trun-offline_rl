#!/bin/bash
# ==============================================================================
# Offline Reinforcement Learning Training Script with PPO
# ==============================================================================
# This script demonstrates how to run offline RL training using pre-collected
# trajectory data without online environment interaction.
#
# Usage:
#   ./run_offline_ppo.sh [additional_hydra_args...]
#
# Example:
#   ./run_offline_ppo.sh data.offline_train_files=/path/to/data.jsonl \
#                        actor_rollout_ref.model.path=/path/to/model
# ==============================================================================

set -e  # Exit on any error
set -x  # Print commands as they are executed

# Configuration file for offline RL training
CONFIG_NAME="offline_ppo_trainer"

# ==============================================================================
# OFFLINE RL TRAINING CONFIGURATION
# ==============================================================================

python3 -m verl.trainer.main_ppo \
    --config-path configs \
    --config-name ${CONFIG_NAME} \
    \
    `# Core offline RL settings` \
    trainer.offline_mode=true \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=true \
    \
    `# Dataset configuration - MODIFY THESE PATHS` \
    data.offline_train_files=[/path/to/your/offline_train_data.jsonl] \
    data.offline_val_files=[/path/to/your/offline_val_data.jsonl] \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    data.train_batch_size=64 \
    data.val_batch_size=32 \
    \
    `# Model configuration - MODIFY MODEL PATH` \
    actor_rollout_ref.model.path=google/gemma-2-2b-it \
    critic.model.path=google/gemma-2-2b-it \
    \
    `# Conservative learning settings for offline RL` \
    actor_rollout_ref.actor.optim.lr=1e-7 \
    critic.optim.lr=1e-6 \
    actor_rollout_ref.actor.clip_ratio=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    critic.ppo_mini_batch_size=16 \
    critic.ppo_micro_batch_size_per_gpu=2 \
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
    `# Training schedule` \
    trainer.total_epochs=10 \
    trainer.test_freq=100 \
    trainer.save_freq=500 \
    trainer.critic_warmup=0 \
    \
    `# Infrastructure settings` \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    \
    `# Logging and experiment tracking` \
    trainer.logger=['console','wandb'] \
    trainer.project_name='offline_rl_experiment' \
    trainer.experiment_name='offline_ppo_training' \
    trainer.log_val_generations=5 \
    trainer.validation_data_dir=./validation_outputs \
    \
    `# Checkpoint management` \
    trainer.default_local_dir=./checkpoints/offline_rl \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.max_critic_ckpt_to_keep=3 \
    \
    `# Remove padding optimization` \
    actor_rollout_ref.model.use_remove_padding=false \
    critic.model.use_remove_padding=false \
    \
    `# Validation settings` \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.1 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    \
    "$@"  # Pass any additional arguments

# ==============================================================================
# TRAINING TIPS AND RECOMMENDATIONS
# ==============================================================================
echo "
===============================================================================
OFFLINE RL TRAINING TIPS
===============================================================================

1. DATA REQUIREMENTS:
   - Ensure your offline data is in JSONL format with proper structure
   - Each line should contain trajectory history and reward information
   - Validate data format before training

2. HYPERPARAMETER TUNING:
   - Start with conservative learning rates (1e-7 to 1e-6)
   - Use lower clip ratios (0.1-0.2) for stable training
   - Enable KL penalties to prevent policy drift
   - Consider behavior cloning coefficient > 0 for very conservative training

3. MONITORING:
   - Watch validation metrics closely for signs of overfitting
   - Monitor KL divergence to ensure policy doesn't drift too far
   - Log validation generations to inspect policy behavior

4. ALGORITHM CHOICES:
   - GRPO is recommended for offline RL
   - reinforce_plus_plus is also a good choice
   - Avoid GAE unless you have good value function estimates

5. TROUBLESHOOTING:
   - If training is unstable, reduce learning rates further
   - If policy doesn't improve, increase behavior cloning coefficient
   - If validation degrades, add more regularization

For more information, see the documentation in examples/offline_rl/README.md
===============================================================================
" 