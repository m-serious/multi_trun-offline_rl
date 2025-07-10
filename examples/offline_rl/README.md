# Offline Reinforcement Learning (Offline RL) Implementation

This directory contains a complete implementation of offline reinforcement learning for the VERL framework. Offline RL enables training policies from pre-collected trajectory data without requiring online environment interaction during training.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Data Format](#data-format)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Algorithm Details](#algorithm-details)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)

## Overview

### What is Offline RL?

Offline Reinforcement Learning (also known as Batch RL) is a paradigm where policies are learned from fixed datasets of previously collected trajectories, without the ability to interact with the environment during training. This approach is particularly valuable when:

- **Online interaction is expensive or risky** (e.g., autonomous vehicles, medical treatments)
- **You have access to large datasets** of expert demonstrations or exploration data
- **Environment interaction is not available** during training time
- **You want to avoid distribution shift** from online exploration

### Key Features

✅ **Complete Offline Training Pipeline**: Train PPO policies purely from offline data  
✅ **Multi-turn Conversation Support**: Handle complex dialogue scenarios with proper loss masking  
✅ **Conservative Policy Updates**: Prevent extrapolation errors with KL penalties and behavior cloning  
✅ **Flexible Data Formats**: Support multiple JSONL data structures  
✅ **Advantage Estimation**: Optimized for offline settings (GRPO, Reinforce++)  
✅ **Comprehensive Validation**: Evaluate trained policies on holdout data  
✅ **Production Ready**: Full integration with VERL's distributed training infrastructure

### Differences from Online RL

| Aspect | Online RL | Offline RL |
|--------|-----------|------------|
| **Data Source** | Generated during training | Pre-collected fixed dataset |
| **Exploration** | Active exploration | No exploration (conservative updates) |
| **Learning Rate** | Standard (1e-6 to 1e-5) | Much lower (1e-8 to 1e-7) |
| **Policy Constraints** | Minimal | Strong (KL penalties, behavior cloning) |
| **Advantage Estimation** | GAE typically | GRPO, Reinforce++ recommended |
| **Risk of Overfitting** | Lower | Higher (requires careful monitoring) |

## Quick Start

### 1. Prepare Your Data

Create offline trajectory data in JSONL format:

```bash
# Example data structure (see data_format_example.jsonl for more examples)
echo '{"trajectory_id": "example_001", "output": {"history": [{"role": "user", "content": "What is 2+2?"}, {"role": "agent", "content": "2+2 equals 4."}], "result": {"reward": 1.0, "success": true}}}' > my_offline_data.jsonl
```

### 2. Run Training

```bash
# Quick start with Gemma model
./run_offline_gemma.sh \
    data.offline_train_files=[/path/to/your/data.jsonl] \
    actor_rollout_ref.model.path=google/gemma-2-2b-it

# Or use the general script
./run_offline_ppo.sh \
    data.offline_train_files=[/path/to/your/data.jsonl] \
    data.offline_val_files=[/path/to/your/val_data.jsonl] \
    actor_rollout_ref.model.path=your/model/path
```

### 3. Monitor Training

```bash
# Watch training progress
tail -f logs/offline_rl_experiment/offline_ppo_training/console.log

# Monitor with wandb (if configured)
wandb login
# Training metrics will appear in your wandb dashboard
```

## Data Format

### Standard Format

The expected JSONL format for offline training data:

```json
{
  "trajectory_id": "unique_identifier",
  "output": {
    "history": [
      {"role": "user", "content": "Question or prompt"},
      {"role": "agent", "content": "Agent response or action"}
    ],
    "result": {
      "reward": 1.0,
      "success": true
    }
  }
}
```

### Alternative Formats

The implementation supports multiple data formats:

```json
// Simplified format
{
  "history": [
    {"role": "user", "content": "Question"},
    {"role": "agent", "content": "Answer"}
  ],
  "reward": 1.0
}

// With different reward field names
{
  "output": {
    "result": {
      "history": [...],
      "result": 0.8  // reward value
    }
  }
}

// With score instead of reward
{
  "history": [...],
  "score": 0.9
}
```

### Multi-turn Conversations

For multi-turn conversations, include multiple user-agent pairs:

```json
{
  "trajectory_id": "multi_turn_001",
  "output": {
    "history": [
      {"role": "user", "content": "What is the capital of France?"},
      {"role": "agent", "content": "The capital of France is Paris."},
      {"role": "user", "content": "What is its population?"},
      {"role": "agent", "content": "Paris has approximately 2.2 million people."}
    ],
    "result": {"reward": 1.0, "success": true}
  }
}
```

### Reward Guidelines

- **High reward (0.8-1.0)**: Correct, helpful, well-formatted responses
- **Medium reward (0.4-0.7)**: Partially correct or somewhat helpful responses  
- **Low reward (0.0-0.3)**: Incorrect, unhelpful, or harmful responses
- **Binary rewards**: Use 1.0 for success, 0.0 for failure (simple but effective)

## Configuration

### Basic Configuration

The main configuration files:

- `verl/trainer/config/offline_ppo_trainer.yaml`: Complete offline RL configuration
- `verl/trainer/config/ppo_trainer.yaml`: Base configuration with offline mode options

### Key Configuration Sections

#### Offline Mode Activation

```yaml
trainer:
  offline_mode: true  # Enable offline RL mode
```

#### Data Configuration

```yaml
data:
  offline_train_files: [/path/to/train.jsonl]  # Required
  offline_val_files: [/path/to/val.jsonl]      # Optional but recommended
  train_batch_size: 64                         # Smaller for stable training
  max_prompt_length: 1024
  max_response_length: 512
```

#### Algorithm Configuration

```yaml
algorithm:
  adv_estimator: grpo           # Recommended: grpo or reinforce_plus_plus
  use_kl_in_reward: true       # Important: prevents policy drift
  kl_ctrl:
    type: adaptive
    kl_coef: 0.01              # Higher for more conservative updates
    target_kl: 0.05            # Lower target for conservative training
  
  offline_rl:
    behavior_cloning_coef: 0.1    # Helps maintain similarity to data policy
    importance_sampling_clip: 1.5  # Conservative IS clipping
```

#### Conservative Learning Settings

```yaml
actor_rollout_ref:
  actor:
    optim:
      lr: 1e-7                 # Much lower than online RL
      weight_decay: 0.01       # Regularization
    clip_ratio: 0.1           # Conservative clipping
    ppo_mini_batch_size: 16   # Smaller batches

critic:
  optim:
    lr: 1e-6                  # Conservative critic learning rate
```

## Usage Examples

### Example 1: Mathematical Reasoning (GSM8K-style)

```bash
# Prepare data with mathematical problems and solutions
./run_offline_gemma.sh \
    data.offline_train_files=[/data/gsm8k_offline_train.jsonl] \
    data.offline_val_files=[/data/gsm8k_offline_test.jsonl] \
    trainer.experiment_name=math_reasoning_offline \
    algorithm.offline_rl.behavior_cloning_coef=0.2  # More conservative
```

### Example 2: Conversational AI

```bash
# Train on conversation data with diverse topics
./run_offline_ppo.sh \
    data.offline_train_files=[/data/conversations.jsonl] \
    actor_rollout_ref.model.path=microsoft/DialoGPT-medium \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    algorithm.adv_estimator=reinforce_plus_plus
```

### Example 3: Code Generation

```bash
# Train on code generation tasks
./run_offline_ppo.sh \
    data.offline_train_files=[/data/code_solutions.jsonl] \
    actor_rollout_ref.model.path=microsoft/CodeBERT-base \
    algorithm.offline_rl.behavior_cloning_coef=0.15 \
    trainer.total_epochs=15
```

### Example 4: Multi-GPU Training

```bash
# Scale to multiple GPUs
./run_offline_ppo.sh \
    data.offline_train_files=[/data/large_dataset.jsonl] \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    data.train_batch_size=256 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64
```

## Algorithm Details

### Offline RL Challenges

1. **Distribution Shift**: Training data may not cover all states the policy might encounter
2. **Extrapolation Error**: Policy may assign high values to unseen state-action pairs
3. **Limited Exploration**: Cannot collect new data to improve policy
4. **Overfitting**: Risk of memorizing training data without generalizing

### Our Solutions

#### 1. Conservative Policy Updates

- **Lower learning rates** (1e-7 vs 1e-6 for online RL)
- **KL penalties** to keep policy close to behavior policy
- **Smaller clip ratios** for more conservative PPO updates

#### 2. Improved Advantage Estimation

- **GRPO**: Better suited for offline settings than GAE
- **Reinforce++**: Handles offline data distributions well
- **Proper masking**: Only compute losses on agent responses

#### 3. Regularization Techniques

- **Behavior cloning loss**: Encourages staying close to data distribution
- **Importance sampling clipping**: Prevents extreme importance weights
- **Weight decay**: Prevents overfitting to training data

### Training Loop Modifications

The offline training loop differs from online RL in several key ways:

1. **No Online Generation**: Skip sequence generation, use actions from data
2. **Policy Evaluation**: Compute log π_current(a_data | s_data) for offline actions
3. **Conservative Updates**: Apply KL penalties and behavior cloning losses
4. **Validation-focused**: Emphasize validation metrics over training metrics

## Best Practices

### Data Preparation

✅ **Quality over Quantity**: Better to have fewer high-quality trajectories than many low-quality ones  
✅ **Diverse Trajectories**: Include successful and unsuccessful examples  
✅ **Proper Rewards**: Ensure reward signals accurately reflect desired behavior  
✅ **Validation Split**: Always hold out validation data to monitor overfitting  
✅ **Data Cleaning**: Remove corrupted, truncated, or inconsistent trajectories

### Hyperparameter Tuning

#### Learning Rates
- Start with **1e-7 for actor**, **1e-6 for critic**
- If training is unstable → reduce to 1e-8 (actor), 1e-7 (critic)
- If convergence is slow → increase to 1e-6 (actor), 1e-5 (critic)

#### KL Control
- Start with **kl_coef=0.01**, **target_kl=0.05**
- If policy drifts too much → increase kl_coef to 0.02-0.05
- If policy doesn't improve → decrease kl_coef to 0.005

#### Behavior Cloning
- Start with **behavior_cloning_coef=0.1**
- For very conservative training → increase to 0.2-0.5
- For more exploratory training → decrease to 0.05 or 0.0

### Monitoring and Debugging

#### Key Metrics to Watch

```python
# Training metrics
"offline/advantage_mean"     # Should be reasonable (not too extreme)
"offline/advantage_std"      # Should be stable
"offline/policy_entropy"     # Should not collapse too quickly
"actor/reward_kl_penalty"    # Should remain reasonable (< 0.1)

# Validation metrics  
"val-core/*/reward/mean*"    # Primary success metric
"val-core/*/reward/best*"    # Best-case performance
```

#### Warning Signs

🚨 **KL divergence > 0.1**: Policy drifting too far from data  
🚨 **Validation decreasing**: Overfitting to training data  
🚨 **Entropy collapse**: Policy becoming too deterministic  
🚨 **Extreme advantages**: Advantage estimation problems  
🚨 **Training instability**: Learning rates too high

### Algorithm Selection

| Algorithm | Best For | Pros | Cons |
|-----------|----------|------|------|
| **GRPO** | Most offline scenarios | Stable, handles offline data well | May be conservative |
| **Reinforce++** | High-quality data | Simple, effective | Less stable than GRPO |
| **GAE** | Good value estimates | Well-understood | Requires good critic |

## Troubleshooting

### Common Issues and Solutions

#### 1. Training is Unstable

**Symptoms**: Loss spikes, NaN values, extreme gradients

**Solutions**:
- Reduce learning rates by 10x
- Increase KL coefficient
- Reduce batch size
- Add gradient clipping

```yaml
actor_rollout_ref.actor.optim.lr: 1e-8
algorithm.kl_ctrl.kl_coef: 0.02
data.train_batch_size: 32
```

#### 2. Policy Doesn't Improve

**Symptoms**: Validation metrics plateaued, low advantages

**Solutions**:
- Decrease behavior cloning coefficient
- Increase learning rates slightly
- Check data quality
- Switch to less conservative algorithm

```yaml
algorithm.offline_rl.behavior_cloning_coef: 0.05
actor_rollout_ref.actor.optim.lr: 1e-6
algorithm.adv_estimator: reinforce_plus_plus
```

#### 3. Overfitting to Training Data

**Symptoms**: Training improves but validation degrades

**Solutions**:
- Increase regularization
- Reduce training epochs
- Add more validation data
- Increase behavior cloning coefficient

```yaml
actor_rollout_ref.actor.optim.weight_decay: 0.02
trainer.total_epochs: 5
algorithm.offline_rl.behavior_cloning_coef: 0.2
```

#### 4. Memory Issues

**Symptoms**: OOM errors, slow training

**Solutions**:
- Reduce batch sizes
- Enable gradient checkpointing
- Use smaller models
- Reduce sequence lengths

```yaml
data.train_batch_size: 16
actor_rollout_ref.model.enable_gradient_checkpointing: true
data.max_prompt_length: 512
```

#### 5. Data Loading Errors

**Symptoms**: Dataset loading failures, format errors

**Solutions**:
- Validate JSONL format
- Check file paths
- Verify data structure
- Use data_format_example.jsonl as reference

```bash
# Validate JSONL format
python -c "
import json
with open('your_data.jsonl', 'r') as f:
    for i, line in enumerate(f):
        try:
            json.loads(line)
        except:
            print(f'Error in line {i+1}')
"
```

## API Reference

### OfflineRLDataset

The core dataset class for loading offline trajectory data.

```python
from verl.utils.dataset.rl_dataset import OfflineRLDataset

dataset = OfflineRLDataset(
    data_files=["/path/to/data.jsonl"],
    tokenizer=tokenizer,
    max_prompt_length=1024,
    max_response_length=512
)
```

#### Parameters

- `data_files` (str or List[str]): Path(s) to JSONL data files
- `tokenizer`: HuggingFace tokenizer instance  
- `max_prompt_length` (int): Maximum prompt tokens
- `max_response_length` (int): Maximum response tokens

#### Methods

- `__len__()`: Returns number of samples
- `__getitem__(idx)`: Returns processed sample at index
- `_load_file(filepath)`: Loads data from single file
- `_process_record(record, filepath, line_idx)`: Processes single record
- `_extract_reward(record)`: Extracts reward from record
- `_extract_conversation_pairs(trajectory)`: Extracts user-agent pairs
- `_build_sample(user_msg, agent_msg, reward, uid)`: Builds training sample

### Configuration Schema

#### trainer section

```yaml
trainer:
  offline_mode: bool              # Enable offline RL mode
  total_epochs: int               # Number of training epochs  
  test_freq: int                  # Validation frequency
  save_freq: int                  # Checkpoint frequency
  critic_warmup: int              # Critic warmup steps
```

#### data section

```yaml
data:
  offline_train_files: List[str]  # Training data paths
  offline_val_files: List[str]    # Validation data paths (optional)
  train_batch_size: int           # Training batch size
  val_batch_size: int             # Validation batch size
  max_prompt_length: int          # Maximum prompt tokens
  max_response_length: int        # Maximum response tokens
```

#### algorithm section

```yaml
algorithm:
  adv_estimator: str              # grpo, reinforce_plus_plus, gae
  use_kl_in_reward: bool          # Enable KL penalty
  offline_rl:
    behavior_cloning_coef: float  # BC loss coefficient (0.0-1.0)
    importance_sampling_clip: float  # IS clipping threshold
```

### Training Pipeline

The offline RL training pipeline consists of:

1. **Data Loading**: OfflineRLDataset loads and preprocesses JSONL data
2. **Policy Evaluation**: Compute log π_current(a_data | s_data)  
3. **Advantage Computation**: Calculate advantages using offline estimators
4. **Conservative Updates**: Apply PPO updates with KL penalties
5. **Validation**: Evaluate policy on holdout data

### Validation Process

Validation in offline RL mode:

1. Generate responses using current policy on validation prompts
2. Evaluate responses using validation reward function
3. Compute validation metrics (accuracy, reward, success rate)
4. Log sample generations for inspection

## Contributing

We welcome contributions to improve the offline RL implementation:

1. **Bug Reports**: Use GitHub issues to report bugs
2. **Feature Requests**: Suggest new features or improvements
3. **Pull Requests**: Submit code improvements
4. **Documentation**: Help improve documentation and examples

### Development Setup

```bash
# Clone repository
git clone https://github.com/volcengine/verl.git
cd verl

# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/ -v
```

### Code Style

- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include type hints where possible
- Write unit tests for new functionality

## License

This implementation is licensed under the Apache License 2.0. See the [LICENSE](../../LICENSE) file for details.

## Citation

If you use this offline RL implementation in your research, please cite:

```bibtex
@software{verl_offline_rl,
  title={VERL Offline Reinforcement Learning Implementation},
  author={VERL Team},
  year={2024},
  url={https://github.com/volcengine/verl}
}
```

## Support

For questions and support:

- 📖 Check this documentation first
- 🐛 Open GitHub issues for bugs
- 💬 Join community discussions
- 📧 Contact the VERL team

---

**Happy training with offline RL!** 🚀 