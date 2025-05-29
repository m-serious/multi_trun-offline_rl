# Multi-Turn Offline Reinforcement Learning (multi_turn-offline_rl)

👋 Hi, everyone!

This repository, `multi_turn-offline_rl`, is a project focused on implementing and experimenting with multi-turn offline reinforcement learning algorithms for Large Language Models (LLMs).

## Project Goal

The primary goal of this project is to adapt and extend existing reinforcement learning frameworks (initially based on `verl`) to effectively train LLM agents using pre-collected, static datasets of multi-turn interactions. This involves:

*   Developing methods to process and load offline multi-turn conversational data.
*   Implementing or modifying RL algorithms (like PPO, GRPO) to learn from these offline datasets.
*   Focusing on how to correctly handle rewards, advantages, and policy updates in an offline, multi-turn setting.
*   Exploring challenges and solutions specific to offline RL for dialogue agents, such as handling distribution shift and ensuring effective learning from fixed trajectories.

## Current Status

This project is currently under development. The initial codebase is an adaptation of the `verl` library, with modifications to support offline data loading and training loops.

## Key Areas of Modification (from original verl)

*   **Data Loading**: Introducing new dataset classes and processing logic to handle offline multi-turn interaction data (e.g., from JSON/JSONL files).
*   **Training Loop**: Modifying the main PPO training loop (`fit` method in `RayPPOTrainer`) to consume offline batches instead of performing online rollouts for training data generation.
*   **Multi-Turn Handling**: Ensuring that `loss_mask` and other multi-turn specific mechanisms are correctly derived from the offline data to guide the learning process.
*   **Configuration**: Adapting configuration files and parameters to support offline training modes and specify offline data sources.

## Future Work (Ideas)

*   Implementation of various offline RL algorithms (e.g., CQL, IQL, TD3+BC).
*   Advanced techniques for `loss_mask` generation from complex multi-turn data.
*   Benchmarking performance on standard offline RL datasets for dialogue.
*   Investigating the impact of data quality and composition on offline RL performance.

## Getting Started

*(This section will be updated as the project matures. For now, refer to the scripts and configurations within the repository to understand how to run experiments.)*

## Contributions

Contributions are welcome! If you are interested in multi-turn offline RL for LLMs, feel free to fork the repository, experiment, and open pull requests.
