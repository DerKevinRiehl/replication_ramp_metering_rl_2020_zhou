# Replication Study of "Ramp Metering for a Distant Downstream Bottleneck Using Reinforcement Learning with Value Function Approximation"

## Authors

**Raphaël Benvenuti**  
Institute for Transport Planning and Systems (IVT), ETH Zürich  
Replication project, Spring Semester 2026

## Introduction

Ramp metering is a traffic management strategy that regulates the flow of vehicles entering a freeway via on-ramps, with the objective of preventing congestion at downstream bottlenecks. When the bottleneck is located far downstream of the metered ramp, conventional linear feedback controllers such as ALINEA and PI-ALINEA suffer from time-delay effects that can cause instability.

Zhou et al. (2020) propose an alternative approach: formulating the ramp metering problem as a Q-learning problem in which an intelligent agent learns a nonlinear feedback policy using an artificial neural network (ANN) as a value function approximator. The learned policy takes as input traffic density measurements at three locations along the freeway stretch, and outputs a discrete metering rate without requiring any explicit traffic prediction.

This repository contains the full replication of that study, implemented from scratch using a Cell Transmission Model (CTM) simulation environment. The replication reproduces the no-control and PI-ALINEA benchmark scenarios, implements the RL agent, and provides an extended quantitative performance comparison that was not present in the original article.

## The Replicated Study

```
Zhou, Y., Ozbay, K., Kachroo, P., & Zuo, F. (2020). Ramp metering for a distant downstream
bottleneck using reinforcement learning with value function approximation. Journal of Advanced
Transportation, 2020, Article 8813467. https://doi.org/10.1155/2020/8813467
```

## What This Repository Includes

```
./
├── 0_original_papers/
│   └── Zhou_et_al_2020.pdf                  # Original paper
├── 0_original_repository/                   # (empty — no code was provided by the authors)
├── 1_code_produce/
│   ├── benchmark_controllers/
│   │   └── benchmark_controllers.py         # No-control, ALINEA, PI-ALINEA simulation,
│   │                                        # grid search optimisation, and results tables
│   └── rl/
│       ├── rl_train.py                      # Train RL agent from scratch
│       ├── rl_resume.py                     # Resume training from a checkpoint
│       ├── rl_resume_bruteforce.py          # Resume with greedy evaluation after every
│       │                                    # episode; saves best-performing ANN
│       ├── rl_scan_checkpoints.py           # Retrospective greedy scan of all saved
│       │                                    # checkpoints; identifies best policy
│       ├── rl_eval.py                       # Load checkpoint, run greedy evaluation,
│       │                                    # noise robustness analysis, generate figures
│       ├── rl_plot_learning_curve.py        # Plot training episode reward and rolling
│       │                                    # standard deviation from episode_rewards.csv
│       ├── rl_plot_learning_curve_bruteforce.py  # Plot deterministic greedy reward from
│       │                                         # greedy_rewards.csv
│       └── rl_fix_reward_csv.py            # Utility: repairs episode_rewards.csv episode
│                                            # numbering after training was resumed
├── 1_data_source/                           # (empty — demand profiles reconstructed from
│                                            # paper figures; no external data required)
├── 2_data_produced/
│   ├── benchmark_controllers/
│   │   ├── table1_full_results.csv          # Performance indicators for all controllers
│   │   │                                    # and demand scenarios
│   │   └── table2_gap.csv                   # Percentage gap from best per indicator
│   └── rl/
│       ├── episode_rewards.csv              # Training episode rewards (700k episodes)
│       ├── greedy_rewards.csv               # Deterministic greedy evaluation rewards
│       └── best_ann.pt                      # Best-performing ANN checkpoint
├── 3_code_visualization/                    # (visualisation is integrated in the
│                                            # production code scripts above)
├── 3_data_visualization/
│   ├── benchmark_controllers/               # Output figures from benchmark_controllers.py
│   └── rl/                                  # Output figures from rl_eval.py
├── requirements.txt
└── README.md
```

## Installation Instructions

Python 3.10 or later is required. Install all dependencies with:

```
pip install -r requirements.txt
```

The `requirements.txt` file contains:

```
numpy
torch
tqdm
scipy
matplotlib
```

No GPU is required. All training was performed on CPU.

## Run Instructions

### Benchmark Controllers

Navigate to `1_code_produce/benchmark_controllers/` and run:

```
python benchmark_controllers.py
```

This will run the no-control simulation, the ALINEA 1D sweep (111 values of K_R), and the PI-ALINEA 2D grid search (111 × 111 combinations of K_R and K_P). Runtime is approximately 20–30 minutes. All results figures are saved to a `pi_alinea_results/` subfolder and the two CSV result tables are saved automatically.

### RL Training

**Starting a new training run from scratch:**

```
python rl_train.py
```

This runs 700,000 training episodes. At the reported training speed of approximately 3.6 episodes/second, the full run takes approximately 55 hours. Checkpoints are saved every 10,000 episodes to a `checkpoints/` subfolder. Training rewards are logged to `checkpoints/episode_rewards.csv`.

**Recommended training strategy** to find the best-performing policy efficiently:

1. Run `rl_train.py` or `rl_resume.py` for 200,000–300,000 episodes first to allow the policy to mature.
2. Switch to `rl_resume_bruteforce.py`, which evaluates the policy greedily after every training episode and saves the best-performing network to `checkpoints/best_ann.pt`. This avoids storing an inferior final checkpoint as the result.

**Controlling the exploration floor:**

In `rl_train.py` and `rl_resume.py`, the `epsilon_end` parameter in the `train_rl_agent` call controls whether exploration is capped:

```python
# With exploration floor (recommended for stability):
epsilon_end=0.05

# Without exploration floor (policy becomes fully deterministic):
epsilon_end=0.0
```

**Resuming from a checkpoint:**

Edit `CHECKPOINT_PATH` at the top of `rl_resume.py` or `rl_resume_bruteforce.py` to point to the desired `.pt` file, then run:

```
python rl_resume.py
```

or

```
python rl_resume_bruteforce.py
```

**Scanning existing checkpoints retrospectively:**

```
python rl_scan_checkpoints.py
```

Set `EP_MIN` and `EP_MAX` at the top of the script to restrict the scan range. The script reports the checkpoint with the highest deterministic greedy reward. This script was used to confirm that the best policy was reached well before episode 700,000 and that no convergence to a stable final policy was observed.

**Evaluating a trained policy:**

```
python rl_eval.py
```

Set `CHECKPOINT_PATH` at the top to point to the desired checkpoint (e.g. `checkpoints/best_ann.pt`). The script runs the greedy policy on the base demand scenario, performs noise robustness evaluation at five noise levels (50, 100, 150, 200, 250 veh/h), prints all performance indicators, and saves all figures to an `eval_results/` subfolder.

**Plotting the learning curve:**

```
python rl_plot_learning_curve.py
```

Reads `episode_rewards.csv` and plots the raw training reward and 1000-episode rolling mean. Note that this plots the **training** episode reward, which includes random actions from ε-greedy exploration and is therefore noisier than the deterministic greedy reward.

```
python rl_plot_learning_curve_bruteforce.py
```

Reads `greedy_rewards.csv` and plots the deterministic greedy evaluation reward over episodes. This reflects true policy quality without exploration noise.

## Replication Notes

The following assumptions and implementation decisions were made during replication, as the original article does not provide code and leaves several parameters unspecified:

**CTM parameters** — The cell length (Δx = 0.5 km) and time step (Δt = 15 s) were identified through trial and error to match the spatio-temporal density contours shown in the original paper. The paper does not state these values explicitly.

**Demand profiles** — The mainline and ramp demand profiles were reconstructed manually from the low-resolution figures in the paper. The reconstructed profiles match the described peak values and durations.

**On-ramp injection cell** — The original paper does not specify which cell receives ramp inflow. The correct cell (cell 4, corresponding to 1.5–2.0 km) was identified by inspecting the spatio-temporal density contours in the original paper.

**Reward scaling constant k** — The paper defines the reward as r = k·|ρ₃ − ρ_des| but does not specify the value of k. The value k = −1.0 was used in this replication.

**Weight initialisation** — The paper states that ANN parameters are initialised with "small random numbers" without specifying a scheme. PyTorch's default Kaiming uniform initialisation, designed for ReLU activations, caused NaN gradients when used with the sigmoid activation specified in the paper. Xavier–Glorot uniform initialisation was used instead, which is the appropriate scheme for sigmoid networks.

**Gradient clipping** — Not mentioned in the paper. Gradient clipping with max norm 1.0 was applied as a numerical stability measure to prevent occasional exploding gradients caused by the combination of sigmoid activation, online SGD, and bootstrapped Q-learning targets.

**ε-decay schedule** — The paper states that ε decays with the number of iterated episodes but does not specify the decay function or rate. An exponential decay factor of 0.99999 per episode was used, reaching approximately 5% after 300,000 episodes. Both capped (ε_min = 0.05) and uncapped (ε_min = 0.0) variants were evaluated.

**Convergence criterion** — The paper states that learning converged after approximately 700,000 episodic iterations but does not specify a convergence criterion. All training campaigns were run for the full 700,000 episodes. No natural convergence was observed; the rolling mean reward plateaued after approximately 200,000–300,000 episodes without further improvement.

**Checkpoint selection** — Since no checkpoint selection rule was reported, all checkpoints were evaluated retrospectively using deterministic greedy action selection. The checkpoint with the highest greedy evaluation reward was selected as the final policy. The best greedy reward of −2062.83 was obtained at episode 263,366.

**Benchmark optimisation** — The original paper benchmarks the RL controller against a PI controller with fixed gains (K_R = 100, K_P = 4). This replication additionally performs a grid search over K_R ∈ [1, 110] and K_P ∈ [1, 110] for PI-ALINEA, and a 1D sweep for ALINEA, optimising for two criteria: average speed without ramp queue and average speed with ramp queue. The optimised PI-ALINEA controller matches the RL controller on episode reward, suggesting the paper's benchmark comparison is not fully fair.

## Citation

Original Paper:
```
Zhou, Y., Ozbay, K., Kachroo, P., & Zuo, F. (2020). Ramp metering for a distant downstream
bottleneck using reinforcement learning with value function approximation. Journal of Advanced
Transportation, 2020, Article 8813467. https://doi.org/10.1155/2020/8813467
```
