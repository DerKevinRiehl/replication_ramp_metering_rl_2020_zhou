# [Re] Ramp Metering for a Distant Downstream Bottleneck Using Reinforcement Learning with Value Function Approximation [Zhou et al. 2020]

## Authors
Raphaël André Maurice Benvenuti [1], Krishna Kanth Vuppala Narasimha [1], Kevin Riehl [1], Anastasios Kouvelas [1], Michail A. Makridis [1]
[1] ETH Zürich, Institute for Transport Planning and Systems, IVT Group, Zürich, Switzerland.

## Introduction

Ramp metering is a traffic management strategy that regulates the flow of vehicles entering a freeway via on-ramps, with the objective of preventing congestion at downstream bottlenecks. When the bottleneck is located far downstream of the metered ramp, conventional linear feedback controllers such as ALINEA and PI-ALINEA can suffer from time-delay effects that can cause instability.

Zhou et al. (2020) propose an alternative approach by formulating the ramp-metering problem as a Q-learning problem. The traffic environment is simulated with a Cell Transmission Model (CTM), while an intelligent agent learns a nonlinear feedback policy using an artificial neural network (ANN) as a value-function approximator. The learned policy takes as input traffic density measurements at three locations along the freeway stretch, as well as the current ramp demand, and outputs a discrete metering rate without requiring any explicit traffic prediction.

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
│   └── Zhou_et_al_2020.pdf                       # Original paper
├── 1_code_produce/
│   ├── benchmark_controllers/
│   │   └── benchmark_controllers.py              # No-control, ALINEA, PI-ALINEA simulation,
│   │                                             # grid search optimisation, and results tables
│   └── rl/
│       ├── rl_train.py                           # Train RL agent from scratch
│       ├── rl_resume.py                          # Resume training from a checkpoint
│       ├── rl_resume_bruteforce.py               # Resume with greedy evaluation after every
│       │                                         # episode; saves best-performing ANN
│       ├── rl_scan_checkpoints.py                # Retrospective greedy scan of all saved
│       │                                         # checkpoints; identifies best policy
│       ├── rl_eval.py                            # Load checkpoint, run greedy evaluation,
│       │                                         # noise robustness analysis, generate figures
│       ├── rl_plot_learning_curve.py             # Plot training episode reward and rolling
│       │                                         # standard deviation from episode_rewards.csv
│       ├── rl_plot_learning_curve_bruteforce.py  # Plot deterministic greedy reward from
│       │                                         # greedy_rewards.csv
│       └── rl_fix_reward_csv.py                  # Utility: repairs episode_rewards.csv episode
│                                                 # numbering after training was resumed
├── 2_data_produced/
│   ├── benchmark_controllers/
│   │   ├── table1_full_results.csv               # Performance indicators for all controllers
│   │   │                                         # and demand scenarios (benchmark only)
│   │   └── table2_pct_gap.csv                    # Percentage gap from best per indicator
│   │                                             # (benchmark only; 0.0% = best)
│   └── rl/
│       ├── episode_rewards.csv                   # Training episode rewards (700k episodes)
│       ├── greedy_rewards.csv                    # Deterministic greedy evaluation rewards
│       ├── ckpt_best_greedy.pt                   # Best-performing ANN checkpoint
│       └── rl_performance_indicators.txt         # Full terminal output of rl_eval.py,
│                                                 # containing all performance indicators
│                                                 # and noise robustness results for the
│                                                 # selected RL policy
├── 3_data_visualization/                         # Output figures are saved automatically
│                                                 # when production scripts are run.
│                                                 # Not uploaded to keep the repository lean.
├── requirements.txt
└── README.md
```

**Note on combined results tables:** The RL controller results were added manually to the benchmark tables for the final comparison presented in the report. The exact values can be extracted from `2_data_produced/rl/rl_performance_indicators.txt`, which contains the complete terminal output of `rl_eval.py` including all performance indicators across base and noisy demand scenarios.

## Installation Instructions

Python 3.10 or later is required. The following packages are needed:

| Package    | Version used | Purpose                              |
|------------|-------------|--------------------------------------|
| Python     | 3.10.10     | Programming language                 |
| NumPy      | 2.2.6       | Numerical computation                |
| PyTorch    | 2.12.0      | Neural network implementation        |
| Matplotlib | 3.10.9      | Visualisation                        |
| SciPy      | 1.15.3      | Signal processing and interpolation  |
| tqdm       | 4.67.3      | Training progress tracking           |

Install all dependencies with:

```
pip install -r requirements.txt
```

## Run Instructions

### Benchmark Controllers

Navigate to `1_code_produce/benchmark_controllers/` and run:

```
python benchmark_controllers.py
```

This will run the no-control simulation, the ALINEA 1D sweep (111 values of K_R), and the PI-ALINEA 2D grid search (111 × 111 combinations of K_R and K_P). Runtime is approximately 1 minute, excluding plot generation. All result figures are saved to a `benchmark_controllers_results/` subfolder and the two CSV result tables are saved automatically.

### RL Training

**Starting a new training run from scratch:**

```
python rl_train.py
```

This runs 700,000 training episodes. At the reported training speed of approximately 3.6 episodes/second, the full run takes approximately 55 hours. Checkpoints are saved every 10,000 episodes to a `checkpoints/` subfolder. Training rewards are logged to `checkpoints/episode_rewards.csv`.

**Recommended training strategy** to find the best-performing policy efficiently:

1. Run `rl_train.py` for 200,000–300,000 episodes first to allow the policy to mature.
2. Switch to `rl_resume_bruteforce.py`, which evaluates the policy greedily after every training episode and saves the best-performing network to `checkpoints/ckpt_best_greedy.pt`. This avoids storing an inferior final checkpoint as the result.

**Controlling the exploration floor:**

In `rl_train.py`, `rl_resume.py`, and `rl_resume_bruteforce.py`, the `epsilon_end` parameter in the training call controls whether exploration is capped:

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

Set `CHECKPOINT_PATH` at the top to point to the desired checkpoint (e.g. `checkpoints/ckpt_best_greedy.pt`). The script runs the greedy policy on the base demand scenario, performs noise robustness evaluation at five noise levels (50, 100, 150, 200, 250 veh/h), prints all performance indicators, and saves all figures to an `eval_results/` subfolder. The terminal output of this script for the selected policy is archived in `2_data_produced/rl/rl_performance_indicators.txt`.

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

The following assumptions and implementation decisions were made during replication, as the original article does not provide code and leaves several parameters unspecified. No original code repository was provided by the authors. No external data is required; all inputs are reconstructed from the paper as described below.

**CTM parameters** — The cell length (Δx = 0.5 km) and time step (Δt = 15 s) were identified through trial and error to match the spatio-temporal density contours shown in the original paper. The paper does not state these values explicitly.

**Demand profiles** — The mainline and ramp demand profiles were reconstructed manually from the low-resolution figures in the paper. The reconstructed profiles match the described peak values and durations.

**On-ramp injection cell** — The original paper does not specify which cell receives ramp inflow. The correct cell (cell 4, corresponding to 1.5–2.0 km) was identified by inspecting the spatio-temporal density contours in the original paper.

**Reward scaling constant k** — The paper defines the reward as r = k·|ρ₃ − ρ_des| but does not specify the value of k. The value k = −1.0 was used in this replication.

**Weight initialisation** — The paper states that ANN parameters are initialised with "small random numbers" without specifying a scheme. PyTorch's default Kaiming uniform initialisation, designed for ReLU activations, caused NaN gradients when used with the sigmoid activation specified in the paper. Xavier–Glorot uniform initialisation was used instead, which is the appropriate scheme for sigmoid networks.

**Gradient clipping** — Not mentioned in the paper. Gradient clipping with max norm 1.0 was applied as a numerical stability measure to prevent occasional exploding gradients caused by the combination of sigmoid activation, online SGD, and bootstrapped Q-learning targets.

**ε-decay schedule** — The paper states that ε decays with the number of iterated episodes but does not specify the decay function or rate. An exponential decay factor of 0.99999 per episode was used, reaching approximately 5% after 300,000 episodes. Both capped (ε_min = 0.05) and uncapped (ε_min = 0.0) variants were evaluated.

**Convergence criterion** — The paper states that learning converged after approximately 700,000 episodic iterations but does not specify a convergence criterion. All training campaigns were run for the full 700,000 episodes. No natural convergence was observed; the rolling mean reward plateaued after approximately 200,000–300,000 episodes without further improvement.

**Policy selection** — Policy selection — Since no checkpoint selection rule was reported in the original article, two complementary strategies were used. First, all periodic checkpoints were scanned retrospectively using rl_scan_checkpoints.py with deterministic greedy evaluation, confirming that the best policy was reached well before episode 700,000 and identifying the approximate episode range of peak performance. Second, targeted training campaigns were run using `rl_resume_bruteforce.py`, which evaluates the policy greedily after every training episode and overwrites `ckpt_best_greedy.pt` whenever a new best is found. The final selected checkpoint, achieving a greedy evaluation reward of −2062.83, was obtained through this bruteforce per-episode evaluation strategy.

**Benchmark optimisation** — The original paper benchmarks the RL controller against a PI controller with fixed gains (K_R = 100, K_P = 4). This replication additionally performs a grid search over K_R ∈ [1, 110] and K_P ∈ [1, 110] for PI-ALINEA, and a 1D sweep for ALINEA, optimising for two criteria: average speed without ramp queue and average speed with ramp queue. The optimised PI-ALINEA controller matches the RL controller on episode reward, suggesting the paper's benchmark comparison is not fully fair.

**Combined results tables** — The benchmark CSV tables (`table1_full_results.csv` and `table2_pct_gap.csv`) contain only the benchmark controller results. RL controller performance indicators were extracted from `rl_performance_indicators.txt` and added manually to the final tables presented in the replication report.

## Citation

Replication Study:
```
Benvenuti, R.A.M., Vuppala Narasimha, K.K., Riehl, K., Kouvelas, A. and Makridis, M.A. (2026). 
[RE] Ramp Metering for a Distant Downstream Bottleneck Using Reinforcement Learning with Value Function Approximation. ReScience C, 202X(X).
DOI: [To be added upon publication]
```

Original Paper:
```
Zhou, Y., Ozbay, K., Kachroo, P., & Zuo, F. (2020). Ramp metering for a distant downstream
bottleneck using reinforcement learning with value function approximation. Journal of Advanced
Transportation, 2020, Article 8813467. https://doi.org/10.1155/2020/8813467
```
