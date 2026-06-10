# 1 - imports

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

# 2 - Parameters

# ── Fundamental diagram (triangular) ──────────────────────────────────────
v_f          = 120           # free-flow speed              [km/h]
rho_cr_lane  = 20            # critical density             [veh/(km·lane)]
rho_jam_lane = 100           # jam density                  [veh/(km·lane)]
Q_lane       = v_f * rho_cr_lane          # capacity/lane = 2400  [veh/h]
w            = Q_lane / (rho_jam_lane - rho_cr_lane)   # back-wave speed [km/h]

# ── Simulation time ────────────────────────────────────────────────────────
t_tot   = 6000               # simulation horizon           [s]
delta_t = 15                 # CTM time step                [s]
T       = delta_t / 3600     # time step                    [h]

# ── Spatial discretisation ─────────────────────────────────────────────────
cell_length = v_f * T        # 0.5 km  (CFL condition satisfied exactly)
l_tot       = 6.0            # total network length         [km]
l_ramp      = 2.0            # on-ramp position             [km]
l_narrow  = 5.5            # lane-drop (bottleneck) pos.  [km]

n_cells = int(round(l_tot / cell_length)) + 1   # cell 0 = upstream boundary

# ── Controller settings ────────────────────────────────────────────────────
rho_des = 13.33              # desired density at control cell [veh/(km·lane)]
r_min   = 200                # minimum metering rate           [veh/h]
r_max   = 1200               # maximum metering rate           [veh/h]
all_actions = np.arange(r_min, r_max + 1, 100)  # possible on-ramp demands [veh/h]
# ── Important cell indices ─────────────────────────────────────────────────
entry_cell    = int(round(l_ramp   / cell_length))  # on-ramp entry cell — note the +1
lanedrop_cell = int(round(l_narrow / cell_length)) + 1   # first 2-lane cell — note the +1
# Control cell = cell just upstream of the lane-drop (Wang et al. 2014, Fig 1, location A)
control_cell  = lanedrop_cell - 1                         # last 3-lane cell
upstream_cell = entry_cell +1
middle_cell   = int((control_cell + upstream_cell) / 2)
# ── Lane count & cell capacities ──────────────────────────────────────────
n_lanes = np.zeros(n_cells)
for i in range(0, lanedrop_cell):
    n_lanes[i] = 3
for i in range(lanedrop_cell, n_cells):
    n_lanes[i] = 2

Q_max   = Q_lane       * n_lanes   # capacity        [veh/h]
rho_jam = rho_jam_lane * n_lanes   # jam density     [veh/km]
rho_cr  = rho_cr_lane  * n_lanes   # critical dens.  [veh/km]


# ── Time vector ────────────────────────────────────────────────────────────
n_t = int(t_tot / delta_t) + 1
t   = np.linspace(0, t_tot, n_t)

# 3 - Demand profiles

# Demand amplitudes (original code values)
d_max = 3000    # mainline peak addition  [veh/h]
x_max = 720     # ramp peak addition      [veh/h]

delta_t_d = 600   # ramp-up / ramp-down duration  [s]
delta_t_x = 600
t_demand  = 1800  # hold duration                 [s]

dx = np.zeros((2, n_t))
dx[0, :] = 1000   # mainline baseline  [veh/h]
dx[1, :] = 500    # ramp baseline      [veh/h]

# Dynamic mainline demand
for i in range(int(600 / delta_t), int((600 + delta_t_d) / delta_t)):
    dx[0, i] += d_max / delta_t_d * (i - int(600 / delta_t)) * delta_t
for i in range(int((600 + delta_t_d) / delta_t), int((600 + t_demand + delta_t_d) / delta_t)):
    dx[0, i] += d_max
for i in range(int((600 + t_demand + delta_t_d) / delta_t),
               int((600 + t_demand + 2 * delta_t_d) / delta_t)):
    dx[0, i] += d_max - (d_max / delta_t_d * (i - int((600 + t_demand + delta_t_d) / delta_t)) * delta_t)

# Dynamic ramp demand
for i in range(int(600 / delta_t), int((600 + delta_t_x) / delta_t)):
    dx[1, i] += x_max / delta_t_x * (i - int(600 / delta_t)) * delta_t
for i in range(int((600 + delta_t_x) / delta_t), int((600 + t_demand + delta_t_x) / delta_t)):
    dx[1, i] += x_max
for i in range(int((600 + t_demand + delta_t_x) / delta_t),
               int((600 + t_demand + 2 * delta_t_x) / delta_t)):
    dx[1, i] += x_max - (x_max / delta_t_x * (i - int((600 + t_demand + delta_t_x) / delta_t)) * delta_t)

d_mainline = dx[0]
d_ramp     = dx[1]

print(f"Mainline demand range: [{d_mainline.min():.0f}, {d_mainline.max():.0f}] veh/h")
print(f"Ramp demand range    : [{d_ramp.min():.0f}, {d_ramp.max():.0f}] veh/h")

# 4 - CTM functions

def initialize_sim_state(n_cells, d_ramp0, r_min, r_max):
    return {
        "rho": np.zeros(n_cells),                 # density at current time step
        "q": np.zeros(n_cells),                   # outflow at current time step
        "r_in": 0.0,                              # actual ramp inflow
        "N1": 0.0,                                # ramp queue
        "N2": 0.0,                                # mainline upstream queue
        "r_meter": np.clip(d_ramp0, r_min, r_max) # current metering command
    }

def compute_ramp_demand(N1, prev_ramp_arrival, T_hours):
    return N1 / T_hours + prev_ramp_arrival

def send(rho_i, j):
    """CTM sending function (triangular FD)."""
    return min(v_f * rho_i, Q_max[j])

def recv(rho_next, j_next):
    """CTM receiving function."""
    return min(Q_max[j_next], w * (rho_jam[j_next] - rho_next))

def ctm_step(sim_state, ramp_meter_rate, d_main_k, d_ramp_k):
    """
    Advance the CTM by one time step using a given metering command.

    Parameters
    ----------
    sim_state : dict
        Current simulation state with keys:
            "rho"     : np.ndarray, shape (n_cells,)
            "q"       : np.ndarray, shape (n_cells,)
            "r_in"    : float
            "N1"      : float
            "N2"      : float
            "r_meter" : float
    ramp_meter_rate : float
        Metering command to apply at this step [veh/h].
    d_main_k : float
        Mainline demand at current step [veh/h].
    d_ramp_k : float
        Ramp demand at current step [veh/h].

    Returns
    -------
    next_state : dict
        Updated simulation state after one CTM step.
    """

    rho = sim_state["rho"].copy()
    N1  = sim_state["N1"]
    N2  = sim_state["N2"]

    q = np.zeros(n_cells)
    rho_next = rho.copy()

    # ---- Ramp demand including queue ----
    # Assumes T is in hours. If T is in seconds, replace N1 / T by N1 / (T / 3600).
    ramp_avail = d_ramp_k + N1 / T
    r_cmd = min(ramp_avail, ramp_meter_rate)
    r_cmd = max(0.0, r_cmd)

    r_in = 0.0

    # ---- Upstream boundary ----
    q[0] = min(d_main_k + N2 / T, recv(rho[1], 1))

    # ---- Interior cells ----
    for j in range(1, n_cells - 1):
        s_j  = send(rho[j], j)
        r_j1 = recv(rho[j + 1], j + 1)

        if j == entry_cell - 1:
            # Merge of upstream mainline flow and ramp inflow
            total = s_j + r_cmd

            if total <= r_j1:
                q[j] = s_j
                r_in = r_cmd
            else:
                if total > 0:
                    q[j]  = s_j  / total * r_j1
                    r_in  = r_cmd / total * r_j1
                else:
                    q[j] = 0.0
                    r_in = 0.0
        else:
            q[j] = min(s_j, r_j1)

    # ---- Last cell: free outflow ----
    q[n_cells - 1] = send(rho[n_cells - 1], n_cells - 1)

    # ---- Queue updates ----
    N1_next = max(0.0, N1 + (d_ramp_k - r_in) * T)
    N2_next = max(0.0, N2 + (d_main_k - q[0]) * T)

    # ---- Density updates ----
    # Cell 1
    rho_next[1] = max(
        0.0,
        rho[1] + T / cell_length * (q[0] - q[1])
    )

    # Cells 2 ... n_cells-1
    for j in range(2, n_cells):
        inflow = q[j - 1] + (r_in if j == entry_cell else 0.0)
        rho_next[j] = max(
            0.0,
            rho[j] + T / cell_length * (inflow - q[j])
        )

    next_state = {
        "rho": rho_next,
        "q": q,
        "r_in": r_in,
        "N1": N1_next,
        "N2": N2_next,
        "r_meter": ramp_meter_rate
    }

    return next_state

# 5 - RL and ANN functions

sample_cells = (upstream_cell, middle_cell, control_cell)

def extract_rl_state(sim_state, sample_cells, prev_ramp_arrival):
    """
    RL state = [rho1, rho2, rho3, Dramp]
    densities are per-lane
    """
    rho = sim_state["rho"]

    rho1 = rho[sample_cells[0]] / n_lanes[sample_cells[0]]
    rho2 = rho[sample_cells[1]] / n_lanes[sample_cells[1]]
    rho3 = rho[sample_cells[2]] / n_lanes[sample_cells[2]]

    Dramp = compute_ramp_demand(sim_state["N1"], prev_ramp_arrival, T)

    return np.array([rho1, rho2, rho3, Dramp], dtype=float)

all_actions = np.arange(200.0, 1201.0, 100.0)

#---------------------- control later
def get_admissible_actions(Dramp):
    idx = np.where(all_actions <= Dramp)[0]
    if len(idx) == 0:
        idx = np.array([0], dtype=int)
    return idx.astype(int)
#----------------------
def compute_reward(state, k_reward=-1.0):
    rho3 = state[2]
    return k_reward * abs(rho3 - rho_des)

def one_hot_bin(value, bin_edges):
    """
    Return one-hot encoding of the interval containing 'value'.
    """
    idx = np.searchsorted(bin_edges, value, side="right") - 1
    idx = np.clip(idx, 0, len(bin_edges) - 2)

    out = np.zeros(len(bin_edges) - 1, dtype=np.float32)
    out[idx] = 1.0
    return out

# Predefine bin edges once
rho_bin_edges = np.linspace(0.0, rho_jam_lane, 41)   # 40 bins
dramp_bin_edges = np.concatenate([
    np.linspace(0.0, 1200.0, 20),   # 19 intervals up to amax
    [np.inf]
])

def encode_state(state):
    """
    state = [rho1, rho2, rho3, Dramp]
    output length = 40 + 40 + 40 + 20 = 140
    """
    rho1, rho2, rho3, Dramp = state

    x1 = one_hot_bin(rho1, rho_bin_edges)
    x2 = one_hot_bin(rho2, rho_bin_edges)
    x3 = one_hot_bin(rho3, rho_bin_edges)
    x4 = one_hot_bin(Dramp, dramp_bin_edges)

    x = np.concatenate([x1, x2, x3, x4]).astype(np.float32)
    return x

def select_action_epsilon_greedy(state, epsilon, q_net):
    """
    Return:
        action_value : selected metering rate [veh/h]
        action_idx   : integer index in all_actions
    """
    Dramp = state[3]
    admissible_idx = np.asarray(get_admissible_actions(Dramp), dtype=int)

    if np.random.rand() < epsilon:
        action_idx = int(np.random.choice(admissible_idx))
    else:
        x = encode_state(state)
        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            q_values = q_net(x_t).squeeze(0).cpu().numpy()

        best_local = int(np.argmax(q_values[admissible_idx]))
        action_idx = int(admissible_idx[best_local])

    return float(all_actions[action_idx]), action_idx

def train_step(q_net, optimizer, loss_fn, state, action_idx, reward, next_state, gamma=0.95):
    """
    One online Q-learning update using the ANN approximator.
    """
    x = encode_state(state)
    x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

    x_next = encode_state(next_state)
    x_next_t = torch.tensor(x_next, dtype=torch.float32).unsqueeze(0)

    q_values = q_net(x_t)
    q_sa = q_values[0, action_idx]

    admissible_next_idx = get_admissible_actions(next_state[3])

    with torch.no_grad():
        q_next_values = q_net(x_next_t)[0]
        max_q_next = torch.max(q_next_values[admissible_next_idx])
        td_target = torch.tensor(reward, dtype=torch.float32) + gamma * max_q_next

    loss = loss_fn(q_sa, td_target)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0)  # [CHANGED] prevent gradient explosion with sigmoid+SGD
    optimizer.step()

    return loss.item(), td_target.item()

class QNetwork(nn.Module):
    def __init__(self, n_features=140, n_hidden=420, n_actions=11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.Sigmoid(),
            nn.Linear(n_hidden, n_actions)
        )
        # Xavier initialisation for sigmoid activation
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)
    
# lr_threshold: number of individual time-step updates at which α drops from 0.05 → 0.01
# Each episode has (n_t - 1) steps; paper converges ~0.7 M total steps.
lr_high      = 0.05    # α for first 100 000 steps  (paper §4.1)
lr_low       = 0.01    # α afterwards               (paper §4.1)
lr_threshold = 100_000  # episode count at which α switches


def save_checkpoint(path, q_net, optimizer, epsilon, total_steps, current_episode, episode_rewards, episode_losses):
    torch.save({
        "model_state":     q_net.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epsilon":         epsilon,
        "total_steps":     total_steps,
        "current_episode": current_episode,
        "episode_rewards": episode_rewards,
        "episode_losses":  episode_losses
    }, path)
    print(f"  → checkpoint saved: {path}")

def load_checkpoint(path, q_net, optimizer):
    ckpt = torch.load(path)
    q_net.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return (
        ckpt["epsilon"],
        ckpt["total_steps"],
        ckpt["current_episode"],
        ckpt["episode_rewards"],
        ckpt["episode_losses"]
    )

def train_rl_agent(
    q_net, optimizer, loss_fn,
    n_episodes, d_mainline, d_ramp, sample_cells,
    gamma=0.95,
    epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.99999,
    k_reward=-1.0,
    lr_threshold=100_000, lr_low=0.01,
    total_steps=0, start_episode=0,
    checkpoint_dir="checkpoints"  # [NEW] folder to write .pt files into
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    reward_log_path = os.path.join(checkpoint_dir, "episode_rewards.csv")
    reward_file = open(reward_log_path, "w")
    reward_file.write("episode,reward\n")

    episode_rewards = []
    episode_losses  = []
    epsilon = epsilon_start
    lr_switched = (start_episode >= lr_threshold)

    for ep in tqdm(range(n_episodes), desc="Training", unit="ep"):
        current_episode = start_episode + ep

        if not lr_switched and current_episode >= lr_threshold:
            optimizer.param_groups[0]['lr'] = lr_low
            lr_switched = True

        sim_state = initialize_sim_state(n_cells, d_ramp[0], r_min, r_max)
        prev_ramp_arrival = d_ramp[0]
        total_reward = 0.0
        total_loss   = 0.0

        for k in range(len(d_mainline) - 1):
            state = extract_rl_state(sim_state, sample_cells, prev_ramp_arrival)
            action_value, action_idx = select_action_epsilon_greedy(state, epsilon, q_net)

            next_sim_state = ctm_step(
                sim_state=sim_state,
                ramp_meter_rate=action_value,
                d_main_k=d_mainline[k],
                d_ramp_k=d_ramp[k]
            )

            next_state = extract_rl_state(next_sim_state, sample_cells, sim_state["r_in"])
            reward     = compute_reward(next_state, k_reward=k_reward)

            loss_value, td_target = train_step(
                q_net=q_net, optimizer=optimizer, loss_fn=loss_fn,
                state=state, action_idx=action_idx,
                reward=reward, next_state=next_state, gamma=gamma
            )

            total_reward += reward
            total_loss   += loss_value
            total_steps  += 1
            prev_ramp_arrival = sim_state["r_in"]
            sim_state = next_sim_state

        episode_rewards.append(total_reward)
        episode_losses.append(total_loss)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        reward_file.write(f"{current_episode + 1},{total_reward:.4f}\n")
        reward_file.flush()  # ensures data is written even if job is killed mid-run

        # [NEW] save checkpoint every 10 000 episodes (overwrites same file each time)
        if (current_episode + 1) % 10_000 == 0:
            fname = os.path.join(checkpoint_dir, f"ckpt_ep{current_episode + 1}.pt")
            save_checkpoint(fname, q_net, optimizer, epsilon, total_steps, current_episode + 1, episode_rewards, episode_losses)

    reward_file.close()
    
    return episode_rewards, episode_losses, total_steps, start_episode + n_episodes

q_net     = QNetwork(n_features=140, n_hidden=420, n_actions=len(all_actions))
optimizer = optim.SGD(q_net.parameters(), lr=lr_high)
loss_fn   = nn.MSELoss()

episode_rewards, episode_losses, total_steps, current_episode = train_rl_agent(
    q_net=q_net,
    optimizer=optimizer,
    loss_fn=loss_fn,
    n_episodes=700000,
    d_mainline=d_mainline,
    d_ramp=d_ramp,
    sample_cells=sample_cells,
    gamma=0.95,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.99999,
    k_reward=-1.0,
    lr_threshold=lr_threshold,
    lr_low=lr_low,
    total_steps=0,
    start_episode=0,
    checkpoint_dir="checkpoints"
)

