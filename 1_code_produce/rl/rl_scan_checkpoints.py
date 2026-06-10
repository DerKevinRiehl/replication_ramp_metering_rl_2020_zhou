# ════════════════════════════════════════════════════════════════════════════
# RL Controller Evaluation Script
# Loads a trained checkpoint, runs the policy, plots results, prints KPIs
# ════════════════════════════════════════════════════════════════════════════

# ── 1. Imports ───────────────────────────────────────────────────────────────
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

# ── 2. Parameters ────────────────────────────────────────────────────────────
v_f          = 120
rho_cr_lane  = 20
rho_jam_lane = 100
Q_lane       = v_f * rho_cr_lane
w            = Q_lane / (rho_jam_lane - rho_cr_lane)

t_tot   = 6000
delta_t = 15
T       = delta_t / 3600

cell_length = v_f * T
l_tot       = 6.0
l_ramp      = 2.0
l_narrow    = 5.5

n_cells = int(round(l_tot / cell_length)) + 1

rho_des = 13.33
r_min   = 200
r_max   = 1200
all_actions = np.arange(r_min, r_max + 1, 100)

entry_cell    = int(round(l_ramp   / cell_length))
lanedrop_cell = int(round(l_narrow / cell_length)) + 1
control_cell  = lanedrop_cell - 1
upstream_cell = entry_cell + 1
middle_cell   = int((control_cell + upstream_cell) / 2)

n_lanes = np.zeros(n_cells)
for i in range(0, lanedrop_cell):
    n_lanes[i] = 3
for i in range(lanedrop_cell, n_cells):
    n_lanes[i] = 2

Q_max   = Q_lane       * n_lanes
rho_jam = rho_jam_lane * n_lanes
rho_cr  = rho_cr_lane  * n_lanes

n_t = int(t_tot / delta_t) + 1
t   = np.linspace(0, t_tot, n_t)

print(f"Cell length      : {cell_length} km")
print(f"Number of cells  : {n_cells}")
print(f"Entry cell       : {entry_cell}  @ {(entry_cell-1)*cell_length:.1f}-{entry_cell*cell_length:.1f} km")
print(f"Lane-drop cell   : {lanedrop_cell} @ {(lanedrop_cell-1)*cell_length:.1f}-{lanedrop_cell*cell_length:.1f} km")
print(f"Control cell     : {control_cell} @ {(control_cell-1)*cell_length:.1f}-{control_cell*cell_length:.1f} km")

# ── 3. Demand Profiles ───────────────────────────────────────────────────────
d_max = 3000
x_max = 720
delta_t_d = 600
delta_t_x = 600
t_demand  = 1800

dx = np.zeros((2, n_t))
dx[0, :] = 1000
dx[1, :] = 500

for i in range(int(600 / delta_t), int((600 + delta_t_d) / delta_t)):
    dx[0, i] += d_max / delta_t_d * (i - int(600 / delta_t)) * delta_t
for i in range(int((600 + delta_t_d) / delta_t), int((600 + t_demand + delta_t_d) / delta_t)):
    dx[0, i] += d_max
for i in range(int((600 + t_demand + delta_t_d) / delta_t),
               int((600 + t_demand + 2 * delta_t_d) / delta_t)):
    dx[0, i] += d_max - (d_max / delta_t_d * (i - int((600 + t_demand + delta_t_d) / delta_t)) * delta_t)

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

# ── 4. CTM Functions ─────────────────────────────────────────────────────────
def initialize_sim_state(n_cells, d_ramp0, r_min, r_max):
    return {
        "rho":     np.zeros(n_cells),
        "q":       np.zeros(n_cells),
        "r_in":    0.0,
        "N1":      0.0,
        "N2":      0.0,
        "r_meter": np.clip(d_ramp0, r_min, r_max)
    }

def compute_ramp_demand(N1, prev_ramp_arrival, T_hours):
    return N1 / T_hours + prev_ramp_arrival

def send(rho_i, j):
    return min(v_f * rho_i, Q_max[j])

def recv(rho_next, j_next):
    return min(Q_max[j_next], w * (rho_jam[j_next] - rho_next))

def ctm_step(sim_state, ramp_meter_rate, d_main_k, d_ramp_k):
    rho = sim_state["rho"].copy()
    N1  = sim_state["N1"]
    N2  = sim_state["N2"]

    q = np.zeros(n_cells)
    rho_next = rho.copy()

    ramp_avail = d_ramp_k + N1 / T
    r_cmd = min(ramp_avail, ramp_meter_rate)
    r_cmd = max(0.0, r_cmd)
    r_in  = 0.0

    q[0] = min(d_main_k + N2 / T, recv(rho[1], 1))

    for j in range(1, n_cells - 1):
        s_j  = send(rho[j], j)
        r_j1 = recv(rho[j + 1], j + 1)

        if j == entry_cell - 1:
            total = s_j + r_cmd
            if total <= r_j1:
                q[j] = s_j
                r_in = r_cmd
            else:
                if total > 0:
                    q[j] = s_j  / total * r_j1
                    r_in = r_cmd / total * r_j1
                else:
                    q[j] = 0.0
                    r_in = 0.0
        else:
            q[j] = min(s_j, r_j1)

    q[n_cells - 1] = send(rho[n_cells - 1], n_cells - 1)

    N1_next = max(0.0, N1 + (d_ramp_k - r_in) * T)
    N2_next = max(0.0, N2 + (d_main_k - q[0]) * T)

    rho_next[1] = max(0.0, rho[1] + T / cell_length * (q[0] - q[1]))

    for j in range(2, n_cells):
        inflow = q[j - 1] + (r_in if j == entry_cell else 0.0)
        rho_next[j] = max(0.0, rho[j] + T / cell_length * (inflow - q[j]))

    return {
        "rho":     rho_next,
        "q":       q,
        "r_in":    r_in,
        "N1":      N1_next,
        "N2":      N2_next,
        "r_meter": ramp_meter_rate
    }

# ── 5. RL and ANN Functions ──────────────────────────────────────────────────
sample_cells = (upstream_cell, middle_cell, control_cell)

def extract_rl_state(sim_state, sample_cells, prev_ramp_arrival):
    rho  = sim_state["rho"]
    rho1 = rho[sample_cells[0]] / n_lanes[sample_cells[0]]
    rho2 = rho[sample_cells[1]] / n_lanes[sample_cells[1]]
    rho3 = rho[sample_cells[2]] / n_lanes[sample_cells[2]]
    Dramp = compute_ramp_demand(sim_state["N1"], prev_ramp_arrival, T)
    return np.array([rho1, rho2, rho3, Dramp], dtype=float)

def get_admissible_actions(Dramp):
    idx = np.where(all_actions <= Dramp)[0]
    if len(idx) == 0:
        idx = np.array([0], dtype=int)
    return idx.astype(int)

def one_hot_bin(value, bin_edges):
    idx = np.searchsorted(bin_edges, value, side="right") - 1
    idx = np.clip(idx, 0, len(bin_edges) - 2)
    out = np.zeros(len(bin_edges) - 1, dtype=np.float32)
    out[idx] = 1.0
    return out

rho_bin_edges = np.linspace(0.0, rho_jam_lane, 41)
dramp_bin_edges = np.concatenate([
    np.linspace(0.0, 1200.0, 20),
    [np.inf]
])

def encode_state(state):
    rho1, rho2, rho3, Dramp = state
    x1 = one_hot_bin(rho1, rho_bin_edges)
    x2 = one_hot_bin(rho2, rho_bin_edges)
    x3 = one_hot_bin(rho3, rho_bin_edges)
    x4 = one_hot_bin(Dramp, dramp_bin_edges)
    return np.concatenate([x1, x2, x3, x4]).astype(np.float32)

def select_action_epsilon_greedy(state, epsilon, q_net):
    Dramp = state[3]
    admissible_idx = np.asarray(get_admissible_actions(Dramp), dtype=int)
    if np.random.rand() < epsilon:
        action_idx = int(np.random.choice(admissible_idx))
    else:
        x   = encode_state(state)
        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = q_net(x_t).squeeze(0).cpu().numpy()
        best_local = int(np.argmax(q_values[admissible_idx]))
        action_idx = int(admissible_idx[best_local])
    return float(all_actions[action_idx]), action_idx

class QNetwork(nn.Module):
    def __init__(self, n_features=140, n_hidden=420, n_actions=11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.Sigmoid(),
            nn.Linear(n_hidden, n_actions)
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)

def load_checkpoint(path, q_net, optimizer):
    ckpt = torch.load(path, weights_only=False)
    q_net.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return (
        ckpt["epsilon"],
        ckpt["total_steps"],
        ckpt["current_episode"],
        ckpt["episode_rewards"],
        ckpt["episode_losses"]
    )

def run_trained_policy(q_net, d_mainline, d_ramp, sample_cells):
    n_t = len(d_mainline)

    rho_hist     = np.zeros((n_t, n_cells))
    q_hist       = np.zeros((n_t, n_cells))
    N1_hist      = np.zeros(n_t)
    N2_hist      = np.zeros(n_t)
    r_in_hist    = np.zeros(n_t)
    r_meter_hist = np.zeros(n_t)

    sim_state = initialize_sim_state(n_cells, d_ramp[0], r_min, r_max)
    prev_ramp_arrival = d_ramp[0]

    rho_hist[0]     = sim_state["rho"]
    q_hist[0]       = sim_state["q"]
    N1_hist[0]      = sim_state["N1"]
    N2_hist[0]      = sim_state["N2"]
    r_in_hist[0]    = sim_state["r_in"]
    r_meter_hist[0] = sim_state["r_meter"]

    for k in range(n_t - 1):
        state = extract_rl_state(sim_state, sample_cells, prev_ramp_arrival)
        action_value, _ = select_action_epsilon_greedy(state, epsilon=0.0, q_net=q_net)

        next_sim_state = ctm_step(
            sim_state=sim_state,
            ramp_meter_rate=action_value,
            d_main_k=d_mainline[k],
            d_ramp_k=d_ramp[k]
        )

        rho_hist[k + 1]     = next_sim_state["rho"]
        q_hist[k + 1]       = next_sim_state["q"]
        N1_hist[k + 1]      = next_sim_state["N1"]
        N2_hist[k + 1]      = next_sim_state["N2"]
        r_in_hist[k + 1]    = next_sim_state["r_in"]
        r_meter_hist[k + 1] = next_sim_state["r_meter"]

        prev_ramp_arrival = sim_state["r_in"]
        sim_state = next_sim_state

    return {
        "rho":     rho_hist,
        "q":       q_hist,
        "N1":      N1_hist,
        "N2":      N2_hist,
        "r_in":    r_in_hist,
        "r_meter": r_meter_hist
    }

# ── 6. Plotting Functions ─────────────────────────────────────────────────────
def rho_per_lane(rho_arr):
    return rho_arr / n_lanes[np.newaxis, :]

def plot_scenario(res, title, save_path=None):
    """
    Three separate figures:
      1) Control-cell density time series
      2) Smoothed space-time density contour
      3) Non-smoothed space-time density diagram
    save_path, if provided, is used as a base: suffixes _density, _smoothed,
    _raw are appended before the extension.
    """
    rpl      = rho_per_lane(res['rho'])
    rho_ctrl = rpl[:, control_cell]
    norm     = mpl.colors.Normalize(vmin=0, vmax=50)

    # Helper to derive per-plot save paths
    def _spath(suffix):
        if save_path is None:
            return None
        base, ext = os.path.splitext(save_path)
        return f"{base}_{suffix}{ext}"

    # ── 1) Control-cell density time series ──────────────────────────────────
    fig1, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, rho_ctrl, linewidth=1.2, label='Actual value')
    ax.axhline(rho_des, color='saddlebrown', linewidth=1.0,
               linestyle=(0, (3, 3)), label='Desired value')
    ax.set_xlim(0, t_tot)
    ax.set_ylim(0, 50)
    ax.set_xlabel('Sec', fontsize=12)
    ax.set_ylabel('Veh/km/ln', fontsize=12)
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.tick_params(direction='in', top=True, right=True)
    ax.legend(loc='upper left', fontsize=10, frameon=False)
    ax.set_title(f"{title} — Control-cell density", fontsize=12)
    plt.tight_layout()
    sp = _spath("density")
    if sp:
        plt.savefig(sp, dpi=150)
        print(f"  → saved: {sp}")
    plt.show()

    # ── 2) Smoothed space-time density contour ────────────────────────────────
    Z      = rpl[:, 1:].T
    n_phys = Z.shape[0]
    y_raw  = (np.arange(n_phys) + 0.5) * cell_length * 1000

    interp = RegularGridInterpolator(
        (y_raw, t), Z, bounds_error=False, fill_value=None
    )
    x_fine = np.linspace(0, t_tot, 600)
    y_fine = np.linspace(y_raw[0], y_raw[-1], 400)
    Xf, Yf = np.meshgrid(x_fine, y_fine)
    Z_fine  = interp(np.stack([Yf.ravel(), Xf.ravel()], axis=-1)).reshape(len(y_fine), len(x_fine))
    Z_fine  = gaussian_filter(np.clip(Z_fine, 0, 50), sigma=(2, 4))

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    cf    = ax2.contourf(Xf, Yf, Z_fine, levels=np.linspace(0, 50, 21), cmap='turbo', norm=norm)
    cbar2 = fig2.colorbar(cf, ax=ax2, ticks=np.arange(0, 51, 10))
    cbar2.ax.set_title('[veh/km/ln]', size=9)
    ax2.axhline(l_narrow * 1000, color='k', lw=1.2, ls='--', alpha=0.7, label='Lane drop')
    ax2.set_xlim(0, t_tot)
    ax2.set_ylim(y_fine[0], y_fine[-1])
    ax2.set_xlabel('Sec', fontsize=12)
    ax2.set_ylabel('Position [m]', fontsize=12)
    ax2.xaxis.set_major_locator(MultipleLocator(1000))
    ax2.tick_params(direction='in', top=True, right=True)
    ax2.set_title(f"{title} — Smoothed space-time density", fontsize=12)
    plt.tight_layout()
    sp = _spath("smoothed")
    if sp:
        plt.savefig(sp, dpi=150)
        print(f"  → saved: {sp}")
    plt.show()

    # ── 3) Non-smoothed space-time density ───────────────────────────────────
    Z_raw      = rpl[:, 1:n_cells].T
    n_phys_raw = Z_raw.shape[0]
    y_edges_m  = np.arange(n_phys_raw + 1) * cell_length * 1000
    x_edges_s  = np.arange(n_t + 1) * delta_t

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    cp    = ax3.pcolormesh(x_edges_s, y_edges_m, Z_raw, cmap='turbo', norm=norm, shading='flat')
    cbar3 = fig3.colorbar(cp, ax=ax3, ticks=np.arange(0, 51, 5))
    cbar3.ax.set_title('[veh/km/ln]', size=9)
    ax3.axhline(l_narrow * 1000, linestyle='--', linewidth=1.2, color='k', alpha=0.7)
    ax3.set_xlim(0, t_tot)
    ax3.set_ylim(0, n_phys_raw * cell_length * 1000)
    ax3.set_xlabel('Time [s]', fontsize=12)
    ax3.set_ylabel('Position [m]', fontsize=12)
    ax3.xaxis.set_major_locator(MultipleLocator(1000))
    ax3.tick_params(direction='in', top=True, right=True)
    ax3.grid(linestyle='--', linewidth=0.3, alpha=0.3)
    ax3.set_title(f"{title} — Non-smoothed space-time density", fontsize=12)
    plt.tight_layout()
    sp = _spath("raw")
    if sp:
        plt.savefig(sp, dpi=150)
        print(f"  → saved: {sp}")
    plt.show()

def plot_metering_rate(res, t, t_tot, title="Metering rate", save_path=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, res['r_meter'], linewidth=1.0)
    ax.set_xlim(0, t_tot)
    ax.set_ylim(0, 1400)
    ax.set_xlabel('Sec', fontsize=11)
    ax.set_ylabel('Veh/h', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.yaxis.set_major_locator(MultipleLocator(200))
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.tick_params(direction='in', top=True, right=True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  → saved: {save_path}")
    plt.show()

def plot_ramp_queue(res, t, t_tot, title="On-ramp queue", save_path=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, res['N1'], linewidth=1.2)
    ax.set_xlim(0, t_tot)
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel('Queue length [veh]', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.tick_params(direction='in', top=True, right=True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  → saved: {save_path}")
    plt.show()

def print_performance_indicators(res, label):
    rho_arr = res["rho"]
    q_arr   = res["q"]
    N1_arr  = res["N1"]

    VKT              = np.sum(q_arr[:, 1:] * cell_length * T)
    VHT              = np.sum(rho_arr[:, 0:] * cell_length * T)
    VHT_with_ramp    = VHT + np.sum(N1_arr * T)
    v_avg            = VKT / VHT          if VHT > 0          else 0
    v_avg_with_ramp  = VKT / VHT_with_ramp if VHT_with_ramp > 0 else 0

    print(f"\n── Performance indicators: {label} ──")
    print(f"  VKT                : {VKT:.1f} veh·km")
    print(f"  VHT                : {VHT:.4f} veh·h")
    print(f"  VHT (with ramp)    : {VHT_with_ramp:.4f} veh·h")
    print(f"  v_avg              : {v_avg:.2f} km/h")
    print(f"  v_avg (with ramp)  : {v_avg_with_ramp:.2f} km/h")


# ════════════════════════════════════════════════════════════════════════════
# CHECKPOINT SCANNER
# Loads every checkpoint in ./checkpoints/ between EP_MIN and EP_MAX,
# runs the trained policy greedily (epsilon=0), computes episode reward,
# and reports the best checkpoint found.
# ════════════════════════════════════════════════════════════════════════════
import glob
import re

# ── Settings — adapt as needed ───────────────────────────────────────────────
EP_MIN      = 200_000   # skip checkpoints before this episode
EP_MAX      = 700_000   # skip checkpoints after this episode
CKPT_DIR    = "checkpoints"

# ── Discover checkpoint files ────────────────────────────────────────────────
pattern     = os.path.join(CKPT_DIR, "ckpt_ep*.pt")
all_files   = sorted(glob.glob(pattern))

# filter by episode range
def extract_ep(path):
    m = re.search(r"ckpt_ep(\d+)\.pt", path)
    return int(m.group(1)) if m else -1

files_in_range = [f for f in all_files if EP_MIN <= extract_ep(f) <= EP_MAX]

if not files_in_range:
    print(f"No checkpoints found in {CKPT_DIR} between ep {EP_MIN} and ep {EP_MAX}.")
else:
    print(f"Found {len(files_in_range)} checkpoints to evaluate "
          f"(ep {extract_ep(files_in_range[0])} → ep {extract_ep(files_in_range[-1])})")
    print()

    best_reward  = -np.inf
    best_file    = None
    best_episode = None

    for ckpt_path in files_in_range:
        ep = extract_ep(ckpt_path)

        # load checkpoint into fresh network
        q_net_scan     = QNetwork(n_features=140, n_hidden=420, n_actions=len(all_actions))
        optimizer_scan = optim.SGD(q_net_scan.parameters(), lr=0.05)
        load_checkpoint(ckpt_path, q_net_scan, optimizer_scan)

        # greedy evaluation
        res = run_trained_policy(q_net_scan, d_mainline, d_ramp, sample_cells)

        # compute episode reward (k=-1.0, same as training)
        rho_ctrl       = res["rho"][:, control_cell] / n_lanes[control_cell]
        episode_reward = float(np.sum(-1.0 * np.abs(rho_ctrl - rho_des)))

        print(f"  ep {ep:>7d}  |  greedy reward = {episode_reward:.4f}")

        if episode_reward > best_reward:
            best_reward  = episode_reward
            best_file    = ckpt_path
            best_episode = ep

    print()
    print("=" * 50)
    print(f"  BEST CHECKPOINT : {best_file}")
    print(f"  EPISODE         : {best_episode}")
    print(f"  GREEDY REWARD   : {best_reward:.4f}")
    print("=" * 50)
