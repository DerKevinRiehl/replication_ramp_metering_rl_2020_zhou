# ════════════════════════════════════════════════════════════════════════════
# CTM with PI-ALINEA Ramp Metering
# Reproduction of Zhou et al. (2020)
# PI-ALINEA formulation follows Wang et al. (2014) equation (3).
# ════════════════════════════════════════════════════════════════════════════

# ── 1. Imports ───────────────────────────────────────────────────────────────
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

OUTPUT_DIR = "benchmark_controllers_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

entry_cell    = int(round(l_ramp   / cell_length))
lanedrop_cell = int(round(l_narrow / cell_length)) + 1
control_cell  = lanedrop_cell - 1

n_lanes = np.zeros(n_cells)
for i in range(0, lanedrop_cell):
    n_lanes[i] = 3
for i in range(lanedrop_cell, n_cells):
    n_lanes[i] = 2

Q_max   = Q_lane       * n_lanes
rho_jam = rho_jam_lane * n_lanes
rho_cr  = rho_cr_lane  * n_lanes

rho_des = 13.33
r_min   = 200
r_max   = 1200

# Original PI-ALINEA gains
K_R_orig = 100
K_P_orig = 4

# ALINEA gain
K_R_alinea = 10

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
def send(rho_i, j):
    return min(v_f * rho_i, Q_max[j])

def recv(rho_next, j_next):
    return min(Q_max[j_next], w * (rho_jam[j_next] - rho_next))

def run_sim(K_R, K_P, d_main, d_ramp_in, use_control=True):
    """
    Run CTM simulation with PI-ALINEA ramp metering.
    PI-ALINEA formula (Wang et al. 2014, eq. 3):
        r(k) = r(k-1) - K_P*[rho(k)-rho(k-1)] + K_R*[rho_des - rho(k)]
    Accepts demand arrays as arguments so noisy demands can be passed in.
    """
    rho     = np.zeros((n_t, n_cells))
    q       = np.zeros((n_t, n_cells))
    r_in    = np.zeros(n_t)
    N1      = np.zeros(n_t)
    N2      = np.zeros(n_t)
    r_meter = np.zeros(n_t)

    r_meter[0] = np.clip(d_ramp_in[0], r_min, r_max)

    for k in range(n_t - 1):
        rho_ctrl      = rho[k,     control_cell] / n_lanes[control_cell]
        rho_ctrl_prev = rho[k - 1, control_cell] / n_lanes[control_cell] if k > 0 else rho_ctrl

        if use_control:
            r_meter[k + 1] = np.clip(
                r_meter[k]
                - K_P * (rho_ctrl - rho_ctrl_prev)
                + K_R * (rho_des  - rho_ctrl),
                r_min, r_max
            )

        ramp_avail = d_ramp_in[k] + N1[k] / T
        r_cmd = min(ramp_avail, r_meter[k + 1]) if use_control else ramp_avail
        r_cmd = max(0.0, r_cmd)

        q[k, 0] = min(d_main[k] + N2[k] / T, recv(rho[k, 1], 1))

        for j in range(1, n_cells - 1):
            s_j  = send(rho[k, j], j)
            r_j1 = recv(rho[k, j + 1], j + 1)

            if j == entry_cell - 1:
                total = s_j + r_cmd
                if total <= r_j1:
                    q[k, j] = s_j
                    r_in[k] = r_cmd
                else:
                    if total > 0:
                        q[k, j] = s_j   / total * r_j1
                        r_in[k] = r_cmd / total * r_j1
            else:
                q[k, j] = min(s_j, r_j1)

        q[k, n_cells - 1] = send(rho[k, n_cells - 1], n_cells - 1)

        N1[k + 1] = max(0.0, N1[k] + (d_ramp_in[k] - r_in[k]) * T)
        N2[k + 1] = max(0.0, N2[k] + (d_main[k]    - q[k, 0]) * T)

        rho[k + 1, 1] = max(0.0, rho[k, 1] + T / cell_length * (q[k, 0] - q[k, 1]))

        for j in range(2, n_cells):
            inflow = q[k, j - 1] + (r_in[k] if j == entry_cell else 0.0)
            rho[k + 1, j] = max(0.0, rho[k, j] + T / cell_length * (inflow - q[k, j]))

    r_in[-1]=r_in[-2]#used only for plotting later

    return dict(rho=rho, q=q, r_in=r_in, N1=N1, r_meter=r_meter)

print("Simulation function defined.")

# ── 5. Performance Indicators ────────────────────────────────────────────────
def print_performance_indicators(res, label):
    rho_arr = res["rho"]
    q_arr   = res["q"]
    N1_arr  = res["N1"]

    VKT             = np.sum(q_arr[:, 1:] * cell_length * T)
    VHT             = np.sum(rho_arr[:, 0:] * cell_length * T)
    VHT_with_ramp   = VHT + np.sum(N1_arr * T)
    v_avg           = VKT / VHT          if VHT > 0          else 0
    v_avg_with_ramp = VKT / VHT_with_ramp if VHT_with_ramp > 0 else 0

    rho_ctrl        = rho_arr[:, control_cell] / n_lanes[control_cell]
    episode_reward  = np.sum(-1.0 * np.abs(rho_ctrl - rho_des))

    print(f"\n── Performance indicators: {label} ──")
    print(f"  VKT                : {VKT:.1f} veh·km")
    print(f"  VHT                : {VHT:.4f} veh·h")
    print(f"  VHT (with ramp)    : {VHT_with_ramp:.4f} veh·h")
    print(f"  v_avg              : {v_avg:.2f} km/h")
    print(f"  v_avg (with ramp)  : {v_avg_with_ramp:.2f} km/h")
    print(f"  Episode reward     : {episode_reward:.4f}")

# ── 6. Plot Helper Functions ─────────────────────────────────────────────────
def rho_per_lane(rho_arr):
    return rho_arr / n_lanes[np.newaxis, :]

def _spath(save_path, suffix):
    if save_path is None:
        return None
    base, ext = os.path.splitext(save_path)
    return f"{base}_{suffix}{ext}"

# Global plotting style for half-page-width figures
FS_LABEL = 10
FS_TICK  = 9
FS_TITLE = 10
FS_LEG   = 8.5

LW_DATA  = 1.4
LW_REF   = 1.3
LW_DROP  = 1.2
LW_SPINE = 0.8
LW_TICK  = 0.8
LEN_TICK = 3

DPI_SAVE = 300


def style_axis(ax):
    ax.tick_params(
        direction="in",
        top=True,
        right=True,
        labelsize=FS_TICK,
        width=LW_TICK,
        length=LEN_TICK
    )
    for spine in ax.spines.values():
        spine.set_linewidth(LW_SPINE)


def style_colorbar(cbar, title="[veh/km/ln]"):
    cbar.ax.set_title(title, size=FS_TICK)
    cbar.ax.tick_params(
        labelsize=FS_TICK,
        width=LW_TICK,
        length=LEN_TICK
    )


def plot_scenario(res, title, save_path=None):
    rpl      = rho_per_lane(res["rho"])
    rho_ctrl = rpl[:, control_cell]
    norm     = mpl.colors.Normalize(vmin=0, vmax=50)

    # ── 1) Control-cell density time series ──────────────────────────────────
    fig1, ax = plt.subplots(figsize=(6.5, 3.5))

    ax.plot(t, rho_ctrl, linewidth=LW_DATA, label="Actual value")
    ax.axhline(
        rho_des,
        color="saddlebrown",
        linewidth=LW_REF,
        linestyle=(0, (3, 3)),
        label="Desired value"
    )

    ax.set_xlim(0, t_tot)
    ax.set_ylim(0, 50)
    ax.set_xlabel("Sec", fontsize=FS_LABEL)
    ax.set_ylabel("Veh/km/ln", fontsize=FS_LABEL)
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.yaxis.set_major_locator(MultipleLocator(5))

    style_axis(ax)

    ax.legend(loc="upper left", fontsize=FS_LEG, frameon=False)
    ax.set_title(f"{title} — Control-cell density", fontsize=FS_TITLE)

    plt.tight_layout()
    sp = _spath(save_path, "density")
    if sp:
        plt.savefig(sp, dpi=DPI_SAVE, bbox_inches="tight")
        print(f"  → saved: {sp}")
    plt.show()

    # ── 2) Smoothed space-time density contour ────────────────────────────────
    Z      = rpl[:, 1:].T
    n_phys = Z.shape[0]
    y_raw  = (np.arange(n_phys) + 0.5) * cell_length * 1000

    interp = RegularGridInterpolator(
        (y_raw, t),
        Z,
        bounds_error=False,
        fill_value=None
    )

    x_fine = np.linspace(0, t_tot, 600)
    y_fine = np.linspace(y_raw[0], y_raw[-1], 400)
    Xf, Yf = np.meshgrid(x_fine, y_fine)

    Z_fine = interp(
        np.stack([Yf.ravel(), Xf.ravel()], axis=-1)
    ).reshape(len(y_fine), len(x_fine))

    Z_fine = gaussian_filter(np.clip(Z_fine, 0, 50), sigma=(2, 4))

    fig2, ax2 = plt.subplots(figsize=(7.2, 3.5))

    cf = ax2.contourf(
        Xf,
        Yf,
        Z_fine,
        levels=np.linspace(0, 50, 41),
        cmap="turbo",
        norm=norm
    )

    cbar2 = fig2.colorbar(cf, ax=ax2, ticks=np.arange(0, 51, 5))
    style_colorbar(cbar2, "[veh/km/ln]")

    ax2.axhline(
        l_narrow * 1000,
        color="k",
        lw=LW_DROP,
        ls="--",
        alpha=0.7,
        label="Lane drop"
    )

    ax2.set_xlim(0, t_tot)
    ax2.set_ylim(y_fine[0], y_fine[-1])
    ax2.set_xlabel("Sec", fontsize=FS_LABEL)
    ax2.set_ylabel("Position [m]", fontsize=FS_LABEL)
    ax2.xaxis.set_major_locator(MultipleLocator(500))
    ax2.yaxis.set_major_locator(MultipleLocator(500))

    style_axis(ax2)

    ax2.set_title(f"{title} — Smoothed space-time density", fontsize=FS_TITLE)

    plt.tight_layout()
    sp = _spath(save_path, "smoothed")
    if sp:
        plt.savefig(sp, dpi=DPI_SAVE, bbox_inches="tight")
        print(f"  → saved: {sp}")
    plt.show()

    # ── 3) Non-smoothed space-time density ───────────────────────────────────
    Z_raw      = rpl[:, 1:n_cells].T
    n_phys_raw = Z_raw.shape[0]
    y_edges_m  = np.arange(n_phys_raw + 1) * cell_length * 1000
    x_edges_s  = np.arange(n_t + 1) * delta_t

    fig3, ax3 = plt.subplots(figsize=(7.2, 3.5))

    cp = ax3.pcolormesh(
        x_edges_s,
        y_edges_m,
        Z_raw,
        cmap="turbo",
        norm=norm,
        shading="flat"
    )

    cbar3 = fig3.colorbar(cp, ax=ax3, ticks=np.arange(0, 51, 5))
    style_colorbar(cbar3, "[veh/km/ln]")

    ax3.axhline(
        l_narrow * 1000,
        linestyle="--",
        linewidth=LW_DROP,
        color="k",
        alpha=0.7
    )

    ax3.set_xlim(0, t_tot)
    ax3.set_ylim(0, n_phys_raw * cell_length * 1000)
    ax3.set_xlabel("Time [s]", fontsize=FS_LABEL)
    ax3.set_ylabel("Position [m]", fontsize=FS_LABEL)
    ax3.xaxis.set_major_locator(MultipleLocator(500))
    ax3.yaxis.set_major_locator(MultipleLocator(500))

    style_axis(ax3)

    ax3.grid(linestyle="--", linewidth=0.3, alpha=0.25)
    ax3.set_title(f"{title} — Non-smoothed space-time density", fontsize=FS_TITLE)

    plt.tight_layout()
    sp = _spath(save_path, "raw")
    if sp:
        plt.savefig(sp, dpi=DPI_SAVE, bbox_inches="tight")
        print(f"  → saved: {sp}")
    plt.show()


def plot_metering_rate(res, title, save_path=None):
    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    ax.plot(t, res["r_in"], linewidth=LW_DATA)

    ax.set_xlim(0, t_tot)
    ax.set_ylim(0, 1400)
    ax.set_xlabel("Sec", fontsize=FS_LABEL)
    ax.set_ylabel("Veh/h", fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_TITLE)
    ax.yaxis.set_major_locator(MultipleLocator(200))
    ax.xaxis.set_major_locator(MultipleLocator(1000))

    style_axis(ax)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=DPI_SAVE, bbox_inches="tight")
        print(f"  → saved: {save_path}")
    plt.show()


def plot_ramp_queue(res, title, save_path=None):
    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    ax.plot(t, res["N1"], linewidth=LW_DATA)

    ax.set_xlim(0, t_tot)
    ax.set_xlabel("Time [s]", fontsize=FS_LABEL)
    ax.set_ylabel("Queue length [veh]", fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_TITLE)
    ax.xaxis.set_major_locator(MultipleLocator(1000))

    style_axis(ax)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=DPI_SAVE, bbox_inches="tight")
        print(f"  → saved: {save_path}")
    plt.show()


def plot_demand(d_main, d_ramp_arr, title, save_path=None):
    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    ax.plot(
        t,
        d_main,
        linewidth=LW_DATA,
        color="steelblue",
        label="Mainline demand"
    )

    ax.plot(
        t,
        d_ramp_arr,
        linewidth=LW_DATA,
        color="darkorange",
        linestyle="--",
        label="Ramp demand"
    )

    ax.set_xlim(0, t_tot)
    ax.set_xlabel("Time [s]", fontsize=FS_LABEL)
    ax.set_ylabel("Veh/h", fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_TITLE)
    ax.xaxis.set_major_locator(MultipleLocator(1000))

    style_axis(ax)

    ax.legend(fontsize=FS_LEG, frameon=False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=DPI_SAVE, bbox_inches="tight")
        print(f"  → saved: {save_path}")
    plt.show()

def compute_kpis(res):
    """Return dict of all performance indicators for a simulation result."""
    rho_arr = res["rho"]
    q_arr   = res["q"]
    N1_arr  = res["N1"]
    VKT             = np.sum(q_arr[:, 1:] * cell_length * T)
    VHT             = np.sum(rho_arr[:, 0:] * cell_length * T)
    VHT_with_ramp   = VHT + np.sum(N1_arr * T)
    v_avg           = VKT / VHT           if VHT > 0           else 0
    v_avg_with_ramp = VKT / VHT_with_ramp if VHT_with_ramp > 0 else 0
    rho_ctrl        = rho_arr[:, control_cell] / n_lanes[control_cell]
    episode_reward  = float(np.sum(-1.0 * np.abs(rho_ctrl - rho_des)))
    return dict(VKT=VKT, VHT=VHT, VHT_with_ramp=VHT_with_ramp,
                v_avg=v_avg, v_avg_with_ramp=v_avg_with_ramp,
                episode_reward=episode_reward)

def run_and_report(K_R, K_P, d_main, d_ramp_in, label, use_control=True, tag=""):
    """Run simulation, print indicators, save all plots. Returns (res, kpis)."""
    res  = run_sim(K_R, K_P, d_main, d_ramp_in, use_control=use_control)
    kpis = compute_kpis(res)
    print_performance_indicators(res, label)
    base = os.path.join(OUTPUT_DIR, tag)
    plot_scenario(res, label, save_path=base + "_scenario.png")
    plot_metering_rate(res, f"{label} — Metering rate",
                       save_path=base + "_metering_rate.png")
    plot_ramp_queue(res, f"{label} — Ramp queue",
                    save_path=base + "_ramp_queue.png")
    return res, kpis

print("Plot helper functions defined.")

# ── 7. Baseline Demand Plot ───────────────────────────────────────────────────
plot_demand(
    d_mainline, d_ramp,
    title="Demand profiles",
    save_path=os.path.join(OUTPUT_DIR, "demand_baseline.png")
)

# ── 8. Grid Search ────────────────────────────────────────────────────────────
print("\nRunning grid search (PI-ALINEA: 111×111, ALINEA: 111)...")

KR_candidates = np.arange(1, 111, 1)
KP_candidates = np.arange(1, 111, 1)

# PI-ALINEA 2D grid
v_avg_grid          = np.zeros((len(KR_candidates), len(KP_candidates)))
v_avg_ramp_grid     = np.zeros((len(KR_candidates), len(KP_candidates)))

best_pi_v_avg       = -np.inf;  best_pi_v_avg_params      = None
best_pi_v_avg_ramp  = -np.inf;  best_pi_v_avg_ramp_params = None

for i_kr, kr in enumerate(KR_candidates):
    for i_kp, kp in enumerate(KP_candidates):
        res = run_sim(kr, kp, d_mainline, d_ramp, use_control=True)
        VKT = np.sum(res["q"][:, 1:] * cell_length * T)
        VHT = np.sum(res["rho"][:, 0:] * cell_length * T)
        VHT_r = VHT + np.sum(res["N1"] * T)
        va  = VKT / VHT   if VHT   > 0 else 0
        var = VKT / VHT_r if VHT_r > 0 else 0
        v_avg_grid[i_kr, i_kp]      = va
        v_avg_ramp_grid[i_kr, i_kp] = var
        if va  > best_pi_v_avg:       best_pi_v_avg       = va;  best_pi_v_avg_params      = (kr, kp)
        if var > best_pi_v_avg_ramp:  best_pi_v_avg_ramp  = var; best_pi_v_avg_ramp_params = (kr, kp)

# ALINEA 1D sweep
best_al_v_avg      = -np.inf; best_al_kr_v_avg      = None
best_al_v_avg_ramp = -np.inf; best_al_kr_v_avg_ramp = None

for kr in KR_candidates:
    res = run_sim(kr, 0, d_mainline, d_ramp, use_control=True)
    VKT = np.sum(res["q"][:, 1:] * cell_length * T)
    VHT = np.sum(res["rho"][:, 0:] * cell_length * T)
    VHT_r = VHT + np.sum(res["N1"] * T)
    va  = VKT / VHT   if VHT   > 0 else 0
    var = VKT / VHT_r if VHT_r > 0 else 0
    if va  > best_al_v_avg:       best_al_v_avg       = va;  best_al_kr_v_avg      = kr
    if var > best_al_v_avg_ramp:  best_al_v_avg_ramp  = var; best_al_kr_v_avg_ramp = kr

print("Grid search complete.")
print(f"  PI-ALINEA best by v_avg           : K_R={best_pi_v_avg_params[0]},  K_P={best_pi_v_avg_params[1]},  v_avg={best_pi_v_avg:.4f}")
print(f"  PI-ALINEA best by v_avg_with_ramp : K_R={best_pi_v_avg_ramp_params[0]},  K_P={best_pi_v_avg_ramp_params[1]},  v_avg_with_ramp={best_pi_v_avg_ramp:.4f}")
print(f"  ALINEA    best by v_avg           : K_R={best_al_kr_v_avg},  v_avg={best_al_v_avg:.4f}")
print(f"  ALINEA    best by v_avg_with_ramp : K_R={best_al_kr_v_avg_ramp},  v_avg_with_ramp={best_al_v_avg_ramp:.4f}")

# ── 9. Define all controllers to evaluate ────────────────────────────────────
controllers = [
    # (K_R, K_P, use_control, label, file_tag)
    (0,                       0,    False, "No control",                                         "no_control"),
    (K_R_alinea,              0,    True,  f"ALINEA (K_R={K_R_alinea})",                         "alinea_orig"),
    (best_al_kr_v_avg,        0,    True,  f"ALINEA opt v_avg (K_R={best_al_kr_v_avg})",         "alinea_opt_vavg"),
    (best_al_kr_v_avg_ramp,   0,    True,  f"ALINEA opt v_avg_ramp (K_R={best_al_kr_v_avg_ramp})","alinea_opt_vavg_ramp"),
    (K_R_orig,                K_P_orig, True, f"PI-ALINEA (K_R={K_R_orig}, K_P={K_P_orig})",    "pi_orig"),
    (best_pi_v_avg_params[0], best_pi_v_avg_params[1], True,
     f"PI-ALINEA opt v_avg (K_R={best_pi_v_avg_params[0]}, K_P={best_pi_v_avg_params[1]})",     "pi_opt_vavg"),
    (best_pi_v_avg_ramp_params[0], best_pi_v_avg_ramp_params[1], True,
     f"PI-ALINEA opt v_avg_ramp (K_R={best_pi_v_avg_ramp_params[0]}, K_P={best_pi_v_avg_ramp_params[1]})", "pi_opt_vavg_ramp"),
]

# ── 10. Clean-demand runs ─────────────────────────────────────────────────────
print("\n" + "="*70)
print("CLEAN DEMAND RUNS")
print("="*70)

results_clean = {}
kpis_clean    = {}
for K_R, K_P, use_ctrl, label, tag in controllers:
    print(f"\nRunning: {label}")
    res, kpis = run_and_report(K_R, K_P, d_mainline, d_ramp, label,
                               use_control=use_ctrl, tag=tag + "_clean")
    results_clean[tag]  = res
    kpis_clean[label]   = kpis

# ── 11. Noise Robustness Runs ─────────────────────────────────────────────────
noise_levels = [50, 100, 150, 200, 250]

# Pre-generate all noisy demand realisations
noisy_demands = {}
for std in noise_levels:
    rng = np.random.default_rng(seed=42 + std)
    noisy_demands[std] = (
        np.clip(d_mainline + rng.normal(0, std, size=n_t), 0, None),
        np.clip(d_ramp     + rng.normal(0, std, size=n_t), 0, None)
    )

# Demand plots — one per noise level
for std in noise_levels:
    d_main_n, d_ramp_n = noisy_demands[std]
    plot_demand(
        d_main_n, d_ramp_n,
        title=f"Demand profile — noise std={std} veh/h",
        save_path=os.path.join(OUTPUT_DIR, f"demand_noise{std}.png")
    )

# Simulation runs for each controller x noise level
kpis_noise = {}
print("\n" + "="*70)
print("NOISE ROBUSTNESS RUNS")
print("="*70)

for std in noise_levels:
    d_main_n, d_ramp_n = noisy_demands[std]
    print(f"\n{'─'*60}")
    print(f"  Noise std = {std} veh/h")
    print(f"{'─'*60}")
    for K_R, K_P, use_ctrl, label, tag in controllers:
        noise_label = f"{label} — noise std={std}"
        noise_tag   = f"{tag}_noise{std}"
        print(f"\n  Running: {noise_label}")
        _, kpis = run_and_report(K_R, K_P, d_main_n, d_ramp_n, noise_label,
                                 use_control=use_ctrl, tag=noise_tag)
        kpis_noise[(label, std)] = kpis

print("\nAll runs complete. Results saved to:", OUTPUT_DIR)

# ════════════════════════════════════════════════════════════════════════════
# 12. Summary Tables
# ════════════════════════════════════════════════════════════════════════════
import csv

noise_levels_all = [0] + noise_levels   # 0 = clean demand

controller_labels = [label for _, _, _, label, _ in controllers]
kpi_keys  = ["VKT", "VHT", "VHT_with_ramp", "v_avg", "v_avg_with_ramp", "episode_reward"]
kpi_names = ["VKT [veh·km]", "VHT [veh·h]", "VHT w/ ramp [veh·h]",
             "v_avg [km/h]", "v_avg w/ ramp [km/h]", "Episode reward"]

# ── KPI direction ─────────────────────────────────────────────────────────────
higher_is_better = {"VKT", "v_avg", "v_avg_with_ramp", "episode_reward"}

# ── Build flat results dict keyed by (label, noise_std) ──────────────────────
all_kpis = {}
for label in controller_labels:
    all_kpis[(label, 0)] = kpis_clean[label]
for (label, std), kpis in kpis_noise.items():
    all_kpis[(label, std)] = kpis

# ── Helper: percentage gap from best ─────────────────────────────────────────
def pct_gap(value, best, higher):
    return (best - value) / abs(best) * 100 if higher else (value - best) / abs(best) * 100

# ── Table 1: Full results ─────────────────────────────────────────────────────
table1_path = os.path.join(OUTPUT_DIR, "table1_full_results.csv")
with open(table1_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Controller", "Noise std [veh/h]"] + kpi_names)
    for std in noise_levels_all:
        for label in controller_labels:
            row_kpis = all_kpis.get((label, std), {})
            writer.writerow(
                [label, std] +
                [f"{row_kpis.get(k, float('nan')):.4f}" for k in kpi_keys]
            )
print(f"\n  → Table 1 saved: {table1_path}")

# ── Table 2: Percentage gap from best ────────────────────────────────────────
# 0.0% = best performer; higher % = worse relative to best
# To add RL results: append rows to table1_full_results.csv manually or re-run
# after adding RL KPIs to all_kpis dict (see README for instructions).
table2_path = os.path.join(OUTPUT_DIR, "table2_pct_gap.csv")
with open(table2_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Controller", "Noise std [veh/h]"] + kpi_names)
    for std in noise_levels_all:
        bests = {}
        for k in kpi_keys:
            vals = [all_kpis[(label, std)][k]
                    for label in controller_labels
                    if (label, std) in all_kpis]
            bests[k] = max(vals) if k in higher_is_better else min(vals)
        for label in controller_labels:
            row_kpis = all_kpis.get((label, std), {})
            gaps = [f"{pct_gap(row_kpis[k], bests[k], k in higher_is_better):.2f}%"
                    for k in kpi_keys]
            writer.writerow([label, std] + gaps)
print(f"  → Table 2 saved: {table2_path}")

# ── Pretty-print Table 1 to terminal ─────────────────────────────────────────
print("\n" + "="*120)
print("TABLE 1 — FULL RESULTS")
print("="*120)
header = f"{'Controller':<52} {'Noise':>6}  " + "  ".join(f"{n:>20}" for n in kpi_names)
print(header)
print("-"*120)
for std in noise_levels_all:
    for label in controller_labels:
        row_kpis = all_kpis.get((label, std), {})
        vals = "  ".join(f"{row_kpis.get(k, float('nan')):>20.4f}" for k in kpi_keys)
        print(f"{label:<52} {std:>6}  {vals}")
    print()

# ── Pretty-print Table 2 to terminal ─────────────────────────────────────────
print("\n" + "="*120)
print("TABLE 2 — PERCENTAGE GAP FROM BEST (0.0% = best)")
print("="*120)
print(header)
print("-"*120)
for std in noise_levels_all:
    bests = {}
    for k in kpi_keys:
        vals = [all_kpis[(label, std)][k]
                for label in controller_labels
                if (label, std) in all_kpis]
        bests[k] = max(vals) if k in higher_is_better else min(vals)
    for label in controller_labels:
        row_kpis = all_kpis.get((label, std), {})
        gaps = "  ".join(
            f"{pct_gap(row_kpis[k], bests[k], k in higher_is_better):>19.2f}%"
            for k in kpi_keys
        )
        print(f"{label:<52} {std:>6}  {gaps}")
    print()
