import pandas as pd
import matplotlib.pyplot as plt

# ── Load ──────────────────────────────────────────────────────────────────
df = pd.read_csv("checkpoints/greedy_rewards.csv")

# Keep only episodes from 0 to 700000
EPISODE_MIN = 0
EPISODE_MAX = 700000
df = df[(df["episode"] >= EPISODE_MIN) & (df["episode"] <= EPISODE_MAX)].copy()

# ── Rolling statistics ────────────────────────────────────────────────────
window = 1000

df["greedy_reward_smooth"] = df["greedy_reward"].rolling(
    window=window, min_periods=1
).mean()

df["greedy_reward_var"] = df["greedy_reward"].rolling(
    window=window, min_periods=1
).var(ddof=0)

df["greedy_reward_std"] = df["greedy_reward"].rolling(
    window=window, min_periods=1
).std(ddof=0)

# ── Best greedy reward ────────────────────────────────────────────────────
best_reward = df["greedy_reward"].max()
best_episode = df.loc[df["greedy_reward"].idxmax(), "episode"]

# ═══════════════════════════════════════════════════════════════════════════
# 1) Greedy learning curve with rolling mean
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    df["episode"], df["greedy_reward"],
    alpha=0.2, color="steelblue", linewidth=0.5,
    label="Greedy reward"
)

ax.plot(
    df["episode"], df["greedy_reward_smooth"],
    color="steelblue", linewidth=1.5,
    label=f"Rolling mean ({window} ep)"
)

ax.axhline(
    y=best_reward,
    color="tomato", linewidth=1.5, linestyle="--",
    label="Best greedy reward"
)

ax.text(
    df["episode"].max(), best_reward,
    f"  {best_reward:.2f} (ep {int(best_episode)})",
    color="tomato", va="bottom", ha="right", fontsize=9
)

ax.set_xlabel("Episode")
ax.set_ylabel("Greedy total reward")
ax.set_title("Greedy evaluation learning curve")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("06_07_greedy_learning_curve.png", dpi=150)
plt.show()

# ═══════════════════════════════════════════════════════════════════════════
# 2) Rolling variance of greedy reward
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    df["episode"], df["greedy_reward_var"],
    color="darkorange", linewidth=1.2,
    label=f"Rolling variance ({window} ep)"
)

ax.set_xlabel("Episode")
ax.set_ylabel("Greedy reward variance")
ax.set_title(f"Rolling greedy reward variance ({window}-episode window)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("06_07_greedy_reward_rolling_variance.png", dpi=150)
plt.show()

# ═══════════════════════════════════════════════════════════════════════════
# 3) Rolling standard deviation of greedy reward
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    df["episode"], df["greedy_reward_std"],
    color="darkorange", linewidth=1.2,
    label=f"Rolling standard deviation ({window} ep)"
)

ax.set_xlabel("Episode")
ax.set_ylabel("Greedy reward standard deviation")
ax.set_title(f"Rolling greedy reward standard deviation ({window}-episode window)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("06_07_greedy_reward_rolling_std.png", dpi=150)
plt.show()

# ── Save enriched greedy reward log for later analysis ────────────────────
df.to_csv("06_07_greedy_rewards_with_rolling_stats.csv", index=False)