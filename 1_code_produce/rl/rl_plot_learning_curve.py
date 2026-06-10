import pandas as pd
import matplotlib.pyplot as plt

# ── Load ──────────────────────────────────────────────────────────────────
df = pd.read_csv("checkpoints/episode_rewards.csv")

# Keep only episodes from 0 to 700000
EPISODE_MIN = 0
EPISODE_MAX = 700000
df = df[(df["episode"] >= EPISODE_MIN) & (df["episode"] <= EPISODE_MAX)].copy()

# ── Rolling statistics ────────────────────────────────────────────────────
window = 1000
# Rolling mean of episode rewards
df["reward_smooth"] = df["reward"].rolling(window=window, min_periods=1).mean()
# Rolling variance over the same window
# ddof=0 avoids NaN for windows with a single element and gives population variance
df["reward_var"] = df["reward"].rolling(window=window, min_periods=1).var(ddof=0)
# Optional: rolling standard deviation, useful because it has the same unit as reward
df["reward_std"] = df["reward"].rolling(window=window, min_periods=1).std(ddof=0)

# ── Best reward ───────────────────────────────────────────────────────────
best_reward = df["reward"].max()
best_episode = df.loc[df["reward"].idxmax(), "episode"]

# ═══════════════════════════════════════════════════════════════════════════
# 1) Learning curve with rolling mean
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    df["episode"], df["reward"],
    alpha=0.2, color="steelblue", linewidth=0.5,
    label="Episode reward"
)
ax.plot(
    df["episode"], df["reward_smooth"],
    color="steelblue", linewidth=1.5,
    label=f"Rolling mean ({window} ep)"
)

ax.axhline(
    y=best_reward,
    color="tomato", linewidth=1.5, linestyle="--",
    label="Best reward"
)
ax.text(
    df["episode"].max(), best_reward,
    f"  {best_reward:.2f} (ep {int(best_episode)})",
    color="tomato", va="bottom", ha="right", fontsize=9
)

ax.set_xlabel("Episode")
ax.set_ylabel("Total reward")
ax.set_title("Learning curve")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("06_07_learning_curve.png", dpi=150)
plt.show()

# ═══════════════════════════════════════════════════════════════════════════
# 2) Rolling variance of training reward
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    df["episode"], df["reward_var"],
    color="darkorange", linewidth=1.2,
    label=f"Rolling variance ({window} ep)"
)

ax.set_xlabel("Episode")
ax.set_ylabel("Reward variance")
ax.set_title(f"Rolling reward variance ({window}-episode window)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("06_07_reward_rolling_variance.png", dpi=150)
plt.show()

# ═══════════════════════════════════════════════════════════════════════════
# 3) Optional: rolling standard deviation of training reward
#    This is often easier to interpret because it has the same unit as reward.
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    df["episode"], df["reward_std"],
    color="darkorange", linewidth=1.2,
    label=f"Rolling standard deviation ({window} ep)"
)

ax.set_xlabel("Episode")
ax.set_ylabel("Reward standard deviation")
ax.set_title(f"Rolling reward standard deviation ({window}-episode window)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("06_07_reward_rolling_std.png", dpi=150)
plt.show()

# ── Save enriched reward log for later analysis ───────────────────────────
df.to_csv("06_07_episode_rewards_with_rolling_stats.csv", index=False)
