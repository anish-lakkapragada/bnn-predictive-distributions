# %% 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from predictive_distributions import compute_predictive_confidence_interval
from cavi import sample_from_cavi_posterior
from blr import sample_from_blr_posterior, generate_data


def compute_predictive_stats(sample_func, X, y, x_vals, K):
    beta_samples = sample_func(X, y, K)
    means, lowers, uppers = [], [], []
    for x_star in x_vals:
        m, lo, hi = compute_predictive_confidence_interval(beta_samples, x_star)
        means.append(m)
        lowers.append(lo)
        uppers.append(hi)
    return np.array(means), np.array(lowers), np.array(uppers)


def init_plot(ax, x_vals):
    line_mean, = ax.plot([], [], lw=2)
    fill = ax.fill_between(x_vals, x_vals, x_vals, alpha=0.2)
    return line_mean, fill


def update_plot(ax, line_mean, x_vals, means, lowers, uppers, color, label_mean, label_ci):
    ax.collections.clear()  # remove old fill
    line_mean.set_data(x_vals, means)
    ax.fill_between(x_vals, lowers, uppers, color=color, alpha=0.2, label=label_ci)
    line_mean.set_label(label_mean)
    return line_mean


def create_single_frame(X, y, x_vals, K, fig, ax):
    # Compute stats
    m_blr, lo_blr, hi_blr = compute_predictive_stats(sample_from_blr_posterior, X, y, x_vals, K)
    m_cavi, lo_cavi, hi_cavi = compute_predictive_stats(sample_from_cavi_posterior, X, y, x_vals, K)

    # Clear axes
    ax.clear()
    ax.plot(x_vals, m_blr, color='blue', lw=2, label=f"Mean (BLR, K={K})")
    ax.fill_between(x_vals, lo_blr, hi_blr, color='blue', alpha=0.2, label="95% CI (BLR)")

    ax.plot(x_vals, m_cavi, color='green', lw=2, label=f"Mean (CAVI, K={K})")
    ax.fill_between(x_vals, lo_cavi, hi_cavi, color='green', alpha=0.2, label="95% CI (CAVI)")

    ax.set_title(f"Predictive Intervals with K = {K}")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\hat{y}$")
    ax.grid(True)
    ax.legend(loc='upper left')


def animate_convergence(X, y, x_vals, K_vals, save_path="predictive_convergence.gif", fps=10):
    fig, ax = plt.subplots(figsize=(10, 6))

    def update(frame):
        K = K_vals[frame]
        create_single_frame(X, y, x_vals, K, fig, ax)
        return fig,

    anim = FuncAnimation(fig, update, frames=len(K_vals), blit=False, repeat=False)
    anim.save(save_path, fps=fps, writer="pillow")
    plt.close(fig)


if __name__ == "__main__":
    K_vals = np.unique(np.logspace(np.log10(2), np.log10(1000), 50, dtype=int))
    x_vals = np.linspace(-3, 3, 1000)

    X, y = generate_data(n=1000) 

    animate_convergence(X, y, x_vals, K_vals)

# %%
