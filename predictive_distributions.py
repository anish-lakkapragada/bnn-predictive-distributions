"""
File to compare exact BLR posterior with CAVI approximated
posterior (Step 3) in the blog. 
"""

# %%
import numpy as np 
import scipy as stats
from blr import generate_data
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression

X, y = generate_data(n = 1000)

"""
First plot our data and line of best fit. 
"""

X_ = X.reshape(-1, 1)

model = LinearRegression()
model.fit(X_, y)

x_line = np.linspace(X_.min(), X_.max(), 100).reshape(-1, 1)
y_line = model.predict(x_line)

plt.figure()
plt.scatter(X_, y, alpha=0.6, label=r"$(X_i, Y_i)$")
plt.plot(x_line, y_line, linewidth=2, label=f"y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}", color="orange")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Data with line of best fit")
plt.legend()
plt.show()

# %%
"""
Now write the code to be able to compare the predictive distributions, 
first starting at x = 0. 
"""
from blr import SIGMA_SQUARED, sample_from_blr_posterior
from cavi import sample_from_cavi_posterior

K = 1000 
def compute_predictive_confidence_interval(beta_samples, x):
    # take in a given sample of betas according to choice of posterior. 
    y_hat_samples = []
    for beta in beta_samples:
        mean = beta[0] + beta[1] * x
        y_sample = np.random.normal(mean, np.sqrt(SIGMA_SQUARED))
        y_hat_samples.append(y_sample)
    y_hat_samples = np.array(y_hat_samples)
    pred_mean = np.mean(y_hat_samples)
    # pred_std = np.std(y_hat_samples)
    ci_lower, ci_upper = np.percentile(y_hat_samples, [2.5, 97.5])
    return pred_mean, ci_lower, ci_upper 

x_star = 0

beta_samples_blr = sample_from_blr_posterior(X, y, K)
beta_samples_cavi = sample_from_cavi_posterior(X, y, K)

mean_blr, lower_blr, upper_blr = compute_predictive_confidence_interval(beta_samples_blr, x_star)
mean_cavi, lower_cavi, upper_cavi = compute_predictive_confidence_interval(beta_samples_cavi, x_star)

# Determine zoomed-in y-limits
y_min = min(lower_blr, lower_cavi)
y_max = max(upper_blr, upper_cavi)
margin = 0.1 * (y_max - y_min)
ylim_low = y_min - margin
ylim_high = y_max + margin

fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# BLR subplot
axs[0].axvline(x_star, color='gray', linestyle='--')
axs[0].errorbar(x_star, mean_blr, yerr=[[mean_blr - lower_blr], [upper_blr - mean_blr]],
                fmt='o', capsize=5, color='blue', label=r"Confidence Interval for $\hat{y}$")
axs[0].plot(x_line, y_line, linewidth=2, label="Line of Best Fit", color="orange")
axs[0].set_title("Predictive Interval from BLR")
axs[0].set_xlim(x_star - 1, x_star + 1)
axs[0].set_ylim(ylim_low, ylim_high)
axs[0].set_ylabel('$\hat{y}$ at $x=0$')
axs[0].grid(True)
axs[0].legend()

# CAVI subplot
axs[1].axvline(x_star, color='gray', linestyle='--')
axs[1].errorbar(x_star, mean_cavi, yerr=[[mean_cavi - lower_cavi], [upper_cavi - mean_cavi]],
                fmt='o', capsize=5, color='green',label=r"Confidence Interval for $\hat{y}$")
axs[1].plot(x_line, y_line, linewidth=2, label="Line of Best Fit", color="orange")
axs[1].set_title("Predictive Interval from CAVI")
axs[1].set_xlim(x_star - 1, x_star + 1)
axs[1].set_ylim(ylim_low, ylim_high)
axs[1].grid(True)
axs[1].legend()

plt.suptitle(r"Zoomed-In 95% Predictive Confidence Intervals at $x = 0$", fontsize=15)
plt.tight_layout()
plt.show()
# %%
"""
Run the above procedure for multiple choices of x. 
"""

K = 1000

x_vals = np.linspace(-3, 3, 1000)

beta_samples_blr = sample_from_blr_posterior(X, y, K)
beta_samples_cavi = sample_from_cavi_posterior(X, y, K)

means_blr, lowers_blr, uppers_blr = [], [], []
means_cavi, lowers_cavi, uppers_cavi = [], [], []

for x_star in x_vals:
    mean_blr, lower_blr, upper_blr = compute_predictive_confidence_interval(beta_samples_blr, x_star)
    means_blr.append(mean_blr)
    lowers_blr.append(lower_blr)
    uppers_blr.append(upper_blr)

    mean_cavi, lower_cavi, upper_cavi = compute_predictive_confidence_interval(beta_samples_cavi, x_star)
    means_cavi.append(mean_cavi)
    lowers_cavi.append(lower_cavi)
    uppers_cavi.append(upper_cavi)

means_blr = np.array(means_blr)
lowers_blr = np.array(lowers_blr)
uppers_blr = np.array(uppers_blr)

means_cavi = np.array(means_cavi)
lowers_cavi = np.array(lowers_cavi)
uppers_cavi = np.array(uppers_cavi)

plt.figure(figsize=(10, 6))

plt.plot(x_vals, means_blr, label="Mean $\hat{y}$ (BLR)", color='blue')
plt.fill_between(x_vals, lowers_blr, uppers_blr, color='blue', alpha=0.2, label="95% CI (BLR)")

plt.plot(x_vals, means_cavi, label="Mean $\hat{y}$ (CAVI)", color='green')
plt.fill_between(x_vals, lowers_cavi, uppers_cavi, color='green', alpha=0.2, label="95% CI (CAVI)")

# plt.scatter(X, y, alpha=0.2, color='black', s=10, label="Training Data")

plt.title(r"Predictive Confidence Intervals for $\hat{y}$ across $x \in [-3, 3]$")
plt.xlabel("x")
plt.ylabel(r"$\hat{y}$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
# %%


# %%
