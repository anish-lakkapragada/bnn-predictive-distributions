# %%
import numpy as np 
from scipy import stats 

SIGMA_SQUARED = 1

# generates D
def generate_data(n=100): 
    np.random.seed(69)
    beta_0_true = np.random.random() * 3
    beta_1_true = np.random.random() * 3
    X = np.random.normal(0, 1, n)
    eps = np.random.normal(0, np.sqrt(SIGMA_SQUARED), n) # true sigma^2 = 1
    y = beta_0_true + beta_1_true * X + eps 
    return X, y 


S = np.array([[1, 0], [0, 1]])

# compute p(beta | D)
def exact_posterior(beta, X, y): 
    # beta is a tuple containing (beta_0, beta_1)
    X_expanded = np.ones((X.shape[0], 2)) # expand data
    X_expanded[:, 1] = X

    Sigma = np.linalg.inv((1 / SIGMA_SQUARED) * np.matmul(X_expanded.T, X_expanded) + np.linalg.inv(S))
    mu = (1 / SIGMA_SQUARED) * np.matmul(Sigma, np.matmul(X_expanded.T, y))
    return stats.multivariate_normal.pdf(beta, mean=mu, cov=Sigma)

# sample from p(beta | D)
def sample_from_blr_posterior(X, y, num_samples=1000): 
    X_expanded = np.ones((X.shape[0], 2))
    X_expanded[:, 1] = X

    Sigma = np.linalg.inv((1 / SIGMA_SQUARED) * np.matmul(X_expanded.T, X_expanded) + np.linalg.inv(S))
    mu = (1 / SIGMA_SQUARED) * np.matmul(Sigma, np.matmul(X_expanded.T, y))
    
    return np.random.multivariate_normal(mean=mu, cov=Sigma, size=num_samples)

# %%
