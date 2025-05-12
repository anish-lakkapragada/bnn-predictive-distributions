"""
Code to run coordinate-ascent variational inference (CAVI). 
"""
# %%
from blr import SIGMA_SQUARED
import numpy as np 
from scipy import stats

TAU_0_SQUARED = TAU_1_SQUARED = 1
def beta_0_update(X, y, phi_1):
    m_1 = phi_1[0] # treat as fixed
    n = X.shape[0]
    s_0_new = 1 / ((1 / TAU_0_SQUARED) + (n / SIGMA_SQUARED)) # s_{0, *}^2
    m_0_new = (s_0_new / SIGMA_SQUARED) * np.sum(y - X * m_1)
    return np.array([m_0_new, s_0_new])


def beta_1_update(X, y, phi_0): 
    m_0 = phi_0[0]
    s_1_new = 1 / ((1 / TAU_1_SQUARED) + (np.sum(np.pow(X, 2)) / SIGMA_SQUARED)) # s_{1, *}^2 
    m_1_new = (s_1_new / SIGMA_SQUARED) * np.sum(X * (y - m_0))
    return np.array([m_1_new, s_1_new])

def run_cavi(X, y): 
    phi_0, phi_1 = [0, 1], [0, 1]
    phi_0, phi_1 = np.array(phi_0), np.array(phi_1)
    while (True): 
        phi_0_new = beta_0_update(X, y, phi_1)
        phi_1_new = beta_1_update(X, y, phi_0_new)
        if (np.sum(np.abs(phi_0_new - phi_0)) < 1e-12): 
            break 
        phi_0 = phi_0_new
        phi_1 = phi_1_new 
    return phi_0, phi_1

# evaluate q_{phi}(beta) by first learning phi
def cavi_posterior(beta, X, y): 
  # beta is a tuple of (beta_0, beta_1)
  phi_0, phi_1 = run_cavi(X, y)
  q_0_pdf = stats.norm.pdf(beta[0], phi_0[0], np.sqrt(phi_0[1])) # q_{phi_0}(beta_0)
  q_1_pdf = stats.norm.pdf(beta[1], phi_1[0], np.sqrt(phi_1[1])) # q_{phi_1}(beta_1)
  return q_0_pdf * q_1_pdf

# sample from p(beta | D)
def sample_from_cavi_posterior(X, y, num_samples=1000): 
    phi_0, phi_1 = run_cavi(X, y)
    beta_0_samples = np.random.normal(loc=phi_0[0], scale=np.sqrt(phi_0[1]), size=num_samples)
    beta_1_samples = np.random.normal(loc=phi_1[0], scale=np.sqrt(phi_1[1]), size=num_samples)
    return np.column_stack((beta_0_samples, beta_1_samples))
# %%
