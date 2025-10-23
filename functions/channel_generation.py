import numpy as np
from .beamforming import beamfocusing

def generate_channel(para):
    """
    Generate communication and sensing channels.
    
    Args:
        para (dict): Dictionary containing simulation parameters.
        
    Returns:
        tuple: A tuple containing:
            - H (np.ndarray): Communication channels (N x K)
            - G (np.ndarray): Target response matrix (N x N)
            - beta_s (complex): Target reflection coefficient
            - r (np.ndarray): User distances (K,)
            - theta (np.ndarray): User angles in radians (K,)
            - r_s (float): Target distance
            - theta_s (float): Target angle in radians
    """
    # Communication channels
    # Generate user locations
    lambda_ = para['c'] / para['f']
    rayleigh_distance = 2 * para['D']**2 / lambda_
    
    # Generate random user positions within the Rayleigh distance
    r = np.random.uniform(0, rayleigh_distance, para['K'])
    theta = np.random.uniform(0, np.pi, para['K'])
    
    # Initialize channel matrix
    N = para['N']
    K = para['K']
    H = np.zeros((N, K), dtype=complex)
    
    # Generate channels for each user
    for k in range(K):
        # Calculate the channel coefficient and ensure it's a scalar
        # *** CORRECTION: Removed 'np.sqrt(1/para['noise'])' from beta calculation.
        # The channel gain should not be dependent on receiver noise.
        beta = (para['rho_0'] / r[k] * np.exp(-1j * 2 * np.pi * para['f']/para['c'] * r[k]))
        
        # Get the beamfocusing vector and ensure it's a 1D array before assignment
        bf_vec = beamfocusing(para, r[k], theta[k])
        H[:, k] = (beta * bf_vec).flatten()
    
    # Target response matrix
    r_s = para['r_s']
    theta_s = para['theta_s']
    
    # Generate random reflection coefficient
    beta_reflection = np.sqrt(1/2) * (np.random.randn() + 1j * np.random.randn())
    beta_s = (np.sqrt(para['rho_0']) / (2 * r_s) * np.exp(-1j * 2 * np.pi * para['f']/para['c'] * 2 * r_s) * beta_reflection)
    
    # Calculate target response matrix
    a = beamfocusing(para, r_s, theta_s)
    G = beta_s * np.outer(a, a.conj())
    
    return H, G, beta_s, r, theta, r_s, theta_s