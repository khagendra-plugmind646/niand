import numpy as np

def rate_calculator(para, H, Rx, f):
    """
    Calculate the achievable rate for each user.
    
    Args:
        para (dict): Dictionary containing simulation parameters
        H (np.ndarray): Channel matrix (N x K)
        Rx (np.ndarray): Transmit covariance matrix (N x N)
        f (np.ndarray): Precoding matrix (N x K)
        
    Returns:
        np.ndarray: Achievable rates for each user (K,)
    """
    K = para['K']
    rate = np.zeros(K)
    
    for k in range(K):
        hk = H[:, k]  # Channel for user k
        fk = f[:, k]   # Precoding vector for user k
        
        # Desired signal power - ensure we're working with real values before power operation
        signal = np.square(np.abs(np.vdot(hk, fk)))  # Equivalent to |h_k^H * f_k|^2
        
        # Interference from other users
        interference = 0
        for j in range(K):
            if j != k:
                fj = f[:, j]
                interference += np.square(np.abs(np.vdot(hk, fj)))  # |h_k^H * f_j|^2
        
        # Noise power
        noise = para['noise']
        
        # Calculate SINR (ensure it's non-negative)
        sinr = signal / (interference + noise + 1e-10)  # Small constant to avoid division by zero
        sinr = max(sinr, 0)  # Ensure non-negative
        
        # Calculate achievable rate
        rate[k] = np.log2(1 + sinr)
    
    return rate
