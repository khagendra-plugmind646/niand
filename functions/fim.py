import numpy as np
import cvxpy as cp
from .beamforming import beamfocusing

def FIM(para, Rx, beta, scale):
    """
    Calculate the Fisher Information Matrix (FIM) blocks for CVXPY.
    
    Args:
        para (dict): Dictionary containing simulation parameters
        Rx: CVXPY variable for the transmit covariance matrix
        beta (complex): Complex channel gain
        scale (float): Scaling factor for optimization
        
    Returns:
        tuple: A tuple containing the 2x2 FIM blocks (J_11, J_12, J_22)
    """
    lambda_ = para['c'] / para['f']
    r = para['r_s']
    theta = para['theta_s']
    
    # Calculate array response vector and target response matrix
    a = beamfocusing(para, r, theta)  # N x 1
    G = np.outer(a, a.conj())        # N x N
    
    # Derivative index vector
    n = (np.arange(-(para['N']-1)/2, (para['N']-1)/2 + 1) * para['d']).reshape(-1, 1)  # N x 1
    r_n = np.sqrt(r**2 + n**2 - 2 * r * n * np.cos(theta))
    
    # Steering derivatives
    a_r = -1j * 2 * np.pi / lambda_ * ((r - n * np.cos(theta)) / r_n - 1) * a
    a_theta = -1j * 2 * np.pi / lambda_ * (r * n * np.sin(theta)) / r_n * a
    
    # Derivatives of G
    G_r = np.outer(a_r, a.conj()) + np.outer(a, a_r.conj())
    G_theta = np.outer(a_theta, a.conj()) + np.outer(a, a_theta.conj())
    
    beta2 = np.abs(beta)**2
    
    # Calculate traces (CVXPY expressions)
    if isinstance(Rx, np.ndarray):
        # If Rx is a numpy array (for testing)
        t_rr = np.trace(G_r @ Rx @ G_r.conj().T)
        t_rtheta = np.trace(G_r @ Rx @ G_theta.conj().T)
        t_thetat = np.trace(G_theta @ Rx @ G_theta.conj().T)
        t_ggr = np.trace(G @ Rx @ G_r.conj().T)
        t_ggtheta = np.trace(G @ Rx @ G_theta.conj().T)
        t_gg = np.trace(G @ Rx @ G.conj().T)
    else:
        # If Rx is a CVXPY variable, ensure proper matrix dimensions
        # Convert G matrices to numpy arrays for proper broadcasting with CVXPY variables
        G_r_np = np.array(G_r)
        G_theta_np = np.array(G_theta)
        G_np = np.array(G)
        
        # Calculate traces using element-wise multiplication and sum for better numerical stability
        def safe_trace(A, B, C):
            # A @ B @ C^H is equivalent to sum_ij A_ij (B @ C^H)_ji
            BC = cp.matmul(B, cp.conj(C).T)
            return cp.sum(cp.multiply(A, BC.T))  # Element-wise multiply and sum
            
        t_rr = cp.real(safe_trace(G_r_np, Rx, G_r_np))
        t_rtheta = cp.real(safe_trace(G_r_np, Rx, G_theta_np))
        t_thetat = cp.real(safe_trace(G_theta_np, Rx, G_theta_np))
        t_ggr = cp.real(safe_trace(G_np, Rx, G_r_np))
        t_ggtheta = cp.real(safe_trace(G_np, Rx, G_theta_np))
        t_gg = cp.real(safe_trace(G_np, Rx, G_np))
    
    # Assemble 2x2 blocks
    J_11 = scale * 2 * beta2 * cp.bmat([
        [t_rr, t_rtheta],
        [t_rtheta, t_thetat]
    ]) if not isinstance(Rx, np.ndarray) else scale * 2 * beta2 * np.array([
        [t_rr, t_rtheta],
        [t_rtheta, t_thetat]
    ])
    
    J_12 = scale * 2 * beta2 * cp.bmat([
        [t_ggr, 1j * t_ggr],
        [t_ggtheta, 1j * t_ggtheta]
    ]) if not isinstance(Rx, np.ndarray) else scale * 2 * beta2 * np.array([
        [t_ggr, 1j * t_ggr],
        [t_ggtheta, 1j * t_ggtheta]
    ])
    
    J_22 = scale * 2 * beta2 * np.eye(2) * t_gg
    
    return J_11, J_12, J_22
