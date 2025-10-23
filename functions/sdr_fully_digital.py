import numpy as np
import cvxpy as cp
from .fim import FIM

def SDR_fully_digital(para, H, beta_s, scale):
    """
    Semidefinite Relaxation (SDR) for fully digital beamforming design.
    
    Args:
        para (dict): Dictionary containing simulation parameters
        H (np.ndarray): Channel matrix (N x K)
        beta_s (complex): Complex sensing channel gain
        scale (float): Scaling factor for optimization
        
    Returns:
        tuple: A tuple containing:
            - Rx (np.ndarray): Covariance matrix of transmit signal (N x N)
            - f (np.ndarray): Beamformers for communication signals (N x K)
    """
    N = para['N']
    K = para['K']
    
    # Define optimization variables
    F = [cp.Variable((N, N), hermitian=True) for _ in range(K)]
    
    # *** CORRECTION: U must be real and symmetric, not complex hermitian,
    # as it corresponds to the real-valued J_11 block.
    U = cp.Variable((2, 2), symmetric=True) 
    
    Rx = cp.Variable((N, N), hermitian=True)
    
    # Get FIM blocks
    J_11, J_12, J_22 = FIM(para, Rx, beta_s, scale)
    
    # Construct constraints
    constraints = []
    
    # FIM constraint
    matrix = cp.bmat([
        [J_11 - U, J_12],
        [J_12.H, J_22]
    ])
    constraints.append(matrix >> 0)  # Positive semidefinite
    
    # Rate constraints with relaxation
    for k in range(K):
        hk = H[:, k]
        Fk = F[k]
        
        # Positive semidefinite constraint with small epsilon for numerical stability
        constraints.append(Fk >> -1e-4 * np.eye(N))
        
        # Rate constraint with relaxation
        hk_herm = hk.conj().T
        rate_factor = float(2**para['Rmin'] - 1) if 'Rmin' in para else 0.1
        signal_term = cp.real(cp.trace(Fk @ np.outer(hk, hk_herm)))
        interference_term = cp.real(cp.trace((Rx - Fk) @ np.outer(hk, hk_herm)))
        
        # *** CORRECTION: Use the actual noise power from parameters, not 1.0.
        # This ensures consistency with rate_calculator.py.
        noise_var = para['noise'] 
        
        constraints.append(signal_term >= rate_factor * (interference_term + noise_var) - 1e-3)
    
    # Power constraint
    # *** CORRECTION: Use 'Pmax' (normalized power) not 'Pt'.
    # Removed arbitrary 1.1x relaxation.
    constraints.append(cp.real(cp.trace(Rx)) <= para['Pmax'])
    
    # Covariance constraint
    F_sum = sum(F)
    constraints.append(Rx - F_sum >> 0)
    
    # Define and solve the problem
    # *** CORRECTION: U is now real, so cp.inv_pos(U) is correct.
    # We no longer need cp.real(U)
    obj = cp.Minimize(cp.trace(cp.inv_pos(U)))
    prob = cp.Problem(obj, constraints)
    
    # Try different solvers
    solvers = []
    solvers.append((cp.SCS, {
        'verbose': False, # Set to True for detailed solver output
        'max_iters': 2500,
        'eps': 1e-3, # Slightly tighter tolerance
    }))
    
    if 'MOSEK' in cp.installed_solvers():
        # MOSEK is the best solver for this if available
        solvers.insert(0, (cp.MOSEK, {'verbose': False}))

    if 'CLARABEL' in cp.installed_solvers():
        solvers.append((cp.CLARABEL, {'verbose': False, 'tol_gap_abs': 1e-3, 'tol_gap_rel': 1e-3}))
    
    solution_found = False
    for solver, solver_params in solvers:
        try:
            solver_name = solver if isinstance(solver, str) else solver.__name__.split('.')[-1]
            # print(f"\n  - Trying {solver_name} solver... ", end="", flush=True) # Uncomment for debug
            
            prob.solve(solver=solver, **solver_params)
            
            if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                # print(f"solved with status: {prob.status}") # Uncomment for debug
                solution_found = True
                break
            else:
                # print(f"failed with status: {prob.status}") # Uncomment for debug
                pass
        except Exception as e:
            # print(f"failed: {str(e)[:100]}") # Uncomment for debug
            pass
    
    if not solution_found:
        # print("\n  - All solvers failed.") # Uncomment for debug
        return None, None
    
    # Check if variables have valid values
    if Rx.value is None or any(Fk.value is None for Fk in F):
        # print("\n  - Solver reported success but solution is invalid.") # Uncomment for debug
        return None, None

    # Construct rank-one solution
    f = np.zeros((N, K), dtype=complex)
    for k in range(K):
        hk = H[:, k]
        Fk = F[k].value
        if Fk is not None:
            try:
                # Handle potential numerical issues with small eigenvalues
                w, v = np.linalg.eigh(Fk)
                # Take the eigenvector corresponding to the largest eigenvalue
                if w[-1] > 1e-10: # Only proceed if eigenvalue is meaningfully positive
                    fk = np.sqrt(w[-1]) * v[:, -1]
                    f[:, k] = fk
                else:
                    f[:, k] = np.zeros(N) # Set to zero if Fk is effectively zero
            except np.linalg.LinAlgError:
                f[:, k] = np.zeros(N)
    
    return Rx.value, f