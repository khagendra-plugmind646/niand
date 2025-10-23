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
    U = cp.Variable((2, 2), hermitian=True)
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
        rate_factor = float(2**para['Rmin'] - 1) if 'Rmin' in para else 0.1  # Default to 0.1 if not specified
        signal_term = cp.real(cp.trace(Fk @ np.outer(hk, hk_herm)))
        interference_term = cp.real(cp.trace((Rx - Fk) @ np.outer(hk, hk_herm)))
        noise_var = 1.0  # Assuming unit noise variance
        constraints.append(signal_term >= rate_factor * (interference_term + noise_var) - 1e-3)
    
    # Power constraint with small relaxation
    if 'Pt' in para:
        constraints.append(cp.real(cp.trace(Rx)) <= para['Pt'] * 1.1)  # 10% relaxation
    else:
        # Default power constraint if not specified
        constraints.append(cp.real(cp.trace(Rx)) <= 1.0)
    
    # Covariance constraint
    F_sum = sum(F)
    constraints.append(Rx - F_sum >> 0)
    
    # Define and solve the problem
    # Ensure we're working with real values for the objective
    obj = cp.Minimize(cp.trace(cp.inv_pos(cp.real(U))))
    prob = cp.Problem(obj, constraints)
    
    # Try different solvers in order of preference with more relaxed parameters
    solvers = []
    # Try SCS with relaxed parameters
    solvers.append((cp.SCS, {
        'verbose': True,
        'max_iters': 2000,  # Increased max iterations
        'eps': 1e-2,       # Relaxed tolerance
        'alpha': 1.2,      # Relaxed step size
        'scale': 5.0,      # Scale parameter
        'normalize': True,
        'use_indirect': False,  # Use direct solver (more stable but slower)
        'warm_start': True,     # Use warm start
        'acceleration_lookback': 5  # Reduce lookback for stability
    }))
    
    # Try other available solvers
    if 'CLARABEL' in cp.installed_solvers():
        solvers.append((cp.CLARABEL, {'verbose': True, 'tol_gap_abs': 1e-3, 'tol_gap_rel': 1e-3}))
    if 'OSQP' in cp.installed_solvers():
        solvers.append((cp.OSQP, {'verbose': True, 'eps_abs': 1e-3, 'eps_rel': 1e-3, 'max_iter': 5000}))
    
    solution_found = False
    for solver, solver_params in solvers:
        try:
            if isinstance(solver, str):
                solver_name = solver
            else:
                solver_name = solver.__name__.split('.')[-1]
            print(f"\n  - Trying {solver_name} solver... ", end="", flush=True)
            
            prob.solve(solver=solver, **solver_params)
            if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                print(f"solved with status: {prob.status}")
                solution_found = True
                break
            else:
                print(f"failed with status: {prob.status}")
        except Exception as e:
            print(f"failed: {str(e)[:100]}")
    
    if not solution_found:
        print("\n  - All solvers failed. Check the problem formulation or constraints.")
        return None, None
    
    # Construct rank-one solution
    f = np.zeros((N, K), dtype=complex)
    for k in range(K):
        hk = H[:, k]
        Fk = F[k].value
        if Fk is not None:
            # Handle potential numerical issues with small eigenvalues
            w, v = np.linalg.eigh(Fk)
            # Take the eigenvector corresponding to the largest eigenvalue
            fk = np.sqrt(w[-1]) * v[:, -1]
            f[:, k] = fk
    
    return Rx.value, f
