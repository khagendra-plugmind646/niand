import numpy as np
from .beamforming import beamfocusing
from scipy.linalg import cholesky, eig
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

def matrix_decomposition(U):
    """
    Perform matrix decomposition for square root calculation.
    
    Args:
        U (np.ndarray): Input matrix
        
    Returns:
        np.ndarray: Square root of the input matrix
    """
    w, v = np.linalg.eigh(U)
    # Ensure numerical stability by setting small negative eigenvalues to zero
    w = np.maximum(w, 0)
    return v @ np.diag(np.sqrt(w)) @ v.conj().T

def music_estimation(para, Rx, f, G, m=500, r_max=40):
    """
    MUSIC algorithm for target parameter estimation.
    
    Args:
        para (dict): Dictionary containing simulation parameters
        Rx (np.ndarray): Covariance matrix of transmit signal (N x N)
        f (np.ndarray): Beamformers of communication signals (N x K)
        G (np.ndarray): Target response matrix (N x N)
        m (int): Number of grid points in each dimension
        r_max (float): Maximum range for the grid search
        
    Returns:
        tuple: A tuple containing:
            - spectrum (np.ndarray): MUSIC spectrum (m x m)
            - X (np.ndarray): X coordinates of the grid (m x m)
            - Y (np.ndarray): Y coordinates of the grid (m x m)
    """
    # Generate signals
    Rs = Rx - f @ f.conj().T
    A = matrix_decomposition(Rs)
    
    # Generate random signals
    T = para['T']
    N = para['N']
    K = para['K']
    
    # Dedicated sensing signal
    L = cholesky(A @ A.conj().T, lower=True)
    s = L.conj().T @ (np.random.randn(N, T) + 1j * np.random.randn(N, T)) / np.sqrt(2)
    
    # Communication signal
    c = (np.random.randn(K, T) + 1j * np.random.randn(K, T)) / np.sqrt(2)
    
    # Noise
    N_s = np.sqrt(para['noise']/2) * (np.random.randn(N, T) + 1j * np.random.randn(N, T))
    
    # Transmit and receive signals
    X_tx = f @ c + s
    Y_s = G @ X_tx + N_s
    
    # MUSIC algorithm
    R = (Y_s @ Y_s.conj().T) / T  # Sample covariance matrix
    
    # Eigenvalue decomposition
    D, U = eig(R)
    # Sort eigenvalues in descending order
    idx = np.argsort(D)[::-1]
    U = U[:, idx]
    
    # Noise subspace (all eigenvectors except the first one)
    Uz = U[:, 1:]
    U = Uz @ Uz.conj().T
    
    # Create search grid
    x = np.linspace(0, r_max, m)
    y = np.linspace(0, r_max, m)
    X, Y = np.meshgrid(x, y)
    
    # Convert to polar coordinates
    theta_all = np.arctan2(Y, X)
    r_all = np.sqrt(X**2 + Y**2)
    
    # Initialize spectrum
    spectrum = np.zeros((m, m), dtype=float)
    
    # Function to compute spectrum at a single point
    def compute_point(i, j):
        aa = beamfocusing(para, r_all[i, j], theta_all[i, j])
        # Ensure we're working with a real, positive value
        denominator = np.real(np.vdot(aa, U @ aa))
        # Add small epsilon to avoid division by zero
        return 1.0 / (denominator + 1e-10)
    
    # Parallel computation of the spectrum
    for i in range(m):
        for j in range(m):
            spectrum[i, j] = compute_point(i, j)
    
    # Normalize the spectrum
    spectrum = spectrum / np.max(spectrum)
    
    return spectrum, X, Y

def plot_music_spectrum(spectrum, X, Y, save_path='results/music_spectrum.png'):
    """
    Plot the MUSIC spectrum.
    
    Args:
        spectrum (np.ndarray): MUSIC spectrum
        X (np.ndarray): X coordinates
        Y (np.ndarray): Y coordinates
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(X, Y, 10 * np.log10(spectrum + 1e-10), shading='auto')
    plt.colorbar(label='Power (dB)')
    plt.xlabel('X coordinate (m)')
    plt.ylabel('Y coordinate (m)')
    plt.title('MUSIC Spectrum')
    plt.grid(True)
    
    # Save the plot
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
