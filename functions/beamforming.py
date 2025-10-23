import numpy as np

def beamfocusing(para, r, theta):
    """
    Calculate the near-field array response vector.
    
    Args:
        para (dict): Dictionary containing simulation parameters
        r (float): Distance from the array
        theta (float): Angle in radians
        
    Returns:
        np.ndarray: Array response vector (N x 1)
    """
    # Create array element positions
    n = np.arange(-(para['N']-1)/2, (para['N']-1)/2 + 1) * para['d']
    n = n.reshape(-1, 1)  # Convert to column vector
    
    # Calculate distance for each element
    # Using the exact spherical wavefront model
    distances = np.sqrt(r**2 + n**2 - 2 * r * n * np.cos(theta)) - r
    
    # Calculate beamfocusing vector and ensure it's a column vector (N x 1)
    a = np.exp(-1j * 2 * np.pi * para['f']/para['c'] * distances)
    
    # Ensure the output is a 2D column vector (N x 1)
    return a.reshape(-1, 1)
