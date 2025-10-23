import numpy as np

def para_init():
    """
    Initialize and return a dictionary containing simulation parameters.
    
    Returns:
        dict: A dictionary containing all simulation parameters.
    """
    para = {}
    
    # Antenna and system parameters
    para['N'] = 65        # Number of transmit antennas
    para['N_RF'] = 5      # Number of RF chains
    para['T'] = 128       # Length of coherent time block
    
    # Power and communication parameters
    para['Pt'] = 10**(20/10)  # Overall transmit power (mW)
    para['K'] = 4            # Number of users
    para['noise'] = 10**(-60/10)  # Noise power in mW
    para['Rmin'] = 5         # Minimum communication rate (bits/s/Hz)
    
    # Physical constants and antenna parameters
    para['c'] = 3e8          # Speed of light (m/s)
    para['f'] = 28e9         # Carrier frequency (Hz)
    para['D'] = 0.5          # Antenna aperture (m)
    para['d'] = para['D'] / (para['N'] - 1)  # Antenna spacing (m)
    
    # Reference pathloss
    para['rho_0'] = 1 / (4 * np.pi * para['f'] / para['c'])
    
    # Target location parameters
    para['r_s'] = 20                 # Distance to target (m)
    para['theta_s'] = 45 * np.pi/180  # Angle to target (radians)
    para['Pmax'] = 1                 # Maximum power constraint
    
    return para
