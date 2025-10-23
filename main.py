import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from joblib import Parallel, delayed
import time
import os
import traceback
from functions.parameters import para_init
from functions.channel_generation import generate_channel
from functions.sdr_fully_digital import SDR_fully_digital
from functions.rate_calculator import rate_calculator

def process_trial(para, scale, trial_num):
    """Process a single trial for a given parameter set."""
    try:
        # Generate channel
        H, G, beta_s, r, theta, r_s, theta_s = generate_channel(para)
        
        # Run SDR optimization
        print(f"  - [Trial {trial_num}] Starting SDR optimization...")
        Rx, f = SDR_fully_digital(para, H, beta_s, scale)
        
        if Rx is None or f is None:
            print(f"  - [Trial {trial_num}] SDR optimization failed: No solution found")
            return None
            
        # Normalize Rx to satisfy power constraint
        # This is a critical step
        Rx_trace = np.trace(Rx)
        if Rx_trace < 1e-6: # Avoid division by zero if Rx is effectively zero
            print(f"  - [Trial {trial_num}] SDR solution is near-zero. Skipping.")
            return 0.0

        Rx = Rx / Rx_trace * para['Pmax']
        
        # Calculate rates
        rates = rate_calculator(para, H, Rx, f)
        sum_rate = np.sum(rates)
        print(f"  - [Trial {trial_num}] Succeeded. Sum rate: {sum_rate:.2f} bits/s/Hz")
        return sum_rate
        
    except Exception as e:
        print(f"\n  - [Trial {trial_num}] Error: {str(e)}")
        print("  - Traceback (most recent call last):")
        traceback.print_exc(limit=2)
        return None

def run_simulation():
    """
    Main function to run the simulation for different numbers of users.
    
    This function simulates the system for different numbers of users,
    calculates the sum rates, and plots the results.
    """
    # Initialize parameters
    para = para_init()
    scale = 1e2  # Scaling factor for optimization
    
    # Simulation parameters (updated to run a full simulation)
    num_users = np.arange(2, 11)  # Number of users from 2 to 10
    num_trials = 30  # Number of trials for averaging
    
    # Store results
    sum_rates = np.zeros_like(num_users, dtype=float)
    
    # Start timing
    start_time = time.time()
    
    # Use all available CPU cores, but cap at a reasonable number (e.g., 8)
    # or just use n_jobs=-1 to use all. -2 means all except one.
    n_jobs = -2 
    print(f"Running simulation with n_jobs={n_jobs}")

    # Run simulation for each number of users
    for i, K in enumerate(num_users):
        para['K'] = K  # Update number of users
        print(f"\n{'='*50}")
        print(f"Simulating for {K} users ({num_trials} trials)...")
        
        # Run trials in parallel
        # Note: 'para' and 'scale' are copied to each worker process
        trial_results = Parallel(n_jobs=n_jobs)(
            delayed(process_trial)(para.copy(), scale, t+1) for t in range(num_trials)
        )
        
        # Filter out failed trials (which returned None)
        successful_results = [r for r in trial_results if r is not None]
        
        if successful_results:
            sum_rates[i] = np.mean(successful_results)
            print(f"\n  - Completed {len(successful_results)}/{num_trials} successful trials")
            print(f"  - Average sum rate for {K} users: {sum_rates[i]:.2f} bits/s/Hz")
        else:
            print(f"\n  - No successful trials for {K} users. Check for optimization errors.")
            sum_rates[i] = 0
    
    # Print total simulation time
    total_time = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"Total simulation time: {total_time/60:.2f} minutes")
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    
    # Only apply Savitzky-Golay filter if we have enough points
    # The filter requires window_length > polynomial_order
    window_len = min(5, len(num_users))
    if window_len % 2 == 0: window_len -= 1 # Window must be odd
    poly_order = 2

    if len(num_users) > poly_order and window_len > poly_order:
        sum_rates_smooth = savgol_filter(sum_rates, window_len, poly_order)
        plt.plot(num_users, sum_rates, 's-', color='gray', alpha=0.5, label='Raw Data')
        plt.plot(num_users, sum_rates_smooth, '-o', linewidth=2, label='Smoothed (Savitzky-Golay)')
        
        # Save results with smoothing
        results = np.column_stack((num_users, sum_rates, sum_rates_smooth))
        np.savetxt('results/sum_rates.csv', results, 
                  delimiter=',', 
                  header='num_users,sum_rate,sum_rate_smooth',
                  comments='',
                  fmt=['%d', '%.6f', '%.6f'])
    else:
        plt.plot(num_users, sum_rates, 'o-', linewidth=1.5, label='Raw Data')
        
        # Save results without smoothing
        results = np.column_stack((num_users, sum_rates))
        np.savetxt('results/sum_rates.csv', results, 
                  delimiter=',', 
                  header='num_users,sum_rate',
                  comments='',
                  fmt=['%d', '%.6f'])
    
    plt.xlabel('Number of Users (K)')
    plt.ylabel('Average Sum Rate (bits/s/Hz)')
    plt.title('SDMA Sum Rate vs Number of Users')
    plt.grid(True)
    plt.legend()
    plt.xticks(num_users)
    
    # Save the plot
    plot_path = 'results/sum_rate_vs_users.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Results saved to results/sum_rates.csv and {plot_path}")
    plt.close()

if __name__ == "__main__":
    run_simulation()