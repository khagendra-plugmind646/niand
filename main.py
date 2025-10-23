import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from joblib import Parallel, delayed
import time
from functions.parameters import para_init
from functions.channel_generation import generate_channel
from functions.sdr_fully_digital import SDR_fully_digital
from functions.rate_calculator import rate_calculator

def run_simulation():
    """
    Main function to run the simulation for different numbers of users.
    
    This function simulates the system for different numbers of users,
    calculates the sum rates, and plots the results.
    """
    # Initialize parameters
    para = para_init()
    scale = 1e2  # Scaling factor for optimization
    
    # Simulation parameters - reduced for initial testing
    num_users = np.array([2])  # Start with just 2 users
    num_trials = 2  # Reduce number of trials for testing
    
    # Store results
    sum_rates = np.zeros_like(num_users, dtype=float)
    
    # Start timing
    start_time = time.time()
    
    def process_trial(para, scale):
        """Process a single trial for a given parameter set."""
        try:
            # Generate channel
            H, G, beta_s, r, theta, r_s, theta_s = generate_channel(para)
            print(f"  - Channel generated. Shape H: {H.shape}, beta_s: {beta_s.shape if hasattr(beta_s, 'shape') else 'scalar'}")
            
            # Run SDR optimization
            print("  - Starting SDR optimization...")
            Rx, f = SDR_fully_digital(para, H, beta_s, scale)
            
            if Rx is None or f is None:
                print("  - SDR optimization failed: No solution found")
                return None
                
            # Normalize Rx to satisfy power constraint
            print(f"  - Normalizing solution. Rx trace before: {np.trace(Rx):.2f}")
            Rx = Rx / np.trace(Rx) * para['Pmax']
            
            # Calculate rates
            print("  - Calculating rates...")
            rates = rate_calculator(para, H, Rx, f)
            sum_rate = np.sum(rates)
            print(f"  - Trial completed successfully. Sum rate: {sum_rate:.2f} bits/s/Hz")
            return sum_rate
            
        except Exception as e:
            import traceback
            print(f"\n  - Error in trial: {str(e)}")
            print("  - Traceback (most recent call last):")
            traceback.print_exc(limit=1)  # Print just the last line of the traceback
            return None
    
    # Run simulation for each number of users
    for i, K in enumerate(num_users):
        para['K'] = K  # Update number of users
        print(f"\n{'='*50}")
        print(f"Simulating for {K} users...")
        
        # Process trials with detailed progress tracking
        trial_results = []
        for t in range(num_trials):
            print(f"\n--- Trial {t+1}/{num_trials} ---")
            result = process_trial(para, scale)
            if result is not None:
                trial_results.append(result)
            else:
                print(f"  - Trial {t+1} failed, skipping...")
        
        if trial_results:
            sum_rates[i] = np.mean(trial_results)
            print(f"\n  - Completed {len(trial_results)}/{num_trials} trials")
            print(f"  - Average sum rate for {K} users: {sum_rates[i]:.2f} bits/s/Hz")
        else:
            print("\n  - No successful trials. Check for optimization errors.")
            sum_rates[i] = 0
    
    # Print total simulation time
    total_time = time.time() - start_time
    print(f"\nTotal simulation time: {total_time/60:.2f} minutes")
    
    # Create results directory if it doesn't exist
    import os
    os.makedirs('results', exist_ok=True)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    
    # Only apply Savitzky-Golay filter if we have enough points
    if len(num_users) > 3:  # Need at least 3 points for window size 3
        sum_rates_smooth = savgol_filter(sum_rates, 3, 2)  # window size 3, polynomial order 2
        plt.plot(num_users, sum_rates_smooth, '-o', linewidth=1.5, label='Smoothed')
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
    
    plt.xlabel('Number of Users')
    plt.ylabel('Average Sum Rate (bits/s/Hz)')
    plt.title('SDMA Sum Rate vs Number of Users')
    plt.grid(True)
    plt.legend()
    
    # Save the plot
    plt.savefig('results/sum_rate_vs_users.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    run_simulation()
