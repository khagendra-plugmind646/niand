import numpy as np
import matplotlib.pyplot as plt
from parameters import para_init
from channel_generation import generate_channel
from sdr_fully_digital import SDR_fully_digital
from rate_calculator import rate_calculator
from joblib import Parallel, delayed
import time

def run_simulation():
    # Initialize parameters
    para = para_init()
    scale = 1e2  # Scaling factor for optimization
    
    # Simulation parameters
    num_users = np.arange(2, 11)  # Number of users from 2 to 10
    num_trials = 30  # Number of trials for averaging
    
    # Store results
    sum_rates = np.zeros_like(num_users, dtype=float)
    
    # Start timing
    start_time = time.time()
    
    # Run simulation for each number of users
    for i, K in enumerate(num_users):
        para['K'] = K  # Update number of users
        print(f"Simulating for {K} users...")
        
        # Function to run a single trial
        def run_trial(_):
            # Generate channels
            H, G, beta_s, r, theta, r_s, theta_s = generate_channel(para)
            
            # Run SDR optimization
            Rx, f = SDR_fully_digital(para, H, beta_s, scale)
            
            # If optimization failed, return 0 rate
            if Rx is None or f is None:
                return 0
                
            # Calculate rates
            rates = rate_calculator(para, H, Rx, f)
            return np.sum(rates)
        
        # Run trials in parallel
        trial_results = Parallel(n_jobs=-1)(
            delayed(run_trial)(trial) for trial in range(num_trials)
        )
        
        # Calculate average sum rate
        sum_rates[i] = np.mean(trial_results)
        print(f"  - Average sum rate for {K} users: {sum_rates[i]:.2f} bits/s/Hz")
    
    # Print total simulation time
    total_time = time.time() - start_time
    print(f"\nTotal simulation time: {total_time/60:.2f} minutes")
    
    # Smooth the results
    from scipy.signal import savgol_filter
    sum_rates_smooth = savgol_filter(sum_rates, 3, 2)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(num_users, sum_rates_smooth, '-o', linewidth=1.5)
    plt.xlabel('Number of Users')
    plt.ylabel('Average Sum Rate (bits/s/Hz)')
    plt.title('SDMA Sum Rate vs Number of Users')
    plt.grid(True)
    
    # Save the plot
    plt.savefig('results/sum_rate_vs_users.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results to file
    results = np.column_stack((num_users, sum_rates, sum_rates_smooth))
    np.savetxt('results/sum_rates.csv', results, 
               delimiter=',', 
               header='num_users,sum_rate,sum_rate_smooth',
               comments='',
               fmt=['%d', '%.6f', '%.6f'])

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    import os
    os.makedirs('results', exist_ok=True)
    
    # Run the simulation
    run_simulation()
