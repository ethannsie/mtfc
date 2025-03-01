import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd

def generate_parameters():
    R_params = {
        'gamma': np.random.normal(0.05, 0.005, 4),
        'delta': np.random.normal(0.05, 0.005, 4),
        'theta': np.random.normal(0.1, 0.01, 4)
    }
    I_params = {
        'lambda': np.random.normal(0.02, 0.002, 4),
        'epsilon': np.random.normal(0.03, 0.003, 4),
        'phi': np.random.normal(0.07, 0.007, 4)
    }
    T_params = {
        'tau': np.random.normal(0.08, 0.008, 3)
    }
    return R_params, I_params, T_params

def combined_model(t, y, R_func, I_func, T_func, seir_params):
    Supplier, Processing, Store, S, E, I, R = y
    
    R_S = R_func(t, Supplier, 'S')
    R_P = R_func(t, Processing, 'P')
    R_St = R_func(t, Store, 'St')
    R_E = R_func(t, E, 'E')
    
    I_S = I_func(t, Supplier, 'S')
    I_P = I_func(t, Processing, 'P')
    I_St = I_func(t, Store, 'St')
    I_E = I_func(t, E, 'E')
    
    T_SP = T_func(t, Supplier, 'SP')
    T_PS = T_func(t, Processing, 'PS')
    T_StE = T_func(t, Store, 'StE')

    dSupplier_dt = I_S - R_S - T_SP
    dProcessing_dt = I_P - R_P + T_SP - T_PS
    dStore_dt = I_St - R_St + T_PS - T_StE

    beta, sigma, gamma = seir_params
    N = S + E + I + R
    
    dS_dt = -beta * S * I / N - (I_E - R_E + T_StE) 
    dE_dt = beta * S * I / N - sigma * E + (I_E - R_E + T_StE) 
    dI_dt = sigma * E - gamma * I
    dR_dt = gamma * I

    return [dSupplier_dt, dProcessing_dt, dStore_dt, dS_dt, dE_dt, dI_dt, dR_dt]

def R_func(t, C, stage, params):
    stages = ['S', 'P', 'St', 'E']
    i = stages.index(stage)
    return params['gamma'][i] * C * (1 + params['delta'][i] * np.cos(params['theta'][i] * t))

def I_func(t, C, stage, params):
    stages = ['S', 'P', 'St', 'E']
    i = stages.index(stage)
    return params['lambda'][i] * C * (1 + params['epsilon'][i] * np.sin(params['phi'][i] * t))

def T_func(t, C, stage, params):
    stages = ['SP', 'PS', 'StE']
    i = stages.index(stage)
    return params['tau'][i] * C

t_max = 180
t_eval = np.linspace(0, t_max, 1000)

N = 5000
I0, E0, R0 = 0, 0, 0
S0 = N - I0 - E0 - R0

initial_supplier_contamination = np.random.uniform(75, 200)
y0 = [initial_supplier_contamination, 0, 0, S0, E0, I0, R0]

beta = 0.3
sigma = 1/5.2
gamma = 1/14
seir_params = (beta, sigma, gamma)

def run_simulation(R_params, I_params, T_params, threshold=10):
    sol = solve_ivp(combined_model, (0, t_max), y0,
                    t_eval=t_eval,
                    args=(lambda t, C, stage: R_func(t, C, stage, R_params),
                          lambda t, C, stage: I_func(t, C, stage, I_params),
                          lambda t, C, stage: T_func(t, C, stage, T_params),
                          seir_params))     

    Supplier, Processing, Store, S, E, I, R = sol.y
    total_infected_time = np.trapz(I, sol.t)
    
    above_threshold = I > threshold
    if np.any(above_threshold):
        start_time = sol.t[np.argmax(above_threshold)]
        end_time = sol.t[np.where(above_threshold)[0][-1]]
        epidemic_duration = end_time - start_time
    else:
        epidemic_duration = 0

    return total_infected_time, epidemic_duration

def generate_tweaked_parameters(base_R_params, base_I_params, base_T_params, param_name, stage, tweak_factor):
    tweaked_R_params, tweaked_I_params, tweaked_T_params = base_R_params.copy(), base_I_params.copy(), base_T_params.copy()
    stages = ['S', 'P', 'St', 'E']
    stages_T = ['SP', 'PS', 'StE']

    if stage in stages:
        stage_index = stages.index(stage)
    elif stage in stages_T:
        stage_index = stages_T.index(stage)
    else:
        stage_index = None

    if param_name in tweaked_R_params:
        if stage_index is not None:
            tweaked_R_params[param_name][stage_index] = base_R_params[param_name][stage_index] * (1 + tweak_factor)
    elif param_name in tweaked_I_params:
        if stage_index is not None:
            tweaked_I_params[param_name][stage_index] = base_I_params[param_name][stage_index] * (1 + tweak_factor)
    elif param_name in tweaked_T_params and param_name == 'tau':
        tweaked_T_params[param_name] = base_T_params[param_name] * (1 + tweak_factor)
    return tweaked_R_params, tweaked_I_params, tweaked_T_params

# Parameters and stages for sensitivity analysis
params_to_analyze = [
    ('gamma', 'S'), ('gamma', 'P'), ('gamma', 'St'), ('gamma', 'E'),
    ('delta', 'S'), ('delta', 'P'), ('delta', 'St'), ('delta', 'E'),
    ('theta', 'S'), ('theta', 'P'), ('theta', 'St'), ('theta', 'E'),
    ('lambda', 'S'), ('lambda', 'P'), ('lambda', 'St'), ('lambda', 'E'),
    ('epsilon', 'S'), ('epsilon', 'P'), ('epsilon', 'St'), ('epsilon', 'E'),
    ('phi', 'S'), ('phi', 'P'), ('phi', 'St'), ('phi', 'E'),
    ('tau', 'SP'), ('tau', 'PS'), ('tau', 'StE')
]

# Tweak factors to apply
tweak_factors = [0.1, 0.25, 0.5]

# Generate base parameters
base_R_params, base_I_params, base_T_params = generate_parameters()
total_infected_time_base, epidemic_duration_base = run_simulation(base_R_params, base_I_params, base_T_params)

# Perform sensitivity analysis for each tweak factor
all_sensitivity_results = []

for tweak_factor in tweak_factors:
    sensitivity_results = []
    print(f"\n--- Sensitivity Analysis with Tweak Factor: {tweak_factor:.2f} ---")
    for param_name, stage in params_to_analyze:
        print(f"Analyzing: {param_name} - {stage}")
        tweaked_R_params, tweaked_I_params, tweaked_T_params = generate_tweaked_parameters(
            base_R_params, base_I_params, base_T_params, param_name, stage, tweak_factor
        )
        total_infected_time, epidemic_duration = run_simulation(tweaked_R_params, tweaked_I_params, tweaked_T_params)
        sensitivity_results.append({
            'parameter': param_name,
            'stage': stage,
            'tweak_factor': tweak_factor,
            'total_infected_time': total_infected_time,
            'epidemic_duration': epidemic_duration
        })
        print(f"Total Infected-Time: {total_infected_time}")
        print(f"Epidemic Duration: {epidemic_duration}")
    all_sensitivity_results.extend(sensitivity_results)

# Analyze and visualize the results
df = pd.DataFrame(all_sensitivity_results)
df['infected_time_change'] = df.groupby(['parameter', 'stage'])['total_infected_time'].transform(lambda x: x - total_infected_time_base)
df['duration_change'] = df.groupby(['parameter', 'stage'])['epidemic_duration'].transform(lambda x: x - epidemic_duration_base)

# Plotting
for metric in ['infected_time_change', 'duration_change']:
    plt.figure(figsize=(14, 8))

    # Prepare data for plotting each tweak factor on the same chart
    bar_width = 0.2  # Width of each individual bar
    group_width = len(tweak_factors) * bar_width  # Total width of each group of bars

    # Unique parameter-stage combinations for x-axis labels
    unique_combinations = sorted(df['parameter'] + '-' + df['stage'])
    num_combinations = len(unique_combinations)

    # X-axis positions for each group of bars
    x = np.arange(num_combinations)

    # Iterate through each tweak factor and plot the bars
    for i, tweak_factor in enumerate(tweak_factors):
        # Extract the data for the current tweak factor
        subset = df[df['tweak_factor'] == tweak_factor]
        
        # Reindex the data frame to use the sorted and unique parameter/stage
        subset['param_stage'] = subset['parameter'] + '-' + subset['stage']
        subset = subset.set_index('param_stage').reindex(unique_combinations)
    
        # Map the 'parameter-stage' combinations to their corresponding change values
        values = subset['infected_time_change' if metric == 'infected_time_change' else 'duration_change'].fillna(0)
        
        # Calculate the x-positions for the current group of bars
        bar_positions = x + i * bar_width
        
        # Plot the bars
        plt.bar(bar_positions, values, width=bar_width, label=f'Tweak {tweak_factor}')

    # Customize the plot
    plt.xlabel('Parameter - Stage')
    plt.ylabel(metric)
    plt.title(f'Sensitivity Analysis: Impact on {metric} (Grouped)')
    plt.xticks(x + group_width / 2 - bar_width/2, unique_combinations, rotation=90)  # Centered x-axis labels
    plt.legend()
    plt.tight_layout()
    plt.show()
