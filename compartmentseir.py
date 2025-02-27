import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

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



# The rest of the code remains the same as in the previous version

# Combined supply chain and SEIR model differential equations
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

# Customizable functions with Monte Carlo parameters
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

# Simulation parameters
t_max = 180  # 30 days
t_eval = np.linspace(0, t_max, 1000)

# Initial conditions
N = 5000
I0, E0, R0 = 0, 0, 0
S0 = N - I0 - E0 - R0

initial_supplier_contamination = np.random.uniform(75, 200)
y0 = [initial_supplier_contamination, 0, 0, S0, E0, I0, R0]

# SEIR parameters
beta = 0.3
sigma = 1/5.2
gamma = 1/14
seir_params = (beta, sigma, gamma)

# Generate Monte Carlo parameters
R_params, I_params, T_params = generate_parameters()

# Solve ODEs
sol = solve_ivp(combined_model, (0, t_max), y0, 
                t_eval=t_eval,
                args=(lambda t, C, stage: R_func(t, C, stage, R_params),
                      lambda t, C, stage: I_func(t, C, stage, I_params),
                      lambda t, C, stage: T_func(t, C, stage, T_params),
                      seir_params))

Supplier, Processing, Store, S, E, I, R = sol.y

# Plot results
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18), sharex=True)

ax1.plot(sol.t, Supplier, label='Supplier Contamination')
ax1.plot(sol.t, Processing, label='Processing/Transport Contamination')
ax1.plot(sol.t, Store, label='Store Contamination')
ax1.set_ylabel('Contamination Level')
ax1.set_title('Contamination Spread in Supply Chain')
ax1.legend()
ax1.grid(True)

ax2.plot(sol.t, E, label='Consumer Contamination (Exposed)', color='purple')
ax2.set_ylabel('Contamination Level / Population')
ax2.set_title('Consumer Contamination')
ax2.legend()
ax2.grid(True)

ax3.plot(sol.t, S, label='Susceptible')
ax3.plot(sol.t, E, label='Exposed (Consumer Contamination)')
ax3.plot(sol.t, I, label='Infectious')
ax3.plot(sol.t, R, label='Recovered')
ax3.set_xlabel('Time (days)')
ax3.set_ylabel('Number of Individuals')
ax3.set_title('SEIR Model Progression')
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.show()

# Print the generated parameters
print("R_params:", R_params)
print("I_params:", I_params)
print("T_params:", T_params)




# import numpy as np
# from scipy.integrate import solve_ivp
# import matplotlib.pyplot as plt

# # Function to generate random contamination events (unchanged)
# def generate_contamination_events(t_max, avg_events=5):
#     num_events = np.random.poisson(avg_events)
#     event_times = np.sort(np.random.uniform(0, t_max, num_events))
#     event_magnitudes = np.random.lognormal(3, 1, num_events)
#     return list(zip(event_times, event_magnitudes))

# # Combined supply chain and SEIR model differential equations (unchanged)
# def combined_model(t, y, R_func, I_func, T_func, seir_params):
#     Supplier, Processing, Store, S, E, I, R = y
    
#     # Supply chain model
#     R_S = R_func(t, Supplier, 'S')
#     R_P = R_func(t, Processing, 'P')
#     R_St = R_func(t, Store, 'St')
#     R_E = R_func(t, E, 'E')  # E is now Consumer
    
#     I_S = I_func(t, Supplier, 'S')
#     I_P = I_func(t, Processing, 'P')
#     I_St = I_func(t, Store, 'St')
#     I_E = I_func(t, E, 'E')  # E is now Consumer
    
#     T_SP = T_func(t, Supplier, 'SP')
#     T_PS = T_func(t, Processing, 'PS')
#     T_StE = T_func(t, Store, 'StE')  # Transfer from Store to E (Consumer)

#     dSupplier_dt = I_S - R_S - T_SP
#     dProcessing_dt = I_P - R_P + T_SP - T_PS
#     dStore_dt = I_St - R_St + T_PS - T_StE

#     # SEIR model
#     beta, sigma, gamma = seir_params
#     N = S + E + I + R
    
#     dS_dt = -beta * S * I / N - (I_E - R_E + T_StE) 
#     dE_dt = beta * S * I / N - sigma * E + (I_E - R_E + T_StE) 
#     dI_dt = sigma * E - gamma * I
#     dR_dt = gamma * I

#     return [dSupplier_dt, dProcessing_dt, dStore_dt, dS_dt, dE_dt, dI_dt, dR_dt]

# # Customizable functions (unchanged)
# def R_func(t, C, stage):
#     gamma = {'S': 0.05, 'P': 0.04, 'St': 0.03, 'E': 0.02}
#     delta = {'S': 0.05, 'P': 0.04, 'St': 0.03, 'E': 0.02}
#     theta = {'S': 0.1, 'P': 0.09, 'St': 0.08, 'E': 0.07}
#     return gamma[stage] * C * (1 + delta[stage] * np.cos(theta[stage] * t))

# def I_func(t, C, stage):
#     lambda_val = {'S': 0.01, 'P': 0.015, 'St': 0.02, 'E': 0.025}
#     epsilon = {'S': 0.02, 'P': 0.025, 'St': 0.03, 'E': 0.035}
#     phi = {'S': 0.05, 'P': 0.06, 'St': 0.07, 'E': 0.08}
#     return lambda_val[stage] * C * (1 + epsilon[stage] * np.sin(phi[stage] * t))

# def T_func(t, C, stage):
#     tau = {'SP': 0.1, 'PS': 0.08, 'StE': 0.06}
#     return tau[stage] * C

# # Simulation parameters
# t_max = 200
# t_eval = np.linspace(0, t_max, 1000)

# # Generate random contamination events
# contamination_events = generate_contamination_events(t_max)

# # Initial conditions (including SEIR)
# N = 1000  # Total population
# I0 = 0  # Initial number of infected individuals
# E0 = 0  # Initial number of exposed individuals (also initial Consumer contamination)
# R0 = 0  # Initial number of recovered individuals
# S0 = N - I0 - E0 - R0  # Initial number of susceptible individuals

# y0 = [100, 0, 0, S0, E0, I0, R0]  # Supplier0, Processing0, Store0, S0, E0, I0, R0

# # SEIR parameters
# beta = 0.3  # Infection rate
# sigma = 1/5.2  # Incubation rate (1/average incubation period)
# gamma = 1/14  # Recovery rate (1/average infectious period)
# seir_params = (beta, sigma, gamma)

# # Solve ODEs with events
# results = []
# t_span = (0, t_max)

# for event_time, event_magnitude in contamination_events:
#     # Solve up to the next event
#     sol = solve_ivp(combined_model, (t_span[0], event_time), y0, 
#                     t_eval=t_eval[(t_eval >= t_span[0]) & (t_eval <= event_time)],
#                     args=(R_func, I_func, T_func, seir_params))
    
#     results.append(sol)
    
#     # Update initial conditions for next integration
#     y0 = sol.y[:, -1]
#     y0[0] += event_magnitude  # Add contamination to Supplier
    
#     # Directly affect Exposed and Recovered values
#     recontamination_rate = 0.1  # Adjust this value as needed
#     total_population = sum(y0[3:])  # S + E + I + R
#     exposed_increase = event_magnitude * recontamination_rate
    
#     # Calculate the proportion of Recovered individuals to be re-exposed
#     if y0[6] > 0:  # If there are Recovered individuals
#         recovered_proportion = y0[6] / total_population
#         recovered_to_exposed = min(y0[6], exposed_increase * recovered_proportion)
#         y0[6] -= recovered_to_exposed  # Decrease Recovered
#         exposed_increase -= recovered_to_exposed
    
#     # The remaining exposed_increase affects the Susceptible population
#     y0[3] -= exposed_increase  # Decrease Susceptible
#     y0[4] += exposed_increase + recovered_to_exposed  # Increase Exposed
    
#     t_span = (event_time, t_max)

# # Solve final part after last event
# sol = solve_ivp(combined_model, t_span, y0, 
#                 t_eval=t_eval[t_eval >= t_span[0]],
#                 args=(R_func, I_func, T_func, seir_params))
# results.append(sol)

# # Combine results
# t = np.concatenate([res.t for res in results])
# y = np.column_stack([res.y for res in results])

# Supplier, Processing, Store, S, E, I, R = y

# # Plot results
# fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18), sharex=True)

# # Supply chain plot
# ax1.plot(t, Supplier, label='Supplier Contamination')
# ax1.plot(t, Processing, label='Processing/Transport Contamination')
# ax1.plot(t, Store, label='Store Contamination')

# # Plot contamination events
# for event_time, event_magnitude in contamination_events:
#     ax1.axvline(x=event_time, color='r', linestyle='--', alpha=0.5)
#     ax1.text(event_time, ax1.get_ylim()[1], f'+{event_magnitude:.1f}', rotation=90, va='top')

# ax1.set_ylabel('Contamination Level')
# ax1.set_title('Contamination Spread in Supply Chain')
# ax1.legend()
# ax1.grid(True)

# # Consumer contamination plot
# ax2.plot(t, E, label='Consumer Contamination (Exposed)', color='purple')
# ax2.set_ylabel('Contamination Level / Population')
# ax2.set_title('Consumer Contamination')
# ax2.legend()
# ax2.grid(True)

# # SEIR plot
# ax3.plot(t, S, label='Susceptible')
# ax3.plot(t, E, label='Exposed (Consumer Contamination)')
# ax3.plot(t, I, label='Infectious')
# ax3.plot(t, R, label='Recovered')

# ax3.set_xlabel('Time')
# ax3.set_ylabel('Number of Individuals')
# ax3.set_title('SEIR Model Progression')
# ax3.legend()
# ax3.grid(True)

# plt.tight_layout()
# plt.show()
