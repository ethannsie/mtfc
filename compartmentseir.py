import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Function to generate random contamination events (unchanged)
def generate_contamination_events(t_max, avg_events=5):
    num_events = np.random.poisson(avg_events)
    event_times = np.sort(np.random.uniform(0, t_max, num_events))
    event_magnitudes = np.random.lognormal(3, 1, num_events)
    return list(zip(event_times, event_magnitudes))

# Combined supply chain and SEIR model differential equations
def combined_model(t, y, G_func, R_func, I_func, T_func, seir_params):
    Supplier, Processing, Store, Consumer, S, E, I, R = y
    
    # Supply chain model (unchanged)
    G_S = G_func(t, Supplier, 'S')
    G_P = G_func(t, Processing, 'P')
    G_St = G_func(t, Store, 'St')
    G_C = G_func(t, Consumer, 'C')
    
    R_S = R_func(t, Supplier, 'S')
    R_P = R_func(t, Processing, 'P')
    R_St = R_func(t, Store, 'St')
    R_C = R_func(t, Consumer, 'C')
    
    I_S = I_func(t, Supplier, 'S')
    I_P = I_func(t, Processing, 'P')
    I_St = I_func(t, Store, 'St')
    I_C = I_func(t, Consumer, 'C')
    
    T_SP = T_func(t, Supplier, 'SP')
    T_PS = T_func(t, Processing, 'PS')
    T_StC = T_func(t, Store, 'StC')

    dSupplier_dt = G_S + I_S - R_S - T_SP
    dProcessing_dt = G_P + I_P - R_P + T_SP - T_PS
    dStore_dt = G_St + I_St - R_St + T_PS - T_StC
    dConsumer_dt = G_C + I_C - R_C + T_StC

    # SEIR model
    beta, sigma, gamma = seir_params
    N = S + E + I + R
    
    dS_dt = -beta * S * I / N - beta * S * Consumer / N  # Added Consumer contamination
    dE_dt = beta * S * I / N + beta * S * Consumer / N - sigma * E
    dI_dt = sigma * E - gamma * I
    dR_dt = gamma * I

    return [dSupplier_dt, dProcessing_dt, dStore_dt, dConsumer_dt, dS_dt, dE_dt, dI_dt, dR_dt]

# Customizable functions (unchanged)
def G_func(t, C, stage):
    alpha = {'S': 0.03, 'P': 0.02, 'St': 0.01, 'C': 0.005}
    beta = {'S': 0.1, 'P': 0.08, 'St': 0.06, 'C': 0.04}
    omega = {'S': 0.1, 'P': 0.09, 'St': 0.08, 'C': 0.07}
    return alpha[stage] * C * (1 + beta[stage] * np.sin(omega[stage] * t))

def R_func(t, C, stage):
    gamma = {'S': 0.05, 'P': 0.04, 'St': 0.03, 'C': 0.02}
    delta = {'S': 0.05, 'P': 0.04, 'St': 0.03, 'C': 0.02}
    theta = {'S': 0.1, 'P': 0.09, 'St': 0.08, 'C': 0.07}
    return gamma[stage] * C * (1 + delta[stage] * np.cos(theta[stage] * t))

def I_func(t, C, stage):
    lambda_val = {'S': 0.01, 'P': 0.015, 'St': 0.02, 'C': 0.025}
    epsilon = {'S': 0.02, 'P': 0.025, 'St': 0.03, 'C': 0.035}
    phi = {'S': 0.05, 'P': 0.06, 'St': 0.07, 'C': 0.08}
    return lambda_val[stage] * C * (1 + epsilon[stage] * np.sin(phi[stage] * t))

def T_func(t, C, stage):
    tau = {'SP': 0.1, 'PS': 0.08, 'StC': 0.06}
    return tau[stage] * C

# Simulation parameters
t_max = 200
t_eval = np.linspace(0, t_max, 1000)

# Generate random contamination events
contamination_events = generate_contamination_events(t_max)

# Initial conditions (including SEIR)
N = 10000  # Total population
I0 = 10  # Initial number of infected individuals
E0 = 100  # Initial number of exposed individuals
R0 = 0  # Initial number of recovered individuals
S0 = N - I0 - E0 - R0  # Initial number of susceptible individuals

y0 = [100, 0, 0, 0, S0, E0, I0, R0]  # Supplier0, Processing0, Store0, Consumer0, S0, E0, I0, R0

# SEIR parameters
beta = 0.3  # Infection rate
sigma = 1/5.2  # Incubation rate (1/average incubation period)
gamma = 1/14  # Recovery rate (1/average infectious period)
seir_params = (beta, sigma, gamma)

# Solve ODEs with events
results = []
t_span = (0, t_max)

for event_time, event_magnitude in contamination_events:
    # Solve up to the next event
    sol = solve_ivp(combined_model, (t_span[0], event_time), y0, 
                    t_eval=t_eval[(t_eval >= t_span[0]) & (t_eval <= event_time)],
                    args=(G_func, R_func, I_func, T_func, seir_params))
    
    results.append(sol)
    
    # Update initial conditions for next integration
    y0 = sol.y[:, -1]
    y0[0] += event_magnitude  # Add contamination to Supplier
    t_span = (event_time, t_max)

# Solve final part after last event
sol = solve_ivp(combined_model, t_span, y0, 
                t_eval=t_eval[t_eval >= t_span[0]],
                args=(G_func, R_func, I_func, T_func, seir_params))
results.append(sol)

# Combine results
t = np.concatenate([res.t for res in results])
y = np.column_stack([res.y for res in results])

Supplier, Processing, Store, Consumer, S, E, I, R = y

# Plot results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16), sharex=True)

# Supply chain plot
ax1.plot(t, Supplier, label='Supplier Contamination')
ax1.plot(t, Processing, label='Processing/Transport Contamination')
ax1.plot(t, Store, label='Store Contamination')
ax1.plot(t, Consumer, label='Consumer Contamination')

# Plot contamination events
for event_time, event_magnitude in contamination_events:
    ax1.axvline(x=event_time, color='r', linestyle='--', alpha=0.5)
    ax1.text(event_time, ax1.get_ylim()[1], f'+{event_magnitude:.1f}', rotation=90, va='top')

ax1.set_ylabel('Contamination Level')
ax1.set_title('Contamination Spread in Supply Chain with Random Events')
ax1.legend()
ax1.grid(True)

# SEIR plot
ax2.plot(t, S, label='Susceptible')
ax2.plot(t, E, label='Exposed')
ax2.plot(t, I, label='Infectious')
ax2.plot(t, R, label='Recovered')

ax2.set_xlabel('Time')
ax2.set_ylabel('Number of Individuals')
ax2.set_title('SEIR Model Progression')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
