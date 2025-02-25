import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Function to generate random contamination events
def generate_contamination_events(t_max, avg_events=5):
    num_events = np.random.poisson(avg_events)
    event_times = np.sort(np.random.uniform(0, t_max, num_events))
    event_magnitudes = np.random.lognormal(3, 1, num_events)
    return list(zip(event_times, event_magnitudes))

# Compartment model differential equations
def supply_chain_model(t, y, G_func, R_func, I_func, T_func):
    Supplier, Processing, Store, Consumer = y
    
    # Growth
    G_S = G_func(t, Supplier, 'S')
    G_P = G_func(t, Processing, 'P')
    G_St = G_func(t, Store, 'St')
    G_C = G_func(t, Consumer, 'C')
    
    # Reduction
    R_S = R_func(t, Supplier, 'S')
    R_P = R_func(t, Processing, 'P')
    R_St = R_func(t, Store, 'St')
    R_C = R_func(t, Consumer, 'C')
    
    # Bad handling (increases contamination)
    I_S = I_func(t, Supplier, 'S')
    I_P = I_func(t, Processing, 'P')
    I_St = I_func(t, Store, 'St')
    I_C = I_func(t, Consumer, 'C')
    
    # Transfer between stages
    T_SP = T_func(t, Supplier, 'SP')
    T_PS = T_func(t, Processing, 'PS')
    T_StC = T_func(t, Store, 'StC')

    dSupplier_dt = G_S + I_S - R_S - T_SP
    dProcessing_dt = G_P + I_P - R_P + T_SP - T_PS
    dStore_dt = G_St + I_St - R_St + T_PS - T_StC
    dConsumer_dt = G_C + I_C - R_C + T_StC

    return [dSupplier_dt, dProcessing_dt, dStore_dt, dConsumer_dt]

# Customizable functions
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

# Initial conditions
y0 = [100, 0, 0, 0]  # Supplier0, Processing0, Store0, Consumer0

# Solve ODEs with events
results = []
t_span = (0, t_max)

for event_time, event_magnitude in contamination_events:
    # Solve up to the next event
    sol = solve_ivp(supply_chain_model, (t_span[0], event_time), y0, 
                    t_eval=t_eval[(t_eval >= t_span[0]) & (t_eval <= event_time)],
                    args=(G_func, R_func, I_func, T_func))
    
    results.append(sol)
    
    # Update initial conditions for next integration
    y0 = sol.y[:, -1]
    y0[0] += event_magnitude  # Add contamination to Supplier
    t_span = (event_time, t_max)

# Solve final part after last event
sol = solve_ivp(supply_chain_model, t_span, y0, 
                t_eval=t_eval[t_eval >= t_span[0]],
                args=(G_func, R_func, I_func, T_func))
results.append(sol)

# Combine results
t = np.concatenate([res.t for res in results])
y = np.column_stack([res.y for res in results])

Supplier, Processing, Store, Consumer = y

# Plot results
plt.figure(figsize=(12,8))
plt.plot(t, Supplier, label='Supplier Contamination')
plt.plot(t, Processing, label='Processing/Transport Contamination')
plt.plot(t, Store, label='Store Contamination')
plt.plot(t, Consumer, label='Consumer Contamination')

# Plot contamination events
for event_time, event_magnitude in contamination_events:
    plt.axvline(x=event_time, color='r', linestyle='--', alpha=0.5)
    plt.text(event_time, plt.ylim()[1], f'+{event_magnitude:.1f}', rotation=90, va='top')

plt.xlabel('Time')
plt.ylabel('Contamination Level')
plt.title('Contamination Spread in Supply Chain with Random Events')
plt.legend()
plt.grid(True)
plt.show()



# import numpy as np
# from scipy.integrate import odeint
# import matplotlib.pyplot as plt

# # SEIR model differential equations
# def seir_model(y, t, N, beta, sigma, gamma):
#     S, E, I, R = y
#     dSdt = -beta * S * I / N
#     dEdt = beta * S * I / N - sigma * E
#     dIdt = sigma * E - gamma * I
#     dRdt = gamma * I
#     return [dSdt, dEdt, dIdt, dRdt]

# # Initial conditions
# N = 1000  # Total population
# E0 = 10   # Initially exposed
# I0 = 5    # Initially infected
# R0 = 0    # Initially recovered
# S0 = N - E0 - I0 - R0  # Susceptible

# # Model parameters
# beta = 0.3   # Transmission rate
# sigma = 1/5  # Incubation rate (1/days)
# gamma = 1/7  # Recovery rate (1/days)

# # Time grid
# t = np.linspace(0, 100, 100)  # Simulating for 100 days

# # Solve ODEs
# solution = odeint(seir_model, [S0, E0, I0, R0], t, args=(N, beta, sigma, gamma))
# S, E, I, R = solution.T

# # Plot results
# plt.figure(figsize=(10,6))
# plt.plot(t, S, label='Susceptible')
# plt.plot(t, E, label='Exposed')
# plt.plot(t, I, label='Infected')
# plt.plot(t, R, label='Recovered')
# plt.xlabel('Days')
# plt.ylabel('Population')
# plt.title('SEIR Model')
# plt.legend()
# plt.grid()
# plt.show()