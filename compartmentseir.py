import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def generate_parameters():
    I_params = {
        'beta1': {
            'S': np.random.normal(0.02, 0.002),
            'P': 0,
            'St': 0,
            'E': np.random.normal(0.02, 0.002)
        },
        'beta2': {
            'S': 0,
            'P': np.random.normal(0.03, 0.003),
            'St': np.random.normal(0.03, 0.003),
            'E': np.random.normal(0.03, 0.003)
        },
        'beta3': {
            'S': np.random.normal(0.07, 0.007),
            'P': np.random.normal(0.07, 0.007),
            'St': np.random.normal(0.07, 0.007),
            'E': np.random.normal(0.07, 0.007)
        }
    }
    R_params = {
        'gamma0': {
            'S': np.random.normal(0.05, 0.005),
            'P': np.random.normal(0.05, 0.005),
            'St': np.random.normal(0.05, 0.005),
            'E': np.random.normal(0.05, 0.005)
        },
        'mu': {
            'S': np.random.normal(0.01, 0.001),
            'P': np.random.normal(0.01, 0.001),
            'St': np.random.normal(0.01, 0.001),
            'E': np.random.normal(0.01, 0.001)
        },
        'alpha': {
            'S': np.random.normal(0.01, 0.001),
            'P': np.random.normal(0.01, 0.001),
            'St': np.random.normal(0.01, 0.001),
            'E': np.random.normal(0.01, 0.001)
        },
        'delta_T': {
            'S': np.random.normal(0, 10),
            'P': np.random.normal(0, 10),
            'St': np.random.normal(0, 10),
            'E': np.random.normal(0, 10)
        }
    }
    T_params = {
        'tau': np.random.normal(0.03, 0.008, 3)
    }
    return R_params, I_params, T_params


# Combined supply chain and SEIR model differential equations
def combined_model(t, y, R_func, I_func, T_func, seir_params):
    Supplier, Processing, Store, S, E, I, R, D = y  # Added D

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

    beta, sigma, gamma, mu = seir_params  # Include mu
    N = S + E + I + R + D  # Account for deaths in total population

    dS_dt = -beta * S * I / N - (I_E - R_E + T_StE)
    dE_dt = beta * S * I / N - sigma * E + (I_E - R_E + T_StE)
    dI_dt = sigma * E - gamma * I - mu * I  # Account for deaths leaving I
    dR_dt = gamma * I
    dD_dt = mu * I  # Death compartment

    return [dSupplier_dt, dProcessing_dt, dStore_dt, dS_dt, dE_dt, dI_dt, dR_dt, dD_dt]  # Added dD_dt


def R_func(t, C, stage, params):
    delta_T = params['delta_T'][stage]
    return C * (params['gamma0'][stage] * np.exp(-params['mu'][stage] * delta_T) + params['alpha'][stage])


def I_func(t, C, stage, params):
    beta1 = params['beta1'][stage]
    beta2 = params['beta2'][stage]
    beta3 = params['beta3'][stage]
    return C * (beta1 + beta2) + beta3


def T_func(t, C, stage, params):
    stages = ['SP', 'PS', 'StE']
    i = stages.index(stage)
    return params['tau'][i] * C


# Simulation parameters
t_max = 180
t_eval = np.linspace(0, t_max, 1000)

# Initial conditions
N = 5000
I0, E0, R0, D0 = 0, 0, 0, 0  # Added initial deaths
S0 = N - I0 - E0 - R0 - D0  # Account for D in S0

#initial_supplier_contamination = np.random.uniform(1, 10)
initial_supplier_contamination = 0
y0 = [initial_supplier_contamination, 0, 0, S0, E0, I0, R0, D0]  # Added D0

# SEIR parameters
beta = 0.3
sigma = 1 / 5.2
gamma = 1 / 14
mu = 0.01  # Death rate
seir_params = (beta, sigma, gamma, mu)  # Added mu

# Generate Monte Carlo parameters
R_params, I_params, T_params = generate_parameters()

# Example: Setting parameters to 0 for specific stages
I_params['beta1']['P'] = 0  # Setting beta1 to 0 for the 'Processing' stage
I_params['beta2']['P'] = 0  # Setting beta2 to 0 for the 'Processing' stage
I_params['beta3']['St'] = 0  # Setting beta3 to 0 for the 'Store' stage

# Solve ODEs
sol = solve_ivp(combined_model, (0, t_max), y0,
                t_eval=t_eval,
                args=(lambda t, C, stage: R_func(t, C, stage, R_params),
                      lambda t, C, stage: I_func(t, C, stage, I_params),
                      lambda t, C, stage: T_func(t, C, stage, T_params),
                      seir_params))

Supplier, Processing, Store, S, E, I, R, D = sol.y  # Added D

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
ax3.plot(sol.t, D, label='Deaths')  # Added Deaths plot
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
