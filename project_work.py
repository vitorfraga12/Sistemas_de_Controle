import numpy as np
from scipy.integrate import odeint
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import sympy as sy
import control as ctrl
import control.matlab as ctrlmatlab

# Tank System - Nonlinear Model
def tank_system_nonlinear(x, t, Q1, Q2, A1, A2, a1, a2):
    h1 = x[0]
    h2 = x[1]
    
    # Outflows based on heights
    v_a1 = a1 * np.sqrt(2 * 9.81 * h1)
    v_a2 = a2 * np.sqrt(2 * 9.81 * h2)
    
    # Nonlinear differential equations
    dh1dt = (Q1 - v_a1) / A1
    dh2dt = (Q2 + v_a1 - v_a2) / A2
    
    return [dh1dt, dh2dt]

# Parameters
A1 = 20.0  # m^2
A2 = 20.0  # m^2
a1 = 0.1   # m^2
a2 = 0.1   # m^2
Q1 = 1.0   # m^3/s
Q2 = 0.0   # m^3/s

# Initial conditions
h1_0 = 0  # m
h2_0 = 0 # m
x0 = [h1_0, h2_0]

# Time range
t = np.arange(0, 1800, 0.3)

# Solve ODE for the nonlinear system
x_nl = odeint(tank_system_nonlinear, x0, t, args=(Q1, Q2, A1, A2, a1, a2))

# Plot results for the nonlinear model
plt.plot(t, x_nl[:, 0], 'b-', label='h1_nl(t)')
plt.plot(t, x_nl[:, 1], 'r-', label='h2_nl(t)')
plt.ylabel('Heights (m)')
plt.xlabel('Time (s)')
plt.legend(loc='best')
plt.show()

# Linearization around equilibrium points
h1_eq = 5.0  # Equilibrium height for Tank 1
h2_eq = 5.0  # Equilibrium height for Tank 2

# Symbolic variables for linearization
h1, h2, Q1_sym, Q2_sym = sy.symbols('h1 h2 Q1 Q2')

# Nonlinear equations
v_a1 = a1 * sy.sqrt(2 * 9.81 * h1)
v_a2 = a2 * sy.sqrt(2 * 9.81 * h2)

dh1dt = (Q1_sym - v_a1) / A1
dh2dt = (Q2_sym + v_a1 - v_a2) / A2

# Vector field
F = sy.Matrix([dh1dt, dh2dt])

# Calculate Jacobians
A_ss = F.jacobian([h1, h2])
B_ss = F.jacobian([Q1_sym, Q2_sym])

# Evaluate Jacobians at equilibrium points
A = A_ss.evalf(subs={h1: h1_eq, h2: h2_eq, Q1_sym: Q1, Q2_sym: Q2})
B = B_ss.evalf(subs={h1: h1_eq, h2: h2_eq, Q1_sym: Q1, Q2_sym: Q2})

# Define C and D matrices
C = np.matrix([[1, 0], [0, 1]])
D = np.matrix([[0, 0], [0, 0]])

# Linear state-space model
tank_system_linear = ctrl.ss(np.array(A).astype(np.float64), np.array(B).astype(np.float64), C, D)
print(tank_system_linear)
# Simulate linear system
yout, T, xout = ctrlmatlab.lsim(tank_system_linear, np.array([np.zeros_like(t),np.zeros_like(t)]).T, t, x0-np.array([h1_eq, h2_eq]))

# Adjust linear results to equilibrium
h1_lin = xout[:, 0] + h1_eq
h2_lin = xout[:, 1] + h2_eq

# Plot comparison between linear and nonlinear models for h1
plt.plot(T, h1_lin, 'r-', label='h1_lin(t)')
plt.plot(t, x_nl[:,0], 'b-', label='h1_nl(t)')
plt.ylabel('h1 (m)')
plt.xlabel('Time (s)')
plt.legend(loc='best')
plt.show()

# Plot comparison between linear and nonlinear models for h2
plt.plot(T, h2_lin, 'r-', label='h2_lin(t)')
plt.plot(t, x_nl[:, 1], 'b-', label='h2_nl(t)')
plt.ylabel('h2 (m)')
plt.xlabel('Time (s)')
plt.legend(loc='best')
plt.show()