import numpy as np
from scipy.integrate import odeint
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# Bioreactor - Nonlinear and Linear model comparison
# Define the nonlinear model for the bioreactor
def biononlin(x,t,mu_max,Ks,Y,u,d):
    X=x[0]; S=x[1]

    par=[mu_max, Ks, Y]; u=Df; d=Sf
    mu=mu_max*S/(Ks+S)
    
    dXdt=-Df*X+mu*X
    dSdt=Df*(Sf-S)-mu*X/Y
    dxdt = [dXdt,dSdt]
    return dxdt

mu_max = 0.5; Ks = 0.1; Y = 0.4; # Parameters
Df = 0.35; Sf = 1.0;             # Influent

# Initial condition
x0 = [0.1,0.1]

# Time range
t = np.arange(0,60,0.0001)

# Solve ODE
x = odeint(biononlin,x0,t,args=(mu_max,Ks,Y,Df,Sf))

# Plot results
plt.plot(t,x[:,0],'b-',label='X_nl(t)')
plt.plot(t,x[:,1],'r-',label='S(_nl(t)')
plt.ylabel('concentrations')
plt.xlabel('time')
plt.legend(loc='best')
plt.show()

# Equilibrium points
xss = fsolve(biononlin, x0, args=(t,mu_max,Ks,Y,Df,Sf))

# Linearise
import sympy as sy

mu_max,Ks,Y,Df,Sf,S,X =sy.symbols('mu_max,Ks,Y,Df,Sf,S,X')
mu=mu_max*S/(Ks+S);

dXdt=-Df*X+mu*X;
dSdt=Df*(Sf-S)-mu*X/Y;
F = sy.Matrix([dXdt,dSdt])
A_ss = F.jacobian([X,S])
B_ss = F.jacobian([Df,Sf])

# Linear model evaluation
import control as ctrl
import control.matlab as ctrlmatlab

A = A_ss.evalf(subs={mu_max:0.5, Ks: 0.1, Y: 0.4, Df: 0.35, Sf: 1.0, X: xss[0], S: xss[1]})
B = B_ss.evalf(subs={mu_max:0.5, Ks: 0.1, Y: 0.4, Df: 0.35, Sf: 1.0, X: xss[0], S: xss[1]})
C = np.matrix([[1, 0],[0, 1]])
D = np.matrix([[0, 0],[0, 0]])

biolin = ctrl.ss(A,B,C,D)
yout, T, xout = ctrlmatlab.lsim(biolin,0,t,x0-xss)

Xlin = np.array(xout[:,0])+xss[0]
Slin = np.array(xout[:,1])+xss[1]

# Model comparison
plt.plot(T,Xlin,'r-',label='X_lin(t)')
plt.plot(t,x[:,0],'b-',label='X_nl(t)')
plt.ylabel('concentrations')
plt.xlabel('time')
plt.legend(loc='best')
plt.show()

plt.plot(T,Slin,'r-',label='S_lin(t)')
plt.plot(t,x[:,1],'b-',label='S_nl(t)')
plt.ylabel('concentrations')
plt.xlabel('time')
plt.legend(loc='best')
plt.show()
