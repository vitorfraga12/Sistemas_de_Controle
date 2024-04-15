from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
#ITEM 3
#Para encontrar a resposta a u(t) =  e^(−3t)*δ(t)
#Sabemos que G(s) = -10/(-5*s**2 -10*s - 25)
num = [4,15]
den = [1,2,1,0]
sys = signal.TransferFunction(num, den)
t, y = signal.impulse(sys)
y_respot = y * 1

plt.figure()
plt.plot(t, y_respot, label='y(t)')
plt.xlabel('Tempo (s)')
plt.ylabel('y(t)')
plt.title('Saída y(t) em resposta a u(t) = (t)')
plt.legend()
plt.grid(True)
plt.show()