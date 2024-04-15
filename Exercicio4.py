import numpy as np
from sympy import inverse_laplace_transform, symbols, Matrix, eye, Function, simplify, fraction, exp, DiracDelta, lambdify, laplace_transform
import matplotlib.pyplot as plt
from scipy import signal

#ITEM 1
#Sabemos que para encontrar a função de transferência dessa questão é necessário utilizar da seguinte equação:
#G(s) = C(Is - A)^-1 * B + D
#Declarando as variáveis de acordo com a equação

s, t = symbols('s t')

A = Matrix([[0, 1], [-5, -2]])
Is = s * eye(2)
B = Matrix([0, 2])
C = Matrix([[1, 0]])

#Após declarar as variáveis, podemos calcular a função de transferência:
G_s = C * (Is - A).inv() * B
G_s = G_s[0]
print(f'A função de transferência G(s) = {G_s}')

#ITEM 2
#Temos que G(s) = Y(s)/U(s), onde Y(s) é a saída e U(s) é a entrada, fazendo então a separação das variáveis:
Y_s, U_s = fraction(G_s)
U_s = simplify(U_s)
y_t = inverse_laplace_transform(Y_s, s, t)
u_t = inverse_laplace_transform(U_s, s, t)
print(f'\nTemos então que o modelo entrada-saída do sistema é:\n {y_t} = {u_t}')

#ITEM 3
#Para encontrar a resposta a u(t) =  e^(−3t)*δ(t)
#Sabemos que G(s) = -10/(-5*s**2 -10*s - 25)
num = [-10]
den = [-5, -10, -25]
sys = signal.TransferFunction(num, den)
t, y = signal.impulse(sys)
y_respot = y * np.exp(-3*t)

plt.figure()
plt.plot(t, y_respot, label='y(t)')
plt.xlabel('Tempo (s)')
plt.ylabel('y(t)')
plt.title('Saída y(t) em resposta a u(t) = e^(-3t)*δ(t)')
plt.legend()
plt.grid(True)
plt.show()
