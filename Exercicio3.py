from scipy import signal
import matplotlib.pyplot as plt
from sympy import symbols, inverse_laplace_transform, exp

# Definindo as variáveis "t" no domínio do Tempo e "s" no domínio de Laplace
s, t = symbols('s t')

# Definindo as funções que eu quero encontrar a transformada inversa de Laplace
F1 = (4*s**2 + 15*s+ 19)/(s**3 + 2*s**2 + s )

# Encontrando as transformada inversas de Laplace
f1 = inverse_laplace_transform(F1, s, t)

# Imprimir o resultado
print(f'Temos que as transformadas inversas de Laplace são:\n f1(t) = {f1}')
##############################################################

num = [4,15,19]
den = [1,2,1]
sys = signal.TransferFunction(num, den)
t, y = signal.impulse(sys)

print('Resposta ao impulso y(t) = ', sys.inputs)

plt.figure()
plt.plot(t, y, label='y(t)')
plt.xlabel('Tempo (s)')
plt.ylabel('y(t)')
plt.title('Saída y(t) em resposta a u(t)')
plt.legend()
plt.grid(True)
plt.show()