from scipy import signal
import matplotlib.pyplot as plt

# Definindo os valores de alpha que você quer usar
alphas = [1, -1, -5, 0, 5, 7]
num = []   
den = [2, 3, 1]
# Criando a função de transferência
sys = []
for i in range(len(alphas)):
    num = [alphas[i], 1]
    sys.append(signal.TransferFunction(num, den))

t, y = [[] for _ in range(len(alphas))], [[] for _ in range(len(alphas))]
# Calculando e plotando a resposta ao impulso
for i in range(len(alphas)):
    t[i], y[i] = signal.impulse(sys[i])    

plt.figure()
for i in range(len(alphas)):
    plt.plot(t[i], y[i])
plt.legend(['α = 1', 'α = -1', 'α = 5', 'α = 0', 'α = 5', 'α = 7'])
plt.title('Resposta ao Impulso para diferentes valores de α')
plt.xlabel('Tempo')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

# Calculando e plotando a resposta ao degrau
for i in range(len(alphas)):
    t[i], y[i] = signal.step(sys[i])    

plt.figure()
for i in range(len(alphas)):
    plt.plot(t[i], y[i])
plt.legend(['α = 1', 'α = -1', 'α = 5', 'α = 0', 'α = 5', 'α = 7'])
plt.title('Resposta ao Degrau para diferentes valores de α')
plt.xlabel('Tempo')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()