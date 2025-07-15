import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parâmetros da planta
R = 1.0
L = 0.5
km = 0.05
kb = 0.05
J = 1e-3

# Perturbação da carga
TL = lambda t: 0.1 * np.sin(t)

# Parâmetros do SMC
lambda_s = 100.0
k = 6.0
epsilon = 0.01

# Função do sistema com SMC baseado em Lyapunov
def motor_dc_smc_lyapunov(t, x):
    i, w = x

    dw_dt_nom = (km * i - TL(t)) / J
    s = dw_dt_nom + lambda_s * w

    u_eq = L * (-lambda_s * dw_dt_nom - (R * i + kb * w) / L)
    u = u_eq - k * np.tanh(s / epsilon)

    di_dt = (-R * i - kb * w + u) / L
    dw_dt = (km * i - TL(t)) / J

    return [di_dt, dw_dt]

# Condições iniciais e tempo estendido
x0 = [0.0, 1.0]
t_span = (0, 10)  # tempo estendido
t_eval = np.linspace(*t_span, 2000)

# Simulação
sol = solve_ivp(motor_dc_smc_lyapunov, t_span, x0, t_eval=t_eval)
t = sol.t
i = sol.y[0]
w = sol.y[1]

# Derivar s(t) e V(t)
dw_dt = np.gradient(w, t)
s = dw_dt + lambda_s * w
V = 0.5 * s**2

# ==============================
# 1. Plot da corrente e velocidade
# ==============================
plt.figure(figsize=(10, 4))
plt.plot(t, i, label='Corrente (i)', color='tab:blue')
plt.plot(t, w, label='Velocidade (ω)', color='tab:orange')
plt.title('Respostas do sistema')
plt.xlabel('Tempo (s)')
plt.ylabel('Magnitude')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ==============================
# 2. Plot do retrato de fase
# ==============================
plt.figure(figsize=(6, 5))
plt.plot(i, w, color='tab:green')
plt.title('Retrato de Fase: i x ω')
plt.xlabel('Corrente i (A)')
plt.ylabel('Velocidade ω (rad/s)')
plt.grid(True)
plt.tight_layout()
plt.show()

# ==============================
# 3. Plot da função de Lyapunov
# ==============================
plt.figure(figsize=(8, 4))
plt.plot(t, V, color='tab:red')
plt.title('Função de Lyapunov: $V = \\frac{1}{2}s^2$')
plt.xlabel('Tempo (s)')
plt.ylabel('Energia V(t)')
plt.grid(True)
plt.tight_layout()
plt.show()
