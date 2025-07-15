import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# =============================================================================
# @brief Parâmetros do Sistema (Notação Matemática)
# =============================================================================
R = 1.0      # Ω (Resistência de armadura)
L = 0.5      # H (Indutância de armadura)
k_m = 0.05   # Nm/A (Constante de torque)
k_b = 0.05   # V/(rad/s) (Constante CEMF)
J = 1e-3     # kg·m² (Momento de inércia)

# =============================================================================
# @brief Perturbação e Parâmetros do Controlador
# =============================================================================
T_L = lambda t: 0.1 * np.sin(t)  # Nm (Torque de carga)
λ = 100.0    # Ganho da superfície deslizante
K = 6.0      # Ganho de chaveamento
ε = 0.01     # Coeficiente de suavização

# =============================================================================
# @brief Modelo Dinâmico com SMC
# =============================================================================
def motor_dynamics(t, x):
    i, ω = x  # Corrente (A), Velocidade (rad/s)
    
    # Dinâmica nominal (inclui T_L que seria desconhecido na prática)
    ω̇_nom = (k_m * i - T_L(t)) / J
    
    # Superfície deslizante (s = ω̇ + λω)
    s = ω̇_nom + λ * ω
    
    # Lei de controle (u = u_eq + u_sw)
    u_eq = L * (-λ * ω̇_nom - (R * i + k_b * ω) / L)
    u_sw = -K * np.tanh(s / ε)
    u = u_eq + u_sw
    
    # Equações diferenciais
    di_dt = (-R * i - k_b * ω + u) / L
    dω_dt = (k_m * i - T_L(t)) / J
    
    return [di_dt, dω_dt]

# =============================================================================
# @brief Configuração da Simulação
# =============================================================================
t_span = (0, 10)       # Intervalo de tempo (s)
x0 = [0.0, 1.0]        # Condições iniciais [i(0), ω(0)]
t_eval = np.linspace(*t_span, 2000)  # Pontos de avaliação

# =============================================================================
# @brief Execução da Simulação
# =============================================================================
sol = solve_ivp(motor_dynamics, t_span, x0, t_eval=t_eval)
t = sol.t
i = sol.y[0]
ω = sol.y[1]

# Cálculo numérico da superfície deslizante (pós-processamento)
ω̇ = np.gradient(ω, t)
s = ω̇ + λ * ω
V = 0.5 * s**2

# =============================================================================
# @brief Visualização dos Resultados
# =============================================================================
plt.figure(figsize=(14, 10))

# Gráfico 1: Variáveis de Estado
plt.subplot(2, 2, 1)
plt.plot(t, i, 'b', label='Corrente (i)')
plt.plot(t, ω, 'r', label='Velocidade (ω)')
plt.title('Resposta Temporal')
plt.xlabel('Tempo (s)')
plt.ylabel('Magnitude')
plt.legend()
plt.grid(True)

# Gráfico 2: Retrato de Fase
plt.subplot(2, 2, 2)
plt.plot(i, ω, 'g')
plt.title('Retrato de Fase (i vs ω)')
plt.xlabel('Corrente (A)')
plt.ylabel('Velocidade (rad/s)')
plt.grid(True)

# Gráfico 3: Função de Lyapunov
plt.subplot(2, 2, 3)
plt.plot(t, V, 'm')
plt.title('Função de Lyapunov (V=0.5s²)')
plt.xlabel('Tempo (s)')
plt.ylabel('V(t)')
plt.grid(True)

# Gráfico 4: Superfície Deslizante
plt.subplot(2, 2, 4)
plt.plot(t, s, 'c')
plt.title('Superfície Deslizante (s)')
plt.xlabel('Tempo (s)')
plt.ylabel('s(t)')
plt.grid(True)

plt.tight_layout()
plt.show()
