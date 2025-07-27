import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parâmetros do motor
R = 1.0
L = 0.5
km = 0.05
kb = km
J = 1e-3

# Perturbação de carga
TL = lambda t: 0.1 * np.sin(t)

# Parâmetros SMC
lambda_s = 100.0
K = 100.0  # Ganho aumentado para robustez

# Condições iniciais
w0 = 1.0  # rad/s
i0 = 0.0  # A

# Armazenamento do sinal de controle
u_history = []

# Lei de controle SMC corrigida
def smc_motor(t, x):
    global u_history
    i, w = x
    
    # Superfície deslizante (independente de T_L)
    s = (km/J) * i + lambda_s * w
    
    # Lei de controle
    u_eq = (R - lambda_s*L)*i + kb*w  # Termo equivalente
    u_sw = -K * np.sign(s)            # Termo descontínuo
    u = u_eq + u_sw
    
    # Armazena o sinal de controle para plotagem
    u_history.append(u)
    
    # Dinâmica do sistema
    dw_dt = (km*i - TL(t)) / J
    di_dt = (-R*i - kb*w + u) / L
    return [di_dt, dw_dt]

# Simulação
t_span = (0, 10)
t_eval = np.linspace(*t_span, 5000)
sol = solve_ivp(smc_motor, t_span, [i0, w0], t_eval=t_eval)

# Resultados
t_sim = sol.t
i_sim = sol.y[0]
omega_sim = sol.y[1]
u_sim = np.array(u_history[:len(t_sim)])  # Ajusta o tamanho

# Superfície deslizante s=0: (km/J)*i + lambda_s*w = 0
w_range = np.linspace(min(omega_sim), max(omega_sim), 100)
i_surf = -(lambda_s * J / km) * w_range

# =============================================
# Gráfico 1: Retrato de Fases
# =============================================
plt.style.use('seaborn-v0_8-whitegrid')
fig1, ax1 = plt.subplots(figsize=(10, 8))
ax1.plot(i_sim, omega_sim, label='Trajetória do Sistema', color='b', lw=2)
ax1.plot(i0, w0, 'go', markersize=10, label=f'Início: (i={i0}A, ω={w0}rad/s)')
ax1.plot(i_sim[-1], omega_sim[-1], 'rs', markersize=10, 
         label=f'Fim: (i={i_sim[-1]:.2f}A, ω={omega_sim[-1]:.2f}rad/s)')
ax1.plot(i_surf, w_range, 'r--', lw=2, label='Superfície Deslizante (s=0)')
ax1.set_title('Retrato de Fases: i-ω com Controle por Modos Deslizantes', fontsize=16)
ax1.set_xlabel('Corrente de Armadura i(t) [A]', fontsize=12)
ax1.set_ylabel('Velocidade Angular ω(t) [rad/s]', fontsize=12)
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True)
plt.tight_layout()
plt.show()

# =============================================
# Gráfico 2: Respostas Temporais
# =============================================
fig2, (ax_w, ax_i, ax_u) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig2.suptitle('Resposta Temporal do Sistema Controlado', fontsize=16)

# Velocidade Angular ω(t)
ax_w.plot(t_sim, omega_sim, label='ω(t)', color='blue')
ax_w.set_ylabel('Velocidade [rad/s]', fontsize=12)
ax_w.legend(fontsize=10)
ax_w.grid(True)

# Corrente de Armadura i(t)
ax_i.plot(t_sim, i_sim, label='i(t)', color='orange')
ax_i.set_ylabel('Corrente [A]', fontsize=12)
ax_i.legend(fontsize=10)
ax_i.grid(True)

# Sinal de Controle u(t)
ax_u.plot(t_sim, u_sim, label='u(t)', color='purple')
ax_u.set_ylabel('Tensão [V]', fontsize=12)
ax_u.set_xlabel('Tempo [s]', fontsize=12)
ax_u.legend(fontsize=10)
ax_u.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

print("Simulação concluída com sucesso!")