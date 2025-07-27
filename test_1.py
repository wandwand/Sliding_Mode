import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parâmetros do sistema
R = 1.0      # Ω
L = 0.5      # H
k_m = 5e-2   # Nm/A
k_b = k_m    # Constante back-EMF
J = 1e-3     # Nms²/rad

# Parâmetros do controlador
c = 1000.0   # Ganho da superfície de deslizamento
K = 1002.0   # Ganho do termo de chaveamento
phi = 0.1    # Largura da zona de saturação

# Função de saturação
def sat(s, phi):
    return np.clip(s / phi, -1.0, 1.0)

# Dinâmica do sistema
def motor_dynamics(t, x):
    w, i = x
    T_L = 0.1 * np.sin(t)  # Torque de carga (perturbação)
    
    # Superfície de deslizamento
    s = c * w + (k_m / J) * i
    
    # Lei de controle (SMC)
    u_eq = R * i + k_b * w - c * L * i
    u_sw = -K * sat(s, phi)
    u = u_eq + u_sw
    
    # Equações diferenciais
    dw_dt = (k_m * i - T_L) / J
    di_dt = (u - R * i - k_b * w) / L
    
    return [dw_dt, di_dt]

# Condições iniciais e tempo de simulação
initial_conditions = [1.0, 0.0]  # ω(0) = 1 rad/s, i(0) = 0 A
t_span = (0, 2.0)  # 0 a 2 segundos
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Resolver as EDOs
solution = solve_ivp(
    motor_dynamics,
    t_span,
    initial_conditions,
    t_eval=t_eval,
    method='RK45'
)

# Extrair resultados
t = solution.t
w = solution.y[0]
i = solution.y[1]

# Calcular superfície de deslizamento e controle
s = c * w + (k_m / J) * i
u_eq = R * i + k_b * w - c * L * i
u_sw = -K * np.array([sat(s_val, phi) for s_val in s])
u = u_eq + u_sw


# --------------------------------------------
# Gráficos individuais
# --------------------------------------------

# 1. Velocidade Angular (ω)
plt.figure(figsize=(8, 4))
plt.plot(t, w, 'b', linewidth=2)
plt.title('Controle SMC: Velocidade Angular (ω)', fontsize=12, pad=20)
plt.xlabel('Tempo (s)', fontsize=10)
plt.ylabel('rad/s', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim([0, 2])
plt.tight_layout()
plt.show()

# 2. Corrente de Armadura (i)
plt.figure(figsize=(8, 4))
plt.plot(t, i, 'r', linewidth=2)
plt.title('Controle SMC: Corrente de Armadura (i)', fontsize=12, pad=20)
plt.xlabel('Tempo (s)', fontsize=10)
plt.ylabel('A', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim([0, 2])
plt.tight_layout()
plt.show()

# 3. Superfície de Deslizamento (s)
plt.figure(figsize=(8, 4))
plt.plot(t, s, 'g', linewidth=2)
plt.axhline(0, color='k', linestyle='--', alpha=0.5)
plt.title('Superfície de Deslizamento (s)', fontsize=12, pad=20)
plt.xlabel('Tempo (s)', fontsize=10)
plt.ylabel('s', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim([0, 2])
plt.tight_layout()
plt.show()

# 4. Sinal de Controle (u)
plt.figure(figsize=(8, 4))
plt.plot(t, u, 'm', linewidth=2)
plt.title('Sinal de Controle Total (u)', fontsize=12, pad=20)
plt.xlabel('Tempo (s)', fontsize=10)
plt.ylabel('V', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim([0, 2])
plt.tight_layout()
plt.show()

# 5. Componentes do Controle
plt.figure(figsize=(8, 4))
plt.plot(t, u_eq, 'c', label='Controle Equivalente (u_eq)', linewidth=2)
plt.plot(t, u_sw, 'y', label='Termo de Chaveamento (u_sw)', linewidth=2)
plt.title('Componentes do Sinal de Controle', fontsize=12, pad=20)
plt.xlabel('Tempo (s)', fontsize=10)
plt.ylabel('V', fontsize=10)
plt.legend(loc='upper right', fontsize=9)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim([0, 2])
plt.tight_layout()
plt.show()

# 6. Plano de Fase (ω vs. i)
plt.figure(figsize=(6, 6))
plt.plot(w, i, 'b', linewidth=2)
plt.scatter(w[0], i[0], color='r', s=100, label='Início (t=0)')
plt.scatter(w[-1], i[-1], color='g', s=100, label='Fim (t=2s)')
plt.title('Plano de Fase: ω vs. i', fontsize=12, pad=20)
plt.xlabel('ω (rad/s)', fontsize=10)
plt.ylabel('i (A)', fontsize=10)
plt.legend(loc='upper right', fontsize=9)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()