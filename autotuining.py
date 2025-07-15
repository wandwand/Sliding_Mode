import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parâmetros do motor
R, L, km, kb, J = 1.0, 0.5, 0.05, 0.05, 1e-3
TL = lambda t: 0.1 * np.sin(t)

epsilon = 0.005
t_span = (0, 8)
t_eval = np.linspace(*t_span, 1600)
x0 = [0.0, 1.0]

# Autotuning de parâmetros
lambdas = np.arange(80, 160, 20)
ks = np.arange(2, 10, 2)

results = []

for lambda_s in lambdas:
    for k in ks:

        def system(t, x):
            i, w = x
            dw = (km * i - TL(t)) / J
            s = dw + lambda_s * w
            u_eq = L * (-lambda_s * dw - (R * i + kb * w) / L)
            u = u_eq - k * np.tanh(s / epsilon)
            di = (-R * i - kb * w + u) / L
            dw = (km * i - TL(t)) / J
            return [di, dw]

        sol = solve_ivp(system, t_span, x0, t_eval=t_eval, method='RK45')
        t = sol.t
        w = sol.y[1]

        # Calcular tempo até ω < 0.01
        abs_w = np.abs(w)
        below_threshold = np.where(abs_w < 0.01)[0]
        t_converge = t[below_threshold[0]] if len(below_threshold) > 0 else np.inf

        # Energia de Lyapunov
        dw_dt = np.gradient(w, t)
        s = dw_dt + lambda_s * w
        V = 0.5 * s**2
        V_final = V[-1]

        results.append({
            'lambda': lambda_s,
            'k': k,
            't_converge': t_converge,
            'V_final': V_final
        })

# Ordenar pelo tempo de convergência
sorted_results = sorted(results, key=lambda r: (r['t_converge'], r['V_final']))

# Mostrar os melhores
print("\n🏆 Melhores parâmetros encontrados:")
for r in sorted_results[:5]:
    print(f"λ = {r['lambda']:.1f}, k = {r['k']:.1f}, t_convergência ≈ {r['t_converge']:.2f}s, V_final = {r['V_final']:.2e}")
