# Controladores por Modos Deslizantes

## Descrição do Problema

O motor DC pode ser modelado por:

<img width="188" height="89" alt="image" src="https://github.com/user-attachments/assets/97ab0715-f6dc-47f6-b218-60b0287d3d62" />


### Parâmetros:
-  J : Momento de inércia
-  i : Corrente de armadura
-  L ,  R : Indutância e resistência de armadura
- w: Velocidade angular do motor
- k_b : Constante da força contra-eletromotriz
- k_m : Torque constante do motor
- u : Sinal de controle (tensão de armadura)
- T_L : Perturbação de torque de carga (desconhecida, mas limitada)

## Objetivos

### a) Projeto do Sistema de Controle 
Projete um controlador por modos deslizantes (SMC) para garantir que w(0) quando t -->∞, assumindo parâmetros conhecidos.

### b) Simulação do Sistema 
Simule o sistema de controle com os seguintes parâmetros:

```python
R = 1.0          # Ω
L = 0.5          # H
k_m = 5e-2       # Nm/A
k_b = k_m        # Constante back-EMF
J = 1e-3         # Nms²/rad
T_L = 0.1*sin(t) # Torque de carga (perturbação)
ω(0) = 1         # rad/s (condição inicial)
i(0) = 0         # A (condição inicial)
