"""
Autorzy: Błażej Majchrzak, Marceli Gosztyła

Opis problemu:
Symulacja wahadła z logiką rozmytą. Celem jest zatrzymanie wahadłą na szczycie, mogąc jedynie bujać je na lewo i prawo

Wejścia:
Cos(x), Sin(x), Prędkość kątowa

Wyjście:
Siła popychania wahadła

Instalacja wymaganych bibliotek:
!pip install gymnasium[classic_control] scikit-fuzzy matplotlib numpy
"""

import numpy as np
import gymnasium as gym
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# === KONFIGURACJA ===
USE_FUZZY = True     # True = sterowanie rozmyte, False = tylko grawitacja
MAX_TORQUE = 1.2      # Maksymalny moment siły [Nm]
SIM_TIME = 12        # Czas symulacji [s]
DT = 0.05            # Krok symulacji [s]

# === ZMIENNE ROZMYTE ===
cos_theta = ctrl.Antecedent(np.linspace(-1, 1, 200), 'cos_theta')
sin_theta = ctrl.Antecedent(np.linspace(-1, 1, 200), 'sin_theta')
angular_velocity = ctrl.Antecedent(np.linspace(-8, 8, 200), 'angular_velocity')
torque = ctrl.Consequent(np.linspace(-MAX_TORQUE, MAX_TORQUE, 200), 'torque')

"""
Definicja funkcji przynależności dla zmiennych rozmytych.
"""
cos_theta['down']      = fuzz.trimf(cos_theta.universe, [-1.0, -1.0, 0.1])
cos_theta['mid']       = fuzz.trimf(cos_theta.universe, [-0.1, 0.4, 0.6])
cos_theta['near_up']   = fuzz.trimf(cos_theta.universe, [0.55, 0.64, 0.98])
cos_theta['up']        = fuzz.trimf(cos_theta.universe, [0.85, 1.0, 1.0])

angular_velocity['neg_big']   = fuzz.trimf(angular_velocity.universe, [-8, -2.5, -0.8])
angular_velocity['neg_small'] = fuzz.trimf(angular_velocity.universe, [-1.2, -0.3, 0])
angular_velocity['zero']      = fuzz.trimf(angular_velocity.universe, [-0.4, 0, 0.4])
angular_velocity['pos_small'] = fuzz.trimf(angular_velocity.universe, [0, 0.3, 1.2])
angular_velocity['pos_big']   = fuzz.trimf(angular_velocity.universe, [0.8, 2.5, 8])

sin_theta['neg'] = fuzz.trimf(sin_theta.universe, [-1.0, -0.3, 0.0])
sin_theta['pos'] = fuzz.trimf(sin_theta.universe, [0.0, 0.3, 1.0])

torque['zero'] = fuzz.trimf(torque.universe, [-0.1*MAX_TORQUE, 0.0, 0.1*MAX_TORQUE])
torque['very_left']    = fuzz.trimf(torque.universe, [-0.5*MAX_TORQUE, -0.5*MAX_TORQUE, -0.2*MAX_TORQUE])
torque['slight_left']  = fuzz.trimf(torque.universe, [-0.2*MAX_TORQUE, -0.15*MAX_TORQUE, 0])
torque['slight_right'] = fuzz.trimf(torque.universe, [0, 0.15*MAX_TORQUE, 0.2*MAX_TORQUE])
torque['very_right']   = fuzz.trimf(torque.universe, [0.2*MAX_TORQUE, 0.5*MAX_TORQUE, 0.5*MAX_TORQUE])
torque['full_break_right']   = fuzz.trimf(torque.universe, [0.5*MAX_TORQUE, MAX_TORQUE, MAX_TORQUE])
torque['full_break_left']    = fuzz.trimf(torque.universe, [-MAX_TORQUE, -MAX_TORQUE, -0.5*MAX_TORQUE])

"""
Reguły sterowania rozmytego:
- Swing-up: agresywne i umiarkowane w zależności od kąta
- Mid: po środku, dodanie energii ale mniej
- Near-up: hamowanie przy zbliżaniu się do pozycji pionowej
- Up: finalna stabilizacja
"""
rules = [
    ctrl.Rule(cos_theta['down'] & angular_velocity['pos_big'], torque['very_right']),
    ctrl.Rule(cos_theta['down'] & angular_velocity['pos_small'], torque['very_right']),
    ctrl.Rule(cos_theta['down'] & angular_velocity['neg_small'], torque['very_left']),
    ctrl.Rule(cos_theta['down'] & angular_velocity['neg_big'], torque['very_left']),
    
    ctrl.Rule(cos_theta['mid'] & sin_theta['pos'] & angular_velocity['pos_big'], torque['slight_right']),
    ctrl.Rule(cos_theta['mid'] & sin_theta['pos'] & angular_velocity['pos_small'], torque['slight_right']),
    ctrl.Rule(cos_theta['mid'] & sin_theta['pos'] & angular_velocity['neg_small'], torque['slight_left']),
    ctrl.Rule(cos_theta['mid'] & sin_theta['pos'] & angular_velocity['neg_big'], torque['slight_left']),
    
    ctrl.Rule(cos_theta['mid'] & sin_theta['neg'] & angular_velocity['pos_big'], torque['slight_right']),
    ctrl.Rule(cos_theta['mid'] & sin_theta['neg'] & angular_velocity['pos_small'], torque['slight_right']),
    ctrl.Rule(cos_theta['mid'] & sin_theta['neg'] & angular_velocity['neg_small'], torque['slight_left']),
    ctrl.Rule(cos_theta['mid'] & sin_theta['neg'] & angular_velocity['neg_big'], torque['slight_left']),
    
    ctrl.Rule(cos_theta['near_up'] & angular_velocity['pos_big'], torque['full_break_left']),
    ctrl.Rule(cos_theta['near_up'] & angular_velocity['pos_small'], torque['very_left']),
    ctrl.Rule(cos_theta['near_up'] & angular_velocity['zero'], torque['zero']),
    ctrl.Rule(cos_theta['near_up'] & angular_velocity['neg_small'], torque['very_right']),
    ctrl.Rule(cos_theta['near_up'] & angular_velocity['neg_big'], torque['full_break_right']),
    
    ctrl.Rule(cos_theta['up'] & sin_theta['pos'] & angular_velocity['pos_big'], torque['full_break_left']),
    ctrl.Rule(cos_theta['up'] & sin_theta['pos'] & angular_velocity['pos_small'], torque['full_break_left']),
    ctrl.Rule(cos_theta['up'] & sin_theta['pos'] & angular_velocity['zero'], torque['very_left']),
    ctrl.Rule(cos_theta['up'] & sin_theta['pos'] & angular_velocity['neg_small'], torque['zero']),

    ctrl.Rule(cos_theta['up'] & sin_theta['neg'] & angular_velocity['neg_big'], torque['full_break_right']),
    ctrl.Rule(cos_theta['up'] & sin_theta['neg'] & angular_velocity['neg_small'], torque['full_break_right']),
    ctrl.Rule(cos_theta['up'] & sin_theta['neg'] & angular_velocity['zero'], torque['very_right']),
    ctrl.Rule(cos_theta['up'] & sin_theta['neg'] & angular_velocity['pos_small'], torque['zero']),
]

torque_ctrl = ctrl.ControlSystem(rules)

def safe_fuzzy_torque(cos_t, sin_t, theta_dot):
    """
    Oblicza moment torque na podstawie logiki rozmytej.
    
    cos_t : float - cosinus kąta wahadła
    sin_t : float - sinus kąta wahadła
    theta_dot : float - prędkość kątowa
    
    Zwraca:
    float - moment siły w zakresie [-MAX_TORQUE, MAX_TORQUE]
    """
    sim = ctrl.ControlSystemSimulation(torque_ctrl)
    sim.input['cos_theta'] = np.clip(cos_t, -1, 1)
    sim.input['sin_theta'] = np.clip(sin_t, -1, 1)
    sim.input['angular_velocity'] = np.clip(theta_dot, -8, 8)
    try:
        sim.compute()
        tau = sim.output.get('torque', 0.0)
        if cos_t > 0.9:
            tau -= 0.2 * sin_t
        return float(np.clip(tau, -MAX_TORQUE, MAX_TORQUE))
    except Exception:
        return 0.0

"""
Inicjalizacja środowiska Gymnasium
Ustawienie początkowego stanu losowego oraz parametrów środowiska.
"""
env = gym.make("Pendulum-v1", render_mode=None)
obs, info = env.reset()
env.unwrapped.max_torque = MAX_TORQUE
env.unwrapped.g = 10.0

random_theta = np.random.uniform(-np.pi, np.pi)
random_velocity = np.random.uniform(-1.0, 1.0)
env.unwrapped.state = np.array([random_theta, random_velocity])
obs = np.array([np.cos(random_theta), np.sin(random_theta), random_velocity], dtype=np.float32)

print(f"   max_torque = {MAX_TORQUE}")
print(f"   start θ₀ = {random_theta:.2f} rad")

"""
Bufory danych do animacji i wykresów.
"""
theta_hist, torque_hist, time_hist = [], [], []
trail_x, trail_y = [], []

"""
Tworzenie wykresów i przygotowanie linii animacji.
"""
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
ax1.set_xlim(-1.2, 1.2)
ax1.set_ylim(-1.2, 1.2)
ax1.set_aspect('equal')
ax1.set_title("Fuzzy Pendulum")
line, = ax1.plot([], [], 'o-', lw=4, color='blue', markersize=10)
trail, = ax1.plot([], [], '-', lw=1, color='gray', alpha=0.5)
text = ax1.text(-1.1, 1.0, '', fontsize=10, family='monospace')

ax2.set_xlim(0, SIM_TIME)
ax2.set_ylim(-1.2, 1.2)
ax2.set_title("sin(θ), cos(θ) i moment (τ)")
ax2.set_xlabel("Czas [s]")
ax2.set_ylabel("Wartość")
sin_line, = ax2.plot([], [], label="sin(θ)", color='tab:red')
cos_line, = ax2.plot([], [], label="cos(θ)", color='tab:green')
torque_line, = ax2.plot([], [], label="τ [Nm]", color='tab:blue')
ax2.legend(loc='upper right')

time_hist, sin_hist, cos_hist, torque_hist = [], [], [], []

def update(frame):
    """
    Aktualizacja klatki animacji:
    - Obliczenie momentu siły
    - Aktualizacja pozycji wahadła
    - Aktualizacja wykresów sin(θ), cos(θ) i τ
    """
    global obs
    cos_t, sin_t, theta_dot = obs
    theta = np.arctan2(sin_t, cos_t)
    theta = -theta

    torque_value = safe_fuzzy_torque(cos_t, sin_t, theta_dot) if USE_FUZZY else 0.0

    obs, _, _, _, _ = env.step([torque_value])

    x = -np.sin(theta)
    y = np.cos(theta)
    line.set_data([0, x], [0, y])

    trail_x.append(x)
    trail_y.append(y)
    trail.set_data(trail_x[-50:], trail_y[-50:])
    text.set_text(f"θ={theta:+.2f} rad\nτ={torque_value:+.2f} Nm")

    t = frame * DT
    time_hist.append(t)
    sin_hist.append(sin_t)
    cos_hist.append(cos_t)
    torque_hist.append(torque_value)

    sin_line.set_data(time_hist, sin_hist)
    cos_line.set_data(time_hist, cos_hist)
    torque_line.set_data(time_hist, torque_hist)

    ax2.set_xlim(0, max(5, t))
    ax2.set_ylim(-1.2, 1.2)

    print(f"[frame={frame:03d}] | cos={cos_t:+.3f} | sin={sin_t:+.3f} | τ={torque_value:+.3f}")
    return line, trail, text, sin_line, cos_line, torque_line

ani = FuncAnimation(fig, update, frames=int(SIM_TIME/DT), interval=DT*1000, blit=False, repeat=False)
plt.close()
HTML(ani.to_jshtml(fps=20, default_mode='loop'))
