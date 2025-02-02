# intervention.py
import matplotlib.pyplot as plt
from euler_sird import euler_sird


def run_intervention():
    # Paramètres
    beta_original = 0.5
    beta_reduced = 0.3
    gamma = 0.15
    mu = 0.015
    S0, I0, R0, D0 = 0.99, 0.01, 0, 0
    dt = 0.01
    t_max = 100

    # Scénario sans intervention
    t, S, I, R, D = euler_sird(
        beta_original, gamma, mu, S0, I0, R0, D0, dt, t_max)

    # Scénario avec intervention
    t2, S2, I2, R2, D2 = euler_sird(
        beta_reduced, gamma, mu, S0, I0, R0, D0, dt, t_max)

    # Visualisation
    plt.figure(figsize=(10, 6))
    plt.plot(t, I, label='Infectés (sans intervention)')
    plt.plot(t2, I2, label='Infectés (avec intervention)')
    plt.xlabel('Temps (jours)')
    plt.ylabel('Proportion de la population')
    plt.title('Impact d\'une intervention sur la propagation')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    run_intervention()
