# simulation.py
import matplotlib.pyplot as plt
from euler_sird import euler_sird


def run_simulation():
    # Paramètres
    beta = 0.5
    gamma = 0.15
    mu = 0.015
    S0, I0, R0, D0 = 0.99, 0.01, 0, 0
    dt = 0.01
    t_max = 100

    # Simulation
    t, S, I, R, D = euler_sird(beta, gamma, mu, S0, I0, R0, D0, dt, t_max)

    # Visualisation
    plt.figure(figsize=(10, 6))
    plt.plot(t, S, label='Susceptibles (S)')
    plt.plot(t, I, label='Infectés (I)')
    plt.plot(t, R, label='Rétablis (R)')
    plt.plot(t, D, label='Décédés (D)')
    plt.xlabel('Temps (jours)')
    plt.ylabel('Proportion de la population')
    plt.title('Modèle SIRD - Simulation')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    run_simulation()
