import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

# === Étape 1 : Implémentation de la méthode d'Euler ===


def euler_sird(beta, gamma, mu, S0, I0, R0, D0, dt, t_max):
    """
    Résout le modèle SIRD avec la méthode d'Euler.

    :param beta: Taux de transmission
    :param gamma: Taux de guérison
    :param mu: Taux de mortalité
    :param S0: Proportion initiale de susceptibles
    :param I0: Proportion initiale d'infectés
    :param R0: Proportion initiale de rétablis
    :param D0: Proportion initiale de décédés
    :param dt: Pas de temps
    :param t_max: Durée de la simulation (en jours)
    :return: Temps t, et arrays S, I, R, D
    """
    n_steps = int(t_max / dt) + 1
    t = np.linspace(0, t_max, n_steps)
    S = np.zeros(n_steps)
    I = np.zeros(n_steps)
    R = np.zeros(n_steps)
    D = np.zeros(n_steps)

    # Conditions initiales
    S[0], I[0], R[0], D[0] = S0, I0, R0, D0

    # Méthode d'Euler
    for n in range(n_steps - 1):
        dS = -beta * S[n] * I[n] * dt
        dI = (beta * S[n] * I[n] - gamma * I[n] - mu * I[n]) * dt
        dR = gamma * I[n] * dt
        dD = mu * I[n] * dt

        S[n + 1] = S[n] + dS
        I[n + 1] = I[n] + dI
        R[n + 1] = R[n] + dR
        D[n + 1] = D[n] + dD

    return t, S, I, R, D


# === Étape 2 : Simulation et analyse ===
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


# === Étape 3 : Ajustement des paramètres ===
def compute_mse(predictions, observations):
    return np.mean((predictions - observations) ** 2)


def run_parameter_estimation():
    # Lecture des données
    data = pd.read_csv('sird_dataset.csv')

    # Plages de paramètres pour la recherche sur grille
    beta_range = np.linspace(0.25, 0.5, 20)
    gamma_range = np.linspace(0.08, 0.15, 20)
    mu_range = np.linspace(0.005, 0.015, 20)

    best_mse = float('inf')
    best_params = None
# Jour,Susceptibles,Infectés,Rétablis,Décès

    for beta, gamma, mu in product(beta_range, gamma_range, mu_range):
        t, S, I, R, D = euler_sird(
            beta, gamma, mu, data['Susceptibles'][0], data['Infectés'][0], data['Rétablis'][0], data['Décès'][0], 0.01, len(data))
        mse = compute_mse(S, data['Susceptibles']) + compute_mse(I, data['Infectés']) + \
            compute_mse(R, data['Rétablis']) + compute_mse(D, data['Décès'])
        if mse < best_mse:
            best_mse = mse
            best_params = (beta, gamma, mu)

    print("Meilleurs paramètres :", best_params)
    print("MSE minimal :", best_mse)


# === Étape 4 : Scénarios de contrôle ===
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


# === Fonction principale ===
def main():
    print("=== Étape 2 : Simulation ===")
    run_simulation()

    print("\n=== Étape 3 : Ajustement des paramètres ===")
    run_parameter_estimation()

    print("\n=== Étape 4 : Scénarios de contrôle ===")
    run_intervention()


# Exécution du programme
if __name__ == "__main__":
    main()
