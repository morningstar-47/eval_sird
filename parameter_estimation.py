# parameter_estimation.py
import numpy as np
import pandas as pd
from itertools import product
from euler_sird import euler_sird


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


if __name__ == "__main__":
    run_parameter_estimation()
