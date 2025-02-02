# euler_sird.py
import numpy as np


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
