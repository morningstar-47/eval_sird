import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product


class SIRDModel:
    def __init__(self, beta, gamma, mu, S0, I0, R0, D0, dt, t_max):
        self.beta = beta
        self.gamma = gamma
        self.mu = mu
        self.S0 = S0
        self.I0 = I0
        self.R0 = R0
        self.D0 = D0
        self.dt = dt
        self.t_max = t_max

    def euler_sird(self):
        n_steps = int(self.t_max / self.dt) + 1
        t = np.linspace(0, self.t_max, n_steps)
        S = np.zeros(n_steps)
        I = np.zeros(n_steps)
        R = np.zeros(n_steps)
        D = np.zeros(n_steps)

        S[0], I[0], R[0], D[0] = self.S0, self.I0, self.R0, self.D0

        for n in range(n_steps - 1):
            dSdt = -self.beta * S[n] * I[n]
            dIdt = self.beta * S[n] * I[n] - self.gamma * I[n] - self.mu * I[n]
            dRdt = self.gamma * I[n]
            dDdt = self.mu * I[n]

            S[n + 1] = S[n] + self.dt * dSdt
            I[n + 1] = I[n] + self.dt * dIdt
            R[n + 1] = R[n] + self.dt * dRdt
            D[n + 1] = D[n] + self.dt * dDdt

        return t, S, I, R, D


class SIRDEvaluator:
    @staticmethod
    def compute_mse(predictions, observations):
        return np.mean((predictions - observations) ** 2)

    @staticmethod
    def evaluate_parameters(beta, gamma, mu, data, dt=0.01):
        S0 = data['Susceptibles'].iloc[0]
        I0 = data['Infectés'].iloc[0]
        R0 = data['Rétablis'].iloc[0]
        D0 = data['Décès'].iloc[0]

        t_max = len(data)
        model = SIRDModel(beta, gamma, mu, S0, I0, R0, D0, dt, t_max)
        t, S, I, R, D = model.euler_sird()

        indices = np.arange(0, len(S), int(1/dt))
        S = S[indices][:len(data)]
        I = I[indices][:len(data)]
        R = R[indices][:len(data)]
        D = D[indices][:len(data)]

        mse_S = SIRDEvaluator.compute_mse(S, data['Susceptibles'])
        mse_I = SIRDEvaluator.compute_mse(I, data['Infectés'])
        mse_R = SIRDEvaluator.compute_mse(R, data['Rétablis'])
        mse_D = SIRDEvaluator.compute_mse(D, data['Décès'])

        total_mse = mse_S + 2*mse_I + mse_R + mse_D

        return total_mse, (S, I, R, D)


class SIRDSimulation:
    @staticmethod
    def simulate_and_plot(beta, gamma, mu, S0, I0, R0, D0, dt, t_max):
        """
        Simule le modèle SIRD avec les paramètres donnés et trace les courbes.

        :param beta: Taux de transmission
        :param gamma: Taux de guérison
        :param mu: Taux de mortalité
        :param S0: Proportion initiale de susceptibles
        :param I0: Proportion initiale d'infectés
        :param R0: Proportion initiale de rétablis
        :param D0: Proportion initiale de décès
        :param dt: Pas de temps
        :param t_max: Temps total de simulation (en jours)
        """
        # Création du modèle
        model = SIRDModel(beta, gamma, mu, S0, I0, R0, D0, dt, t_max)

        # Simulation
        t, S, I, R, D = model.euler_sird()

        # Tracé des courbes
        plt.figure(figsize=(12, 6))
        plt.plot(t, S, 'b-', label='Susceptibles (S)', linewidth=2)
        plt.plot(t, I, 'r-', label='Infectés (I)', linewidth=2)
        plt.plot(t, R, 'g-', label='Rétablis (R)', linewidth=2)
        plt.plot(t, D, 'k-', label='Décès (D)', linewidth=2)

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Temps (jours)', fontsize=12)
        plt.ylabel('Proportion de la population', fontsize=12)
        plt.title(f'Simulation SIRD avec β={beta}, γ={
                  gamma}, μ={mu}', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()


class SIRDVisualizer:
    @staticmethod
    def visualize_results(data, best_curves, best_params, best_mse):
        plt.figure(figsize=(15, 6))

        plt.scatter(data['Jour'], data['Susceptibles'],
                    color='blue', alpha=0.5, label='S (réels)', s=20)
        plt.scatter(data['Jour'], data['Infectés'],
                    color='red', alpha=0.5, label='I (réels)', s=20)
        plt.scatter(data['Jour'], data['Rétablis'],
                    color='green', alpha=0.5, label='R (réels)', s=20)
        plt.scatter(data['Jour'], data['Décès'], color='black',
                    alpha=0.5, label='D (réels)', s=20)

        S_opt, I_opt, R_opt, D_opt = best_curves
        plt.plot(data['Jour'], S_opt, 'b-', label='S (simulées)', linewidth=2)
        plt.plot(data['Jour'], I_opt, 'r-', label='I (simulées)', linewidth=2)
        plt.plot(data['Jour'], R_opt, 'g-', label='R (simulées)', linewidth=2)
        plt.plot(data['Jour'], D_opt, 'k-', label='D (simulées)', linewidth=2)

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Temps (jours)', fontsize=12)
        plt.ylabel('Proportion de la population', fontsize=12)
        plt.title('Ajustement du modèle SIRD aux données empiriques', fontsize=14)

        info_text = (f'Paramètres optimaux:\n'
                     f'β = {best_params[0]:.3f}\n'
                     f'γ = {best_params[1]:.3f}\n'
                     f'μ = {best_params[2]:.3f}\n'
                     f'MSE = {best_mse:.6f}')
        plt.text(0.02, 0.98, info_text,
                 transform=plt.gca().transAxes,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()


def main():
    # Paramètres de simulation
    beta = 0.5
    gamma = 0.15
    mu = 0.015
    S0 = 0.99  # 99%
    I0 = 0.01  # 1%
    R0 = 0.0   # 0%
    D0 = 0.0   # 0%
    dt = 0.01  # Pas de temps
    t_max = 50  # Temps total de simulation (50 jours)

    # Simulation et tracé des courbes
    SIRDSimulation.simulate_and_plot(
        beta, gamma, mu, S0, I0, R0, D0, dt, t_max)

    # Recherche sur grille et visualisation (partie existante)
    data = pd.read_csv('sird_dataset.csv')

    beta_range = [0.355]
    gamma_range = [0.113]
    mu_range = [0.013]

    best_mse = float('inf')
    best_params = None
    best_curves = None

    total_iterations = len(beta_range) * len(gamma_range) * len(mu_range)
    print(f"Début de la recherche sur grille ({
          total_iterations} combinaisons)...")

    for i, (beta, gamma, mu) in enumerate(product(beta_range, gamma_range, mu_range)):
        mse, curves = SIRDEvaluator.evaluate_parameters(beta, gamma, mu, data)

        if mse < best_mse:
            best_mse = mse
            best_params = (beta, gamma, mu)
            best_curves = curves

        if (i + 1) % 1000 == 0:
            print(f"Progression : {100 * (i + 1) / total_iterations:.1f}%")

    beta_opt, gamma_opt, mu_opt = best_params
    print("\nMeilleurs paramètres trouvés :")
    print(f"β = {beta_opt:.3f}")
    print(f"γ = {gamma_opt:.3f}")
    print(f"μ = {mu_opt:.3f}")
    print(f"MSE totale = {best_mse:.6f}")

    SIRDVisualizer.visualize_results(data, best_curves, best_params, best_mse)


if __name__ == "__main__":
    main()
