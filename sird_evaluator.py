import numpy as np
from sird_model import SIRDModel


class SIRDEvaluator:
    @staticmethod
    def compute_mse(predictions, observations):
        """Calcul de l'erreur quadratique moyenne (MSE)."""
        return np.mean((predictions - observations) ** 2)

    @staticmethod
    def evaluate_parameters(beta, gamma, mu, data, dt=0.01):
        """Évalue le modèle avec des paramètres spécifiques et renvoie l'erreur MSE."""
        S0, I0, R0, D0 = data['Susceptibles'][0], data['Infectés'][0], data['Rétablis'][0], data['Décès'][0]
        t_max = len(data)

        model = SIRDModel(beta, gamma, mu, S0, I0, R0, D0, dt, t_max)
        t, S, I, R, D = model.euler_sird()

        indices = np.arange(0, len(S), int(1/dt))
        S, I, R, D = S[indices][:len(data)], I[indices][:len(
            data)], R[indices][:len(data)], D[indices][:len(data)]

        total_mse = sum([
            SIRDEvaluator.compute_mse(S, data['Susceptibles']),
            2 * SIRDEvaluator.compute_mse(I, data['Infectés']),
            SIRDEvaluator.compute_mse(R, data['Rétablis']),
            SIRDEvaluator.compute_mse(D, data['Décès'])
        ])

        return total_mse, (S, I, R, D)

    @staticmethod
    def grid_search(data, beta_range, gamma_range, mu_range, dt=0.01):
        """Effectue un Grid Search pour trouver les meilleurs paramètres (β, γ, μ)."""
        best_mse = float('inf')
        best_params = None
        best_curves = None

        # Recherche des meilleures valeurs dans les plages données
        for beta in beta_range:
            for gamma in gamma_range:
                for mu in mu_range:
                    mse, curves = SIRDEvaluator.evaluate_parameters(
                        beta, gamma, mu, data, dt)
                    if mse < best_mse:
                        best_mse = mse
                        best_params = (beta, gamma, mu)
                        best_curves = curves

        return best_mse, best_params, best_curves
