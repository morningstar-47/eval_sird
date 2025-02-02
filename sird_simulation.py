import matplotlib.pyplot as plt
from sird_model import SIRDModel


class SIRDSimulation:
    @staticmethod
    def simulate_and_plot(beta, gamma, mu, S0, I0, R0, D0, dt, t_max):
        model = SIRDModel(beta, gamma, mu, S0, I0, R0, D0, dt, t_max)
        t, S, I, R, D = model.euler_sird()

        plt.figure(figsize=(12, 6))
        plt.plot(t, S, 'b-', label='Susceptibles', linewidth=2)
        plt.plot(t, I, 'r-', label='Infectés', linewidth=2)
        plt.plot(t, R, 'g-', label='Rétablis', linewidth=2)
        plt.plot(t, D, 'k-', label='Décès', linewidth=2)

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Temps (jours)')
        plt.ylabel('Proportion de la population')
        plt.title(f'Simulation SIRD: β={beta}, γ={gamma}, μ={mu}')
        plt.legend()
        plt.show()
