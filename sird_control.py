import matplotlib.pyplot as plt
from sird_model import SIRDModel


class SIRDControl:
    @staticmethod
    def calculate_R0(beta, gamma, mu):
        return beta / (gamma + mu)

    @staticmethod
    def control_scenarios(beta, gamma, mu, S0, I0, R0, D0, dt, t_max, intervention_beta=None):
        R0_initial = SIRDControl.calculate_R0(beta, gamma, mu)
        print(f"R0 initial : {R0_initial:.2f}")

        model_no_intervention = SIRDModel(
            beta, gamma, mu, S0, I0, R0, D0, dt, t_max)
        t, S_no, I_no, R_no, D_no = model_no_intervention.euler_sird()

        plt.figure(figsize=(14, 8))
        plt.plot(t, S_no, 'b-', label='S sans intervention',
                 linewidth=2, alpha=0.7)
        plt.plot(t, I_no, 'r-', label='I sans intervention',
                 linewidth=2, alpha=0.7)
        plt.plot(t, R_no, 'g-', label='R sans intervention',
                 linewidth=2, alpha=0.7)
        plt.plot(t, D_no, 'k-', label='D sans intervention',
                 linewidth=2, alpha=0.7)

        if intervention_beta is not None:
            R0_intervention = SIRDControl.calculate_R0(
                intervention_beta, gamma, mu)
            print(f"R0 après intervention : {R0_intervention:.2f}")

            model_intervention = SIRDModel(
                intervention_beta, gamma, mu, S0, I0, R0, D0, dt, t_max)
            t, S_int, I_int, R_int, D_int = model_intervention.euler_sird()

            plt.plot(t, S_int, 'b--', label='S avec intervention',
                     linewidth=2, alpha=0.7)
            plt.plot(t, I_int, 'r--', label='I avec intervention',
                     linewidth=2, alpha=0.7)
            plt.plot(t, R_int, 'g--', label='R avec intervention',
                     linewidth=2, alpha=0.7)
            plt.plot(t, D_int, 'k--', label='D avec intervention',
                     linewidth=2, alpha=0.7)

        plt.xlabel('Temps (jours)')
        plt.ylabel('Proportion de la population')
        plt.title('Comparaison des scénarios')
        plt.legend()
        plt.show()
