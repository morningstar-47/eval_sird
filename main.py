import pandas as pd
from sird_simulation import SIRDSimulation
from sird_control import SIRDControl
from sird_visualization import SIRDVisualizer
from sird_evaluator import SIRDEvaluator


def main():
    beta, gamma, mu = 0.5, 0.15, 0.015
    S0, I0, R0, D0 = 0.99, 0.01, 0.0, 0.0
    dt, t_max = 0.01, 50

    data = pd.read_csv('sird_dataset.csv')
    best_mse, best_curves = SIRDEvaluator.evaluate_parameters(
        beta, gamma, mu, data)

    SIRDSimulation.simulate_and_plot(
        beta, gamma, mu, S0, I0, R0, D0, dt, t_max)
    SIRDControl.control_scenarios(
        beta, gamma, mu, S0, I0, R0, D0, dt, t_max, intervention_beta=0.3)
    SIRDVisualizer.visualize_results(data, best_curves)
    print(f"Meilleure erreur MSE : {best_mse:.2f}")


if __name__ == "__main__":
    main()
