# main.py
from simulation import run_simulation
from parameter_estimation import run_parameter_estimation
from intervention import run_intervention


def main():
    print("=== Étape 2 : Simulation ===")
    # run_simulation()

    print("\n=== Étape 3 : Ajustement des paramètres ===")
    run_parameter_estimation()

    print("\n=== Étape 4 : Scénarios de contrôle ===")
    run_intervention()


if __name__ == "__main__":
    main()
