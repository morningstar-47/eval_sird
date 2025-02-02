import matplotlib.pyplot as plt


class SIRDVisualizer:
    @staticmethod
    def visualize_results(data, best_curves):
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
        plt.xlabel('Temps (jours)')
        plt.ylabel('Proportion de la population')
        plt.legend()
        plt.title('Comparaison des données réelles et simulées')
        plt.show()
