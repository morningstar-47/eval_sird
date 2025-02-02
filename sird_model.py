import numpy as np


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
        S, I, R, D = np.zeros(n_steps), np.zeros(
            n_steps), np.zeros(n_steps), np.zeros(n_steps)
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
