import time
import numpy as np
from scipy.stats import multivariate_normal as normal
from scipy.integrate import solve_ivp

class LQ(object):
    def __init__(self, eqn_config):
        np.random.seed(seed=eqn_config.seed)
        self.eqn_config = eqn_config
        self.delta = eqn_config.delta
        self.T = eqn_config.T
        self.nt = eqn_config.nt
        self.lambd = eqn_config.lambd
        self.dim_x = eqn_config.dim_x
        self.dim_pi = eqn_config.dim_pi
        self.dim_w = eqn_config.dim_w
        
        self.dt = self.T / self.nt
        self.sqrt_dt = np.sqrt(self.dt)
        self.n_lag = int(np.round(self.delta/self.dt))
        # self.n_lag = self.delta // self.dt
        # assert self.n_lag * self.dt == self.delta, "The time discretization is inconsistent."
        
        self.A1 = np.identity(self.dim_x) * 0.5
        self.A3 = np.identity(self.dim_x) * 5
        self.Q = np.identity(self.dim_x) / self.dim_x / 10
        self.R = np.identity(self.dim_pi) / self.dim_pi / 10
        self.G = np.identity(self.dim_x) / self.dim_x / 10
        self.B = np.random.normal(size=(self.dim_x, self.dim_pi), scale=1)
        self.sigma = np.random.normal(size=(self.dim_x, self.dim_w), scale=1)
        # self.x_init = np.random.normal(size=(self.dim_x, 1)) * -np.arange(self.n_lag+1) * self.dt  # of shape (dx, n_lag+1)
        self.x_init = 1 * np.arange(1, self.dim_x+1)[:, None] * -np.arange(self.n_lag+1) * self.dt  # of shape (dx, n_lag+1)
        
        self.Rinv = np.linalg.inv(self.R)
        self.exp_fac = np.exp(self.lambd * self.delta)
        self.exp_array = np.exp(-np.arange(-self.n_lag, 1, 1) * self.dt * self.lambd)
        self.wgt_x_init = self.exp_array * self.x_init # of shape (dx, n_lag+1)
        self.A2 = self.exp_fac * (self.lambd * np.identity(self.dim_x) + self.A1 + self.exp_fac * self.A3) @ self.A3
        self.Pt = self.riccati_solu()
        np.random.seed(int(time.time()))
        
    def riccati_solu(self):
        def full_riccati(t, y):
            P = np.reshape(y, (self.dim_x, self.dim_x))
            coeff = self.A1 + self.exp_fac * self.A3
            dP = self.Q + P @ coeff + coeff.transpose() @ P - P @ self.B @ self.Rinv @ self.B.transpose() @ P
            return np.reshape(dP, -1)
        
        sol = solve_ivp(full_riccati, [0, self.T], np.reshape(self.G, -1),
                        t_eval=np.linspace(0, self.T, self.nt+1))
        Pt = np.flip(sol.y, axis=-1)
        Pt = np.reshape(Pt.transpose(), (self.nt+1, self.dim_x, self.dim_x))
        # print(Pt[0])
        # print(Pt[1])
        # print(Pt[-1])
        return Pt
    
    def sample(self, num_sample, fixseed=False):
        if fixseed:
            np.random.seed(seed=self.eqn_config.seed)
        dw_sample = normal.rvs(size=[num_sample, self.dim_w, self.nt]) * self.sqrt_dt
        x_hist = np.repeat(self.x_init[None, :, :], [num_sample], axis=0)
        wgt_x_hist = np.repeat(self.wgt_x_init[None, :, :], [num_sample], axis=0)
        if fixseed:
            np.random.seed(int(time.time()))
        return dw_sample, x_hist, wgt_x_hist
        
    def simulate_true(self, num_sample, fixseed=False):
        if fixseed:
            np.random.seed(seed=self.eqn_config.seed)
        dw_sample = normal.rvs(size=[num_sample, self.dim_w, self.nt]) * self.sqrt_dt
        if fixseed:
            np.random.seed(int(time.time()))
        x_sample = np.zeros([num_sample, self.dim_x, self.nt+1])
        x_sample[:, :, 0] = self.x_init[:, -1]
        y_sample = np.zeros_like(x_sample)
        reward = np.zeros([num_sample])
        value = np.zeros([num_sample])
        
        for t in range(self.nt+1):
            if t == 0:
                x_hist = np.repeat(self.x_init[None, :, :], [num_sample], axis=0) # of shape (B, dx, n_lag+1)
                wgt_x_hist = np.repeat(self.wgt_x_init[None, :, :], [num_sample], axis=0)  # of shape (B, dx, n_lag+1)
            else:
                x_hist[:, :, :-1] = x_hist[:, :, 1:]
                x_hist[:, :, -1] = x_sample[:, :, t]
                wgt_x_hist[:, :, :-1] = wgt_x_hist[:, :, 1:] * self.exp_array[-2]
                wgt_x_hist[:, :, -1] = x_sample[:, :, t]
            zeta = x_hist[:, :, 0]
            y_sample[..., t] = (np.sum(wgt_x_hist, axis=-1) - 0.5*(wgt_x_hist[..., 0] + wgt_x_hist[..., -1])) * self.dt
            x_common = x_sample[..., t] + self.exp_fac * y_sample[..., t] @ self.A3
            pi = - x_common @ (self.Rinv @ self.B.transpose() @ self.Pt[t]).transpose()
            # print(pi[0])
            inst_r = np.sum((x_common @ self.Q) * x_common, axis=-1) + np.sum((pi @ self.R) * pi, axis=-1)
            if t == 0:
                reward += inst_r * self.dt / 2
                value = np.sum((x_common @ self.Pt[0]) * x_common, axis=-1)
                Psum = np.sum(self.Pt, axis=0) - self.Pt[0]/2 - self.Pt[-1]/2
                value += np.sum(Psum * (self.sigma @ self.sigma.transpose())) * self.dt
            elif t == self.nt:
                reward += inst_r * self.dt / 2
                reward += np.sum((x_common @ self.G) * x_common, axis=-1)
            else:
                reward += inst_r * self.dt

            if t < self.nt:
                dx = (x_sample[..., t] @ self.A1.transpose() + y_sample[..., t] @ self.A2.transpose() \
                  + zeta @ self.A3.transpose() + pi @ self.B.transpose())
                x_sample[..., t+1] = x_sample[..., t] + dx * self.dt + dw_sample[..., t] @ self.sigma.transpose()

        return x_sample, reward, value

    def simulate(self, num_sample, policy, fixseed=False):
        if fixseed:
            np.random.seed(seed=self.eqn_config.seed)
        dw_sample = normal.rvs(size=[num_sample, self.dim_w, self.nt]) * self.sqrt_dt
        if fixseed:
            np.random.seed(int(time.time()))
        x_sample = np.zeros([num_sample, self.dim_x, self.nt+1])
        x_sample[:, :, 0] = self.x_init[:, -1]
        pi_sample = np.zeros([num_sample, self.dim_pi, self.nt+1])
        y_sample = np.zeros_like(x_sample)
        reward = np.zeros([num_sample])

        for t in range(self.nt+1):
            if t == 0:
                x_hist = np.repeat(self.x_init[None, :, :], [num_sample], axis=0) # of shape (B, dx, n_lag+1)
                wgt_x_hist = np.repeat(self.wgt_x_init[None, :, :], [num_sample], axis=0)  # of shape (B, dx, n_lag+1)
            else:
                x_hist[:, :, :-1] = x_hist[:, :, 1:]
                x_hist[:, :, -1] = x_sample[:, :, t]
                wgt_x_hist[:, :, :-1] = wgt_x_hist[:, :, 1:] * self.exp_array[-2]
                wgt_x_hist[:, :, -1] = x_sample[:, :, t]
            zeta = x_hist[:, :, 0]
            y_sample[..., t] = (np.sum(wgt_x_hist, axis=-1) - 0.5*(wgt_x_hist[..., 0] + wgt_x_hist[..., -1])) * self.dt
            x_common = x_sample[..., t] + self.exp_fac * y_sample[..., t] @ self.A3
            pi_sample[..., t] = policy(t, x_hist, wgt_x_hist)
            pi = pi_sample[..., t]
            inst_r = np.sum((x_common @ self.Q) * x_common, axis=-1) + np.sum((pi @ self.R) * pi, axis=-1)
            if t == 0:
                reward += inst_r * self.dt / 2
                Psum = np.sum(self.Pt, axis=0) - self.Pt[0]/2 - self.Pt[-1]/2
            elif t == self.nt:
                reward += inst_r * self.dt / 2
                reward += np.sum((x_common @ self.G) * x_common, axis=-1)
            else:
                reward += inst_r * self.dt

            if t < self.nt:
                dx = (x_sample[..., t] @ self.A1.transpose() + y_sample[..., t] @ self.A2.transpose() \
                  + zeta @ self.A3.transpose() + pi @ self.B.transpose())
                x_sample[..., t+1] = x_sample[..., t] + dx * self.dt + dw_sample[..., t] @ self.sigma.transpose()

        return x_sample, pi_sample, reward

    def true_policy(self, t, x_hist, wgt_x_hist=None):
        y_sample = (np.sum(wgt_x_hist, axis=-1) - 0.5*(wgt_x_hist[..., 0] + wgt_x_hist[..., -1])) * self.dt
        x_common = x_hist[..., -1] + self.exp_fac * y_sample @ self.A3
        pi = - x_common @ (self.Rinv @ self.B.transpose() @ self.Pt[t]).transpose()
        return pi