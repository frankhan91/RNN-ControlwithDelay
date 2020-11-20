import time
import numpy as np
from scipy.stats import multivariate_normal as normal
from scipy.integrate import solve_ivp

class LQ(object):
    def __init__(self, eqn_config):
        np.random.seed(seed=eqn_config.seed)
        self.eqn_config = eqn_config
        self.fixinit = self.eqn_config.fixinit
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
        
        self.Rinv = np.linalg.inv(self.R)
        self.exp_fac = np.exp(self.lambd * self.delta)
        self.exp_array = np.exp(-np.arange(-self.n_lag, 1, 1) * self.dt * self.lambd)
        self.A2 = self.exp_fac * (self.lambd * np.identity(self.dim_x) + self.A1 + self.exp_fac * self.A3) @ self.A3
        self.Pt, v_integral = self.riccati_soln()
        dw_sample, x_init, wgt_x_init = self.sample(4096)
        y_init = (np.sum(wgt_x_init, axis=-1) - 0.5*(wgt_x_init[..., 0] + wgt_x_init[..., -1])) * self.dt
        x_common = x_init[..., -1] + self.exp_fac * y_init @ self.A3
        self.value = v_integral + np.mean(np.sum((x_common @ self.Pt[0]) * x_common, axis=-1))
        np.random.seed(int(time.time()))
    
    def sample(self, num_sample, fixseed=False):
        if fixseed:
            np.random.seed(seed=self.eqn_config.seed)
        dw_sample = normal.rvs(size=[num_sample, self.dim_w, self.nt]) * self.sqrt_dt
        if self.fixinit:
            x_init = 1 * np.arange(1, self.dim_x+1)[:, None]/self.dim_x * -np.arange(self.n_lag+1) * self.dt    # of shape (dx, n_lag+1)
            wgt_x_init = self.exp_array * x_init    # of shape (dx, n_lag+1)
            x_init = np.repeat(x_init[None, :, :], [num_sample], axis=0)    # of shape (B, dx, n_lag+1)
            wgt_x_hist = np.repeat(wgt_x_init[None, :, :], [num_sample], axis=0)    # of shape (B, dx, n_lag+1)
        if fixseed:
            np.random.seed(int(time.time()))
        return dw_sample, x_init, wgt_x_hist
        
    def simulate(self, num_sample, policy, fixseed=False, hidden_init_fn=None):
        dw_sample, x_init, wgt_x_hist = self.sample(num_sample, fixseed)
        x_sample = np.zeros([num_sample, self.dim_x, self.nt+1])
        x_hist = x_init.copy()
        x_sample[:, :, 0] = x_hist[:, :, -1]
        pi_sample = np.zeros([num_sample, self.dim_pi, self.nt])
        y_sample = np.zeros_like(x_sample)
        reward = np.zeros([num_sample])
        if hidden_init_fn is None:
            hidden = None
        else:
            hidden = hidden_init_fn(x_init) # used for LSTM model only

        reward = 0
        for t in range(self.nt+1):
            if t > 0:
                x_hist[:, :, :-1] = x_hist[:, :, 1:]
                x_hist[:, :, -1] = x_sample[:, :, t]
                wgt_x_hist[:, :, :-1] = wgt_x_hist[:, :, 1:] * self.exp_array[-2]
                wgt_x_hist[:, :, -1] = x_sample[:, :, t]
            zeta = x_hist[:, :, 0]
            y_sample[..., t] = (np.sum(wgt_x_hist, axis=-1) - 0.5*(wgt_x_hist[..., 0] + wgt_x_hist[..., -1])) * self.dt
            x_common = x_sample[..., t] + self.exp_fac * y_sample[..., t] @ self.A3
            if t == self.nt:
                reward += np.sum((x_common @ self.G) * x_common, axis=-1)
            else:
                if t == 0:
                    dw_inst = dw_sample[..., 0] * 0 + 0.1
                else:
                    dw_inst = dw_sample[..., t-1]
                pi, hidden = policy(t, x_hist, wgt_x_hist, dw_inst, hidden)
                pi_sample[..., t] = pi
                inst_r = np.sum((x_common @ self.Q) * x_common, axis=-1) + np.sum((pi @ self.R) * pi, axis=-1)
                reward = reward + inst_r * self.dt

            if t < self.nt:
                dx = (x_sample[..., t] @ self.A1.transpose() + y_sample[..., t] @ self.A2.transpose() \
                  + zeta @ self.A3.transpose() + pi @ self.B.transpose())
                x_sample[..., t+1] = x_sample[..., t] + dx * self.dt + dw_sample[..., t] @ self.sigma.transpose()

        return x_sample, pi_sample, reward

    def riccati_soln(self):
        def full_riccati(t, y):
            dy = np.zeros_like(y)
            P = np.reshape(y[:-1], (self.dim_x, self.dim_x))
            coeff = self.A1 + self.exp_fac * self.A3
            dP = self.Q + P @ coeff + coeff.transpose() @ P - P @ self.B @ self.Rinv @ self.B.transpose() @ P
            dy[:-1] = np.reshape(dP, -1)
            dy[-1] = np.sum(P * (self.sigma @ self.sigma.transpose()))
            return dy

        sol = solve_ivp(full_riccati, [0, self.T], np.concatenate([np.reshape(self.G, -1), [0]]),
                        t_eval=np.linspace(0, self.T, self.nt+1))
        y = np.flip(sol.y, axis=-1) # (dim_P + 1) * (nt+1)
        Pt = np.reshape(y[:-1].transpose(), (self.nt+1, self.dim_x, self.dim_x))
        v_integral = y[-1, 0]
        # print(Pt[0])
        # print(Pt[1])
        # print(Pt[-1])
        return Pt, v_integral

    def true_policy(self, t, x_hist, wgt_x_hist, dw_inst, hidden=None):
        y_sample = (np.sum(wgt_x_hist, axis=-1) - 0.5*(wgt_x_hist[..., 0] + wgt_x_hist[..., -1])) * self.dt
        x_common = x_hist[..., -1] + self.exp_fac * y_sample @ self.A3
        pi = - x_common @ (self.Rinv @ self.B.transpose() @ self.Pt[t]).transpose()
        return pi, None


class Csmp(object):
    def __init__(self, eqn_config):
        np.random.seed(seed=eqn_config.seed)
        self.eqn_config = eqn_config
        self.fixinit = self.eqn_config.fixinit
        self.delta = eqn_config.delta
        self.T = eqn_config.T
        self.nt = eqn_config.nt
        self.lambd = eqn_config.lambd
        self.beta = eqn_config.beta
        self.gamma = eqn_config.gamma
        self.a = eqn_config.a
        self.mu = eqn_config.mu
        self.sigma = eqn_config.sigma
        self.dim_pi = eqn_config.dim_pi

        self.dt = self.T / self.nt
        self.sqrt_dt = np.sqrt(self.dt)
        self.n_lag = int(np.round(self.delta/self.dt))

        self.final_disc = np.exp(-self.beta*self.T)
        self.util_fn = lambda x: x**self.gamma / self.gamma
        self.exp_fac = np.exp(self.lambd * self.delta)
        self.drift_coeff = self.a * self.exp_fac
        self.exp_array = np.exp(-np.arange(-self.n_lag, 1, 1) * self.dt * self.lambd)
        self.pt = self.riccati_soln()
        dw_sample, x_init, wgt_x_init = self.sample(4096)
        y_sample = (np.sum(wgt_x_init, axis=-1) - 0.5*(wgt_x_init[..., 0] + wgt_x_init[..., -1])) * self.dt
        x_common = x_init[..., -1] + self.a * self.exp_fac * y_sample
        self.value = np.mean(self.pt[0]**(1-self.gamma) / self.gamma * x_common**(self.gamma))
        np.random.seed(int(time.time()))

    def sample(self, num_sample, fixseed=False):
        if fixseed:
            np.random.seed(seed=self.eqn_config.seed)
        dw_sample = normal.rvs(size=[num_sample, self.nt]) * self.sqrt_dt
        if self.fixinit:
            x_init = 2 + 5 * np.arange(self.n_lag+1) * self.dt  # of shape (n_lag+1,)
            wgt_x_init = self.exp_array * x_init   # of shape (n_lag+1,)
            x_init = np.repeat(x_init[None, :], [num_sample], axis=0)   # of shape (B, n_lag+1)
            wgt_x_hist = np.repeat(wgt_x_init[None, :], [num_sample], axis=0)   # of shape (B, n_lag+1)
        else:
            x_init = np.zeros([num_sample, self.n_lag+1])
            x_init[:, 0] = np.random.uniform(5, 10, size=[num_sample,])
            for t in range(1, self.n_lag+1):
                x_init[:, t] = x_init[:, t-1] * (1 + 0.03*self.dt + 1*self.sigma*np.random.normal(size=[num_sample]) * self.sqrt_dt)
            # x_init[:, 0] = np.random.uniform(3, 5, size=[num_sample,])
            # for t in range(1, self.n_lag+1):
            #     x_init[:, t] = x_init[:, t-1] * (1 + 0.03*self.dt + 0.5*self.sigma*np.random.normal(size=[num_sample]) * self.sqrt_dt)
            wgt_x_hist = self.exp_array * x_init
        if fixseed:
            np.random.seed(int(time.time()))
        return dw_sample, x_init, wgt_x_hist

    def simulate(self, num_sample, policy, fixseed=False, hidden_init_fn=None):
        dw_sample, x_init, wgt_x_hist = self.sample(num_sample, fixseed)
        x_sample = np.zeros([num_sample, self.nt+1])
        x_hist = x_init.copy()
        x_sample[:, 0] = x_hist[:, -1]
        pi_sample = np.zeros([num_sample, self.nt])
        y_sample = np.zeros_like(x_sample)
        reward = np.zeros([num_sample])
        if hidden_init_fn is None:
            hidden = None
        else:
            hidden = hidden_init_fn(x_init) # used for LSTM model only

        reward = 0
        for t in range(self.nt+1):
            if t > 0:
                x_hist[:, :-1] = x_hist[:, 1:]
                x_hist[:, -1] = x_sample[:, t]
                wgt_x_hist[:, :-1] = wgt_x_hist[:, 1:] * self.exp_array[-2]
                wgt_x_hist[:, -1] = x_sample[:, t]
            zeta = x_hist[:, 0]
            y_sample[..., t] = (np.sum(wgt_x_hist, axis=-1) - 0.5*(wgt_x_hist[..., 0] + wgt_x_hist[..., -1])) * self.dt
            x_common = x_sample[..., t] + self.a * self.exp_fac * y_sample[..., t]
            if t == self.nt:
                reward += self.util_fn(x_common)*self.final_disc
            else:
                if t == 0:
                    dw_inst = dw_sample[..., 0:1] * 0 + 0.1
                else:
                    dw_inst = dw_sample[..., t-1:t]
                pi, hidden = policy(t, x_hist, wgt_x_hist, dw_inst, hidden)
                pi = np.maximum(pi, 0)
                pi_sample[..., t] = pi
                inst_r = self.util_fn(pi) * np.exp(-self.beta * t * self.dt)
                reward = reward + inst_r * self.dt

            if t < self.nt:
                dx = self.drift_coeff*(self.drift_coeff+self.lambd) * y_sample[..., t] + self.mu * x_common \
                    + self.a * zeta - pi
                x_sample[..., t+1] = x_sample[..., t] + dx * self.dt + dw_sample[..., t] * self.sigma * x_common
                if x_sample[..., t+1].min() < 0:
                    print("Nagative x: {}".format(x_sample[..., t+1].min()))

        return x_sample, pi_sample, reward

    def riccati_soln(self):
        def full_riccati(t, p):
            coeff = 0.5*self.gamma*self.sigma**2 - self.gamma/(1-self.gamma)*(self.mu+self.a*self.exp_fac)
            dp = np.exp(-self.beta*t/(1-self.gamma)) - coeff * p
            return dp

        sol = solve_ivp(full_riccati, [0, self.T], [self.final_disc**(1/(1-self.gamma))],
                        t_eval=np.linspace(0, self.T, self.nt+1))
        pt = np.flip(sol.y, axis=-1)[0] # of shape(nt+1)
        return pt

    def true_policy(self, t, x_hist, wgt_x_hist, dw_inst, hidden=None):
        y_sample = (np.sum(wgt_x_hist, axis=-1) - 0.5*(wgt_x_hist[..., 0] + wgt_x_hist[..., -1])) * self.dt
        x_common = x_hist[..., -1] + self.a * self.exp_fac * y_sample
        pi = np.exp(-self.beta * t * self.dt/(1-self.gamma)) * x_common / self.pt[t]
        return pi, None


class POlog(object):
    def __init__(self, eqn_config):
        np.random.seed(seed=eqn_config.seed)
        self.eqn_config = eqn_config
        self.fixinit = self.eqn_config.fixinit
        self.delta = eqn_config.delta
        self.T = eqn_config.T
        self.nt = eqn_config.nt
        self.lambd = eqn_config.lambd
        self.beta = eqn_config.beta
        self.mu1 = eqn_config.mu1
        self.mu2 = eqn_config.mu2
        self.r = eqn_config.r
        self.sigma = eqn_config.sigma
        self.dim_pi = eqn_config.dim_pi
        self.logeps = eqn_config.logeps

        self.eta = (np.sqrt((self.r+self.lambd)**2 + 4*self.mu2) - (self.r+self.lambd)) / 2
        assert (self.mu1 - self.r)**2/2/self.sigma**2 + self.r + self.eta - self.beta*(1+np.log(self.beta)) > 0, \
            "Parameter condition is violated"
        self.dt = self.T / self.nt
        self.sqrt_dt = np.sqrt(self.dt)
        self.n_lag = int(np.round(self.delta/self.dt))
        self.Lambd2 = (self.mu1 - self.r)**2/2/self.sigma**2/self.beta + np.log(self.beta) - 1 + (self.r+self.eta) / self.beta

        self.x_init = 0.2 + 5 * np.arange(self.n_lag+1) * self.dt  # of shape (n_lag+1,)

        self.final_disc = np.exp(-self.beta*self.T)
        t_grid = np.linspace(0, self.T, self.nt+1)
        self.pt = self.Lambd2 / self.beta * (1 - np.exp(-self.beta*(self.T - t_grid)))
        self.exp_array = np.exp(-np.arange(-self.n_lag+1, 1, 1) * self.dt * self.lambd)  # of shape (n_lag,)
        # assume x_init are constants before -delta
        self.geometric_sum = np.exp(-self.lambd * self.T) * self.dt / (1-np.exp(-self.lambd * self.dt))
        dw_sample, x_init = self.sample(4096)
        y_init = np.sum(self.exp_array * x_init[:, 1:], axis=-1) * self.dt + self.geometric_sum * x_init[:, 0]
        self.value = np.mean(self.pt[0] + np.log(x_init[:, -1] + self.eta * y_init) / self.beta)
        np.random.seed(int(time.time()))

    def sample(self, num_sample, fixseed=False):
        if fixseed:
            np.random.seed(seed=self.eqn_config.seed)
        dw_sample = normal.rvs(size=[num_sample, self.nt]) * self.sqrt_dt
        if self.fixinit:
            x_init = 0.2 + 5 * np.arange(self.n_lag+1) * self.dt  # of shape (n_lag+1,)
            x_init = np.repeat(x_init[None, :], [num_sample], axis=0)
        else:
            x_init = np.zeros([num_sample, self.n_lag+1])
            x_init[:, 0] = np.random.uniform(1, 5, size=[num_sample,])
            for t in range(1, self.n_lag+1):
                x_init[:, t] = x_init[:, t-1] * (1 + 0.4*self.dt + 0.1*self.sigma*np.random.normal(size=[num_sample]) * self.sqrt_dt)
        if fixseed:
            np.random.seed(int(time.time()))
        return dw_sample, x_init

    def simulate(self, num_sample, policy, fixseed=False, hidden_init_fn=None):
        dw_sample, x_init = self.sample(num_sample, fixseed)
        x_sample = np.zeros([num_sample, self.nt+1])
        x_hist = x_init.copy()
        x_sample[:, 0] = x_hist[:, -1]
        pi_sample = np.zeros([num_sample, self.dim_pi, self.nt])
        reward = np.zeros([num_sample])
        if hidden_init_fn is None:
            hidden = None
        else:
            hidden = hidden_init_fn(x_init) # used for LSTM model only

        reward = 0
        y = np.sum(self.exp_array * x_hist[:, 1:], axis=-1) * self.dt + self.geometric_sum * x_hist[:, 0]
        for t in range(self.nt+1):
            if t > 0:
                x_hist[:, :-1] = x_hist[:, 1:]
                x_hist[:, -1] = x_sample[:, t]
            if t == self.nt:
                reward += np.log(x_sample[:, t] + self.eta * y) / self.beta * self.final_disc
            else:
                if t == 0:
                    dw_inst = dw_sample[..., 0:1] * 0 + 0.1
                else:
                    dw_inst = dw_sample[..., t-1:t]
                pi, hidden = policy(t, x_hist, y, dw_inst, hidden)
                pi[:, 0] = np.maximum(pi[:, 0], 0)
                pi_sample[..., t] = pi
                inst_r = np.log(pi[:, 0] * x_sample[:, t] + self.logeps) * np.exp(-self.beta * t * self.dt)
                reward = reward + inst_r * self.dt

            if t < self.nt:
                dx = ((self.mu1-self.r)*pi[:, 1] - pi[:, 0] + self.r) * x_sample[..., t] + self.mu2 * y
                x_sample[..., t+1] = x_sample[..., t] + dx * self.dt + dw_sample[..., t] * self.sigma * pi[:, 1] * x_sample[:, t]
                y = y * np.exp(-self.dt * self.lambd) + x_sample[..., t+1] * self.dt
                # print(np.mean(x_sample[..., t+1]), np.mean(y), np.mean(reward))
                if x_sample[..., t+1].min() < 0:
                    print("Nagative x: {}".format(x_sample[..., t+1].min()))

        return x_sample, pi_sample, reward

    def true_policy(self, t, x_hist, y, dw_inst, hidden=None):
        x_common = (x_hist[..., -1] + self.eta * y) / x_hist[..., -1]
        c = self.beta * x_common
        pi = (self.mu1 - self.r) / self.sigma**2 * x_common
        pi = np.concatenate([c[:, None], pi[:, None]], axis=-1)
        return pi, None
