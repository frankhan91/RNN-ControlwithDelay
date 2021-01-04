import sys
import logging
import time
import numpy as np
import tensorflow as tf


class Solver(object):
    """The fully connected neural network model."""
    def __init__(self, config, eqn):
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.eqn = eqn

        self.model = getattr(
            sys.modules[__name__],
            self.eqn_config.eqn_name + self.net_config.model_name)(config, eqn)
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            self.net_config.lr_boundaries, self.net_config.lr_values)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-8)

    def train(self):
        start_time = time.time()
        training_history = []
        valid_data = self.eqn.sample(self.net_config.valid_size, fixseed=True)
        _, _, reward = self.eqn.simulate(self.net_config.valid_size, self.eqn.true_policy, fixseed=True)
        logging.info('Reward of valid data with analytic policy: %.4e' % (np.mean(reward)))

        # begin sgd iteration
        for step in range(self.net_config.num_iterations+1):
            if step % self.net_config.logging_frequency == 0:
                reward = self.loss_fn(valid_data, training=False).numpy()
                elapsed_time = time.time() - start_time
                training_history.append([step, reward, elapsed_time])
                if self.net_config.verbose:
                    logging.info("step: %5u,    reward: %.4e,   elapsed time: %3u" % (
                        step, reward, elapsed_time))
                    if step == 0:
                        print(self.model.summary())
            self.train_step(self.eqn.sample(self.net_config.batch_size))
        return np.array(training_history)


    def loss_fn(self, inputs, training):
        reward = self.model(inputs, training)
        # use linear approximation outside the clipped range
        loss = tf.reduce_mean(reward)

        return loss

    def grad(self, inputs, training):
        with tf.GradientTape(persistent=True) as tape:
            loss = self.loss_fn(inputs, training)
        grad = tape.gradient(loss, self.model.trainable_variables)
        del tape
        return grad

    @tf.function
    def train_step(self, train_data):
        grad = self.grad(train_data, training=True)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))


class LQPolicyModel(tf.keras.Model):
    def __init__(self, config, eqn):
        super(LQPolicyModel, self).__init__()
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.eqn = eqn

    def call(self, inputs, training):
        dw_sample, x_init, wgt_x_hist = inputs
        x = x_init[:, :, -1] # of shape (B, dx)
        hidden = self.hidden_init_tf(x_init)
        x_hist = x_init

        reward = 0
        for t in range(self.eqn.nt+1):
            if t > 0:
                x_hist = tf.concat([x_hist[:, :, 1:], x[:, :, None]], axis=-1)
                wgt_x_hist = tf.concat([wgt_x_hist[:, :, 1:] * self.eqn.exp_array[-2], x[:, :, None]], axis=-1)
            zeta = x_hist[:, :, 0]
            y = (tf.reduce_sum(wgt_x_hist, axis=-1) - 0.5*(wgt_x_hist[..., 0] + wgt_x_hist[..., -1])) * self.eqn.dt
            x_common = x + self.eqn.exp_fac * y @ self.eqn.A3
            if t == self.eqn.nt:
                reward += tf.reduce_sum((x_common @ self.eqn.G) * x_common, axis=-1)
            else:
                pi, hidden = self.policy_tf(training, t, x_hist, hidden)
                inst_r = tf.reduce_sum((x_common @ self.eqn.Q) * x_common, axis=-1) + tf.reduce_sum((pi @ self.eqn.R) * pi, axis=-1)
                reward += inst_r * self.eqn.dt

            if t < self.eqn.nt:
                dx = (x @ self.eqn.A1.transpose() + y @ self.eqn.A2.transpose() \
                  + zeta @ self.eqn.A3.transpose() + pi @ self.eqn.B.transpose())
                x = x + dx * self.eqn.dt + dw_sample[..., t] @ self.eqn.sigma.transpose()

        return reward

    def hidden_init_tf(self, x_init):
        raise NotImplementedError

    def hidden_init_np(self, x_init):
        raise NotImplementedError

    def policy_tf(self, training, t, x_hist, hidden=None):
        raise NotImplementedError

    def policy(self, t, x_hist, wgt_x_hist=None, hidden=None):
        raise NotImplementedError


class LQSharedFFModel(LQPolicyModel):
    def __init__(self, config, eqn):
        super(LQSharedFFModel, self).__init__(config, eqn)
        self.n_lag_state = config.net_config.n_lag_state
        if self.eqn.fixinit:
            self.pi_init = tf.Variable(
                np.random.uniform(
                    low=0,
                    high=0,
                    size=[1, self.eqn.dim_pi])
            )
        self.subnet = FeedForwardSubNet(config)

    def hidden_init_tf(self, x_init):
        return None

    def hidden_init_np(self, x_init):
        return None

    def policy_tf(self, training, t, x_hist, hidden=None):
        if t == 0 and self.eqn.fixinit:
            return self.pi_init, None
        else:
            state = x_hist[:, :, -(self.n_lag_state+1):]
            state = tf.reshape(state, [state.shape[0], -1])
            t = tf.broadcast_to(
                tf.cast(t*self.eqn.dt, dtype=self.net_config.dtype),
                shape=[state.shape[0], 1]
            )
            state = tf.concat([state, t], axis=-1)
            pi = self.subnet(state, training)
        return pi, None

    def policy(self, t, x_hist, wgt_x_hist=None, dw_inst=None, hidden=None):
        if t == 0 and self.eqn.fixinit:
            return self.pi_init.numpy(), None
        else:
            state = x_hist[:, :, -(self.n_lag_state+1):]
            state = tf.reshape(state, [state.shape[0], -1])
            t = tf.broadcast_to(
                tf.cast(t*self.eqn.dt, dtype=self.net_config.dtype),
                shape=[state.shape[0], 1]
            )
            state = tf.concat([state, t], axis=-1)
            pi = self.subnet(state, training=False)
        return pi, None


class LQLSTMModel(LQPolicyModel):
    def __init__(self, config, eqn):
        super(LQLSTMModel, self).__init__(config, eqn)
        if self.eqn.fixinit:
            self.h_init = tf.Variable(
                np.random.uniform(
                    low=0,
                    high=0,
                    size=[1, self.net_config.dim_h])
            )
            self.C_init = tf.Variable(
                np.random.uniform(
                    low=0,
                    high=0,
                    size=[1, self.net_config.dim_h])
            )
        self.lstm = LSTMCell(config)

    def hidden_init_tf(self, x_init):
        num_sample = tf.shape(x_init)[0]
        self.all_one_vec = tf.ones(shape=tf.stack([num_sample, 1]), dtype=self.net_config.dtype)
        if self.eqn.fixinit:
            h = tf.matmul(self.all_one_vec, self.h_init)
            C = tf.matmul(self.all_one_vec, self.C_init)
            hidden = (h, C)
        else:
            zeros = tf.zeros(
                shape=[num_sample, self.net_config.dim_h-x_init.shape[-2]],
                dtype=self.net_config.dtype
            )
            h = tf.concat([x_init[:, :, 0], zeros], axis=-1)
            C = tf.concat([x_init[:, :, 0], zeros], axis=-1)
            hidden = (h, C)
            for t in range(-self.eqn.n_lag, 0):
                _, hidden = self.lstm(
                    t*self.eqn.dt*self.all_one_vec,
                    x_init[:, :, t+self.eqn.n_lag+1], hidden
                )
        return hidden

    def hidden_init_np(self, x_init):
        num_sample = x_init.shape[0]
        if self.eqn.fixinit:
            return (
                np.broadcast_to(self.h_init.numpy(), [num_sample, self.net_config.dim_h]),
                np.broadcast_to(self.C_init.numpy(), [num_sample, self.net_config.dim_h])
            )
        else:
            zeros = tf.zeros(
                shape=[num_sample, self.net_config.dim_h-x_init.shape[-2]],
                dtype=self.net_config.dtype
            )
            h = tf.concat([x_init[:, :, 0], zeros], axis=-1)
            C = tf.concat([x_init[:, :, 0], zeros], axis=-1)
            hidden = (h, C)
            for t in range(-self.eqn.n_lag, 0):
                _, hidden = self.lstm(
                    t*self.eqn.dt*np.ones([num_sample, 1]),
                    x_init[:, :, t+self.eqn.n_lag+1], hidden
                )
            return (hidden[0].numpy(), hidden[1].numpy())

    def policy_tf(self, training, t, x_hist, hidden=None):
        return self.lstm(t*self.eqn.dt*self.all_one_vec, x_hist[..., -1], hidden)

    def policy(self, t, x_hist, wgt_x_hist=None, dw_inst=None, hidden=None):
        t = np.broadcast_to(t*self.eqn.dt, [x_hist.shape[0], 1])
        pi, hidden = self.lstm(t, x_hist[..., -1], hidden)
        return pi, hidden


class LQLSTMModel_W(LQLSTMModel):
    def __init__(self, config, eqn):
        super(LQLSTMModel_W, self).__init__(config, eqn)
        self.h_init = tf.Variable(
            np.random.uniform(
                low=0,
                high=0,
                size=[1, self.net_config.dim_h])
        )
        self.C_init = tf.Variable(
            np.random.uniform(
                low=0,
                high=0,
                size=[1, self.net_config.dim_h])
        )
        self.lstm = LSTMCell(config)

    def call(self, inputs, training):
        dw_sample, x_hist, wgt_x_hist = inputs
        x = x_hist[:, :, -1] # of shape (B, dx)
        hidden = self.hidden_init_tf(tf.shape(dw_sample)[0])

        reward = 0
        for t in range(self.eqn.nt+1):
            if t > 0:
                x_hist = tf.concat([x_hist[:, :, 1:], x[:, :, None]], axis=-1)
                wgt_x_hist = tf.concat([wgt_x_hist[:, :, 1:] * self.eqn.exp_array[-2], x[:, :, None]], axis=-1)
            zeta = x_hist[:, :, 0]
            y = (tf.reduce_sum(wgt_x_hist, axis=-1) - 0.5*(wgt_x_hist[..., 0] + wgt_x_hist[..., -1])) * self.eqn.dt
            x_common = x + self.eqn.exp_fac * y @ self.eqn.A3
            if t == self.eqn.nt:
                reward += tf.reduce_sum((x_common @ self.eqn.G) * x_common, axis=-1)
            else:
                if t == 0:
                    dw_inst = dw_sample[..., 0] * 0 + 0.1
                else:
                    dw_inst = dw_sample[..., t-1]
                pi, hidden = self.policy_tf(training, t, dw_inst, hidden)
                inst_r = tf.reduce_sum((x_common @ self.eqn.Q) * x_common, axis=-1) + tf.reduce_sum((pi @ self.eqn.R) * pi, axis=-1)
                reward += inst_r * self.eqn.dt

            if t < self.eqn.nt:
                dx = (x @ self.eqn.A1.transpose() + y @ self.eqn.A2.transpose() \
                  + zeta @ self.eqn.A3.transpose() + pi @ self.eqn.B.transpose())
                x = x + dx * self.eqn.dt + dw_sample[..., t] @ self.eqn.sigma.transpose()

        return reward

    def policy_tf(self, training, t, dw_inst, hidden=None):
        return self.lstm(t*self.eqn.dt*self.all_one_vec, dw_inst, hidden)

    def policy(self, t, x_hist, wgt_x_hist=None, dw_inst=None, hidden=None):
        t = np.broadcast_to(t*self.eqn.dt, [x_hist.shape[0], 1])
        pi, hidden = self.lstm(t, dw_inst, hidden)
        return pi, hidden


class CsmpPolicyModel(tf.keras.Model):
    def __init__(self, config, eqn):
        super(CsmpPolicyModel, self).__init__()
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.eqn = eqn
        # self.zero = tf.constant(0.0, dtype=self.net_config.dtype)

    def call(self, inputs, training):
        dw_sample, x_init, wgt_x_hist = inputs
        x = x_init[:, -1] # of shape (B,)
        hidden = self.hidden_init_tf(x_init)
        x_hist = x_init

        reward = 0
        for t in range(self.eqn.nt+1):
            if t > 0:
                x_hist = tf.concat([x_hist[:, 1:], x[:, None]], axis=-1)
                wgt_x_hist = tf.concat([wgt_x_hist[:, 1:] * self.eqn.exp_array[-2], x[:, None]], axis=-1)
            zeta = x_hist[:, 0]
            y = (tf.reduce_sum(wgt_x_hist, axis=-1) - 0.5*(wgt_x_hist[:, 0] + wgt_x_hist[:, -1])) * self.eqn.dt
            x_common = x + self.eqn.a * self.eqn.exp_fac * y
            if t == self.eqn.nt:
                reward += self.util_tf(x_common)*self.eqn.final_disc
                # penalty on x
                reward -= tf.nn.relu(-x)*self.net_config.util_penalty
            else:
                pi, hidden = self.policy_tf(training, t, x_hist, hidden)
                inst_r = self.util_tf(pi) * np.exp(-self.eqn.beta * t * self.eqn.dt)
                # penalty on x
                inst_r -= tf.nn.relu(-x)*self.net_config.util_penalty
                reward += inst_r * self.eqn.dt

            if t < self.eqn.nt:
                dx = self.eqn.drift_coeff*(self.eqn.drift_coeff+self.eqn.lambd) * y + self.eqn.mu * x_common \
                    + self.eqn.a * zeta - pi
                x = x + dx * self.eqn.dt + self.eqn.sigma * x_common * dw_sample[:, t]
        return -reward

    def util_tf(self, pi):
        return tf.pow(tf.nn.relu(pi)+1e-12, self.eqn_config.gamma)/self.eqn_config.gamma
        # return tf.where(
        #     tf.math.greater(pi, self.zero),
        #     pi**self.eqn_config.gamma/self.eqn_config.gamma,
        #     self.net_config.util_penalty*pi
        # )

    # def util_penalty_tf(self, pi):
    #     pi_pos = tf.nn.relu(pi)
    #     util = tf.pow(pi_pos+1e-12, self.eqn_config.gamma)/self.eqn_config.gamma
    #     util += (-tf.math.sign(pi)+1)/2 * self.net_config.util_penalty * pi
    #     return util

    def hidden_init_tf(self, x_init):
        raise NotImplementedError

    def hidden_init_np(self, x_init):
        raise NotImplementedError

    def policy_tf(self, training, t, x_hist, hidden=None):
        raise NotImplementedError

    def policy(self, t, x_hist, wgt_x_hist=None, hidden=None):
        raise NotImplementedError


class CsmpSharedFFModel(CsmpPolicyModel):
    def __init__(self, config, eqn):
        super(CsmpSharedFFModel, self).__init__(config, eqn)
        self.n_lag_state = config.net_config.n_lag_state
        if self.eqn.fixinit:
            self.pi_init = tf.Variable(
                np.random.uniform(
                    low=1.0,
                    high=1.0,
                    size=[1,])
            )
        self.subnet = FeedForwardSubNet(config)

    def hidden_init_tf(self, x_init):
        return None

    def hidden_init_np(self, x_init):
        return None

    def policy_tf(self, training, t, x_hist, hidden=None):
        if t == 0 and self.eqn.fixinit:
            return tf.nn.relu(self.pi_init), None
        else:
            state = x_hist[:, -(self.n_lag_state+1):]
            state = tf.reshape(state, [state.shape[0], -1])
            t = tf.broadcast_to(
                tf.cast(t*self.eqn.dt, dtype=self.net_config.dtype),
                shape=[state.shape[0], 1]
            )
            state = tf.concat([state, t], axis=-1)
            pi = tf.nn.relu(self.subnet(state, training)[:, 0])
        return pi, None

    def policy(self, t, x_hist, wgt_x_hist=None, dw_inst=None, hidden=None):
        if t == 0 and self.eqn.fixinit:
            return self.pi_init.numpy(), None
        else:
            state = x_hist[:, -(self.n_lag_state+1):]
            state = tf.reshape(state, [state.shape[0], -1])
            t = tf.broadcast_to(
                tf.cast(t*self.eqn.dt, dtype=self.net_config.dtype),
                shape=[state.shape[0], 1]
            )
            state = tf.concat([state, t], axis=-1)
            pi = tf.nn.relu(self.subnet(state, training=False)[:, 0])
        return pi, None


class CsmpLSTMModel(CsmpPolicyModel):
    def __init__(self, config, eqn):
        super(CsmpLSTMModel, self).__init__(config, eqn)
        if self.eqn.fixinit:
            self.h_init = tf.Variable(
                np.random.uniform(
                    low=0,
                    high=0,
                    size=[1, self.net_config.dim_h])
            )
            self.C_init = tf.Variable(
                np.random.uniform(
                    low=0,
                    high=0,
                    size=[1, self.net_config.dim_h])
            )
        self.lstm = LSTMCell(config)

    def hidden_init_tf(self, x_init):
        num_sample = tf.shape(x_init)[0]
        self.all_one_vec = tf.ones(shape=tf.stack([num_sample, 1]), dtype=self.net_config.dtype)
        if self.eqn.fixinit:
            h = tf.matmul(self.all_one_vec, self.h_init)
            C = tf.matmul(self.all_one_vec, self.C_init)
            hidden = (h, C)
        else:
            zeros = tf.zeros(
                shape=[num_sample, self.net_config.dim_h-1],
                dtype=self.net_config.dtype
            )
            h = tf.concat([x_init[:, 0:1], zeros], axis=-1)
            C = tf.concat([x_init[:, 0:1], zeros], axis=-1)
            hidden = (h, C)
            for t in range(-self.eqn.n_lag, 0):
                _, hidden = self.lstm(
                    t*self.eqn.dt*self.all_one_vec,
                    x_init[:, t+self.eqn.n_lag+1:t+self.eqn.n_lag+2], hidden
                )
        return hidden

    def hidden_init_np(self, x_init):
        num_sample = x_init.shape[0]
        if self.eqn.fixinit:
            return (
                np.broadcast_to(self.h_init.numpy(), [num_sample, self.net_config.dim_h]),
                np.broadcast_to(self.C_init.numpy(), [num_sample, self.net_config.dim_h])
            )
        else:
            zeros = tf.zeros(
                shape=[num_sample, self.net_config.dim_h-1],
                dtype=self.net_config.dtype
            )
            h = tf.concat([x_init[:, 0:1], zeros], axis=-1)
            C = tf.concat([x_init[:, 0:1], zeros], axis=-1)
            hidden = (h, C)
            for t in range(-self.eqn.n_lag, 0):
                _, hidden = self.lstm(
                    t*self.eqn.dt*np.ones([num_sample, 1]),
                    x_init[:, t+self.eqn.n_lag+1:t+self.eqn.n_lag+2], hidden
                )
            return (hidden[0].numpy(), hidden[1].numpy())

    def policy_tf(self, training, t, x_hist, hidden=None):
        pi, hidden = self.lstm(t*self.eqn.dt*self.all_one_vec, x_hist[:, -1:], hidden)
        pi = tf.nn.relu(pi)[:, 0]
        return pi, hidden

    def policy(self, t, x_hist, wgt_x_hist=None, dw_inst=None, hidden=None):
        t = np.broadcast_to(t*self.eqn.dt, [x_hist.shape[0], 1])
        pi, hidden = self.lstm(t, x_hist[:, -1:], hidden)
        pi = tf.nn.relu(pi)[:, 0]
        return pi, hidden


class CsmpLSTMModel_W(CsmpLSTMModel):
    def __init__(self, config, eqn):
        super(CsmpLSTMModel_W, self).__init__(config, eqn)
        self.h_init = tf.Variable(
            np.random.uniform(
                low=0,
                high=0,
                size=[1, self.net_config.dim_h])
        )
        self.C_init = tf.Variable(
            np.random.uniform(
                low=0,
                high=0,
                size=[1, self.net_config.dim_h])
        )
        self.lstm = LSTMCell(config)

    def call(self, inputs, training):
        dw_sample, x_hist, wgt_x_hist = inputs
        x = x_hist[:, -1] # of shape (B,)
        hidden = self.hidden_init_tf(tf.shape(dw_sample)[0])

        reward = 0
        for t in range(self.eqn.nt+1):
            if t > 0:
                x_hist = tf.concat([x_hist[:, 1:], x[:, None]], axis=-1)
                wgt_x_hist = tf.concat([wgt_x_hist[:, 1:] * self.eqn.exp_array[-2], x[:, None]], axis=-1)
            zeta = x_hist[:, 0]
            y = (tf.reduce_sum(wgt_x_hist, axis=-1) - 0.5*(wgt_x_hist[:, 0] + wgt_x_hist[:, -1])) * self.eqn.dt
            x_common = x + self.eqn.a * self.eqn.exp_fac * y
            if t == self.eqn.nt:
                reward += self.util_tf(x_common)*self.eqn.final_disc
                # penalty on x
                reward -= tf.nn.relu(-x)*self.net_config.util_penalty
            else:
                if t == 0:
                    dw_inst = dw_sample[..., 0:1] * 0 + 0.1
                else:
                    dw_inst = dw_sample[..., t-1:t]
                pi, hidden = self.policy_tf(training, t, dw_inst, hidden)
                inst_r = self.util_tf(pi) * np.exp(-self.eqn.beta * t * self.eqn.dt)
                # penalty on x
                inst_r -= tf.nn.relu(-x)*self.net_config.util_penalty
                reward += inst_r * self.eqn.dt

            if t < self.eqn.nt:
                dx = self.eqn.drift_coeff*(self.eqn.drift_coeff+self.eqn.lambd) * y + self.eqn.mu * x_common \
                    + self.eqn.a * zeta - pi
                x = x + dx * self.eqn.dt + self.eqn.sigma * x_common * dw_sample[:, t]
        return -reward

    def policy_tf(self, training, t, dw_inst, hidden=None):
        pi, hidden = self.lstm(t*self.eqn.dt*self.all_one_vec, dw_inst, hidden)
        pi = tf.nn.relu(pi)[:, 0]
        return pi, hidden

    def policy(self, t, x_hist, wgt_x_hist=None, dw_inst=None, hidden=None):
        t = np.broadcast_to(t*self.eqn.dt, [x_hist.shape[0], 1])
        pi, hidden = self.lstm(t, dw_inst, hidden)
        pi = tf.nn.relu(pi)[:, 0]
        return pi, hidden


class POlogPolicyModel(tf.keras.Model):
    def __init__(self, config, eqn):
        super(POlogPolicyModel, self).__init__()
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.eqn = eqn
        # self.zero = tf.constant(0.0, dtype=self.net_config.dtype)

    def call(self, inputs, training):
        dw_sample, x_init = inputs
        x = x_init[:, -1] # of shape (B,)
        hidden = self.hidden_init_tf(x_init)
        x_hist = x_init

        reward = 0
        y = tf.reduce_sum(self.eqn.exp_array * x_hist[:, 1:], axis=-1) * self.eqn.dt + \
            self.eqn.geometric_sum * x_hist[:, 0]
        for t in range(self.eqn.nt+1):
            if t > 0:
                x_hist = tf.concat([x_hist[:, 1:], x[:, None]], axis=-1)
            if t == self.eqn.nt:
                reward += self.util_tf(x + self.eqn.eta*y)/self.eqn.beta*self.eqn.final_disc
                # penalty on x
                reward -= tf.nn.relu(-x)*self.net_config.util_penalty
            else:
                pi, hidden = self.policy_tf(training, t, x_hist, hidden)
                inst_r = self.util_tf(pi[:, 0]*x) * np.exp(-self.eqn.beta * t * self.eqn.dt)
                # penalty on x
                inst_r -= tf.nn.relu(-x)*self.net_config.util_penalty
                reward += inst_r * self.eqn.dt

            if t < self.eqn.nt:
                dx = ((self.eqn.mu1 - self.eqn.r)*pi[:, 1] - pi[:, 0] + self.eqn.r) * x + self.eqn.mu2 * y
                x = x + dx * self.eqn.dt + self.eqn.sigma * x * pi[:, 1] * dw_sample[:, t]
                y = y * np.exp(-self.eqn.dt * self.eqn.lambd) + x * self.eqn.dt
        return -reward

    def util_tf(self, pi):
        return tf.math.log(tf.nn.relu(pi)+self.eqn.logeps)

    def final_pi_tf(self, pi):
        # return tf.concat([tf.nn.relu(pi[:, 0:1]), pi[:, 1:2]], axis=-1)
        return tf.nn.sigmoid(pi) * 2

    def hidden_init_tf(self, x_init):
        raise NotImplementedError

    def hidden_init_np(self, x_init):
        raise NotImplementedError

    def policy_tf(self, training, t, x_hist, hidden=None):
        raise NotImplementedError

    def policy(self, t, x_hist, wgt_x_hist=None, hidden=None):
        raise NotImplementedError


class POlogSharedFFModel(POlogPolicyModel):
    def __init__(self, config, eqn):
        super(POlogSharedFFModel, self).__init__(config, eqn)
        self.n_lag_state = config.net_config.n_lag_state
        if self.eqn.fixinit:
            self.pi_init = tf.Variable(
                np.random.uniform(
                    low=0.3,
                    high=0.3,
                    size=[1, self.eqn.dim_pi])
            )
        self.subnet = FeedForwardSubNet(config)

    def hidden_init_tf(self, x_init):
        return None

    def hidden_init_np(self, x_init):
        return None

    def policy_tf(self, training, t, x_hist, hidden=None):
        if t == 0 and self.eqn.fixinit:
            return self.pi_init, None
        else:
            state = x_hist[:, -(self.n_lag_state+1):]
            state = tf.reshape(state, [state.shape[0], -1])
            t = tf.broadcast_to(
                tf.cast(t*self.eqn.dt, dtype=self.net_config.dtype),
                shape=[state.shape[0], 1]
            )
            state = tf.concat([state, t], axis=-1)
            pi = self.subnet(state, training)
            pi = self.final_pi_tf(pi)
        return pi, None

    def policy(self, t, x_hist, wgt_x_hist=None, dw_inst=None, hidden=None):
        if t == 0 and self.eqn.fixinit:
            return self.pi_init.numpy(), None
        else:
            state = x_hist[:, -(self.n_lag_state+1):]
            state = tf.reshape(state, [state.shape[0], -1])
            t = tf.broadcast_to(
                tf.cast(t*self.eqn.dt, dtype=self.net_config.dtype),
                shape=[state.shape[0], 1]
            )
            state = tf.concat([state, t], axis=-1)
            pi = self.subnet(state, training=False)
            pi = self.final_pi_tf(pi)
        return pi.numpy(), None


class POlogLSTMModel(POlogPolicyModel):
    def __init__(self, config, eqn):
        super(POlogLSTMModel, self).__init__(config, eqn)
        if self.eqn.fixinit:
            self.h_init = tf.Variable(
                np.random.uniform(
                    low=0,
                    high=0,
                    size=[1, self.net_config.dim_h])
            )
            self.C_init = tf.Variable(
                np.random.uniform(
                    low=0,
                    high=0,
                    size=[1, self.net_config.dim_h])
            )
        self.lstm = LSTMCell(config)

    def hidden_init_tf(self, x_init):
        num_sample = tf.shape(x_init)[0]
        self.all_one_vec = tf.ones(shape=tf.stack([num_sample, 1]), dtype=self.net_config.dtype)
        if self.eqn.fixinit:
            h = tf.matmul(self.all_one_vec, self.h_init)
            C = tf.matmul(self.all_one_vec, self.C_init)
            hidden = (h, C)
        else:
            zeros = tf.zeros(
                shape=[num_sample, self.net_config.dim_h-1],
                dtype=self.net_config.dtype
            )
            h = tf.concat([x_init[:, 0:1], zeros], axis=-1)
            C = tf.concat([x_init[:, 0:1], zeros], axis=-1)
            hidden = (h, C)
            for t in range(-self.eqn.n_lag, 0):
                _, hidden = self.lstm(
                    t*self.eqn.dt*self.all_one_vec,
                    x_init[:, t+self.eqn.n_lag+1:t+self.eqn.n_lag+2], hidden
                )
        return hidden

    def hidden_init_np(self, x_init):
        num_sample = x_init.shape[0]
        if self.eqn.fixinit:
            return (
                np.broadcast_to(self.h_init.numpy(), [num_sample, self.net_config.dim_h]),
                np.broadcast_to(self.C_init.numpy(), [num_sample, self.net_config.dim_h])
            )
        else:
            zeros = tf.zeros(
                shape=[num_sample, self.net_config.dim_h-1],
                dtype=self.net_config.dtype
            )
            h = tf.concat([x_init[:, 0:1], zeros], axis=-1)
            C = tf.concat([x_init[:, 0:1], zeros], axis=-1)
            hidden = (h, C)
            for t in range(-self.eqn.n_lag, 0):
                _, hidden = self.lstm(
                    t*self.eqn.dt*np.ones([num_sample, 1]),
                    x_init[:, t+self.eqn.n_lag+1:t+self.eqn.n_lag+2], hidden
                )
            return (hidden[0].numpy(), hidden[1].numpy())

    def policy_tf(self, training, t, x_hist, hidden=None):
        pi, hidden = self.lstm(t*self.eqn.dt*self.all_one_vec, x_hist[:, -1:], hidden)
        pi = self.final_pi_tf(pi)
        return pi, hidden

    def policy(self, t, x_hist, wgt_x_hist=None, dw_inst=None, hidden=None):
        t = np.broadcast_to(t*self.eqn.dt, [x_hist.shape[0], 1])
        pi, hidden = self.lstm(t, x_hist[:, -1:], hidden)
        pi = self.final_pi_tf(pi)
        return pi.numpy(), hidden


class FeedForwardBNSubNet(tf.keras.Model):
    def __init__(self, config):
        super(FeedForwardSubNet, self).__init__()
        dim = config.eqn_config.dim_pi
        num_hiddens = config.net_config.num_hiddens
        self.bn_layers = [
            tf.keras.layers.BatchNormalization(
                momentum=0.99,
                epsilon=1e-6,
                beta_initializer=tf.random_normal_initializer(0.0, stddev=0.1),
                gamma_initializer=tf.random_uniform_initializer(0.1, 0.5)
            )
            for _ in range(len(num_hiddens) + 2)]
        self.dense_layers = [tf.keras.layers.Dense(num_hiddens[i],
                                                   use_bias=False,
                                                   activation=None)
                             for i in range(len(num_hiddens))]
        # final output should be gradient of size dim
        self.dense_layers.append(tf.keras.layers.Dense(dim, activation=None))

    def call(self, x, training):
        """structure: bn -> (dense -> bn -> relu) * len(num_hiddens) -> dense -> bn"""
        x = self.bn_layers[0](x, training)
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            x = self.bn_layers[i+1](x, training)
            x = tf.nn.relu(x)
        x = self.dense_layers[-1](x)
        x = self.bn_layers[-1](x, training)
        return x


class FeedForwardSubNet(tf.keras.Model):
    def __init__(self, config):
        super(FeedForwardSubNet, self).__init__()
        dim = config.eqn_config.dim_pi
        num_hiddens = config.net_config.num_hiddens
        self.dense_layers = [tf.keras.layers.Dense(num_hiddens[i],
                                                   use_bias=True,
                                                   activation="relu")
                             for i in range(len(num_hiddens))]
        # final output should be gradient of size dim
        self.dense_layers.append(tf.keras.layers.Dense(dim, activation=None))

    def call(self, x, training):
        """structure: bn -> (dense -> bn -> relu) * len(num_hiddens) -> dense -> bn"""
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
        x = self.dense_layers[-1](x)
        return x


class LSTMCell(tf.keras.Model):
    def __init__(self, config):
        super(LSTMCell, self).__init__()
        dim_pi = config.eqn_config.dim_pi
        dim_h = config.net_config.dim_h
        self.f_layer = tf.keras.layers.Dense(dim_h, activation='sigmoid')
        self.i_layer = tf.keras.layers.Dense(dim_h, activation='sigmoid')
        self.o_layer = tf.keras.layers.Dense(dim_h, activation='sigmoid')
        self.C_layer = tf.keras.layers.Dense(dim_h, activation='tanh')
        self.pi_layer = tf.keras.layers.Dense(dim_pi, activation=None)

    def call(self, t, x, hidden_prev):
        h_prev, C_prev = hidden_prev
        z = tf.concat([x, h_prev, t], axis=1)
        f = self.f_layer(z)
        i = self.i_layer(z)
        o = self.o_layer(z)
        C_bar = self.C_layer(z)
        C = C_prev * f + C_bar * i
        h = o * tf.nn.tanh(C)
        pi = self.pi_layer(h)
        return pi, (h, C)
