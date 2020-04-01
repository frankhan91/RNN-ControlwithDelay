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
                if self.net_config.verbose:
                    logging.info("step: %5u,    reward: %.4e,   elapsed time: %3u" % (
                        step, reward, elapsed_time))
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
        dw_sample, x_hist, wgt_x_hist = inputs
        x = x_hist[:, :, -1] # of shape (B, dx)
        hidden = self.hidden_init_tf(tf.shape(dw_sample)[0])

        for t in range(self.eqn.nt+1):
            if t > 0:
                x_hist = tf.concat([x_hist[:, :, 1:], x[:, :, None]], axis=-1)
                wgt_x_hist = tf.concat([wgt_x_hist[:, :, 1:] * self.eqn.exp_array[-2], x[:, :, None]], axis=-1)
            zeta = x_hist[:, :, 0]
            y = (tf.reduce_sum(wgt_x_hist, axis=-1) - 0.5*(wgt_x_hist[..., 0] + wgt_x_hist[..., -1])) * self.eqn.dt
            x_common = x + self.eqn.exp_fac * y @ self.eqn.A3
            pi, hidden = self.policy_tf(training, t, x_hist, hidden)
            inst_r = tf.reduce_sum((x_common @ self.eqn.Q) * x_common, axis=-1) + tf.reduce_sum((pi @ self.eqn.R) * pi, axis=-1)
            if t == 0:
                reward = inst_r * self.eqn.dt / 2
            elif t == self.eqn.nt:
                reward += inst_r * self.eqn.dt / 2
                reward += tf.reduce_sum((x_common @ self.eqn.G) * x_common, axis=-1)
            else:
                reward += inst_r * self.eqn.dt

            if t < self.eqn.nt:
                dx = (x @ self.eqn.A1.transpose() + y @ self.eqn.A2.transpose() \
                  + zeta @ self.eqn.A3.transpose() + pi @ self.eqn.B.transpose())
                x = x + dx * self.eqn.dt + dw_sample[..., t] @ self.eqn.sigma.transpose()

        return reward

    def hidden_init_tf(self, num_sample):
        raise NotImplementedError

    def hidden_init(self, num_sample):
        raise NotImplementedError

    def policy_tf(self, training, t, x_hist, hidden=None):
        raise NotImplementedError

    def policy(self, t, x_hist, wgt_x_hist=None, hidden=None):
        raise NotImplementedError


class LQNonsharedFFModel(LQPolicyModel):
    def __init__(self, config, eqn):
        super(LQNonsharedFFModel, self).__init__(config, eqn)
        self.n_lag_state = config.net_config.n_lag_state
        self.pi_init = tf.Variable(
            np.random.uniform(
                low=0,
                high=0,
                size=[1, self.eqn.dim_pi])
        )
        self.subnet = [FeedForwardSubNet(config) for _ in range(self.eqn.nt)]

    def hidden_init_tf(self, num_sample):
        return None

    def hidden_init(self, num_sample):
        return None

    def policy_tf(self, training, t, x_hist, hidden=None):
        if t == 0:
            return self.pi_init, None
        else:
            state = x_hist[:, :, -(self.n_lag_state+1):]
            state = tf.reshape(state, [state.shape[0], -1])
            pi = self.subnet[t-1](state, training)
        return pi, None

    def policy(self, t, x_hist, wgt_x_hist=None, hidden=None):
        if t == 0:
            return self.pi_init.numpy(), None
        else:
            state = x_hist[:, :, -(self.n_lag_state+1):]
            state = tf.reshape(state, [state.shape[0], -1])
            pi = self.subnet[t-1](state, training=False)
        return pi, None


class LQLSTMModel(LQPolicyModel):
    def __init__(self, config, eqn):
        super(LQLSTMModel, self).__init__(config, eqn)
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

    def hidden_init_tf(self, num_sample):
        self.all_one_vec = tf.ones(shape=tf.stack([num_sample, 1]), dtype=self.net_config.dtype)
        h = tf.matmul(self.all_one_vec, self.h_init)
        C = tf.matmul(self.all_one_vec, self.C_init)
        return (h, C)

    def hidden_init(self, num_sample):
        return (
            np.broadcast_to(self.h_init.numpy(), [num_sample, self.net_config.dim_h]),
            np.broadcast_to(self.C_init.numpy(), [num_sample, self.net_config.dim_h])
        )

    def policy_tf(self, training, t, x_hist, hidden=None):
        return self.lstm(t*self.eqn.dt*self.all_one_vec, x_hist[..., -1], hidden)

    def policy(self, t, x_hist, wgt_x_hist=None, hidden=None):
        t = np.broadcast_to(t*self.eqn.dt, [x_hist.shape[0], 1])
        pi, hidden = self.lstm(t, x_hist[..., -1], hidden)
        return pi, hidden


class FeedForwardSubNet(tf.keras.Model):
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
