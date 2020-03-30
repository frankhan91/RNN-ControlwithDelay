import logging
import time
import numpy as np
import tensorflow as tf


class LQSolver(object):
    """The fully connected neural network model."""
    def __init__(self, config, eqn):
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.eqn = eqn

        self.model = NonsharedModel(config, eqn)
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            self.net_config.lr_boundaries, self.net_config.lr_values)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-8)

    def train(self):
        start_time = time.time()
        training_history = []
        valid_data = self.eqn.sample(self.net_config.valid_size, fixseed=True)
        _, reward, _ = self.eqn.simulate_true(self.net_config.valid_size, fixseed=True)
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


class NonsharedModel(tf.keras.Model):
    def __init__(self, config, eqn):
        super(NonsharedModel, self).__init__()
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.eqn = eqn
        self.n_lag_state = config.net_config.n_lag_state
        self.pi_init = tf.Variable(
            np.random.uniform(
                low=0,
                high=0,
                size=[1, self.eqn.dim_pi])
        )
        self.subnet = [FeedForwardSubNet(config) for _ in range(self.eqn.nt)]

    def call(self, inputs, training):
        dw_sample, x_hist, wgt_x_hist = inputs
        x = x_hist[:, :, -1] # of shape (B, dx)

        for t in range(self.eqn.nt+1):
            if t > 0:
                x_hist = tf.concat([x_hist[:, :, 1:], x[:, :, None]], axis=-1)
                wgt_x_hist = tf.concat([wgt_x_hist[:, :, 1:] * self.eqn.exp_array[-2], x[:, :, None]], axis=-1)
            zeta = x_hist[:, :, 0]
            y = (tf.reduce_sum(wgt_x_hist, axis=-1) - 0.5*(wgt_x_hist[..., 0] + wgt_x_hist[..., -1])) * self.eqn.dt
            x_common = x + self.eqn.exp_fac * y @ self.eqn.A3
            if t == 0:
                pi = self.pi_init
            else:
                state = x_hist[:, :, -(self.n_lag_state+1):]
                state = tf.reshape(state, [state.shape[0], -1])
                pi = self.subnet[t-1](state, training)
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

    def policy(self, t, x_hist, wgt_x_hist=None):
        if t == 0:
            return self.pi_init.numpy()
        else:
            state = x_hist[:, :, -(self.n_lag_state+1):]
            state = tf.reshape(state, [state.shape[0], -1])
            pi = self.subnet[t-1](state, training=False)
        return pi


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