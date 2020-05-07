import json
import munch
import os
import logging

from absl import app
from absl import flags
from absl import logging as absl_logging
import numpy as np
import tensorflow as tf

import equation
from solver import Solver


flags.DEFINE_string('config_path', 'configs/csmp_shff.json',
                    """The path to load json file.""")
flags.DEFINE_string('exp_name', 'test',
                    """The name of numerical experiments, prefix for logging""")
FLAGS = flags.FLAGS
FLAGS.log_dir = './logs'


def main(argv):
    del argv
    absl_logging.get_absl_handler().setFormatter(logging.Formatter('%(levelname)-6s %(message)s'))
    absl_logging.set_verbosity('info')

    with open(FLAGS.config_path) as json_data_file:
        config = json.load(json_data_file)
    config = munch.munchify(config)
    eqn = getattr(equation, config.eqn_config.eqn_name)(config.eqn_config)
    tf.keras.backend.set_floatx(config.net_config.dtype)

    logging.info('Begin to solve %s ' % config.eqn_config.eqn_name)
    x_sample, _, reward = eqn.simulate(4096, eqn.true_policy)
    logging.info('Value in continuous time: %.4e, reward with analytic policy: %.4e' % (eqn.value, np.mean(reward)))

    sol = Solver(config, eqn)
    sol.train()

    n_save = 100
    x_sample, pi_sample, reward = eqn.simulate(n_save, eqn.true_policy, fixseed=True)
    xhat_sample, pihat_sample, reward_hat = eqn.simulate(
        n_save, sol.model.policy,
        fixseed=True, hidden_init=sol.model.hidden_init(n_save)
    )

    print(reward.mean(), reward_hat.mean())
    print(
        np.sqrt(np.mean((x_sample-xhat_sample)**2)/np.mean(x_sample**2)),
        np.sqrt(np.mean((pi_sample-pihat_sample)**2)/np.mean(pi_sample**2))
    )
    print(reward[:10])
    print(reward_hat[:10])

    prob, nn = ((FLAGS.config_path.split('.')[0]).split('/')[-1]).split('_')
    lag = int(config.eqn_config.delta/config.eqn_config.T*10)
    np.savez(
        file=os.path.join('../data/{}'.format(prob), '{}_lag{}_test.npz'.format(nn, lag)),
        x_sample=x_sample, pi_sample=pi_sample, reward=reward,
        xhat_sample=xhat_sample, pihat_sample=pihat_sample, reward_hat=reward_hat
    )


if __name__ == '__main__':
    app.run(main)
