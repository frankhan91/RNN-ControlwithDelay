import json
import munch
import os
import logging

from absl import app
from absl import flags
from absl import logging as absl_logging
import numpy as np
import tensorflow as tf

import solver
import equation


flags.DEFINE_string('config_path', 'configs/lq.json',
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
    x_sample, reward, value = eqn.simulate_true(4096)
    logging.info('Value in continuous time: %.4e, reward with analytic policy: %.4e' % (value[-1], np.mean(reward)))

    sol = solver.LQSolver(config, eqn)
    sol.train()

if __name__ == '__main__':
    app.run(main)
