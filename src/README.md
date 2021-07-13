# [Recurrent neural networks for stochastic control problems with delay](https://arxiv.org/abs/2101.01385) in TensorFlow (2.0)


## Training

```
python main.py --config_path=configs/polog_lstm.json
```

Command-line flags:

* `config_path`: Config path corresponding to the control problem to solve. 
There are three control problems implemented so far. See [Problems and Configs](#problems-and-configs) section below.
* `exp_name`: Name of numerical experiment, prefix of logging and output.


## Problems and Configs

`equation.py` and `configs` now support the following three problems, corresponding to examples in Section 4.1, 4.2, and 4.3 of ref [1]:

* `LQ`: Linear-quadratic problem with delay (3-dimensional or 10-dimensional state variable). 
* `Csmp`:  Optimal consumption in a delayed financial market.
* `POlog`: Portfolio optimization with complete memory and log utility.

Suffix `_lstm` means using long short-term memory (LSTM) networks and `_shff` means using feedforward networks with shared parameters.


## Dependencies

* [TensorFlow >=2.0](https://www.tensorflow.org/)
* [Python package munch](https://github.com/Infinidat/munch)


## Reference
[1] Han, J. and Hu, R. Recurrent neural networks for stochastic control problems with delay, (2021),  [[arXiv]](https://arxiv.org/abs/2101.01385)
