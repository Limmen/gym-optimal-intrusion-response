# Optimal Intrusion Response

An OpenAI Gym interface to a MDP/Markov Game model for optimal intrusion response of a realistic infrastructure simulated using system traces.

<p align="center">
    <a href="https://img.shields.io/badge/license-CC%20BY--SA%204.0-green">
        <img src="https://img.shields.io/badge/license-CC%20BY--SA%204.0-green" /></a>
    <a href="https://img.shields.io/badge/version-1.0.0-blue">
        <img src="https://img.shields.io/badge/version-1.0.0-blue" /></a>
</p>

## Included Environments

- `optimal-intrusion-response-v1`
- `optimal-intrusion-response-v2`
- `optimal-intrusion-response-v3`

## Requirements
- Python 3.5+
- OpenAI Gym
- NumPy
- jsonpickle (for configuration files)
- torch (for baseline algorithms)


## Installation

```bash
# install from pip
pip install gym-optimal-intrusion-response==1.0.0
# local install from source
$ pip install -e gym-optimal-intrusion-response
# force upgrade deps
$ pip install -e gym-optimal-intrusion-response --upgrade

# git clone and install from source
git clone https://github.com/Limmen/gym-optimal-intrusion-response
cd gym-optimal-intrusion-response
pip3 install -e .
```

## Usage
The environment can be accessed like any other OpenAI environment with `gym.make`.
Once the environment has been created, the API functions
`step()`, `reset()`, `render()`, and `close()` can be used to train any RL algorithm of
your preference.
```python
import gym
from gym_idsgame.envs import IdsGameEnv
env_name = "optimal-intrusion-response-v1"
env = gym.make(env_name)
```

## Infrastructure

<p align="center">
<img src="docs/env.png" width="600">
</p>

## Traces

Alert/login traces from the emulated infrastructure are available in ([./traces](./traces)).

## Publications

- [CNSM21](http://dl.ifip.org/db/conf/cnsm/cnsm2021/1570732932.pdf)
Preprint is available ([here](https://arxiv.org/abs/2106.07160))

``` bash
@INPROCEEDINGS{hammar_stadler_cnsm_21,
AUTHOR="Kim Hammar and Rolf Stadler",
TITLE="Learning Intrusion Prevention Policies through Optimal Stopping",
BOOKTITLE="International Conference on Network and Service Management (CNSM 2021)",
ADDRESS="Izmir, Turkey",
DAYS=1,
YEAR=2021,
note={\url{http://dl.ifip.org/db/conf/cnsm/cnsm2021/1570732932.pdf}},
KEYWORDS="Network Security, automation, optimal stopping, reinforcement learning, Markov Decision Processes",
ABSTRACT="We study automated intrusion prevention using reinforcement learning. In a novel approach, we formulate the problem of intrusion prevention as an optimal stopping problem. This formulation allows us insight into the structure of the optimal policies, which turn out to be threshold based. Since the computation of the optimal defender policy using dynamic programming is not feasible for practical cases, we approximate the optimal policy through reinforcement learning in a simulation environment. To define the dynamics of the simulation, we emulate the target infrastructure and collect measurements. Our evaluations show that the learned policies are close to optimal and that they indeed can be expressed using thresholds."
}

```

- [CNSM20](https://ieeexplore.ieee.org/document/9269092)
```
@INPROCEEDINGS{Hamm2011:Finding,
AUTHOR="Kim Hammar and Rolf Stadler",
TITLE="Finding Effective Security Strategies through Reinforcement Learning and
{Self-Play}",
BOOKTITLE="International Conference on Network and Service Management (CNSM 2020)
(CNSM 2020)",
ADDRESS="Izmir, Turkey",
DAYS=1,
MONTH=nov,
YEAR=2020,
KEYWORDS="Network Security; Reinforcement Learning; Markov Security Games",
ABSTRACT="We present a method to automatically find security strategies for the use
case of intrusion prevention. Following this method, we model the
interaction between an attacker and a defender as a Markov game and let
attack and defense strategies evolve through reinforcement learning and
self-play without human intervention. Using a simple infrastructure
configuration, we demonstrate that effective security strategies can emerge
from self-play. This shows that self-play, which has been applied in other
domains with great success, can be effective in the context of network
security. Inspection of the converged policies show that the emerged
policies reflect common-sense knowledge and are similar to strategies of
humans. Moreover, we address known challenges of reinforcement learning in
this domain and present an approach that uses function approximation, an
opponent pool, and an autoregressive policy representation. Through
evaluations we show that our method is superior to two baseline methods but
that policy convergence in self-play remains a challenge."
}
```

## See also

- [gym-idsgame](https://github.com/Limmen/gym-idsgame)

## Author & Maintainer

Kim Hammar <kimham@kth.se>

## Copyright and license

[LICENSE](LICENSE.md)

Creative Commons

(C) 2021, Kim Hammar
