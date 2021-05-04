import time
import torch
import math
from gym_optimal_intrusion_response.dao.experiment.experiment_result import ExperimentResult
from gym_optimal_intrusion_response.agents.train_agent import TrainAgent
from gym_optimal_intrusion_response.agents.config.agent_config import AgentConfig
from gym_optimal_intrusion_response.agents.policy_gradient.ppo_baseline.impl.ppo.ppo import PPO
from gym_optimal_intrusion_response.dao.agent.train_mode import TrainMode


class PPOBaselineAgent(TrainAgent):

    def __init__(self, env, attacker_agent_config: AgentConfig,
                 defender_agent_config: AgentConfig,
                 train_mode: TrainMode = TrainMode.TRAIN_ATTACKER):
        """
        Initialize environment and hyperparameters

        :param attacker_agent_config: the configuration
        """
        super(PPOBaselineAgent, self).__init__(env, attacker_agent_config,
                                               defender_agent_config,
                                               train_mode)

    def train(self) -> ExperimentResult:
        """
        Starts the training loop and returns the result when complete

        :return: the training result
        """

        # Setup Attacker
        if self.attacker_config is not None:
            # Custom MLP policy for attacker
            attacker_net_arch = []
            attacker_pi_arch = []
            attacker_vf_arch = []
            for l in range(self.attacker_config.shared_layers):
                attacker_net_arch.append(self.attacker_config.shared_hidden_dim)
            for l in range(self.attacker_config.pi_hidden_layers):
                attacker_pi_arch.append(self.attacker_config.pi_hidden_dim)
            for l in range(self.attacker_config.vf_hidden_layers):
                attacker_vf_arch.append(self.attacker_config.vf_hidden_dim)

            net_dict_attacker = {"pi": attacker_pi_arch, "vf": attacker_vf_arch}
            attacker_net_arch.append(net_dict_attacker)

            policy_kwargs_attacker = dict(activation_fn=self.get_hidden_activation_attacker(), net_arch=attacker_net_arch)
            device_attacker = "cpu" if not self.attacker_config.gpu else "cuda:" + str(self.attacker_config.gpu_id)
            policy_attacker = "MlpPolicy"

            if self.attacker_config.lr_progress_decay:
                temp = self.attacker_config.alpha
                lr_decay_func = lambda x: temp * math.pow(x, self.attacker_config.lr_progress_power_decay)
                self.attacker_config.alpha = lr_decay_func

        # Setup Defender
        if self.defender_config is not None:
            # Custom MLP policy for attacker
            defender_net_arch = []
            defender_pi_arch = []
            defender_vf_arch = []
            for l in range(self.defender_config.shared_layers):
                defender_net_arch.append(self.defender_config.shared_hidden_dim)
            for l in range(self.defender_config.pi_hidden_layers):
                defender_pi_arch.append(self.defender_config.pi_hidden_dim)
            for l in range(self.defender_config.vf_hidden_layers):
                defender_vf_arch.append(self.defender_config.vf_hidden_dim)

            net_dict_defender = {"pi": defender_pi_arch, "vf": defender_vf_arch}
            defender_net_arch.append(net_dict_defender)

            policy_kwargs_defender = dict(activation_fn=self.get_hidden_activation_defender(), net_arch=defender_net_arch)
            device_defender = "cpu" if not self.defender_config.gpu else "cuda:" + str(self.defender_config.gpu_id)
            policy_defender = "MlpPolicy"

        # Create model
        model = PPO(policy_attacker, policy_defender,
                    self.env,
                    batch_size=self.attacker_config.mini_batch_size,
                    attacker_learning_rate=self.attacker_config.alpha,
                    defender_learning_rate=self.defender_config.alpha,
                    n_steps=self.attacker_config.batch_size,
                    n_epochs=self.attacker_config.optimization_iterations,
                    attacker_gamma=self.attacker_config.gamma,
                    defender_gamma=self.defender_config.gamma,
                    attacker_gae_lambda=self.attacker_config.gae_lambda,
                    defender_gae_lambda=self.defender_config.gae_lambda,
                    attacker_clip_range=self.attacker_config.eps_clip,
                    defender_clip_range=self.defender_config.eps_clip,
                    max_grad_norm=self.attacker_config.max_gradient_norm,
                    seed=self.attacker_config.random_seed,
                    attacker_policy_kwargs=policy_kwargs_attacker,
                    defender_policy_kwargs=policy_kwargs_defender,
                    device=device_attacker,
                    attacker_agent_config=self.attacker_config,
                    defender_agent_config=self.defender_config,
                    attacker_vf_coef=self.attacker_config.vf_coef,
                    defender_vf_coef=self.defender_config.vf_coef,
                    attacker_ent_coef=self.attacker_config.ent_coef,
                    defender_ent_coef=self.defender_config.ent_coef,
                    train_mode = self.train_mode
                    )

        if self.attacker_config.load_path is not None:
            PPO.load(self.attacker_config.load_path, policy_attacker, agent_config=self.attacker_config)

        elif self.defender_config.load_path is not None:
            PPO.load(self.defender_config.load_path, policy_defender, agent_config=self.defender_config)

        # Eval config
        if self.train_mode == TrainMode.TRAIN_ATTACKER or self.train_mode == TrainMode.SELF_PLAY:
            total_timesteps = self.attacker_config.num_episodes
            train_log_frequency = self.attacker_config.train_log_frequency
            eval_frequency = self.attacker_config.eval_frequency
            eval_episodes = self.attacker_config.eval_episodes
            save_dir = self.attacker_config.save_dir
        else:
            total_timesteps = self.defender_config.num_episodes
            train_log_frequency = self.defender_config.train_log_frequency
            eval_frequency = self.defender_config.eval_frequency
            eval_episodes = self.defender_config.eval_episodes
            save_dir = self.defender_config.save_dir

        model.learn(total_timesteps=total_timesteps,
                    log_interval=train_log_frequency,
                    eval_freq=eval_frequency,
                    n_eval_episodes=eval_episodes)

        if self.attacker_config is not None:
            self.attacker_config.logger.info("Training Complete")
        if self.defender_config is not None:
            self.defender_config.logger.info("Training Complete")

        # Save networks
        try:
            model.save_model()
        except Exception as e:
            print("There was en error saving the model:{}".format(str(e)))

        # Save other game data
        if save_dir is not None:
            time_str = str(time.time())
            model.train_result.to_csv(save_dir + "/" + time_str + "_train_results_checkpoint.csv")
            model.eval_result.to_csv(save_dir + "/" + time_str + "_eval_results_checkpoint.csv")

        self.train_result = model.train_result
        self.eval_result = model.eval_result
        return model.train_result

    def get_hidden_activation_attacker(self):
        """
        Interprets the hidden activation

        :return: the hidden activation function
        """
        return torch.nn.Tanh
        if self.attacker_config.hidden_activation == "ReLU":
            return torch.nn.ReLU
        elif self.attacker_config.hidden_activation == "LeakyReLU":
            return torch.nn.LeakyReLU
        elif self.attacker_config.hidden_activation == "LogSigmoid":
            return torch.nn.LogSigmoid
        elif self.attacker_config.hidden_activation == "PReLU":
            return torch.nn.PReLU
        elif self.attacker_config.hidden_activation == "Sigmoid":
            return torch.nn.Sigmoid
        elif self.attacker_config.hidden_activation == "Softplus":
            return torch.nn.Softplus
        elif self.attacker_config.hidden_activation == "Tanh":
            return torch.nn.Tanh
        else:
            raise ValueError("Activation type: {} not recognized".format(self.attacker_config.hidden_activation))

    def get_hidden_activation_defender(self):
        """
        Interprets the hidden activation

        :return: the hidden activation function
        """
        return torch.nn.Tanh
        if self.defender_config.hidden_activation == "ReLU":
            return torch.nn.ReLU
        elif self.defender_config.hidden_activation == "LeakyReLU":
            return torch.nn.LeakyReLU
        elif self.defender_config.hidden_activation == "LogSigmoid":
            return torch.nn.LogSigmoid
        elif self.defender_config.hidden_activation == "PReLU":
            return torch.nn.PReLU
        elif self.defender_config.hidden_activation == "Sigmoid":
            return torch.nn.Sigmoid
        elif self.defender_config.hidden_activation == "Softplus":
            return torch.nn.Softplus
        elif self.defender_config.hidden_activation == "Tanh":
            return torch.nn.Tanh
        else:
            raise ValueError("Activation type: {} not recognized".format(self.defender_config.hidden_activation))


    def get_action(self, s, eval=False, attacker=True) -> int:
        raise NotImplemented("not implemented")

    def eval(self, log=True) -> ExperimentResult:
        raise NotImplemented("not implemented")