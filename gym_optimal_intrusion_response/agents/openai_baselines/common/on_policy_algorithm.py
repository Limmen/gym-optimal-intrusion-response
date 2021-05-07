from typing import Any, Dict, List, Optional, Tuple, Type, Union
import gym
import numpy as np
import torch as th
import time
from gym_optimal_intrusion_response.agents.openai_baselines.common.base_class import BaseAlgorithm
from gym_optimal_intrusion_response.agents.openai_baselines.common.buffers import RolloutBuffer
from gym_optimal_intrusion_response.agents.openai_baselines.common.policies import ActorCriticPolicy
from gym_optimal_intrusion_response.agents.openai_baselines.common.type_aliases import GymEnv, Schedule
from gym_optimal_intrusion_response.agents.openai_baselines.common.vec_env import VecEnv
from gym_optimal_intrusion_response.dao.agent.train_agent_log_dto import TrainAgentLogDTO
from gym_optimal_intrusion_response.dao.agent.rollout_data_dto import RolloutDataDTO
from gym_optimal_intrusion_response.dao.agent.train_mode import TrainMode
from gym_optimal_intrusion_response.agents.config.agent_config import AgentConfig
from gym_optimal_intrusion_response.agents.openai_baselines.common.evaluation import quick_evaluate_policy


class OnPolicyAlgorithm(BaseAlgorithm):

    def __init__(
        self,
        attacker_policy: Union[str, Type[ActorCriticPolicy]],
        defender_policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        attacker_learning_rate: Union[float, Schedule],
        defender_learning_rate: Union[float, Schedule],
        n_steps: int,
        attacker_gamma: float,
        defender_gamma: float,
        attacker_gae_lambda: float,
        defender_gae_lambda: float,
        attacker_ent_coef: float,
        defender_ent_coef: float,
        attacker_vf_coef: float,
        defender_vf_coef: float,
        max_grad_norm: float,        
        tensorboard_log: Optional[str] = None,
        attacker_policy_kwargs: Optional[Dict[str, Any]] = None,
        defender_policy_kwargs: Optional[Dict[str, Any]] = None,
        attacker_clip_range: float = 0.2,
        defender_clip_range: float = 0.2,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        train_mode: TrainMode = TrainMode.TRAIN_ATTACKER,
        attacker_agent_config: AgentConfig = None,
        defender_agent_config: AgentConfig = None
    ):

        super(OnPolicyAlgorithm, self).__init__(
            attacker_policy=attacker_policy,
            defender_policy=defender_policy,
            env=env,
            policy_base=ActorCriticPolicy,
            attacker_learning_rate=attacker_learning_rate,
            defender_learning_rate=defender_learning_rate,
            attacker_policy_kwargs=attacker_policy_kwargs,
            defender_policy_kwargs=defender_policy_kwargs,
            device=device,            
            seed=seed,
            tensorboard_log=tensorboard_log,
            train_mode=train_mode,
            attacker_agent_config=attacker_agent_config,
            defender_agent_config=defender_agent_config
        )

        self.n_steps = n_steps
        self.attacker_gamma = attacker_gamma
        self.defender_gamma = defender_gamma
        self.attacker_gae_lambda = attacker_gae_lambda
        self.defender_gae_lambda = defender_gae_lambda
        self.attacker_ent_coef = attacker_ent_coef
        self.defender_ent_coef = defender_ent_coef
        self.attacker_vf_coef = attacker_vf_coef
        self.defender_vf_coef = defender_vf_coef
        self.attacker_clip_range = attacker_clip_range
        self.defender_clip_range = defender_clip_range
        self.max_grad_norm = max_grad_norm
        self.attacker_rollout_buffer = None
        self.defender_rollout_buffer = None
        self.train_mode = train_mode
        self.attacker_agent_config = attacker_agent_config
        self.defender_agent_config = defender_agent_config
        self.iteration = 0
        self.num_episodes = 0
        self.num_episodes_total = 0

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self.set_random_seed(self.seed)

        self.attacker_rollout_buffer = RolloutBuffer(
            self.n_steps,
            self.attacker_observation_space,
            self.attacker_action_space,
            self.device,
            gamma=self.attacker_gamma,
            gae_lambda=self.attacker_gae_lambda,
            n_envs=self.n_envs,
        )
        self.defender_rollout_buffer = RolloutBuffer(
            self.n_steps,
            self.defender_observation_space,
            self.defender_action_space,
            self.device,
            gamma=self.defender_gamma,
            gae_lambda=self.defender_gae_lambda,
            n_envs=self.n_envs,
        )
        self.attacker_policy = self.attacker_policy_class(
            self.attacker_observation_space,
            self.attacker_action_space,
            self.attacker_learning_rate,
            agent_config = self.attacker_agent_config,
            **self.attacker_policy_kwargs
        )
        self.attacker_policy = self.attacker_policy.to(self.device)

        self.defender_policy = self.defender_policy_class(
            self.defender_observation_space,
            self.defender_action_space,
            self.defender_learning_rate,
            agent_config=self.defender_agent_config,
            **self.defender_policy_kwargs
        )
        self.defender_policy = self.defender_policy.to(self.device)

    def collect_rollouts(
        self, env: VecEnv, attacker_rollout_buffer: RolloutBuffer,
            defender_rollout_buffer: RolloutBuffer, n_rollout_steps: int) -> Tuple[bool, RolloutDataDTO]:
        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0

        attacker_rollout_buffer.reset()
        defender_rollout_buffer.reset()

        # Avg metrics
        rollout_data_dto = RolloutDataDTO()
        rollout_data_dto.initialize()

        # Per episode metrics
        episode_reward_attacker = np.zeros(env.num_envs)
        episode_reward_defender = np.zeros(env.num_envs)
        episode_step = np.zeros(env.num_envs)

        while n_steps < n_rollout_steps:

            new_obs, attacker_rewards, dones, infos, attacker_values, attacker_log_probs, attacker_actions, \
            action_pred_time_s, env_step_time, defender_values, defender_log_probs, defender_actions, \
            defender_rewards = \
                self.step_policy(env, attacker_rollout_buffer, defender_rollout_buffer)

            n_steps += 1
            episode_reward_attacker += attacker_rewards
            episode_reward_defender += defender_rewards
            episode_step += 1

            if dones.any():
                for i in range(len(dones)):
                    if dones[i]:
                        # Record episode metrics
                        rollout_data_dto.attacker_episode_rewards.append(episode_reward_attacker[i])
                        rollout_data_dto.defender_episode_rewards.append(episode_reward_defender[i])
                        rollout_data_dto.episode_steps.append(episode_step[i])
                        rollout_data_dto.episode_flags.append(infos[i]["flags"])
                        rollout_data_dto.episode_caught.append(infos[i]["caught_attacker"])
                        rollout_data_dto.episode_early_stopped.append(infos[i]["early_stopped"])
                        rollout_data_dto.episode_successful_intrusion.append(infos[i]["successful_intrusion"])
                        rollout_data_dto.episode_snort_severe_baseline_rewards.append(
                            infos[i]["snort_severe_baseline_reward"])
                        rollout_data_dto.episode_snort_warning_baseline_rewards.append(
                            infos[i]["snort_warning_baseline_reward"])
                        rollout_data_dto.episode_snort_critical_baseline_rewards.append(
                            infos[i]["snort_critical_baseline_reward"])
                        rollout_data_dto.episode_var_log_baseline_rewards.append(infos[i]["var_log_baseline_reward"])
                        rollout_data_dto.attacker_action_costs.append(infos[i]["attacker_cost"])
                        rollout_data_dto.attacker_action_costs_norm.append(infos[i]["attacker_cost_norm"])
                        rollout_data_dto.attacker_action_alerts.append(infos[i]["attacker_alerts"])
                        rollout_data_dto.attacker_action_alerts_norm.append(infos[i]["attacker_alerts_norm"])
                        rollout_data_dto.episode_flags_percentage.append(
                            infos[i]["flags"] / 1)
                        # print("reset reward:{}".format(episode_reward_attacker))
                        episode_reward_attacker[i] = 0
                        episode_reward_defender[i] = 0
                        episode_step[i] = 0

        if self.train_mode == TrainMode.TRAIN_ATTACKER or self.train_mode == TrainMode.SELF_PLAY:
            attacker_rollout_buffer.compute_returns_and_advantage(attacker_values, dones=dones)
        if self.train_mode == TrainMode.TRAIN_DEFENDER or self.train_mode == TrainMode.SELF_PLAY:
            defender_rollout_buffer.compute_returns_and_advantage(defender_values, dones=dones)

        for i in range(len(dones)):
            if not dones[i]:
                rollout_data_dto.attacker_episode_rewards.append(episode_reward_attacker[i])
                rollout_data_dto.defender_episode_rewards.append(episode_reward_defender[i])
                rollout_data_dto.episode_steps.append(episode_step[i])

        return True, rollout_data_dto

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def learn(
        self,
        total_timesteps: int,
        log_interval: int = 1,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "OnPolicyAlgorithm",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "OnPolicyAlgorithm":

        print("Setting up Training Configuration")

        self.iteration = 0
        total_timesteps = self._setup_learn(
            total_timesteps, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )

        # Tracking metrics
        train_log_dto = TrainAgentLogDTO()
        train_log_dto.initialize()
        train_log_dto.train_result = self.train_result
        train_log_dto.eval_result = self.eval_result
        train_log_dto.iteration = self.iteration
        train_log_dto.start_time = self.training_start

        num_iterations = self.attacker_agent_config.num_iterations
        while self.num_timesteps < num_iterations:

            continue_training, rollout_data_dto = self.collect_rollouts(
                self.env, self.attacker_rollout_buffer, self.defender_rollout_buffer, n_rollout_steps=self.n_steps)

            train_log_dto.attacker_episode_rewards.extend(rollout_data_dto.attacker_episode_rewards)
            train_log_dto.defender_episode_rewards.extend(rollout_data_dto.defender_episode_rewards)
            train_log_dto.episode_steps.extend(rollout_data_dto.episode_steps)
            train_log_dto.episode_flags.extend(rollout_data_dto.episode_flags)
            train_log_dto.episode_caught.extend(rollout_data_dto.episode_caught)
            train_log_dto.episode_successful_intrusion.extend(rollout_data_dto.episode_successful_intrusion)
            train_log_dto.episode_early_stopped.extend(rollout_data_dto.episode_early_stopped)
            train_log_dto.episode_flags_percentage.extend(rollout_data_dto.episode_flags_percentage)
            train_log_dto.episode_snort_severe_baseline_rewards.extend(
                rollout_data_dto.episode_snort_severe_baseline_rewards)
            train_log_dto.episode_snort_warning_baseline_rewards.extend(
                rollout_data_dto.episode_snort_warning_baseline_rewards)
            train_log_dto.episode_snort_critical_baseline_rewards.extend(
                rollout_data_dto.episode_snort_critical_baseline_rewards)
            train_log_dto.episode_var_log_baseline_rewards.extend(rollout_data_dto.episode_var_log_baseline_rewards)
            train_log_dto.attacker_action_costs.extend(rollout_data_dto.attacker_action_costs)
            train_log_dto.attacker_action_costs_norm.extend(rollout_data_dto.attacker_action_costs_norm)
            train_log_dto.attacker_action_alerts.extend(rollout_data_dto.attacker_action_alerts)
            train_log_dto.attacker_action_alerts_norm.extend(rollout_data_dto.attacker_action_alerts_norm)

            if continue_training is False:
                break

            self.iteration += 1
            train_log_dto.iteration = self.iteration

            if self.iteration % self.attacker_agent_config.train_log_frequency == 0 or self.iteration == 1:
                train_log_dto = quick_evaluate_policy(
                    attacker_model=self.attacker_policy, defender_model=self.defender_policy,
                    env=self.env, n_eval_episodes_train=self.attacker_agent_config.eval_episodes,
                    deterministic=self.attacker_agent_config.eval_deterministic,  train_mode=self.train_mode,
                    train_dto=train_log_dto)

                n_af, n_d = 0, 0

                train_log_dto.n_af = n_af
                train_log_dto.n_d = n_d
                if self.train_mode == TrainMode.TRAIN_ATTACKER or self.train_mode == TrainMode.SELF_PLAY:
                    train_log_dto = self.log_metrics_attacker(train_log_dto=train_log_dto, eval=False)
                if self.train_mode == TrainMode.TRAIN_DEFENDER or self.train_mode == TrainMode.SELF_PLAY:
                    train_log_dto = self.log_metrics_defender(train_log_dto=train_log_dto, eval=False)
                self.train_result = train_log_dto.train_result
                self.eval_result = train_log_dto.eval_result
                train_log_dto.initialize()
                train_log_dto.train_result = self.train_result
                train_log_dto.eval_result = self.eval_result
                train_log_dto.iteration = self.iteration
                train_log_dto.start_time = self.training_start
                self.num_episodes = 0

            # Save models every <self.config.checkpoint_frequency> iterations
            if self.iteration % self.attacker_agent_config.checkpoint_freq == 0:
                try:
                    self.save_model(self.iteration)
                except Exception as e:
                    print("There was an error saving the model: {}".format(str(e)))
                if self.attacker_agent_config.save_dir is not None:
                    time_str = str(time.time())
                    self.train_result.to_csv(
                        self.attacker_agent_config.save_dir + "/" + time_str + "_train_results_checkpoint.csv")
                    self.eval_result.to_csv(
                        self.attacker_agent_config.save_dir + "/" + time_str + "_eval_results_checkpoint.csv")

            entropy_loss_attacker, pg_loss_attacker, value_loss_attacker, lr_attacker, \
            entropy_loss_defender, pg_loss_defender, value_loss_defender, lr_defender, = self.train()

            train_log_dto.attacker_episode_avg_loss.append(entropy_loss_attacker + pg_loss_attacker + value_loss_attacker)
            train_log_dto.defender_episode_avg_loss.append(entropy_loss_defender + pg_loss_defender + value_loss_defender)


        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["attacker_policy", "attacker_policy.optimizer",
                       "defender_policy", "defender_policy.optimizer"]

        return state_dicts, []



    def step_policy(self, env, attacker_rollout_buffer, defender_rollout_buffer):
        action_pred_time = 0.0
        env_step_time = 0.0

        with th.no_grad():
            # Convert to pytorch tensor
            if isinstance(self._last_obs, tuple):
                attacker_obs, defender_obs = self._last_obs
            obs_tensor_attacker = th.as_tensor(attacker_obs).to(self.device)
            obs_tensor_defender = th.as_tensor(defender_obs).to(self.device)
            attacker_actions = None
            attacker_values = None
            attacker_log_probs = None
            defender_actions = None
            defender_values = None
            defender_log_probs = None
            if self.train_mode == TrainMode.TRAIN_ATTACKER or self.train_mode == TrainMode.SELF_PLAY:
                attacker_actions, attacker_values, attacker_log_probs = \
                    self.attacker_policy.forward(obs_tensor_attacker, attacker=True, env=env)
                attacker_actions = attacker_actions.cpu().numpy()
            if self.train_mode == TrainMode.TRAIN_DEFENDER or self.train_mode == TrainMode.SELF_PLAY:
                defender_actions, defender_values, defender_log_probs = \
                    self.defender_policy.forward(obs_tensor_defender, attacker=False, env=env)
                defender_actions = defender_actions.cpu().numpy()

        if attacker_actions is None:
            attacker_actions = np.array([None]*len(defender_actions))

        if defender_actions is None:
            defender_actions = np.array([None]*len(attacker_actions))

        actions = []
        for i in range(len(attacker_actions)):
            actions.append((attacker_actions[i], defender_actions[i]))

        new_obs, rewards, dones, infos = env.step(actions)

        attacker_actions = attacker_actions.reshape(-1, 1)
        defender_actions = defender_actions.reshape(-1, 1)

        attacker_obs, defender_obs = self.get_attacker_and_defender_obs(self._last_obs)
        attacker_rewards, defender_rewards = self.get_attacker_and_defender_reward(rewards)
        if self.train_mode == TrainMode.TRAIN_ATTACKER or self.train_mode == TrainMode.SELF_PLAY:
            attacker_rollout_buffer.add(attacker_obs, attacker_actions, attacker_rewards, self._last_dones,
                                        attacker_values, attacker_log_probs)
        if self.train_mode == TrainMode.TRAIN_DEFENDER or self.train_mode == TrainMode.SELF_PLAY:
            defender_rollout_buffer.add(defender_obs, defender_actions, defender_rewards, self._last_dones,
                                        defender_values, defender_log_probs)

        self._last_obs = new_obs
        self._last_dones = dones
        self._last_infos = infos

        return new_obs, attacker_rewards, dones, infos, \
               attacker_values, attacker_log_probs, attacker_actions, action_pred_time, env_step_time, \
               defender_values, defender_log_probs, defender_actions, defender_rewards


    def get_attacker_and_defender_obs(self, obs):
        if isinstance(obs, tuple):
            return obs[0], obs[1]
        else:
            attacker_obs = []
            defender_obs = []
            for i in range(len(obs)):
                a_o = obs[i][0]
                d_o = obs[i][1]
                attacker_obs.append(a_o)
                defender_obs.append(d_o)
            attacker_obs = np.array(attacker_obs)
            defender_obs = np.array(defender_obs)
            attacker_obs = attacker_obs.astype("float64")
            defender_obs = defender_obs.astype("float64")
            return attacker_obs, defender_obs

    def get_attacker_and_defender_reward(self, rewards):
        if isinstance(rewards, tuple):
            return rewards[0], rewards[1]
        else:
            attacker_reward = []
            defender_reward = []
            for i in range(len(rewards)):
                a_r = rewards[i][0]
                d_r = rewards[i][1]
                attacker_reward.append(a_r)
                defender_reward.append(d_r)
            attacker_reward = np.array(attacker_reward)
            defender_reward = np.array(defender_reward)
            attacker_reward = attacker_reward.astype("float64")
            defender_reward = defender_reward.astype("float64")
            return attacker_reward, defender_reward