import warnings
from typing import Any, Dict, Optional, Type, Union
import time
import numpy as np
import torch as th
from gym import spaces
from torch.nn import functional as F


from gym_optimal_intrusion_response.agents.openai_baselines.common.on_policy_algorithm import OnPolicyAlgorithm
from gym_optimal_intrusion_response.agents.openai_baselines.common.policies import ActorCriticPolicy
from gym_optimal_intrusion_response.agents.openai_baselines.common.type_aliases import GymEnv, MaybeCallback, Schedule
from gym_optimal_intrusion_response.agents.openai_baselines.common.utils import explained_variance, get_schedule_fn
from gym_optimal_intrusion_response.agents.config.agent_config import AgentConfig
from gym_optimal_intrusion_response.dao.agent.train_mode import TrainMode


class PPO(OnPolicyAlgorithm):

    def __init__(
        self,
        attacker_policy: Union[str, Type[ActorCriticPolicy]],
        defender_policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        attacker_learning_rate: Union[float, Schedule] = 3e-4,
        defender_learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: Optional[int] = 64,
        n_epochs: int = 10,
        attacker_gamma: float = 0.99,
        defender_gamma: float = 0.99,
        attacker_gae_lambda: float = 0.95,
        defender_gae_lambda: float = 0.95,
        attacker_ent_coef: float = 0.0,
        defender_ent_coef: float = 0.0,
        attacker_vf_coef: float = 0.5,
        defender_vf_coef: float = 0.5,
        attacker_clip_range: float = 0.2,
        defender_clip_range: float = 0.2,
        max_grad_norm: float = 0.5,
        tensorboard_log: Optional[str] = None,
        attacker_policy_kwargs: Optional[Dict[str, Any]] = None,
        defender_policy_kwargs: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        train_mode: TrainMode = TrainMode.TRAIN_ATTACKER,
        attacker_agent_config: AgentConfig = None,
        defender_agent_config: AgentConfig = None
    ):

        super(PPO, self).__init__(
            attacker_policy,
            defender_policy,
            env,
            attacker_learning_rate=attacker_learning_rate,
            defender_learning_rate=defender_learning_rate,
            n_steps=n_steps,
            attacker_gamma=attacker_gamma,
            defender_gamma=defender_gamma,
            attacker_gae_lambda=attacker_gae_lambda,
            defender_gae_lambda=defender_gae_lambda,
            attacker_ent_coef=attacker_ent_coef,
            defender_ent_coef=defender_ent_coef,
            attacker_vf_coef=attacker_vf_coef,
            defender_vf_coef=defender_vf_coef,
            max_grad_norm=max_grad_norm,
            tensorboard_log=tensorboard_log,
            attacker_policy_kwargs=attacker_policy_kwargs,
            defender_policy_kwargs=defender_policy_kwargs,
            device=device,
            seed=seed,
            _init_setup_model=False,
            attacker_agent_config=attacker_agent_config,
            defender_agent_config=defender_agent_config,
            train_mode=train_mode
        )
        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert (
                buffer_size > 1
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a multiple of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.attacker_clip_range = attacker_clip_range
        self.defender_clip_range = defender_clip_range

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(PPO, self)._setup_model()

    def train(self) -> None:
        """
        Update policy using the currently gathered
        rollout buffer.
        """
        # Update optimizer learning rate
        lr_attacker = self.attacker_policy.optimizer.param_groups[0]["lr"]
        lr_defender = self.defender_policy.optimizer.param_groups[0]["lr"]

        # Optional: clip range for the value function
        attacker_clip_range_vf = None
        defender_clip_range_vf = None

        entropy_losses_attacker, all_kl_divs_attacker = [], []
        pg_losses_attacker, value_losses_attacker = [], []
        clip_fractions_attacker = []
        grad_comp_times_attacker = []
        weight_update_times_attacker = []

        entropy_losses_defender, all_kl_divs_defender = [], []
        pg_losses_defender, value_losses_defender = [], []
        clip_fractions_defender = []
        grad_comp_times_defender = []
        weight_update_times_defender = []

        # train for gradient_steps epochs
        for epoch in range(self.n_epochs):

            if self.train_mode == TrainMode.TRAIN_ATTACKER or self.train_mode == TrainMode.SELF_PLAY:
                attacker_clip_range, pg_losses_attacker, clip_fractions_attacker, \
                value_losses_attacker, entropy_losses_attacker  =  \
                    self.attacker_rollout_buffer_pass(
                    self.attacker_clip_range, pg_losses_attacker, clip_fractions_attacker,
                    value_losses_attacker, entropy_losses_attacker)

            if self.train_mode == TrainMode.TRAIN_DEFENDER or self.train_mode == TrainMode.SELF_PLAY:
                defender_clip_range, pg_losses_defender, clip_fractions_defender, \
                value_losses_defender, entropy_losses_defender = \
                    self.defender_rollout_buffer_pass(
                        self.defender_clip_range, pg_losses_defender, clip_fractions_defender,
                        value_losses_defender, entropy_losses_defender)


        self._n_updates += self.n_epochs
        if self.train_mode == TrainMode.TRAIN_ATTACKER or self.train_mode == TrainMode.SELF_PLAY:
            explained_var_attacker = \
                explained_variance(self.attacker_rollout_buffer.returns.flatten(),
                                   self.attacker_rollout_buffer.values.flatten())

        if self.train_mode == TrainMode.TRAIN_DEFENDER or self.train_mode == TrainMode.SELF_PLAY:
            explained_var_defender = \
                explained_variance(self.defender_rollout_buffer.returns.flatten(),
                                   self.defender_rollout_buffer.values.flatten())

        return np.mean(entropy_losses_attacker), \
               np.mean(pg_losses_attacker), np.mean(value_losses_attacker), \
               lr_attacker, \
               np.mean(entropy_losses_defender), \
               np.mean(pg_losses_defender), np.mean(value_losses_defender), \
               lr_defender

    def attacker_rollout_buffer_pass(self, attacker_clip_range, pg_losses_attacker, clip_fractions_attacker,
                                     value_losses_attacker, entropy_losses_attacker):
        # Do a complete pass on the attacker's rollout buffer
        for rollout_data in self.attacker_rollout_buffer.get(self.batch_size):
            if self.attacker_agent_config.performance_analysis:
                start = time.time()

            actions = rollout_data.actions
            if isinstance(self.attacker_action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = rollout_data.actions.long().flatten()

            values, log_prob, entropy = self.attacker_policy.evaluate_actions(rollout_data.observations, actions)
            values = values.flatten()
            # Normalize advantage
            advantages = rollout_data.advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # ratio between old and new policy, should be one at the first iteration
            ratio = th.exp(log_prob - rollout_data.old_log_prob)

            # clipped surrogate loss
            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * th.clamp(ratio, 1 - attacker_clip_range, 1 + attacker_clip_range)
            policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

            # Logging
            pg_losses_attacker.append(policy_loss.item())
            clip_fraction = th.mean((th.abs(ratio - 1) > attacker_clip_range).float()).item()
            clip_fractions_attacker.append(clip_fraction)

            values_pred = values

            # Value loss using the TD(gae_lambda) target
            value_loss = F.mse_loss(rollout_data.returns, values_pred)
            value_losses_attacker.append(value_loss.item())

            # Entropy loss favor exploration
            if entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -th.mean(-log_prob)
            else:
                entropy_loss = -th.mean(entropy)

            entropy_losses_attacker.append(entropy_loss.item())

            loss = policy_loss + self.attacker_ent_coef * entropy_loss + self.attacker_vf_coef * value_loss

            # Optimization step
            self.attacker_policy.optimizer.zero_grad()
            loss.backward()

            # Clip grad norm
            th.nn.utils.clip_grad_norm_(self.attacker_policy.parameters(), self.max_grad_norm)
            self.attacker_policy.optimizer.step()

        return attacker_clip_range, pg_losses_attacker, clip_fractions_attacker, \
               value_losses_attacker, entropy_losses_attacker

    def defender_rollout_buffer_pass(self, defender_clip_range, pg_losses_defender, clip_fractions_defender,
                                     value_losses_defender, entropy_losses_defender):

        # Do a complete pass on the defender's rollout buffer
        for rollout_data in self.defender_rollout_buffer.get(self.batch_size):
            if self.defender_agent_config.performance_analysis:
                start = time.time()

            actions = rollout_data.actions
            if isinstance(self.defender_action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = rollout_data.actions.long().flatten()

            values, log_prob, entropy = self.defender_policy.evaluate_actions(rollout_data.observations, actions)
            values = values.flatten()
            # Normalize advantage
            advantages = rollout_data.advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # ratio between old and new policy, should be one at the first iteration
            ratio = th.exp(log_prob - rollout_data.old_log_prob)

            # clipped surrogate loss
            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * th.clamp(ratio, 1 - defender_clip_range, 1 + defender_clip_range)
            policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

            # Logging
            pg_losses_defender.append(policy_loss.item())
            clip_fraction = th.mean((th.abs(ratio - 1) > defender_clip_range).float()).item()
            clip_fractions_defender.append(clip_fraction)

            # No clipping
            values_pred = values

            # Value loss using the TD(gae_lambda) target
            value_loss = F.mse_loss(rollout_data.returns, values_pred)
            value_losses_defender.append(value_loss.item())

            # Entropy loss favor exploration
            if entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -th.mean(-log_prob)
            else:
                entropy_loss = -th.mean(entropy)

            entropy_losses_defender.append(entropy_loss.item())

            loss = policy_loss + self.defender_ent_coef * entropy_loss + self.defender_vf_coef * value_loss

            # Optimization step
            self.defender_policy.optimizer.zero_grad()
            loss.backward()

            # Clip grad norm
            if self.defender_agent_config.performance_analysis:
                start = time.time()
            th.nn.utils.clip_grad_norm_(self.defender_policy.parameters(), self.max_grad_norm)

            self.defender_policy.optimizer.step()

        return defender_clip_range, pg_losses_defender, clip_fractions_defender, \
               value_losses_defender, entropy_losses_defender

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "PPO",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "PPO":

        return super(PPO, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )