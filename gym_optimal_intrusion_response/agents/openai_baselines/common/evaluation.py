
from typing import Union
import gym
import numpy as np
from gym_optimal_intrusion_response.agents.openai_baselines.common.vec_env import VecEnv
from gym_optimal_intrusion_response.dao.agent.train_mode import TrainMode
from gym_optimal_intrusion_response.dao.agent.train_agent_log_dto import TrainAgentLogDTO


def quick_evaluate_policy(attacker_model, defender_model,
                          env: Union[gym.Env, VecEnv],
                          n_eval_episodes_train : int=10,
                          deterministic : bool= True,
                          train_mode: TrainMode = TrainMode.TRAIN_ATTACKER,
                          train_dto : TrainAgentLogDTO = None
                          ):

    train_dto = _quick_eval_helper(
        env=env, attacker_model=attacker_model, defender_model=defender_model,
        n_eval_episodes=n_eval_episodes_train, deterministic=deterministic,
        train_mode=train_mode,
        train_log_dto=train_dto)

    return train_dto


def _quick_eval_helper(env, attacker_model, defender_model,
                       n_eval_episodes, deterministic, train_mode,
                       train_log_dto : TrainAgentLogDTO = None):

    for episode in range(n_eval_episodes):
        for i in range(env.num_envs):
            obs = env.envs[i].reset()
            done = False
            state = None
            attacker_episode_reward = 0.0
            defender_episode_reward = 0.0
            episode_length = 0
            while not done:
                obs_attacker, obs_defender = obs
                attacker_actions = None
                defender_actions = [None]
                if train_mode == train_mode.TRAIN_ATTACKER or train_mode == train_mode.SELF_PLAY:
                    attacker_actions, state = attacker_model.predict(np.array([obs_attacker]), state=state,
                                                                     deterministic=deterministic,
                                                                     attacker=True, env=env)
                if train_mode == train_mode.TRAIN_DEFENDER or train_mode == train_mode.SELF_PLAY:
                    defender_actions, state = defender_model.predict(np.array([obs_defender]), state=state,
                                                                     deterministic=deterministic,
                                                                     attacker=False, env=env)
                    if attacker_actions is None:
                        attacker_actions = np.array([None])
                defender_action = defender_actions[0]
                attacker_action = attacker_actions[0]
                action = (attacker_action, defender_action)
                obs, reward, done, _info = env.envs[i].step(action)
                attacker_reward, defender_reward = reward
                attacker_episode_reward += attacker_reward
                defender_episode_reward += defender_reward
                episode_length += 1

            # Record episode metrics
            train_log_dto.attacker_eval_episode_rewards.append(attacker_episode_reward)
            train_log_dto.defender_eval_episode_rewards.append(defender_episode_reward)
            train_log_dto.eval_episode_steps.append(episode_length)
            train_log_dto.eval_episode_flags.append(_info["flags"])
            train_log_dto.eval_episode_caught.append(_info["caught_attacker"])
            train_log_dto.eval_episode_early_stopped.append(_info["early_stopped"])
            train_log_dto.eval_episode_successful_intrusion.append(_info["successful_intrusion"])
            train_log_dto.eval_episode_snort_severe_baseline_rewards.append(_info["snort_severe_baseline_reward"])
            train_log_dto.eval_episode_snort_warning_baseline_rewards.append(_info["snort_warning_baseline_reward"])
            train_log_dto.eval_episode_snort_critical_baseline_rewards.append(_info["snort_critical_baseline_reward"])
            train_log_dto.eval_episode_var_log_baseline_rewards.append(_info["var_log_baseline_reward"])
            train_log_dto.eval_episode_flags_percentage.append(_info["flags"]/1)
            train_log_dto.eval_attacker_action_costs.append(_info["attacker_cost"])
            train_log_dto.eval_attacker_action_costs_norm.append(_info["attacker_cost_norm"])
            train_log_dto.eval_attacker_action_alerts.append(_info["attacker_alerts"])
            train_log_dto.eval_attacker_action_alerts_norm.append(_info["attacker_alerts_norm"])

            obs = env.envs[i].reset()

    return train_log_dto
