import numpy as np
import sys
import time
from gym_optimal_intrusion_response.agents.openai_baselines.common.vec_env import SubprocVecEnv
import gym_optimal_intrusion_response.util.util as os_util
from gym_optimal_intrusion_response.agents.config.agent_config import AgentConfig
from gym_optimal_intrusion_response.dao.agent.train_mode import TrainMode
from gym_optimal_intrusion_response.dao.agent.train_agent_log_dto import TrainAgentLogDTO
from gym_optimal_intrusion_response.dao.agent.tensorboard_data_dto import TensorboardDataDTO


class LogUtil:
    """
    Utility class for logging training progress
    """

    @staticmethod
    def log_metrics_attacker(train_log_dto: TrainAgentLogDTO, eps: float = None, eval: bool = False,
                             attacker_agent_config : AgentConfig = None,
                             tensorboard_writer = None) -> TrainAgentLogDTO:
        if eps is None:
            eps = 0.0

        if not eval:
            result = train_log_dto.train_result
        else:
            result = train_log_dto.eval_result

        training_time = time.time() - train_log_dto.start_time
        training_time_hours = training_time/3600
        avg_episode_rewards = np.mean(train_log_dto.attacker_episode_rewards)
        avg_episode_flags = np.mean(train_log_dto.episode_flags)
        avg_episode_flags_percentage = np.mean(train_log_dto.episode_flags_percentage)
        avg_episode_steps = np.mean(train_log_dto.episode_steps)
        avg_episode_costs = np.mean(train_log_dto.attacker_action_costs)
        avg_episode_costs_norm = np.mean(train_log_dto.attacker_action_costs_norm)
        avg_episode_alerts = np.mean(train_log_dto.attacker_action_alerts)
        avg_episode_alerts_norm = np.mean(train_log_dto.attacker_action_alerts_norm)

        if train_log_dto.episode_caught is not None and train_log_dto.episode_early_stopped is not None \
                and train_log_dto.episode_successful_intrusion is not None:
            total_c_s_i = sum(list(map(lambda x: int(x), train_log_dto.episode_caught))) \
                          + sum(list(map(lambda x: int(x), train_log_dto.episode_early_stopped))) \
                          + sum(list(map(lambda x: int(x), train_log_dto.episode_successful_intrusion)))
        else:
            total_c_s_i = 1
        if train_log_dto.eval_episode_caught is not None and train_log_dto.eval_episode_early_stopped is not None \
                and train_log_dto.eval_episode_successful_intrusion is not None:
            eval_total_c_s_i = sum(list(map(lambda x: int(x), train_log_dto.eval_episode_caught))) \
                               + sum(list(map(lambda x: int(x), train_log_dto.eval_episode_early_stopped))) \
                               + sum(list(map(lambda x: int(x), train_log_dto.eval_episode_successful_intrusion)))
        else:
            eval_total_c_s_i = 1
        if train_log_dto.episode_caught is not None:
            episode_caught_frac = sum(list(map(lambda x: int(x), train_log_dto.episode_caught))) / max(1, total_c_s_i)
        else:
            episode_caught_frac = 0

        if train_log_dto.episode_early_stopped is not None:
            episode_early_stopped_frac = sum(list(map(lambda x: int(x),
                                                      train_log_dto.episode_early_stopped))) / max(1, total_c_s_i)
        else:
            episode_early_stopped_frac = 0

        if train_log_dto.episode_successful_intrusion is not None:
            episode_successful_intrusion_frac = sum(list(map(lambda x: int(x),
                                                             train_log_dto.episode_successful_intrusion))) / max(1,
                                                                                                                 total_c_s_i)
        else:
            episode_successful_intrusion_frac = 0

        if train_log_dto.eval_episode_caught is not None:
            eval_episode_caught_frac = sum(list(map(lambda x: int(x),
                                                    train_log_dto.eval_episode_caught))) / max(1, eval_total_c_s_i)
        else:
            eval_episode_caught_frac = 0

        if train_log_dto.eval_episode_successful_intrusion is not None:
            eval_episode_successful_intrusion_frac = sum(list(map(lambda x: int(x),
                                                                  train_log_dto.eval_episode_successful_intrusion))) / max(
                1, eval_total_c_s_i)
        else:
            eval_episode_successful_intrusion_frac = 0

        if train_log_dto.eval_episode_early_stopped is not None:
            eval_episode_early_stopped_frac = sum(list(map(lambda x: int(x),
                                                           train_log_dto.eval_episode_early_stopped))) / max(1,
                                                                                                             eval_total_c_s_i)
        else:
            eval_episode_early_stopped_frac = 0

        if result.attacker_avg_episode_rewards is not None:
            rolling_avg_rewards = os_util.running_average(result.attacker_avg_episode_rewards + [avg_episode_rewards],
                                                            attacker_agent_config.running_avg)
        else:
            rolling_avg_rewards = 0.0

        if result.avg_episode_steps is not None:
            rolling_avg_steps = os_util.running_average(result.avg_episode_steps + [avg_episode_steps],
                                                          attacker_agent_config.running_avg)
        else:
            rolling_avg_steps = 0.0

        if train_log_dto.attacker_lr is None:
            lr = 0.0
        else:
            lr = train_log_dto.attacker_lr
        if not eval and train_log_dto.attacker_episode_avg_loss is not None:
            avg_episode_loss = np.mean(train_log_dto.attacker_episode_avg_loss)
        else:
            avg_episode_loss = 0.0

        if not eval and train_log_dto.attacker_eval_episode_rewards is not None:
            eval_avg_episode_rewards = np.mean(train_log_dto.attacker_eval_episode_rewards)
        else:
            eval_avg_episode_rewards = 0.0

        avg_regret = 0.0
        avg_eval_regret = 0.0
        avg_opt_frac = 0.0
        eval_avg_opt_frac = 0.0

        if not eval and train_log_dto.eval_episode_flags is not None:
            eval_avg_episode_flags = np.mean(train_log_dto.eval_episode_flags)
        else:
            eval_avg_episode_flags = 0.0
        if not eval and train_log_dto.eval_episode_flags_percentage is not None:
            eval_avg_episode_flags_percentage = np.mean(train_log_dto.eval_episode_flags_percentage)
        else:
            eval_avg_episode_flags_percentage = 0.0
        if not eval and train_log_dto.eval_episode_steps is not None:
            eval_avg_episode_steps = np.mean(train_log_dto.eval_episode_steps)
        else:
            eval_avg_episode_steps = 0.0

        if not eval and train_log_dto.eval_attacker_action_costs is not None:
            eval_avg_episode_costs = np.mean(train_log_dto.eval_attacker_action_costs)
        else:
            eval_avg_episode_costs = 0.0

        if not eval and train_log_dto.eval_attacker_action_costs_norm is not None:
            eval_avg_episode_costs_norm = np.mean(train_log_dto.eval_attacker_action_costs_norm)
        else:
            eval_avg_episode_costs_norm = 0.0

        if not eval and train_log_dto.eval_attacker_action_alerts is not None:
            eval_avg_episode_alerts = np.mean(train_log_dto.eval_attacker_action_alerts)
        else:
            eval_avg_episode_alerts = 0.0

        if not eval and train_log_dto.eval_attacker_action_alerts_norm is not None:
            eval_avg_episode_alerts_norm = np.mean(train_log_dto.eval_attacker_action_alerts_norm)
        else:
            eval_avg_episode_alerts_norm = 0.0

        tensorboard_data_dto = TensorboardDataDTO(
            iteration=train_log_dto.iteration, avg_episode_rewards=avg_episode_rewards,
            avg_episode_steps=avg_episode_steps,
            avg_episode_loss=avg_episode_loss, eps=eps, lr=lr, eval=eval,
            avg_flags_catched=avg_episode_flags, avg_episode_flags_percentage=avg_episode_flags_percentage,
            eval_avg_episode_rewards=eval_avg_episode_rewards, eval_avg_episode_steps=eval_avg_episode_steps,
            eval_avg_episode_flags=eval_avg_episode_flags,
            eval_avg_episode_flags_percentage=eval_avg_episode_flags_percentage,
            rolling_avg_episode_rewards=rolling_avg_rewards,
            rolling_avg_episode_steps=rolling_avg_steps,
            tensorboard_writer=tensorboard_writer,
            episode_caught_frac=episode_caught_frac,
            episode_early_stopped_frac=episode_early_stopped_frac,
            episode_successful_intrusion_frac=episode_successful_intrusion_frac,
            eval_episode_caught_frac=eval_episode_caught_frac,
            eval_episode_early_stopped_frac=eval_episode_early_stopped_frac,
            eval_episode_successful_intrusion_frac=eval_episode_successful_intrusion_frac,
            avg_regret=avg_regret, avg_opt_frac=avg_opt_frac, rolling_avg_rewards=rolling_avg_rewards,
            rolling_avg_steps=rolling_avg_steps, avg_episode_flags=avg_episode_flags,
            n_af=train_log_dto.n_af, n_d=train_log_dto.n_d, avg_episode_costs=avg_episode_costs,
            avg_episode_costs_norm=avg_episode_costs_norm,
            avg_episode_alerts=avg_episode_alerts, avg_episode_alerts_norm=avg_episode_alerts_norm,
            eval_avg_episode_costs=eval_avg_episode_costs, eval_avg_episode_costs_norm=eval_avg_episode_costs_norm,
            eval_avg_episode_alerts=eval_avg_episode_alerts, eval_avg_episode_alerts_norm=eval_avg_episode_alerts_norm,
            total_num_episodes=train_log_dto.total_num_episodes, avg_eval_regret=avg_eval_regret,
            eval_avg_opt_frac=eval_avg_opt_frac,
            epsilon=attacker_agent_config.epsilon,
            training_time_hours=training_time_hours)
        log_str = tensorboard_data_dto.log_str_attacker()
        attacker_agent_config.logger.info(log_str)
        print(log_str)
        sys.stdout.flush()
        if attacker_agent_config.tensorboard:
            tensorboard_data_dto.log_tensorboard_attacker()

        result.avg_episode_steps.append(avg_episode_steps)
        result.attacker_avg_episode_rewards.append(avg_episode_rewards)
        result.epsilon_values.append(attacker_agent_config.epsilon)
        result.attacker_avg_episode_loss.append(avg_episode_loss)
        result.avg_episode_flags.append(avg_episode_flags)
        result.avg_episode_flags_percentage.append(avg_episode_flags_percentage)
        result.attacker_eval_avg_episode_rewards.append(eval_avg_episode_rewards)
        result.eval_avg_episode_steps.append(eval_avg_episode_steps)
        result.eval_avg_episode_flags.append(eval_avg_episode_flags)
        result.eval_avg_episode_flags_percentage.append(eval_avg_episode_flags_percentage)
        result.lr_list.append(train_log_dto.attacker_lr)
        result.attacker_avg_regret.append(avg_regret)
        result.attacker_avg_opt_frac.append(avg_opt_frac)
        result.attacker_eval_avg_regret.append(avg_eval_regret)
        result.attacker_eval_avg_opt_frac.append(eval_avg_opt_frac)
        result.caught_frac.append(episode_caught_frac)
        result.early_stopping_frac.append(episode_early_stopped_frac)
        result.intrusion_frac.append(episode_successful_intrusion_frac)
        result.eval_caught_frac.append(eval_episode_caught_frac)
        result.eval_early_stopping_frac.append(eval_episode_early_stopped_frac)
        result.eval_intrusion_frac.append(eval_episode_successful_intrusion_frac)
        result.attacker_action_costs.append(avg_episode_costs)
        result.attacker_action_costs_norm.append(avg_episode_costs_norm)
        result.attacker_action_alerts.append(avg_episode_alerts)
        result.attacker_action_alerts_norm.append(avg_episode_alerts_norm)
        result.eval_attacker_action_costs.append(eval_avg_episode_costs)
        result.eval_attacker_action_costs_norm.append(eval_avg_episode_costs_norm)
        result.eval_attacker_action_alerts.append(eval_avg_episode_alerts)
        result.eval_attacker_action_alerts_norm.append(eval_avg_episode_alerts_norm)
        result.time_elapsed.append(training_time)

        if not eval:
            train_log_dto.train_result = result
        else:
            train_log_dto.eval_result = result
        return train_log_dto


    @staticmethod
    def log_metrics_defender(train_log_dto: TrainAgentLogDTO, eps: float = None, eval: bool = False,
                             defender_agent_config : AgentConfig = None,
                             tensorboard_writer = None, train_mode: TrainMode = TrainMode.TRAIN_ATTACKER) \
            -> TrainAgentLogDTO:
        if eps is None:
            eps = 0.0

        if not eval:
            result = train_log_dto.train_result
        else:
            result = train_log_dto.eval_result

        training_time = time.time() - train_log_dto.start_time
        training_time_hours = training_time / 3600

        avg_episode_rewards = np.mean(train_log_dto.defender_episode_rewards)
        avg_episode_steps = np.mean(train_log_dto.episode_steps)
        avg_episode_snort_severe_baseline_rewards = np.mean(train_log_dto.episode_snort_severe_baseline_rewards)
        avg_episode_snort_warning_baseline_rewards = np.mean(train_log_dto.episode_snort_warning_baseline_rewards)
        avg_episode_snort_critical_baseline_rewards = np.mean(train_log_dto.episode_snort_critical_baseline_rewards)
        avg_episode_var_log_baseline_rewards = np.mean(train_log_dto.episode_var_log_baseline_rewards)
        avg_episode_costs = np.mean(train_log_dto.attacker_action_costs)
        avg_episode_costs_norm = np.mean(train_log_dto.attacker_action_costs_norm)
        avg_episode_alerts = np.mean(train_log_dto.attacker_action_alerts)
        avg_episode_alerts_norm = np.mean(train_log_dto.attacker_action_alerts_norm)

        avg_episode_optimal_rewards = np.mean(train_log_dto.optimal_rewards)
        avg_episode_optimal_steps = np.mean(train_log_dto.optimal_steps)

        avg_episode_flags = np.mean(train_log_dto.episode_flags)
        avg_episode_flags_percentage = np.mean(train_log_dto.episode_flags_percentage)

        if train_log_dto.episode_caught is not None and train_log_dto.episode_early_stopped is not None \
                and train_log_dto.episode_successful_intrusion is not None:
            total_c_s_i = sum(list(map(lambda x: int(x), train_log_dto.episode_caught))) \
                          + sum(list(map(lambda x: int(x), train_log_dto.episode_early_stopped))) \
                          + sum(list(map(lambda x: int(x), train_log_dto.episode_successful_intrusion)))
        else:
            total_c_s_i = 1
        if train_log_dto.eval_episode_caught is not None and train_log_dto.eval_episode_early_stopped is not None \
                and train_log_dto.eval_episode_successful_intrusion is not None:
            eval_total_c_s_i = sum(list(map(lambda x: int(x), train_log_dto.eval_episode_caught))) \
                               + sum(list(map(lambda x: int(x), train_log_dto.eval_episode_early_stopped))) \
                               + sum(list(map(lambda x: int(x), train_log_dto.eval_episode_successful_intrusion)))
        else:
            eval_total_c_s_i = 1
        if train_log_dto.episode_caught is not None:
            episode_caught_frac = sum(list(map(lambda x: int(x), train_log_dto.episode_caught))) / max(1, total_c_s_i)
        else:
            episode_caught_frac = 0

        if train_log_dto.episode_early_stopped is not None:
            episode_early_stopped_frac = sum(list(map(lambda x: int(x),
                                                      train_log_dto.episode_early_stopped))) / max(1, total_c_s_i)
        else:
            episode_early_stopped_frac = 0

        if train_log_dto.episode_successful_intrusion is not None:
            episode_successful_intrusion_frac = sum(list(map(lambda x: int(x),
                                                             train_log_dto.episode_successful_intrusion))) / max(1,
                                                                                                                 total_c_s_i)
        else:
            episode_successful_intrusion_frac = 0

        if train_log_dto.eval_episode_caught is not None:
            eval_episode_caught_frac = sum(list(map(lambda x: int(x),
                                                    train_log_dto.eval_episode_caught))) / max(1, eval_total_c_s_i)
        else:
            eval_episode_caught_frac = 0

        if train_log_dto.eval_episode_successful_intrusion is not None:
            eval_episode_successful_intrusion_frac = sum(list(map(lambda x: int(x),
                                                                  train_log_dto.eval_episode_successful_intrusion))) / max(
                1, eval_total_c_s_i)
        else:
            eval_episode_successful_intrusion_frac = 0

        if train_log_dto.eval_episode_early_stopped is not None:
            eval_episode_early_stopped_frac = sum(list(map(lambda x: int(x),
                                                           train_log_dto.eval_episode_early_stopped))) / max(1,
                                                                                                             eval_total_c_s_i)
        else:
            eval_episode_early_stopped_frac = 0

        if not eval and train_log_dto.eval_episode_flags is not None:
            eval_avg_episode_flags = np.mean(train_log_dto.eval_episode_flags)
        else:
            eval_avg_episode_flags = 0.0
        if not eval and train_log_dto.eval_episode_flags_percentage is not None:
            eval_avg_episode_flags_percentage = np.mean(train_log_dto.eval_episode_flags_percentage)
        else:
            eval_avg_episode_flags_percentage = 0.0

        if not eval and train_log_dto.eval_episode_steps is not None:
            eval_avg_episode_steps = np.mean(train_log_dto.eval_episode_steps)
        else:
            eval_avg_episode_steps = 0.0

        if not eval and train_log_dto.eval_attacker_action_costs is not None:
            eval_avg_episode_costs = np.mean(train_log_dto.eval_attacker_action_costs)
        else:
            eval_avg_episode_costs = 0.0

        if not eval and train_log_dto.eval_attacker_action_costs_norm is not None:
            eval_avg_episode_costs_norm = np.mean(train_log_dto.eval_attacker_action_costs_norm)
        else:
            eval_avg_episode_costs_norm = 0.0

        if not eval and train_log_dto.eval_attacker_action_alerts is not None:
            eval_avg_episode_alerts = np.mean(train_log_dto.eval_attacker_action_alerts)
        else:
            eval_avg_episode_alerts = 0.0

        if not eval and train_log_dto.eval_attacker_action_alerts_norm is not None:
            eval_avg_episode_alerts_norm = np.mean(train_log_dto.eval_attacker_action_alerts_norm)
        else:
            eval_avg_episode_alerts_norm = 0.0

        if result.defender_avg_episode_rewards is not None:
            rolling_avg_rewards = os_util.running_average(result.defender_avg_episode_rewards + [avg_episode_rewards],
                                                            defender_agent_config.running_avg)
        else:
            rolling_avg_rewards = 0.0

        if result.avg_episode_steps is not None:
            rolling_avg_steps = os_util.running_average(result.avg_episode_steps + [avg_episode_steps],
                                                          defender_agent_config.running_avg)
        else:
            rolling_avg_steps = 0.0

        if train_log_dto.defender_lr is None:
            lr = 0.0
        else:
            lr = train_log_dto.defender_lr
        if not eval and train_log_dto.defender_episode_avg_loss is not None:
            avg_episode_loss = np.mean(train_log_dto.defender_episode_avg_loss)
        else:
            avg_episode_loss = 0.0

        if not eval and train_log_dto.defender_eval_episode_rewards is not None:
            eval_avg_episode_rewards = np.mean(train_log_dto.defender_eval_episode_rewards)
        else:
            eval_avg_episode_rewards = 0.0

        if not eval and train_log_dto.eval_episode_snort_severe_baseline_rewards is not None:
            eval_avg_episode_snort_severe_baseline_rewards = np.mean(
                train_log_dto.eval_episode_snort_severe_baseline_rewards)
        else:
            eval_avg_episode_snort_severe_baseline_rewards = 0.0

        if not eval and train_log_dto.eval_episode_snort_warning_baseline_rewards is not None:
            eval_avg_episode_snort_warning_baseline_rewards = np.mean(
                train_log_dto.eval_episode_snort_warning_baseline_rewards)
        else:
            eval_avg_episode_snort_warning_baseline_rewards = 0.0

        if not eval and train_log_dto.eval_episode_snort_critical_baseline_rewards is not None:
            eval_avg_episode_snort_critical_baseline_rewards = np.mean(
                train_log_dto.eval_episode_snort_critical_baseline_rewards)
        else:
            eval_avg_episode_snort_critical_baseline_rewards = 0.0

        if not eval and train_log_dto.eval_episode_var_log_baseline_rewards is not None:
            eval_avg_episode_var_log_baseline_rewards = np.mean(
                train_log_dto.eval_episode_var_log_baseline_rewards)
        else:
            eval_avg_episode_var_log_baseline_rewards = 0.0
        avg_regret = 0.0
        avg_eval_regret = 0.0
        avg_opt_frac = 0.0
        eval_avg_opt_frac = 0.0

        tensorboard_data_dto = TensorboardDataDTO(
            iteration=train_log_dto.iteration, avg_episode_rewards=avg_episode_rewards,
            avg_episode_steps=avg_episode_steps,
            avg_episode_loss=avg_episode_loss, eps=eps, lr=lr, eval=eval,
            eval_avg_episode_rewards=eval_avg_episode_rewards, eval_avg_episode_steps=eval_avg_episode_steps,
            rolling_avg_episode_rewards=rolling_avg_rewards,
            rolling_avg_episode_steps=rolling_avg_steps,
            tensorboard_writer=tensorboard_writer,
            episode_caught_frac=episode_caught_frac,
            episode_early_stopped_frac=episode_early_stopped_frac,
            episode_successful_intrusion_frac=episode_successful_intrusion_frac,
            eval_episode_caught_frac=eval_episode_caught_frac,
            eval_episode_early_stopped_frac=eval_episode_early_stopped_frac,
            eval_episode_successful_intrusion_frac=eval_episode_successful_intrusion_frac,
            avg_regret=avg_regret, avg_opt_frac=avg_opt_frac, rolling_avg_rewards=rolling_avg_rewards,
            rolling_avg_steps=rolling_avg_steps,
            n_af=train_log_dto.n_af, n_d=train_log_dto.n_d, avg_episode_costs=avg_episode_costs,
            avg_episode_costs_norm=avg_episode_costs_norm,
            avg_episode_alerts=avg_episode_alerts, avg_episode_alerts_norm=avg_episode_alerts_norm,
            eval_avg_episode_costs=eval_avg_episode_costs, eval_avg_episode_costs_norm=eval_avg_episode_costs_norm,
            eval_avg_episode_alerts=eval_avg_episode_alerts, eval_avg_episode_alerts_norm=eval_avg_episode_alerts_norm,
            total_num_episodes=train_log_dto.total_num_episodes, avg_eval_regret=avg_eval_regret,
            eval_avg_opt_frac=eval_avg_opt_frac,
            epsilon=defender_agent_config.epsilon, training_time_hours=training_time_hours,
            avg_episode_snort_severe_baseline_rewards=avg_episode_snort_severe_baseline_rewards,
            avg_episode_snort_warning_baseline_rewards=avg_episode_snort_warning_baseline_rewards,
            eval_avg_episode_snort_severe_baseline_rewards=eval_avg_episode_snort_severe_baseline_rewards,
            eval_avg_episode_snort_warning_baseline_rewards=eval_avg_episode_snort_warning_baseline_rewards,
            avg_episode_snort_critical_baseline_rewards=avg_episode_snort_critical_baseline_rewards,
            avg_episode_var_log_baseline_rewards=avg_episode_var_log_baseline_rewards,
            eval_avg_episode_snort_critical_baseline_rewards=eval_avg_episode_snort_critical_baseline_rewards,
            eval_avg_episode_var_log_baseline_rewards=eval_avg_episode_var_log_baseline_rewards,
            avg_flags_catched=avg_episode_flags, avg_episode_flags_percentage=avg_episode_flags_percentage,
            eval_avg_episode_flags=eval_avg_episode_flags,
            eval_avg_episode_flags_percentage=eval_avg_episode_flags_percentage,
            avg_optimal_reward=avg_episode_optimal_rewards,
            avg_optimal_steps=avg_episode_optimal_steps
        )
        log_str = tensorboard_data_dto.log_str_defender()
        defender_agent_config.logger.info(log_str)
        print(log_str)
        sys.stdout.flush()
        if defender_agent_config.tensorboard:
            tensorboard_data_dto.log_tensorboard_defender()

        # Defender specific metrics
        result.defender_avg_episode_rewards.append(avg_episode_rewards)
        result.defender_avg_episode_loss.append(avg_episode_loss)
        result.defender_eval_avg_episode_rewards.append(eval_avg_episode_rewards)
        result.defender_avg_regret.append(avg_regret)
        result.defender_avg_opt_frac.append(avg_opt_frac)
        result.defender_eval_avg_regret.append(avg_eval_regret)
        result.defender_eval_avg_opt_frac.append(eval_avg_opt_frac)
        result.snort_severe_baseline_rewards.append(avg_episode_snort_severe_baseline_rewards)
        result.snort_warning_baseline_rewards.append(avg_episode_snort_warning_baseline_rewards)
        result.eval_snort_severe_baseline_rewards.append(eval_avg_episode_snort_severe_baseline_rewards)
        result.eval_snort_warning_baseline_rewards.append(eval_avg_episode_snort_warning_baseline_rewards)
        result.snort_critical_baseline_rewards.append(avg_episode_snort_critical_baseline_rewards)
        result.var_log_baseline_rewards.append(avg_episode_var_log_baseline_rewards)
        result.eval_snort_critical_baseline_rewards.append(eval_avg_episode_snort_critical_baseline_rewards)
        result.eval_var_log_baseline_rewards.append(eval_avg_episode_var_log_baseline_rewards)


        # General metrics
        if not train_mode == TrainMode.SELF_PLAY:
            result.avg_episode_steps.append(avg_episode_steps)
            result.epsilon_values.append(defender_agent_config.epsilon)
            result.eval_avg_episode_steps.append(eval_avg_episode_steps)
            result.lr_list.append(train_log_dto.defender_lr)
            result.caught_frac.append(episode_caught_frac)
            result.early_stopping_frac.append(episode_early_stopped_frac)
            result.intrusion_frac.append(episode_successful_intrusion_frac)
            result.eval_caught_frac.append(eval_episode_caught_frac)
            result.eval_early_stopping_frac.append(eval_episode_early_stopped_frac)
            result.eval_intrusion_frac.append(eval_episode_successful_intrusion_frac)
            result.attacker_action_costs.append(avg_episode_costs)
            result.attacker_action_costs_norm.append(avg_episode_costs_norm)
            result.attacker_action_alerts.append(avg_episode_alerts)
            result.attacker_action_alerts_norm.append(avg_episode_alerts_norm)
            result.eval_attacker_action_costs.append(eval_avg_episode_costs)
            result.eval_attacker_action_costs_norm.append(eval_avg_episode_costs_norm)
            result.eval_attacker_action_alerts.append(eval_avg_episode_alerts)
            result.eval_attacker_action_alerts_norm.append(eval_avg_episode_alerts_norm)
            result.avg_episode_flags.append(avg_episode_flags)
            result.avg_episode_flags_percentage.append(avg_episode_flags_percentage)
            result.eval_avg_episode_flags.append(eval_avg_episode_flags)
            result.eval_avg_episode_flags_percentage.append(eval_avg_episode_flags_percentage)
            result.optimal_rewards.append(avg_episode_optimal_rewards)
            result.optimal_steps.append(avg_episode_optimal_steps)

            if not eval:
                train_log_dto.train_result = result
            else:
                train_log_dto.eval_result = result
            return train_log_dto

    @staticmethod
    def compute_opt_frac(r: float, opt_r: float) -> float:
        """
        Utility function for computing fraction of optimal reward

        :param r: reward
        :param opt_r: optimal reward
        :return: fraction of optimal reward
        """
        abs_difference = abs(opt_r - r)
        if (r >= 0 and opt_r >= 0) or (r <= 0 and opt_r <= 0):
            return r/opt_r
        elif r < 0 and opt_r > 0:
            return 1/abs_difference
        else:
            return 1/abs_difference

    @staticmethod
    def compute_regret(r: float, opt_r : float) -> float:
        """
        Utility function for computing the regret

        :param r: the reward
        :param opt_r: the optimal reward
        :return: the regret
        """
        return abs(opt_r - r)
