import os
from gym_optimal_intrusion_response.agents.config.agent_config import AgentConfig
from gym_optimal_intrusion_response.dao.experiment.client_config import ClientConfig
from gym_optimal_intrusion_response.dao.agent.agent_type import AgentType
from gym_optimal_intrusion_response.util import experiments_util
from gym_optimal_intrusion_response.dao.experiment.runner_mode import RunnerMode
from gym_optimal_intrusion_response.dao.agent.train_mode import TrainMode


def default_config() -> ClientConfig:
    """
    :return: Default configuration for the experiment
    """
    agent_config = AgentConfig(gamma=1, alpha=0.0005, epsilon=1, render=False,
                               min_epsilon=0.01, eval_episodes=10, train_log_frequency=1,
                               epsilon_decay=0.9999, video=False, eval_log_frequency=1,
                               video_fps=5, video_dir=experiments_util.default_output_dir() + "/results/videos",
                               num_iterations=1000,
                               gif_dir=experiments_util.default_output_dir() + "/results/gifs",
                               save_dir=experiments_util.default_output_dir() + "/results/data",
                               checkpoint_freq=25, input_dim=2,
                               output_dim=2,
                               pi_hidden_dim=64, pi_hidden_layers=1,
                               vf_hidden_dim=64, vf_hidden_layers=1,
                               shared_hidden_layers=2, shared_hidden_dim=64,
                               batch_size=2000,
                               gpu=False, tensorboard=True,
                               tensorboard_dir=experiments_util.default_output_dir() + "/results/tensorboard",
                               optimizer="Adam", lr_exp_decay=False, lr_decay_rate=0.999,
                               state_length=1, gpu_id=0, sde_sample_freq=4, use_sde=False,
                               lr_progress_decay=False, lr_progress_power_decay=4, ent_coef=0.0005,
                               vf_coef=0.5, features_dim=512, gae_lambda=0.95, max_gradient_norm=0.5,
                               eps_clip=0.2, optimization_iterations=10,
                               render_steps=100, illegal_action_logit=-1000,
                               filter_illegal_actions=True,
                               running_avg=50,  eval_deterministic=False
                               )
    env_name = "optimal-intrusion-response-v3"

    client_config = ClientConfig(env_name=env_name, defender_agent_config=agent_config,
                                 agent_type=AgentType.PPO_BASELINE.value,
                                 output_dir=experiments_util.default_output_dir(),
                                 title="Optimal Intrusion Response V3",
                                 run_many=True, random_seeds=[0, 999, 299],
                                 random_seed=299,
                                 mode=RunnerMode.TRAIN_ATTACKER.value,train_mode=TrainMode.TRAIN_DEFENDER,
                                 )
    return client_config


def write_default_config(path:str = None) -> None:
    """
    Writes the default configuration to a json file

    :param path: the path to write the configuration to
    :return: None
    """
    if path is None:
        path = experiments_util.default_config_path()
    config = default_config()
    experiments_util.write_config_file(config, path)


# Program entrypoint
if __name__ == '__main__':

    # Setup
    args = experiments_util.parse_args(experiments_util.default_config_path())
    experiment_title = "Optimal Intrusion Response V1"
    if args.configpath is not None and not args.noconfig:
        if not os.path.exists(args.configpath):
            write_default_config()
        config = experiments_util.read_config(args.configpath)
    else:
        config = default_config()

    train_csv_path, eval_csv_path = experiments_util.run_experiment(config, config.random_seed)