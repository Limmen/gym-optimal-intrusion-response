"""
Client configuration for running experiments (parsed from JSON)
"""
from gym_optimal_intrusion_response.agents.config.agent_config import AgentConfig
from gym_optimal_intrusion_response.dao.experiment.runner_mode import RunnerMode
from gym_optimal_intrusion_response.dao.agent.train_mode import TrainMode

class ClientConfig:
    """
    DTO with client config for running experiments
    """

    def __init__(self, env_name:str,
                 attacker_agent_config: AgentConfig = None,
                 defender_agent_config: AgentConfig = None,
                 output_dir:str = None, title = None,
                 env_config = None, run_many :bool = False,
                 random_seeds : list = None, random_seed = 0,
                 agent_type : int = 0,
                 env_checkpoint_dir : str = None,
                 mode: RunnerMode = RunnerMode.TRAIN_ATTACKER,
                 eval_env: bool = None,
                 n_envs : int = 1,
                 train_mode: TrainMode = TrainMode.TRAIN_ATTACKER
                 ):
        self.env_name = env_name
        self.logger = None
        self.output_dir = output_dir
        self.title = title
        self.env_config = env_config
        self.run_many = run_many
        self.random_seeds = random_seeds
        self.random_seed = random_seed
        self.attacker_agent_config = attacker_agent_config
        self.defender_agent_config = defender_agent_config
        self.agent_type = agent_type
        self.env_checkpoint_dir = env_checkpoint_dir
        self.mode = mode
        self.eval_env = eval_env
        self.n_envs = n_envs
        self.train_mode = train_mode