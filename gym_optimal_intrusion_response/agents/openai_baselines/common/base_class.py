"""Abstract base classes for RL algorithms."""

import io
import pathlib
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th

from gym_optimal_intrusion_response.agents.openai_baselines.common.callbacks import BaseCallback, CallbackList, ConvertCallback, EvalCallback
from gym_optimal_intrusion_response.agents.openai_baselines.common.policies import BasePolicy, get_policy_from_name
from gym_optimal_intrusion_response.agents.openai_baselines.common.save_util import load_from_zip_file, recursive_getattr, recursive_setattr, save_to_zip_file
from gym_optimal_intrusion_response.agents.openai_baselines.common.type_aliases import GymEnv, MaybeCallback, Schedule
from gym_optimal_intrusion_response.dao.experiment.experiment_result import ExperimentResult
from stable_baselines3.common.utils import (
    check_for_correct_spaces,
    get_device,
    set_random_seed,
)
from gym_optimal_intrusion_response.agents.openai_baselines.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    unwrap_vec_normalize,
)

from gym_optimal_intrusion_response.dao.agent.train_mode import TrainMode
from gym_optimal_intrusion_response.agents.config.agent_config import AgentConfig
from gym_optimal_intrusion_response.dao.agent.train_agent_log_dto import TrainAgentLogDTO
from gym_optimal_intrusion_response.agents.util.log_util import LogUtil

def maybe_make_env(env: Union[GymEnv, str, None]) -> Optional[GymEnv]:
    """If env is a string, make the environment; otherwise, return env.

    :param env: The environment to learn from.
    :return A Gym (vector) environment.
    """
    if isinstance(env, str):
        env = gym.make(env)
    return env


class BaseAlgorithm(ABC):

    def __init__(
        self,
        attacker_policy: Type[BasePolicy],
        defender_policy: Type[BasePolicy],
        env: Union[GymEnv, str, None],
        policy_base: Type[BasePolicy],
        attacker_learning_rate: Union[float, Schedule],
        defender_learning_rate: Union[float, Schedule],
        attacker_policy_kwargs: Dict[str, Any] = None,
        defender_policy_kwargs: Dict[str, Any] = None,
        tensorboard_log: Optional[str] = None,
        device: Union[th.device, str] = "auto",
        seed: Optional[int] = None,
        train_mode: TrainMode = TrainMode.TRAIN_ATTACKER,
        attacker_agent_config: AgentConfig = None,
        defender_agent_config: AgentConfig = None
    ):

        if isinstance(attacker_policy, str) and policy_base is not None:
            self.attacker_policy_class = get_policy_from_name(policy_base, attacker_policy)
        else:
            self.attacker_policy_class = attacker_policy

        if isinstance(defender_policy, str) and policy_base is not None:
            self.defender_policy_class = get_policy_from_name(policy_base, defender_policy)
        else:
            self.defender_policy_class = defender_policy

        self.device = get_device(device)
        self.env = None
        self._vec_normalize_env = unwrap_vec_normalize(env)
        self.attacker_policy_kwargs = {} if attacker_policy_kwargs is None else attacker_policy_kwargs
        self.defender_policy_kwargs = {} if defender_policy_kwargs is None else defender_policy_kwargs
        self.attacker_observation_space = None
        self.attacker_action_space = None
        self.defender_observation_space = None
        self.defender_action_space = None
        self.n_envs = None
        self.num_timesteps = 0
        # Used for updating schedules
        self._total_timesteps = 0
        self.seed = seed
        self.start_time = None
        self.attacker_policy = None
        self.defender_policy = None
        self.attacker_learning_rate = attacker_learning_rate
        self.defender_learning_rate = defender_learning_rate
        self.tensorboard_log = tensorboard_log
        self._last_obs = None
        self._last_episode_starts = None
        self._last_original_obs = None
        self._episode_num = 0
        self._current_progress_remaining = 1
        self.ep_info_buffer = None
        self.ep_success_buffer = None
        self._n_updates = 0
        self.attacker_agent_config = attacker_agent_config
        self.defender_agent_config = defender_agent_config
        self.train_mode = train_mode
        self.train_result = ExperimentResult()
        self.eval_result = ExperimentResult()
        self.training_start = time.time()

        # Create and wrap the env if needed
        if env is not None:
            env = maybe_make_env(env)
            env = self._wrap_env(env)
            self.attacker_observation_space = env.observation_space
            self.attacker_action_space = env.action_space
            self.n_envs = env.num_envs
            self.env = env

    @staticmethod
    def _wrap_env(env: GymEnv) -> VecEnv:
        if not isinstance(env, VecEnv):
            env = DummyVecEnv([lambda: env])
        return env

    @abstractmethod
    def _setup_model(self) -> None:
        """Create networks, buffer and optimizers."""

    def _update_current_progress_remaining(self, num_timesteps: int, total_timesteps: int) -> None:
        """
        Compute current progress remaining (starts from 1 and ends to 0)

        :param num_timesteps: current number of timesteps
        :param total_timesteps:
        """
        self._current_progress_remaining = 1.0 - float(num_timesteps) / float(total_timesteps)

    def _excluded_save_params(self) -> List[str]:
        """
        Returns the names of the parameters that should be excluded from being
        saved by pickling. E.g. replay buffers are skipped by default
        as they take up a lot of space. PyTorch variables should be excluded
        with this so they can be stored with ``th.save``.

        :return: List of parameters that should be excluded from being saved with pickle.
        """
        return [
            "policy",
            "device",
            "env",
            "replay_buffer",
            "rollout_buffer",
            "_vec_normalize_env",
        ]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy"]
        return state_dicts, []

    def _init_callback(
        self,
        callback: MaybeCallback
    ) -> BaseCallback:
        # Convert a list of callbacks into a callback
        if isinstance(callback, list):
            callback = CallbackList(callback)

        # Convert functional callback to object
        if not isinstance(callback, BaseCallback):
            callback = ConvertCallback(callback)

        callback.init_callback(self)
        return callback

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
    ) -> Tuple[int, BaseCallback]:

        self.start_time = time.time()
        if self.ep_info_buffer is None or reset_num_timesteps:
            # Initialize buffers if they don't exist, or reinitialize if resetting counters
            self.ep_info_buffer = deque(maxlen=100)
            self.ep_success_buffer = deque(maxlen=100)

        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
        else:
            # Make sure training timesteps are ahead of the internal counter
            total_timesteps += self.num_timesteps
        self._total_timesteps = total_timesteps

        # Avoid resetting the environment when calling ``.learn()`` consecutive times
        if reset_num_timesteps or self._last_obs is None:
            self._last_obs = self.env.reset()
            self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)
            # Retrieve unnormalized observation for saving into the buffer
            if self._vec_normalize_env is not None:
                self._last_original_obs = self._vec_normalize_env.get_original_obs()

        # Create eval callback if needed
        callback = self._init_callback(callback, eval_freq, n_eval_episodes, log_path)

        return total_timesteps, callback

    def get_env(self) -> Optional[VecEnv]:
        """
        Returns the current environment (can be None if not defined).

        :return: The current environment
        """
        return self.env

    def set_env(self, env: GymEnv) -> None:
        env = self._wrap_env(env)
        self.n_envs = env.num_envs
        self.env = env

    @abstractmethod
    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 100,
        tb_log_name: str = "run",
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "BaseAlgorithm":
        """
        Return a trained model.

        :param total_timesteps: The total number of samples (env steps) to train on
        :param callback: callback(s) called at every step with state of the algorithm.
        :param log_interval: The number of timesteps before logging.
        :param tb_log_name: the name of the run for TensorBoard logging
        :param eval_freq: Evaluate the agent every ``eval_freq`` timesteps (this may vary a little)
        :param n_eval_episodes: Number of episode to evaluate the agent
        :param eval_log_path: Path to a folder where the evaluations will be saved
        :param reset_num_timesteps: whether or not to reset the current timestep number (used in logging)
        :return: the trained model
        """

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
        attacker : bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if attacker:
            return self.attacker_policy.predict(observation, state, mask, deterministic, attacker=attacker)
        else:
            return self.defender_policy.predict(observation, state, mask, deterministic, attacker=attacker)

    def set_random_seed(self, seed: Optional[int] = None) -> None:
        """
        Set the seed of the pseudo-random generators
        (python, numpy, pytorch, gym, action_space)

        :param seed:
        """
        if seed is None:
            return
        set_random_seed(seed, using_cuda=self.device.type == th.device("cuda").type)
        self.attacker_action_space.seed(seed)
        if self.env is not None:
            self.env.seed(seed)

    def set_parameters(
        self,
        load_path_or_dict: Union[str, Dict[str, Dict]],
        exact_match: bool = True,
        device: Union[th.device, str] = "auto",
    ) -> None:
        """
        Load parameters from a given zip-file or a nested dictionary containing parameters for
        different modules (see ``get_parameters``).

        :param load_path_or_iter: Location of the saved data (path or file-like, see ``save``), or a nested
            dictionary containing nn.Module parameters used by the policy. The dictionary maps
            object names to a state-dictionary returned by ``torch.nn.Module.state_dict()``.
        :param exact_match: If True, the given parameters should include parameters for each
            module and each of their parameters, otherwise raises an Exception. If set to False, this
            can be used to update only specific parameters.
        :param device: Device on which the code should run.
        """
        params = None
        if isinstance(load_path_or_dict, dict):
            params = load_path_or_dict
        else:
            _, params, _ = load_from_zip_file(load_path_or_dict, device=device)

        # Keep track which objects were updated.
        # `_get_torch_save_params` returns [params, other_pytorch_variables].
        # We are only interested in former here.
        objects_needing_update = set(self._get_torch_save_params()[0])
        updated_objects = set()

        for name in params:
            attr = None
            try:
                attr = recursive_getattr(self, name)
            except Exception:
                # What errors recursive_getattr could throw? KeyError, but
                # possible something else too (e.g. if key is an int?).
                # Catch anything for now.
                raise ValueError(f"Key {name} is an invalid object name.")

            if isinstance(attr, th.optim.Optimizer):
                # Optimizers do not support "strict" keyword...
                # Seems like they will just replace the whole
                # optimizer state with the given one.
                # On top of this, optimizer state-dict
                # seems to change (e.g. first ``optim.step()``),
                # which makes comparing state dictionary keys
                # invalid (there is also a nesting of dictionaries
                # with lists with dictionaries with ...), adding to the
                # mess.
                #
                # TL;DR: We might not be able to reliably say
                # if given state-dict is missing keys.
                #
                # Solution: Just load the state-dict as is, and trust
                # the user has provided a sensible state dictionary.
                attr.load_state_dict(params[name])
            else:
                # Assume attr is th.nn.Module
                attr.load_state_dict(params[name], strict=exact_match)
            updated_objects.add(name)

        if exact_match and updated_objects != objects_needing_update:
            raise ValueError(
                "Names of parameters do not match agents' parameters: "
                f"expected {objects_needing_update}, got {updated_objects}"
            )

    @classmethod
    def load(
        cls,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        env: Optional[GymEnv] = None,
        device: Union[th.device, str] = "auto",
        custom_objects: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> "BaseAlgorithm":
        """
        Load the model from a zip-file

        :param path: path to the file (or a file-like) where to
            load the agent from
        :param env: the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model) has priority over any saved environment
        :param device: Device on which the code should run.
        :param custom_objects: Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            ``keras.models.load_model``. Useful when you have an object in
            file that can not be deserialized.
        :param kwargs: extra arguments to change the model when loading
        """
        data, params, pytorch_variables = load_from_zip_file(path, device=device, custom_objects=custom_objects)

        # Remove stored device information and replace with ours
        if "policy_kwargs" in data:
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]

        if "policy_kwargs" in kwargs and kwargs["policy_kwargs"] != data["policy_kwargs"]:
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs."
                f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
            )

        if "observation_space" not in data or "action_space" not in data:
            raise KeyError("The observation_space and action_space were not given, can't verify new environments")

        if env is not None:
            # Wrap first if needed
            env = cls._wrap_env(env, data["verbose"])
            # Check if given env is valid
            check_for_correct_spaces(env, data["observation_space"], data["action_space"])
        else:
            # Use stored env, if one exists. If not, continue as is (can be used for predict)
            if "env" in data:
                env = data["env"]

        # noinspection PyArgumentList
        model = cls(  # pytype: disable=not-instantiable,wrong-keyword-args
            policy=data["policy_class"],
            env=env,
            device=device,
            _init_setup_model=False,  # pytype: disable=not-instantiable,wrong-keyword-args
        )

        # load parameters
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()

        # put state_dicts back in place
        model.set_parameters(params, exact_match=True, device=device)

        # put other pytorch variables back in place
        if pytorch_variables is not None:
            for name in pytorch_variables:
                # Set the data attribute directly to avoid issue when using optimizers
                # See https://github.com/DLR-RM/stable-baselines3/issues/391
                recursive_setattr(model, name + ".data", pytorch_variables[name].data)

        # Sample gSDE exploration matrix, so it uses the right device
        # see issue #44
        if model.use_sde:
            model.attacker_policy.reset_noise()  # pytype: disable=attribute-error
        return model

    def get_parameters(self) -> Dict[str, Dict]:
        """
        Return the parameters of the agent. This includes parameters from different networks, e.g.
        critics (value functions) and policies (pi functions).

        :return: Mapping of from names of the objects to PyTorch state-dicts.
        """
        state_dicts_names, _ = self._get_torch_save_params()
        params = {}
        for name in state_dicts_names:
            attr = recursive_getattr(self, name)
            # Retrieve state dict
            params[name] = attr.state_dict()
        return params

    def save(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        exclude: Optional[Iterable[str]] = None,
        include: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Save all the attributes of the object and the model parameters in a zip-file.

        :param path: path to the file where the rl agent should be saved
        :param exclude: name of parameters that should be excluded in addition to the default ones
        :param include: name of parameters that might be excluded but should be included anyway
        """
        # Copy parameter list so we don't mutate the original dict
        data = self.__dict__.copy()

        # Exclude is union of specified parameters (if any) and standard exclusions
        if exclude is None:
            exclude = []
        exclude = set(exclude).union(self._excluded_save_params())

        # Do not exclude params if they are specifically included
        if include is not None:
            exclude = exclude.difference(include)

        state_dicts_names, torch_variable_names = self._get_torch_save_params()
        all_pytorch_variables = state_dicts_names + torch_variable_names
        for torch_var in all_pytorch_variables:
            # We need to get only the name of the top most module as we'll remove that
            var_name = torch_var.split(".")[0]
            # Any params that are in the save vars must not be saved by data
            exclude.add(var_name)

        # Remove parameter entries of parameters which are to be excluded
        for param_name in exclude:
            data.pop(param_name, None)

        # Build dict of torch variables
        pytorch_variables = None
        if torch_variable_names is not None:
            pytorch_variables = {}
            for name in torch_variable_names:
                attr = recursive_getattr(self, name)
                pytorch_variables[name] = attr

        # Build dict of state_dicts
        params_to_save = self.get_parameters()

        save_to_zip_file(path, data=data, params=params_to_save, pytorch_variables=pytorch_variables)


    def save_model(self, iteration : int = 1) -> None:
        """
        Saves the PyTorch Model Weights

        :return: None
        """
        time_str = str(time.time())
        if self.train_mode == TrainMode.TRAIN_ATTACKER or self.train_mode == TrainMode.SELF_PLAY:
            save_dir = self.attacker_agent_config.save_dir
            seed = self.attacker_agent_config.random_seed
        else:
            save_dir = self.defender_agent_config.save_dir
            seed = self.defender_agent_config.random_seed

        if save_dir is not None:
            path = save_dir + "/" + time_str + "_" + str(seed)  + "_" + str(iteration) + "_policy_network.zip"
            env_config = None
            env_configs = None
            eval_env_config = None
            eval_env_configs = None
            if self.attacker_agent_config is not None:
                self.attacker_agent_config.logger.info("Saving policy-network to: {}".format(path))
                env_config = self.attacker_agent_config.env_config
                env_configs = self.attacker_agent_config.env_configs
                eval_env_config = self.attacker_agent_config.eval_env_config
                eval_env_configs = self.attacker_agent_config.eval_env_configs
                self.attacker_agent_config.env_config = None
                self.attacker_agent_config.env_configs = None
                self.attacker_agent_config.eval_env_config = None
                self.attacker_agent_config.eval_env_configs = None
            if self.defender_agent_config is not None:
                self.defender_agent_config.logger.info("Saving policy-network to: {}".format(path))
                if self.defender_agent_config.env_config is not None:
                    env_config = self.defender_agent_config.env_config
                    env_configs = self.defender_agent_config.env_configs
                    eval_env_config = self.defender_agent_config.eval_env_config
                    eval_env_configs = self.defender_agent_config.eval_env_configs
                    self.defender_agent_config.env_config = None
                    self.defender_agent_config.env_configs = None
                    self.defender_agent_config.eval_env_config = None
                    self.defender_agent_config.eval_env_configs = None
            self.save(path, exclude=["tensorboard_writer", "eval_env", "env_2", "env"])
            if self.attacker_agent_config is not None:
                self.attacker_agent_config.env_config = env_config
                self.attacker_agent_config.env_configs = env_configs
                self.attacker_agent_config.eval_env_config = eval_env_config
                self.attacker_agent_config.eval_env_configs = eval_env_configs
            if self.defender_agent_config is not None:
                self.defender_agent_config.env_config = env_config
                self.defender_agent_config.env_configs = env_configs
                self.defender_agent_config.eval_env_config = eval_env_config
                self.defender_agent_config.eval_env_configs = eval_env_configs
        else:
            if self.attacker_agent_config is not None:
                self.attacker_agent_config.logger.warning("Save path not defined, not saving policy-networks to disk")
                print("Save path not defined, not saving policy-networks to disk")
            if self.defender_agent_config is not None:
                self.defender_agent_config.logger.warning("Save path not defined, not saving policy-networks to disk")
                print("Save path not defined, not saving policy-networks to disk")


    def log_metrics_attacker(self, train_log_dto: TrainAgentLogDTO, eps: float = None, eval: bool = False) \
            -> TrainAgentLogDTO:
        """
        Logs average metrics for the last <self.config.log_frequency> episodes

        :param train_log_dto: DTO with the information to log
        :param eps: machine eps
        :param eval: flag whether it is evaluation or not
        :return: the updated train agent log dto
        """
        return LogUtil.log_metrics_attacker(train_log_dto=train_log_dto, eps=eps, eval=eval,
                                     attacker_agent_config=self.attacker_agent_config, env=self.env,
                                     env_2=self.env_2, tensorboard_writer=self.tensorboard_writer)

    def log_metrics_defender(self, train_log_dto: TrainAgentLogDTO, eps: float = None, eval: bool = False) \
            -> TrainAgentLogDTO:
        """
        Logs average metrics for the last <self.config.log_frequency> episodes

        :param train_log_dto: DTO with the information to log
        :param eps: machine eps
        :param eval: flag whether it is evaluation or not
        :return: the updated train agent log dto
        """
        return LogUtil.log_metrics_defender(train_log_dto=train_log_dto, eps=eps, eval=eval, env=self.env,
                                     env_2=self.env_2, defender_agent_config=self.defender_agent_config,
                                     tensorboard_writer=self.tensorboard_writer)