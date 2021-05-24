"""
Register OpenAI Envs
"""
from gym.envs.registration import register

# -------- Difficulty Version: V1 ------------
register(
    id='optimal-intrusion-response-v1',
    entry_point='gym_optimal_intrusion_response.envs.derived_envs.optimal_intrusion_response_env_v1:OptimalIntrusionResponseEnvV1',
    kwargs={"traces_dir": "", "traces_filename": ""}
)


# -------- Difficulty Version: V2 ------------
register(
    id='optimal-intrusion-response-v2',
    entry_point='gym_optimal_intrusion_response.envs.derived_envs.optimal_intrusion_response_env_v2:OptimalIntrusionResponseEnvV2',
    kwargs={"traces_dir": "", "traces_filename": ""}
)

# -------- Difficulty Version: V3 ------------
register(
    id='optimal-intrusion-response-v3',
    entry_point='gym_optimal_intrusion_response.envs.derived_envs.optimal_intrusion_response_env_v3:OptimalIntrusionResponseEnvV3',
    kwargs={"traces_dir": "", "traces_filename": ""}
)