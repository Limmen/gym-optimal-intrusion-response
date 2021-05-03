"""
Register OpenAI Envs
"""
from gym.envs.registration import register

# -------- Difficulty Version: V1 ------------
register(
    id='optimal-intrusion-response-v1',
    entry_point='gym_optimal_intrusion_response.envs.derived_envs.optimal_intrusion_response_env_v1:OptimalIntrusionResponseEnvV1',
    kwargs={}
)