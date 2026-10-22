
from gymnasium.envs.registration import register

register(
    id="G1-walker",
    entry_point="G1_tele.envs.walker:G1Env",
)

register(
    id="G1-hand-tracking",
    entry_point="G1_tele.envs.hand_tracking:HandTrackingEnv",     # generic: pass kwargs at make()
)