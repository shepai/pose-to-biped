"""
src/G1_tele/policies/hand_tracking.py
 
Stable-Baselines3 policy class for the hand-tracking task, used exactly like the custom
policy in your other project:
 
    from G1_tele.policies.hand_tracking import HandTrackingPolicy
    model = PPO(HandTrackingPolicy, env, ...)
    model = PPO.load("models/hand_tracking", device="cpu")     # reloads fine, class is stored by reference
 
Not imported in policies/__init__.py on purpose (it needs stable-baselines3, and the
keyboard example shouldn't).
 
It is a plain actor-critic MLP with defaults suited to joint-target control: separate
256-128-128 ELU networks for actor and critic and a small initial action std, so the
new policy starts close to "do nothing" and does not destabilise the frozen leg policy.
Override any of them with policy_kwargs=dict(net_arch=..., log_std_init=..., ...).
"""
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
 
 
class HandTrackingPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("net_arch", dict(pi=[256, 128, 128], vf=[256, 128, 128]))
        kwargs.setdefault("activation_fn", nn.ELU)
        kwargs.setdefault("log_std_init", -1.0)
        super().__init__(*args, **kwargs)
 