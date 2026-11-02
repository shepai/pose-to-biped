"""
src/G1_tele/policies/pose_tracking.py

Asymmetric actor-critic for PoseTrackingEnv (Stable-Baselines3):

    from G1_tele.policies.pose_tracking import PoseTrackingPolicy
    n_actor = venv.get_attr("n_actor_obs", indices=[0])[0]
    model = PPO(PoseTrackingPolicy, venv, policy_kwargs=dict(actor_obs_dim=n_actor))

The env's observation is [actor part | privileged part]. The actor MLP only receives the first
actor_obs_dim entries (what the real robot can measure); the critic receives everything, including
the privileged balance state (base velocity, centre of mass, capture point margin, foot contacts,
friction, pushes). A critic that can see an imminent fall gives much better advantage estimates,
and the actor stays deployable. With actor_obs_dim=None it is a plain symmetric actor-critic.

Not imported in policies/__init__.py on purpose (it needs stable-baselines3).
"""
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy


def _mlp(in_dim, arch, activation_fn):
    layers = []
    for h in arch:
        layers += [nn.Linear(in_dim, h), activation_fn()]
        in_dim = h
    return nn.Sequential(*layers), in_dim


class AsymmetricMlpExtractor(nn.Module):
    """Drop-in replacement for SB3's MlpExtractor: actor sees features[..., :actor_dim], critic all."""

    def __init__(self, feature_dim, actor_dim, pi_arch, vf_arch, activation_fn):
        super().__init__()
        self.actor_dim = int(actor_dim)
        self.policy_net, self.latent_dim_pi = _mlp(self.actor_dim, pi_arch, activation_fn)
        self.value_net, self.latent_dim_vf = _mlp(feature_dim, vf_arch, activation_fn)

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features):
        return self.policy_net(features[..., :self.actor_dim])

    def forward_critic(self, features):
        return self.value_net(features)


class PoseTrackingPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, actor_obs_dim=None, **kwargs):
        self.actor_obs_dim = actor_obs_dim                  # must exist before super() builds the networks
        kwargs.setdefault("net_arch", dict(pi=[512, 256, 128], vf=[512, 256, 128]))
        kwargs.setdefault("activation_fn", nn.ELU)
        kwargs.setdefault("log_std_init", -1.0)
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build_mlp_extractor(self):
        arch = self.net_arch
        if isinstance(arch, dict):
            pi, vf = list(arch.get("pi", [])), list(arch.get("vf", []))
        else:
            pi = vf = list(arch)
        n_actor = self.features_dim if self.actor_obs_dim is None else int(self.actor_obs_dim)
        if not 0 < n_actor <= self.features_dim:
            raise ValueError(f"actor_obs_dim={n_actor} but the observation has {self.features_dim} entries")
        self.mlp_extractor = AsymmetricMlpExtractor(self.features_dim, n_actor, pi, vf, self.activation_fn)

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data["actor_obs_dim"] = self.actor_obs_dim
        return data
