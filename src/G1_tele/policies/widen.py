"""
Grow a trained policy's input layer for a bigger observation, without changing what it does.

PoseTrackingEnv(locomotion=True) inserts new entries after the old actor part:

    old obs  [ pose actor part (n_a)                   | privileged (n_p) ]
    new obs  [ pose actor part (n_a) | locomotion (n_l) | privileged (n_p) ]

The first layer of the actor and of the critic get zero weights for the new entries, so the widened
network gives exactly the old outputs whatever the new entries hold; training then learns to use them.
The normalisation statistics (VecNormalize) are widened the same way. Everything here works on numpy
arrays, so it can be checked without torch; train_locomotion.py converts to and from torch tensors.
"""
import numpy as np

ACTOR_IN = "mlp_extractor.policy_net.0.weight"
CRITIC_IN = "mlp_extractor.value_net.0.weight"


def obs_index_map(old_n_actor, old_n_obs, new_n_actor, new_n_obs):
    """New position of every old observation entry."""
    if new_n_obs - new_n_actor != old_n_obs - old_n_actor:
        raise ValueError("the privileged part must keep its size")
    if new_n_actor < old_n_actor:
        raise ValueError("the new actor part is smaller than the old one")
    m = np.arange(old_n_obs)
    m[old_n_actor:] += new_n_actor - old_n_actor
    return m


def widen_columns(W, new_in, col_map):
    W = np.asarray(W)
    out = np.zeros((W.shape[0], new_in), W.dtype)
    out[:, col_map] = W
    return out


def widen_state_dict(sd, old_n_actor, old_n_obs, new_n_actor, new_n_obs):
    """sd: {name: numpy array} of a PoseTrackingPolicy. Returns a new dict for the bigger observation."""
    m = obs_index_map(old_n_actor, old_n_obs, new_n_actor, new_n_obs)
    out = {k: np.array(v, copy=True) for k, v in sd.items()}
    if out[ACTOR_IN].shape[1] != old_n_actor or out[CRITIC_IN].shape[1] != old_n_obs:
        raise ValueError(f"this policy's input sizes ({out[ACTOR_IN].shape[1]}, {out[CRITIC_IN].shape[1]}) are not "
                         f"({old_n_actor}, {old_n_obs})")
    out[ACTOR_IN] = widen_columns(out[ACTOR_IN], new_n_actor, m[:old_n_actor])
    out[CRITIC_IN] = widen_columns(out[CRITIC_IN], new_n_obs, m)
    return out


def widen_stats(mean, var, old_n_actor, old_n_obs, new_n_actor, new_n_obs, new_mean=None, new_var=None):
    """Observation mean/var for the bigger observation: old entries keep their statistics, new entries
    get new_mean/new_var (length new_n_actor - old_n_actor; default 0 and 1)."""
    m = obs_index_map(old_n_actor, old_n_obs, new_n_actor, new_n_obs)
    k = new_n_actor - old_n_actor
    mu, va = np.zeros(new_n_obs), np.ones(new_n_obs)
    mu[m], va[m] = mean, var
    if new_mean is not None:
        mu[old_n_actor:new_n_actor] = np.asarray(new_mean, float).reshape(k)
    if new_var is not None:
        va[old_n_actor:new_n_actor] = np.maximum(np.asarray(new_var, float).reshape(k), 1e-4)
    return mu, va


def floor_log_std(log_std, min_std):
    """Raise any action noise below min_std to min_std (more exploration for the new skill)."""
    if not min_std:
        return np.asarray(log_std)
    return np.maximum(np.asarray(log_std), np.log(min_std))
