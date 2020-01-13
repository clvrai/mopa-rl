from .mlp_actor_critic import MlpActor, MlpCritic
from .manual_subdiv_actor import ManualSubdivActor


def get_actor_critic_by_name(name):
    if name == 'mlp':
        return MlpActor, MlpCritic
    elif name == 'manual':
        return ManualSubdivActor, MlpCritic
    else:
        raise NotImplementedError()

