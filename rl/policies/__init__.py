from .mlp_actor_critic import MlpActor, MlpCritic
from .cnn_actor_critic import CNNActor, CNNCritic


def get_actor_critic_by_name(name):
    if name == 'mlp':
        return MlpActor, MlpCritic
    elif name == 'cnn':
        return CNNActor, CNNCritic
    else:
        raise NotImplementedError()

