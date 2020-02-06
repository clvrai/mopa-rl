from .mlp_actor_critic import MlpActor, MlpCritic
from .cnn_actor_critic import CNNActor, CNNCritic


def get_actor_critic_by_name(name, ae=False):
    if name == 'mlp':
        return MlpActor, MlpCritic
    elif name == 'cnn' and not ae:
        return CNNActor, CNNCritic
    elif name == 'cnn' and ae:
        return MlpActor, MlpCritic
    else:
        raise NotImplementedError()

