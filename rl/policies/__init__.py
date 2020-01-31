from .mlp_actor_critic import MlpActor, MlpCritic


def get_actor_critic_by_name(name):
    if name == 'mlp':
        return MlpActor, MlpCritic
    elif name == 'cnn':
        raise NotImplementedError()
    else:
        raise NotImplementedError()

