
SAMPLER_REGISTRY = {}

def register_sampler(name):
    def wrapper(fn):
        SAMPLER_REGISTRY[name.lower()] = fn
        return fn
    return wrapper


def get_sampler(name):
    name = name.lower()
    if name not in SAMPLER_REGISTRY:
        raise ValueError(f"Sampler '{name}' is not registered.")
    return SAMPLER_REGISTRY[name]










