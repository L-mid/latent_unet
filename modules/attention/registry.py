
ATTENTION_REGISTRY = {}

def register_attention(name):
    def decorator(cls):
        ATTENTION_REGISTRY[name] = cls
        return cls
    return decorator

def get_attention(cfg):
    name = cfg.kind
    if name not in ATTENTION_REGISTRY:
        raise ValueError(f"Unknown attention type: {name}")
    params = cfg.get("params", {})
    return ATTENTION_REGISTRY[name](**params)
