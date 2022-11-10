import yaml
from scipy.stats import loguniform, uniform, randint


def load_config(config_path):
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    return config


def parse_config(config, use_gpu) -> dict:
    dist = {"i": randint, "l": loguniform, "f": uniform}
    params = {}
    print(config)
    for k, v in config.items():
        if v["type"] == "c":
            params[f"{k}" if use_gpu else f"{k}"] = v["choices"]
        else:
            params[f"{k}" if use_gpu else f"{k}"] = dist[v["type"]](v["lo"], v["hi"])
    return params