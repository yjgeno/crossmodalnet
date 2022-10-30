import yaml
from scipy.stats import loguniform, uniform, randint


def load_config(config_path):
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    return config


def parse_config(config) -> dict:
    dist = {"i": randint, "l": loguniform, "f": uniform}
    params = {}
    for k, v in config.items():
        if v["type"] == "c":
            params[f"estimator__{k}"] = v["choices"]
        params[f"estimator__{k}"] = dist[v["type"]](v["lo"], v["hi"])
    return params