import importlib
import sys

from addict import Dict
import yaml


if __name__ == "__main__":
    print("ONLY FOR TEST")
    path = sys.argv[1]
    with open(path, "r") as f:
        cfg = Dict(yaml.safe_load(f))
    tmp = cfg.callable.split(".")
    module_name = ".".join(tmp[:-1])
    callable_name = tmp[-1]
    module = importlib.import_module(module_name)
    func = getattr(module, callable_name)
    func(**cfg["kwargs"])
