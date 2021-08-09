import importlib
from typing import Dict, List, Optional

from addict import Dict
from airflow.operators.python import PythonOperator
import yaml


class CustomPythonOperator(PythonOperator):
    def __init__(
        self,
        config_path,
        op_args: Optional[List] = None,
        op_kwargs: Optional[Dict] = None,
        templates_dict: Optional[Dict] = None,
        templates_exts: Optional[List[str]] = None,
        **kwargs
    ):
        self.templates_dict = templates_dict
        self.parse_config(config_path)
        self.kwargs.update({"task_id": kwargs["task_id"]})
        super().__init__(**self.kwargs)

    def parse_config(self, config_path):
        with open(config_path, "r") as f:
            cfg = Dict(yaml.safe_load(f))
        tmp = cfg.callable.split(".")
        module_name = ".".join(tmp[:-1])
        callable_name = tmp[-1]
        module = importlib.import_module(module_name)
        python_callable = getattr(module, callable_name)
        self.python_callable = python_callable
        self.op_kwargs = cfg["kwargs"]
        self.op_args = []
        self.kwargs = {
            "op_kwargs": self.op_kwargs,
            "op_args": self.op_args,
            "python_callable": self.python_callable,
            "templates_dict": self.templates_dict,
        }
