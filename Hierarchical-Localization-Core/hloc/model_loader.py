from . import extractors, logger
from . import logger, matchers
from .utils.base_model import dynamic_load
import torch 

class ModelLocalLoader:
    def __init__(self):
        self.models = {}
    def load_model(self,conf):
        """Load and cache models."""
        model_name = conf["model"]["name"]
        if model_name not in self.models:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            Model = dynamic_load(extractors, model_name)
            self.models[model_name] = Model(conf["model"]).eval().to(device)
        return self.models[model_name]
class ModelGlobalLoader:
    def __init__(self):
        self.models = {}
    def load_model(self,conf):
        """Load and cache models."""
        model_name = conf["model"]["name"]
        if model_name not in self.models:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            Model = dynamic_load(extractors, model_name)
            self.models[model_name] = Model(conf["model"]).eval().to(device)
        return self.models[model_name]
class ModelMatchLoader():
    def __init__(self):
        self.models = {}
    def load_model(self,conf):
        """Load and cache models."""
        model_name = conf["model"]["name"]
        if model_name not in self.models:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            Model = dynamic_load(matchers, conf["model"]["name"])
            self.models[model_name] = Model(conf["model"]).eval().to(device)
        return self.models[model_name]

model_local=ModelLocalLoader()
model_global=ModelGlobalLoader()
model_match=ModelMatchLoader()