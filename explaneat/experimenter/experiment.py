from jsonschema import validate
from explaneat.experimenter.schemas.experiment import experiment as EXPERIMENT_SCHEMA
from pathlib import Path
import json

class GenericExperiment(object):



    def __init__(self, config):
        if Path(config).is_file():
            with open(config, 'r') as fp:
                self.config = json.load(fp)
        else:
            raise FileNotFoundError("Config file not found")


    @property
    def config(self):
        return self._config
    
    @config.setter
    def config(self, configuration):
        validate(configuration, EXPERIMENT_SCHEMA)
        self._config = configuration
    
    @config.getter
    def config(self):
        return self._config

    def _validate_configuration(self):
        validate(self._config, EXPERIMENT_SCHEMA)

    # def _create_experiment_folders(self):
        