from jsonschema import validate
from explaneat.experimenter.schemas.experiment import experiment as EXPERIMENT_SCHEMA
from pathlib import Path
import json
import logging

LOGGING_LEVEL = logging.INFO

logger = logging.getLogger("experimenter")
logger.setLevel(LOGGING_LEVEL)

ch = logging.StreamHandler()
ch.setLevel(LOGGING_LEVEL)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)




class obj:
      
    # constructor
    def __init__(self, dict1):
        self.__dict__.update(dict1)


class GenericExperiment(object):



    def __init__(self, config):
        if Path(config).is_file():
            with open(config, 'r') as fp:
                self.config = json.load(fp)
        else:
            raise FileNotFoundError("Config file not found")


    def dict2obj(self, dict1):
        # https://www.geeksforgeeks.org/convert-nested-python-dictionary-to-object/
        # using json.loads method and passing json.dumps
        # method and custom object hook as arguments
        return json.loads(json.dumps(dict1), object_hook=obj)

    def validate_configuration(self, configuration):
        # check json schema
        logger.info("Validating configuration schema")
        validate(configuration, EXPERIMENT_SCHEMA)
        logger.info("Schema validation passed")

        # Check data file locations
        for location_name, location in configuration['data']['locations'].items():
            logger.info("Checking `%s` for existence" % (location_name))
            for data_set in ['xs', 'ys']:
                if not Path(location[data_set]).is_file():
                    logger.warning("`{}` is not a file for `{}` - `{}`".format(
                    location[data_set], location, data_set
                ))

        # check results location name
        if not Path(configuration['results']['location']).is_dir():
            logger.warning("`%s` does not exist to put results into" % (
                configuration['results']['location']
                )
            )

    @property
    def config(self):
        return self._config
    
    @config.setter
    def config(self, configuration):
        self.validate_configuration(configuration)

        self._config = configuration

    
    @config.getter
    def config(self):
        return self._config

    def _validate_configuration(self):
        validate(self._config, EXPERIMENT_SCHEMA)

    # def _create_experiment_folders(self):
        