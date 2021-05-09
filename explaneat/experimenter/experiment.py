from jsonschema import validate
from explaneat.experimenter.schemas.experiment import experiment as EXPERIMENT_SCHEMA
from pathlib import Path
import json
import logging
import os
import datetime

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

    folders = [
            'results',
            'results/interim',
            'results/final',
            'configurations',
            'logs'
        ]


    def __init__(self, config):
        if Path(config).is_file():
            with open(config, 'r') as fp:
                self.config = json.load(fp)
        else:
            raise FileNotFoundError("Config file not found")

        self.generate_folder_structure()

        self.save_configurations()

    def generate_folder_structure(self):
        ## This will create the standard folder structure for experiments
        ## including results, experiment information, etc.
        logger.info("Starting to create folder structures")

        self.experiment_folder_name = "%s_%s" % (
            self._config['experiment']['codename'],
            datetime.datetime.now().strftime("%y%m%dT%H%M%S")
            )
        logger.info("Experiment folder name is %s" %self.experiment_folder_name)

        self.root_path = os.path.join(
            self.config['experiment']['base_location'],
            self.experiment_folder_name
            )
        logger.info("Experiment root path is %s" % self.root_path)

        if not os.path.exists(self.config['experiment']['base_location']):
            if not (input("The base location does not exist, continue? Y/N\nBase location is %s"%self.config['experiment']['base_location']).lower() == "y"):
                raise FileNotFoundError("The base location does not exist!")
        ## baselocation/[experiment_name]_[time]
        if not os.path.exists(self.root_path):
            logger.info("Creating the root path")
            os.makedirs(self.root_path)
        logger.info('Root path created')

        for folder in self.folders:
            folder_name = self.path(folder)
            logger.info('Creating %s' % folder)
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
        logger.info('All folders created')

    def path(self, *args):
        return os.path.join(self.root_path, *args)

    def save_configurations(self):
        logger.info('Saving experiment configuration')
        with open(self.path('configurations', 'experiment.json'), 'w') as fp:
            json.dump(self.config, fp, indent=4, sort_keys=True)



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


        # Check base location exists
        if not Path(configuration['experiment']['base_location']).is_dir():
            logger.error("`%s` does not exist as base location" % (
                configuration['experiment']['base_location']
                )
            )

        # Check data file locations
        for location_name, location in configuration['data']['locations'].items():
            logger.info("Checking `%s` for existence" % (location_name))
            for data_set in ['xs', 'ys']:
                if not Path(location[data_set]).is_file():
                    logger.error("`{}` is not a file for `{}` - `{}`".format(
                    location[data_set], location, data_set
                ))

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