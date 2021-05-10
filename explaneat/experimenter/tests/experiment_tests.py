import unittest
from explaneat.experimenter import experiment
import os
import shutil

import json

test_path_location = '/Users/mike/dev-mtm/phd-neat-experiments/data/experiments/tests'
remove_paths = True

class TestExperimentMethods(unittest.TestCase):
    config_file = './test_config.json'
    def setUp(self):
        self.experiment = experiment.GenericExperiment(self.config_file, confirm_path_creation=False)
    
    def test_all_folders_exist(self):
        self.assertTrue(os.path.exists(test_path_location))
        for folder in self.experiment.folders:
            self.assertTrue(os.path.exists(
                os.path.join(self.experiment.root_path, folder)))

    def test_experiment_config_exists(self):
        with open(self.config_file, 'r') as fp:
            my_config = json.load(fp)
        with open(os.path.join(self.experiment.path('configurations','experiment.json'))) as fp:
            saved_config = json.load(fp)
        self.assertEqual(my_config, saved_config)

    def tearDown(self):
        if remove_paths:
            shutil.rmtree(test_path_location)

if __name__ == '__main__':
    unittest.main()
