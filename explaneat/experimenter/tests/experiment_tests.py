import unittest
from explaneat.experimenter import experiment

class TestExperimentMethods(unittest.TestCase):
    def setUp(self):
        self.experiment = experiment.GenericExperiment()
