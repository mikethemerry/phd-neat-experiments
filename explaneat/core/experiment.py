from neat.reporting import ReporterSet


class ExperimentReporterSet(ReporterSet):
    """ Extends neat.reporting.ReporterSet to have safe methods for ExplaNEAT
    implementation, especially around backprop methods
    """

    def __init__(self, *args, **kwargs):
        super(ExperimentReporterSet, self).__init__(*args, **kwargs)

    
    def start_experiment(self, config):
        for r in self.reporters:
            try:
                r.start_experiment(config)
            except AttributeError:
                continue
    def start_backprop(self, config, population, species):
        for r in self.reporters:
            try:
                r.start_backprop(config, population, species)
            except AttributeError:
                continue
    def end_backprop(self, config, population, species):
        for r in self.reporters:
            try:
                r.end_backprop(config, population, species)
            except AttributeError:
                continue
    
 

class ExperimentReporter(object):
    """ ExplaNEAT Experiment Reporter that captures relevant performance data
    about the running of the explaneat algorithm. Works with the
    ExperimentReporterSet
    """
    
    def start_experiment(self, config):
        pass
    
    def start_backprop(self, config, population, species):
        pass

    def end_backprop(self, config, population, species):
        pass

    def start_generation(self, generation):
        pass
    
    def end_generation(self, config, population, species_set):
        pass

    def post_evaluate(self, config, population, species, best_genome):
        pass

    def post_reproduction(self, config, population, species):
        pass

    def complete_extinction(self):
        pass

    def found_solution(self, config, generation, best):
        pass

    def species_stagnant(self, sid, species):
        pass

    def info(self, msg):
        pass