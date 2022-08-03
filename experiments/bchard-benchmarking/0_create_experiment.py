import argparse

from explaneat.experimenter.experiment import GenericExperiment


parser = argparse.ArgumentParser(description="Provide the experiment config")
parser.add_argument('conf_file',
                    metavar='experiment_config_file',
                    type=str,
                    help="Path to experiment config")
parser.add_argument("ref_file",
                    metavar='experiment_reference_file',
                    type=str,
                    help="Path to experiment ref file")

args = parser.parse_args()

experiment = GenericExperiment(
    args.conf_file,
    confirm_path_creation=False)


experiment.create_logging_header("Starting 0_create_experiment", 50)

experiment.write_run_to_file(args.ref_file)

experiment.create_logging_header("Ending 0_create_experiment", 50)
