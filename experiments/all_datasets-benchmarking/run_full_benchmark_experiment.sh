#! /bin/bash
# This script will run the end to end experiment, ensuring that data
# processing, experiment runs, data capture, etc., are all run in order
# end to end. The goal of this script is to go from raw, downloaded data
# and produce the final processed results.


############################################################
# Help                                                     #
############################################################
Help()
{
   # Display Help
   echo "This script will run the end to end experiment, ensuring that data"
   echo "processing, experiment runs, data capture, etc., are all run in order"
   echo "end to end. The goal of this script is to go from raw, downloaded data"
   echo "and produce the final processed results."

   echo "Syntax: scriptTemplate [-s]"
   echo "options:"
   echo "s:     The step to start at."
   echo "d:     The dataset list to work with."
   echo
}

############################################################
############################################################
# Main program                                             #
############################################################
############################################################

# Set variables
STEP=1
CONFIG=./all_data_benchmarking_experiment_config.json
SHAFILE=./sha_file.json

############################################################
# Process the input options. Add options as needed.        #
############################################################
# Get the options
while getopts "s:d:" opt; do
  case ${opt} in
    s ) # process -s option
        STEP=$OPTARG
        ;;
    d ) # process -d option
        FILEPATH=$OPTARG
        ;;
    \? ) echo "Usage: command [-s step] [-d data]" 1>&2
        exit 1
        ;;
    : ) echo "Invalid option: $OPTARG requires an argument" 1>&2
        exit 1
        ;;
  esac
done

echo "*********************************"
echo "Starting at $STEP!"

## STEP 1
if (($STEP <= 0));
then
   ipython 0_create_experiment.py $CONFIG $SHAFILE
fi

while IFS= read -r DATA; do

   if (($STEP <= 1));
   then
      ipython 1_prepare_data.py $CONFIG $SHAFILE "$DATA"
   fi


   if (($STEP <= 2));
   then
      ipython 2_train_svm.py $CONFIG $SHAFILE "$DATA"
      ipython 2_train_rf.py $CONFIG $SHAFILE "$DATA"
      ipython 2_train_regression.py $CONFIG $SHAFILE "$DATA"
      ipython 2_train_nn.py $CONFIG $SHAFILE "$DATA"
      ipython 2_train_explaneat.py $CONFIG $SHAFILE "$DATA"
   fi

done < "$FILEPATH"
