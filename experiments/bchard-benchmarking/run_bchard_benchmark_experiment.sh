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
   echo
}

############################################################
############################################################
# Main program                                             #
############################################################
############################################################

# Set variables
Step=1
CONFIG=./bchard_benchmarking_experiment_config.json
SHAFILE=./sha_file.json

############################################################
# Process the input options. Add options as needed.        #
############################################################
# Get the options
while getopts ":s:" option; do
   case $option in
      s) # Enter a name
         Step=$OPTARG;;
     \?) # Invalid option
         echo "Error: Invalid option"
         exit;;
   esac
done


echo "*********************************"
echo "Starting at $Step!"

## Step 1
if (($Step <= 0));
then
   ipython 0_create_experiment.py $CONFIG $SHAFILE
fi

if (($Step <= 1));
then
   ipython 1_prepare_data.py $CONFIG $SHAFILE
fi


if (($Step <= 2));
then
   ipython 2_train_svm.py $CONFIG $SHAFILE
   ipython 2_train_rf.py $CONFIG $SHAFILE
   ipython 2_train_regression.py $CONFIG $SHAFILE
   ipython 2_train_nn.py $CONFIG $SHAFILE
   ipython 2_train_explaneat.py $CONFIG $SHAFILE
fi