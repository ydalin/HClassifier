# ARGUMENTS SHOULD BE IN THE FOLLOWING FORMAT:
# python pace_training_generator.py DATA_FILE OUTPUT_DIR HF_DIR
# arg DATA_FILE: the path to the .txt file containing the joke dataset,
#     as it was uploaded
# arg OUTPUT_DIR: the path to the directory where the trained model
#     should be saved
# arg HF_DIR: the path to the directory where the huggingface transformers
#     library is cloned
#     ./transformers/examples/tensorflow/language-modeling

# Import Steps
import subprocess
import sys

# Getting the arguments from the command line
num_args = len(sys.argv)
args = sys.argv

data_file = args[1]
output_dir = args[2]
hf_dir = args[3]
log_err = args[4]
log_out = args[5]

output = "--output_dir=" + output_dir
data = "--train_file=" + data_file

# Defining some variables so that they can easily be modified later
epochs = "--num_train_epochs=2"
save_steps = "--save_steps=1"

log_err = open(log_err, 'w+')
log_out = open(log_out, 'w+')

# First, change into the language-modeling directory
# Next, actually run the command
command = subprocess.Popen(["python",
                            "run_clm.py",
                            "--model_name_or_path=gpt2",
                            "--model_type=gpt2",
                            output,
                            data,
                            "--line_by_line=TRUE",
                            "--do_train",
                            epochs,
                            save_steps,
                            ],
                           cwd=hf_dir,
                           stderr=log_err,
                           stdout=log_out)
                           #shell=True)

# Ideally, the program would have finished training here.
# Output is currently being printed directly to the command line,
# but it is possible to have it print to a logging file as well.
