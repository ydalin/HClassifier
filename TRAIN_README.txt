To run the script, the following should be run in the command line:

python pace_training_generator.py $training_file $output_dir $huggingface_library

Each parameter should be as follows:

$training_file
    The file path to the file with the jokes.

$output_dir
    The file path to the directory where the trained model will be saved with a .h5 and .json file.

$huggingface_library
    The file path to the folder within the huggingface/transformers GitHub repository.
    This must be directed to: transformers/examples/tensorflow/language-modeling/
    within the github repository
    within the github repository


Note: to install the hugginface library, we should have the following:

git clone https://github.com/huggingface/transformers
