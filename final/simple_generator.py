# To run this, simply type "python simple_generator.py" and the output will be
# printed to the console from the array "output"

import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import csv


reader = csv.reader(open("positive_data_file.csv"))
jokes = []
count = 0
for row in reader:
    if count == 50:
        break
    count += 1
    jokes.extend(row)

def generate_jokes():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
    # model = TFGPT2LMHeadModel.from_pretrained("gpt2") #removed pad_token_id

    # N is the total number of sequences we want to return. Currently set to 5
    N = 5
    # MAX_LEN is the max length of each generated sequence
    MAX_LEN = 50
    # Padding text can be provided to make input longer so that the model has more context for generation.
    PADDING_TEXT = ""
    # Modify the input_text variable to provide the starting text for the generator
    input_text = "Knock Knock! Who's there?"
    input_ids = tokenizer.encode(PADDING_TEXT + input_text, return_tensors='tf')

    tf.random.set_seed(0)

    sample_outputs = model.generate(
        input_ids,
        do_sample=True,
        max_length=MAX_LEN,
        top_k=50,
        top_p=0.95,
        num_return_sequences=N
    )

    output = []
    for i, sample_output in enumerate(sample_outputs):
        s = (tokenizer.decode(sample_output, skip_special_tokens=True))
        output.append(s)
    return output





'''
Earlier Models (not used in the actual generation, just remaining in case we want to use them later)

greedy_output = model.generate(input_ids, max_length=50)

beam_output = model.generate(
    input_ids,
    max_length=50,
    num_beams=5,
    no_repeat_ngram_size=2,
    num_return_sequences=5,
    early_stopping=True
)
'''

