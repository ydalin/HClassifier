# To run this, simply type "python simple_generator.py" and the output will be
# printed to the console from the array "output"

import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer


def generate_jokes():
    tokenizer = GPT2Tokenizer.from_pretrained("msharma95/joke-generator",
                                              use_auth_token=True)

    model = TFGPT2LMHeadModel.from_pretrained("msharma95/joke-generator",
                                              use_auth_token=True)

    # N is the total number of sequences we want to return.
    N = 5
    # MAX_LEN is the max length of each generated sequence
    MAX_LEN = 150
    # Padding text can be provided to make input longer so
    # that the model has more context for generation.
    PADDING_TEXT = ""
    # Modify the input_text variable to provide the starting
    # text for the generator
    input_text = "Knock Knock! Who's there?"
    input_ids = tokenizer.encode(PADDING_TEXT + input_text,
                                 return_tensors='tf')

    # tf.random.set_seed(0)

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


def main():
    print(generate_jokes())


if __name__ == "__main__":
    main()
