import numpy as np
from classifier_model import get_model
from gather_data import gather_data
import tensorflow as tf
import pickle
from matplotlib import pyplot as plt
import seaborn as sns


def run_model(directly=False):
    """
    Runs the following operations:
    1. Opens stored in "stats.pkl" file, which was generated by gather_data(),
        or from directly compiled data from get_stats()
    2. Splits data into training and test data
    3. Compiles and fits model from get_model(), saves it to "results" folder
    4. Runs model on test data
    5. Outputs graph to results folder
    6. Saves result data to "results/final_data.pkl"
    """

    if directly:
        # Generate daa from scratch
        train_data, test_data = gather_data()
    else:
        # Get data from "stats.pkl" file
        file_name = "stats.pkl"
        open_file = open(file_name, "rb")
        stats = pickle.load(open_file)
        open_file.close()
        train_data, test_data = stats

    x = []
    y = []
    j = []
    for i in range(len(train_data)):
        d = train_data[i]
        x.append(d[0])
        y.append(d[1])
        j.append(d[2])

    x_test = []
    y_test = []
    j_test = []
    for i in range(len(test_data)):
        d = test_data[i]
        x_test.append(d[0])
        y_test.append(d[1])
        j_test.append(d[2])

    x = np.asarray(x).astype('float32')
    y = np.asarray(y).astype(int)
    x_test = np.asarray(x_test).astype('float32')
    y_test = np.asarray(y_test).astype(int)

    # Compiles and Trains Model
    print('training')
    model = get_model(x)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', tf.keras.metrics.BinaryCrossentropy(
        name="binary_crossentropy", dtype=None)])
    model.fit(x, y, validation_split=0.1, epochs=5)
    model.save('results')

    # Runs model on test data
    predictions = model.predict(x_test)
    predictions = np.mean(predictions, axis=1).transpose()[0]
    predictions = np.vstack([predictions, ((predictions >= 0.5).astype(int) == y_test).astype(int)])

    correct = predictions[:, predictions[1] == 1][0]
    incorrect = predictions[:, predictions[1] == 0][0]

    # Output graph
    sns.kdeplot(correct, common_norm=True, color='green', label='correct')
    sns.kdeplot(incorrect, common_norm=True, color='red', label='incorrect')
    plt.legend()
    plt.title("Classifier Accuracy Density Plot")
    plt.xlabel("Confidence")
    plt.ylabel("Density")
    plt.savefig('results/kdeplot.png')

    # # Output graph displot
    # sns.displot(
    #     data=[correct, incorrect],
    #     common_norm=True,
    #     # hue_order=(0, 1),
    #     # kind="kde",
    #     multiple="fill",
    #     # palette="ch:rot=-.25,hue=1,light=.75",
    #     label=("correct", "incorrect"),
    # )
    plt.clf()
    plt.hist(correct, bins=10, density=True, label='correct')
    plt.hist(incorrect, bins=10, density=True, label='incorrect')

    # plt.xlim([0.3, .7])
    plt.legend()
    plt.title("Classifier Accuracy Conditional Density Plot")
    plt.xlabel("Confidence")
    plt.ylabel("Density")
    plt.savefig('results/Ckdeplot.png')

    correct = np.array(sorted(correct))
    incorrect = np.array(sorted(incorrect))
    print('correct:')
    print(correct[correct>.6].shape)
    print('\nincorrect:')
    print(incorrect[incorrect>.6].shape)
    # Save results in pickle file "results/final_data.pkl"
    final_data = (train_data, test_data, predictions, correct, incorrect)
    file_name = "results/final_data.pkl"
    open_file = open(file_name, "wb")
    pickle.dump(final_data, open_file)
    open_file.close()