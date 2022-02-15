import numpy as np
from classifier_model import get_model
from gather_data import gather_data
import tensorflow as tf
import pickle
from matplotlib import pyplot as plt
import seaborn as sns
from pandas import read_csv
import timeit


def run_model(directly=False, data=None):
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
        # Generate data from scratch
        if data is None:
            negative_data = read_csv('datasets/ML_Inter_Puns/negative_data_file.csv', names=['joke', 'funny'])
            Puns_positive_data = read_csv('datasets/ML_Inter_Puns/positive_data_file.csv', names=['joke', 'funny'])
            dataset_name_original = 'puns'
            start = timeit.default_timer()
            train_data, test_data = gather_data(negative_data, Puns_positive_data)
            stop = timeit.default_timer()
            print('Gather_data() runtime for dataset: ' + dataset_name + ' is: ' + str(stop-start))
        else:
            dataset_name = data[2]
            train_data, test_data = gather_data(data[0], data[1])
    else:
        # Get data from "stats.pkl" file
        file_name = "stats.pkl"
        open_file = open(file_name, "rb")
        stats = pickle.load(open_file)
        open_file.close()
        train_data, test_data = stats
        dataset_name_original = 'from .pkl file'

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
    print('training dataset: ' + dataset_name_original)
    model = get_model(x)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', tf.keras.metrics.BinaryCrossentropy(
        name="binary_crossentropy", dtype=None)])
    model.fit(x, y, validation_split=0.1, epochs=5)
    model.save('results')

    # Runs model on test data
    predictions_original = model.predict(x_test)
    # print(predictions[:10])
    # print(predictions[:, 0].flatten())

    if data is None:
        # predictions = np.mean(predictions_original.copy(), axis=1).transpose()[0]
        # dataset_name += '_Metric-mean'
        predict_types = ['mean']
    else:
        predict_types = ['mean', 'median', 'maximum', 'Path only', 'L-Ch only', 'Wu-P only']

    for predict_type in predict_types:
        dataset_name = dataset_name_original + '_Metric-' + predict_type
        if predict_type == 'mean':
            predictions = np.mean(predictions_original.copy(), axis=1).transpose()[0]
        elif predict_type == 'median':
            predictions = np.median(predictions_original.copy(), axis=1).transpose()[0]
        elif predict_type == 'maximum':
            predictions = np.maximum(predictions_original.copy(), axis=1).transpose()[0]
        elif predict_type == 'Path only':
            predictions = predictions_original.copy()[:, 0].flatten()
        elif predict_type == 'L-Ch only':
            predictions = predictions_original.copy()[:, 2].flatten()
        else:
            predictions = predictions_original.copy()[:, 1].flatten()

        predictions = np.vstack([predictions, ((predictions >= 0.5).astype(int) == y_test).astype(int)])

        correct = predictions[:, predictions[1] == 1][0]
        incorrect = predictions[:, predictions[1] == 0][0]

        # Output graph
        sns.kdeplot(correct, common_norm=True, color='green', label='correct', fill=True)
        sns.kdeplot(incorrect, common_norm=True, color='red', label='incorrect', fill=True)
        plt.legend()
        plt.title("Classifier Accuracy Density Plot, dataset: " + dataset_name)
        plt.xlabel("Confidence")
        plt.ylabel("Density")
        plt.savefig("results/kdeplot_" + dataset_name + ".png")

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
        plt.title("Classifier Histogram, dataset: " + dataset_name)
        plt.xlabel("Confidence")
        plt.ylabel("Density")
        plt.savefig("results/hist_" + dataset_name + ".png")

        correct = np.array(sorted(correct))
        incorrect = np.array(sorted(incorrect))
        predictions = predictions[:, predictions[0, :].argsort()]

        print('Stats for dataset: ' + dataset_name)
        print('Entire dataset % Correct: ' + str(correct.shape[0]/(correct.shape[0]+incorrect.shape[0])))


        correct5 = predictions[:, -5:][:, predictions[1][-5:] == 1].shape[1]*100/5
        correct10 = predictions[:, -10:][:, predictions[1][-10:] == 1].shape[1]*100/10
        correct20 = predictions[:, -20:][:, predictions[1][-20:] == 1].shape[1]*100/20
        std_devpt2 = predictions[:, np.argwhere(predictions[0] >= np.max(predictions[0])-np.std(predictions[0])*0.2)]
        std_devpt5 = predictions[:, np.argwhere(predictions[0] >= np.max(predictions[0])-np.std(predictions[0])*0.5)]
        std_dev = predictions[:, np.argwhere(predictions[0] >= np.max(predictions[0])-np.std(predictions[0]))]
        pct_1 = predictions[:, -int(.1*predictions.shape[1]):]
        pct_05 = predictions[:, -int(.05 * predictions.shape[1]):]
        pct_01 = predictions[:, -int(.01 * predictions.shape[1]):]
        print('\nTop 20, in pct correct: ' + str(correct20))
        print('Top 10, in pct correct: ' + str(correct10))
        print('Top 5, in pct correct: ' + str(correct5))

        print('\nTop 1 percent, in pct correct: ' + str(pct_1[:, pct_1[1]==1].shape[1]*100/pct_1.shape[1]))
        print('Top .05 percent, in pct correct: ' + str(pct_05[:, pct_05[1] == 1].shape[1] * 100 / pct_05.shape[1]))
        print('Top .01 percent, in pct correct: ' + str(pct_01[:, pct_01[1] == 1].shape[1] * 100 / pct_01.shape[1]))

        print('\nTop 1 std dev, in pct correct: ' + str(std_dev[:, std_dev[1] == 1].shape[1]*100/std_dev.shape[1]))
        print('Top .5 std dev, in pct correct: ' + str(std_devpt5[:, std_devpt5[1] == 1].shape[1]*100/std_devpt5.shape[1]))
        print('Top .2 std dev, in pct correct: ' + str(std_devpt2[:, std_devpt2[1] == 1].shape[1]*100/std_devpt2.shape[1]))

        # Save results in pickle file "results/final_data.pkl"
        final_data = (train_data, test_data, predictions, correct, incorrect)
        file_name = "results/final_data_" + dataset_name + ".pkl"
        open_file = open(file_name, "wb")
        pickle.dump(final_data, open_file)
        open_file.close()
        print('*********************************END DATASET STATS***************************************\n')
