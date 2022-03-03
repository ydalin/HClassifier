from run_model import run_model
from gather_data import write_to_pickle
from pandas import read_csv
from gather_data import gather_data
import timeit


def main():
    """
    Runs model, outputs data
    """
    prompt = "How to run? Input 1 or 2 or 3 or 4:" + \
             "\n1 Directly (very slow)" + \
             "\n2 From saved 'stats.pkl' file (not as slow)" + \
             "\n3 run cross-training full experiment (very very slow)" + \
             "\n4 Write data to 'stats.pkl' file\n"
    action = input(prompt)

    if action not in ['1', '2', '3', '4']:
        raise Exception("Please input 1 or 2 or 3")

    if action == '1':
        print("Running Directly")
        run_model(directly=True)
    elif action == '2':
        print("Running from saved stats.pkl file")
        run_model(directly=False)
    elif action == '3':
        negative_data = read_csv('datasets/ML_Inter_Puns/negative_data_file.csv', names=['joke', 'funny'])
        Puns_positive_data = read_csv('datasets/ML_Inter_Puns/positive_data_file.csv', names=['joke', 'funny'])
        Shortjokes_positive_data = read_csv('datasets/ML_Inter_ShortJokes/positive_data_file.csv', names=['joke', 'funny'])
        start = timeit.default_timer()
        puns_train_data, puns_test_data = gather_data(negative_data=negative_data, positive_data=Puns_positive_data)
        stop = timeit.default_timer()
        print('Gather_data() runtime for puns dataset is: ' + str(stop - start))
        print('Puns train data shape: ' + str(len(puns_train_data)) + '\nPuns test data shape: ' + str(
            len(puns_test_data)))
        start = timeit.default_timer()
        shortjokes_train_data, shortjokes_test_data = gather_data(negative_data=negative_data, positive_data=Shortjokes_positive_data)
        stop = timeit.default_timer()
        print('Gather_data() runtime for shortjokes dataset is: ' + str(stop - start))
        print('Shortjokes train data shape: ' + str(len(shortjokes_train_data)) + '\nShortjokes test data shape: ' + str(len(shortjokes_test_data)))
        run_model(directly=True, data=(puns_train_data, puns_test_data, 'puns train, puns test'))
        run_model(directly=True, data=(shortjokes_train_data, shortjokes_test_data, 'shortjokes train, shortjokes test'))
        run_model(directly=True, data=(shortjokes_train_data, puns_test_data, 'shortjokes train, puns test'))
        run_model(directly=True, data=(puns_train_data, shortjokes_test_data, 'puns train, shortjokes test'))
    else:
        prompt = 'What test/train split to use (input a float)?'
        split = input(prompt)
        try:
            split = float(split)
        except ValueError:
            print("Please enter a float.")
        if split <= 0 or split >= 1:
            raise Exception("Please enter a float between 0 and 1")
        write_to_pickle(split=split)


if __name__ == "__main__":
    main()
