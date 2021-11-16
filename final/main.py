from simple_generator import generate_jokes
import keras


def get_joke(input='1'):
    jokes = generate_jokes()
    if input == '1':
        model = keras.models.load_model(r"C:\Users\super\PycharmProjects\A-NN_Classifier\final")
        predictions = model.predict(jokes)
        predictions = np.mean(predictions, axis=1)
        final_prediction = np.max(predictions)
        return final_prediction

joke = get_joke()
print(joke)