import generateJokes
import saved_model_stats.pb


def getJokes(input='1'):
    jokes = generateJokes()
    if input == '1':
        predictions = model.predict(jokes)
        predictions = np.mean(predictions, axis=1)
        final_prediction = np.max(predictions)
        return final_prediction
