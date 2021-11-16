from simple_generator import generateJokes


def getJoke(input='1'):
    jokes = generateJokes()
    if input == '1':
        model = keras.models.load_model("saved_model_stats.pb")
        print(model)
        predictions = model.predict(jokes)
        predictions = np.mean(predictions, axis=1)
        return final_prediction

joke = getJoke()
print(joke)