import markovify # https://github.com/jsvine/markovify

MODEL_NAME = "donald"
MARKOV_SIZE = 2 # State size of markov model for training

def train():
    with open('' + "donald" + '.txt', encoding="utf8") as f:
        text = f.read()
        # Build the model.
        # Don't retain original corpus. (Means sentences are often repeated instead of original)
        model = markovify.Text(text, state_size=MARKOV_SIZE, retain_original=False)

    ## Save model
    model_json = model.to_json()
    with open('models/' + MODEL_NAME + str(MARKOV_SIZE) + '.json', 'w+', encoding="utf8") as f:
        f.write(model_json)
    print('Model is trained and saved!')

def load(model_name=MODEL_NAME):
    with open('models/' + model_name + str(MARKOV_SIZE) + '.json', encoding="utf8") as f:
        model_json = f.read()

    model = markovify.Text.from_json(model_json)

    return model

def test(model):
    # Testing
    # Print five randomly-generated sentences
    return model.make_sentence()
    #return model.make_short_sentence(500)
    # Print three randomly-generated sentences of no more than 140 characters
    # for i in range(3):
    #     print(model.make_short_sentence(140))

def loop_until_short(model):
    s_length = 1000
    sentence=""
    #while len(sentence.split())!=10:
    while s_length>500:
        sentence = test(model)
        s_length = len(sentence)
    return sentence

def train_all():
    MARKOV_SIZE = 10
    train()

train_all()

model = load()



test_sentence = test(model)

test_sentence = loop_until_short(model)
len(test_sentence)
print(test_sentence)
