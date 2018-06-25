import markovify # https://github.com/jsvine/markovify

MODEL_NAME = "donald"
MARKOV_SIZE = 2 # State size of markov model for training

def train():
    with open('' + "donald" + '.txt', encoding="utf8") as f:
        model = markovify.Text(f, state_size=MARKOV_SIZE, retain_original=True)
    return model
def test(model):
    return model.make_sentence()


def loop_until_short(model):
    s_length = 1000
    sentence=""
    #while len(sentence.split())!=10:
    while s_length>1000:
        sentence = test(model)
        s_length = len(sentence)
    return sentence

def train_all():
    MARKOV_SIZE = 10
    return train()

model = train_all()

test_sentence = test(model)

test_sentence = loop_until_short(model)
len(test_sentence)
print(test_sentence)
