import os
import random

import telebot
import markovify
from textblob import TextBlob

import generator.markov_model as markov_model
#from classifier.classifier import classifier

MARKOV_SIZE = 2

# Init text  CRAZY SLOW
blob = TextBlob('Hello')
noun_phrases = blob.noun_phrases
del blob, noun_phrases

with open("token") as file:
    bot = telebot.TeleBot(file.readline()[:-1])

weight_republican = 0.5
weight_democrat = 0.5

The_Donald_model = markov_model.load('The_Donald')
TwoXChromosomes_model = markov_model.load('TwoXChromosomes')

hardcoded_responses = {
    'Hi': 'Hi there!\nHow do you feel about what Trump said yesterday?',
    'Hello': 'Hello to you too. How do you feel about what Trump said yesterday?',
    'Goodbye': 'See you again soon!'
}

keyword_responses = [
    'I believe that ',
]

no_keyword_responses = [
    'I totally agree with you.',
    'I know right?',
    'Preach it.',
    'Couldn`t agree more',
]

def extract_keywords(message: str) -> list:
    blob = TextBlob(message)
    return blob.noun_phrases
    # return ['gun control']

def respond(message: str) -> str:
    # Generate merged model based on current evaluation. Might be too slow for every sentence.
    weight_republican = 0.5
    weight_democrat = 0.5
    merged_model = markovify.combine([The_Donald_model, TwoXChromosomes_model], [weight_republican, weight_democrat])
    keywords = extract_keywords(message)

    response = ''
    if keywords:
        # Respond to keyword
        # Randomly select response formula
        formula = keyword_responses[0]

        # Add keywords
        formula += keywords[0]

        # Complete sentence
        if formula.count(' ') > MARKOV_SIZE - 1:
            try:
                before_start = formula.split()[0:-MARKOV_SIZE]
                end = merged_model.make_sentence_with_start(' '.join(formula.split()[-MARKOV_SIZE:]))
                response = ' '.join(before_start) + ' ' + end
            except:
                print('Could not handle start: ' + ' '.join(formula.split()[-MARKOV_SIZE:]))
        else:
            try:
                response = merged_model.make_sentence_with_start(formula)
            except:
                print('Could not handle start: ' + formula)

    if response == '':
        # Respond without keyword.
        response = random.choice(no_keyword_responses)

    return response

#commands=['start', 'help']
@bot.message_handler(func=lambda m: True)
def send_welcome(message):
    print('>' + message.text)
    if message.text in hardcoded_responses:
        response = hardcoded_responses[message.text]
    else:
        response = respond(message.text)
    print(response)
    bot.reply_to(message, response)

print('Running!')
# Test
test = 'Hello there. I hate gun control.'
print('(Test) >' + test)
print(respond(test))


bot.polling()
