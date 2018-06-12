# Cognitive Modeling Week4

## Our Bot
We set out to create a BubbleBot: A chatbot which recognizes your political views and then responds with agreeable statements.
Although an anti-bubble bot would be a nicer product, we felt that disagreements and arguments tend to require more sophisticated
language than agreement.

We managed to technically meet our goal, albeit in a very limited form. Our bot tries to classify the statements made by the user
into either 'Democratic' or 'Republican' views based on Twitter data.
The bot then has a corresponding generative model for each view. Based on the classification, with bot creates a composite model of
the two, weighted depending on our classification. It generates responses from this model which try to take the input topic into
account.

The bot analyses the incoming text data using a naïve bayesian classifier. We have
trained this model on a dataset of political tweets. The dataset consists
of a large amount of tweets made by political accounts. By using google queries
we were able to link these accounts to Wikipedia pages. This allowed us to label the dataset into
two categories: Democratic politician and Republic politician. We then trained the network on the tweets (1.2 million).
As the classifier has to make sense of this input data, we used the google word embeddings to
encode the data.

The generated responses are simple Markov chains, and because we did little cleaning of the source data from Twitter and Reddit,
fairly ungrammatical. Responses also relate to the user input in a very superficial way, and are often generic.

## Examples
It should be coming with stuff like:
>Hello
Hi!
Hey, so how do you feel about gun control?
>Damn libs will have to pry my guns from my cold, dead hands.
I know right? Lock her up!

But instead it looks more like:

>Hello
Hi!
Hey, so how do you feel about gun control?
>As technology increases, so should the power of the civilian weapons.

The bot generally only makes sense when allowed to construct sentences almost identical or at least very similar to the source corpus.
Hence the readability from this reply; it's actually from a real post.

In a lot of cases the bot does not correctly guess if your original comment was democratic or republican.
This is a limit caused by the classifier. While it does train on a vast dataset, within the scope
of the exercise we haven't been able to perfect it.

We had a lot of fun experimenting with both dataset. We ended up using both as we were
both scouting for datasets and we figured we could extract some useful information from both.

In practice the bot generates some really vague sentences, but honestly, we would recommend
you to give it a spin yourself :).



## Setup
pip install ...
python -m textblob.download_corpora
We used the pol_accounts tweet dataset.
Make sure to download stopwords for gensim.
