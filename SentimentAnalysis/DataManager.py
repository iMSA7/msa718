import pandas as pd
import json
import re
import datetime
import string
import nltk
from nltk import stem
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer


nltk.download('stopwords')
nltk.download('vader_lexicon')


def stemmer_stop(text):
    stemmer = stem.PorterStemmer("NLTK_EXTENSIONS")
    stemmed_tokens = [stemmer.stem(token) for token in text.split()]
    stopwords_set = set(stopwords.words('english'))
    tokens = [token for token in stemmed_tokens if token not in stopwords_set]
    text = " ".join(tokens)
    return text


def convert_date_format(str_date):
    """
    convert twitter date format to YYYY-MM-DD format
    :param str_date: raw tweet date format
    :return: str date in YYYY-MM-DD format
    """
    converted = datetime.datetime.strftime(datetime.datetime.strptime(str_date, '%a %b %d %H:%M:%S +0000 %Y'),
                                           '%Y-%m-%d')
    return converted


def convert_date(text_date, int_diff):
    """
        get a date that is different from 'text_date' by 'int_diff'
        :param text_date: string date in YYYY-MM-DD format
        :param int_diff: days different from the given date, negative integers return date in the past.
        :return: Date String of the input format
    """
    diff = abs(int_diff)
    date = datetime.datetime.strptime(text_date, '%Y-%m-%d')
    if int_diff > 0:
        date = date + datetime.timedelta(days=diff)
        return str(date)[:10]
    elif int_diff < 0:
        date = date - datetime.timedelta(days=diff)
        return str(date)[:10]
    else:
        return text_date


def clear_text(text):
    """
    filters a given tweet from links, usernames, hashtags, and empty lines
    :param text: input tweet
    :return: str of filtered tweet
    """
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    text = text.lower()
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub(r'\s?@+\s?', ' ', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'rt :[\s]+', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\.\.+', ' ', text)
    text = re.sub(r'-+[\s]', ' ', text)
    text = re.sub(r'\s[A-Za-z0-9]+…', '', text)
    text = regex.sub('', text)
    text = re.sub(r"\s*…", '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def vader_labelling(tweets):
    positive = pd.read_csv("Data/positive.txt",
                           sep='\t', names=['txt', 'liked'])

    negative = pd.read_csv("Data/negative.txt",
                           sep='\t', names=['txt', 'liked'])

    # nltk vader:
    clf_vader = SentimentIntensityAnalyzer()

    # updating vader's lexicon with the following stock related words
    token_words = {
        'crushes': 10,
        'undervalued': 10,
        'rocket': 10,
        'dividends': 10,
        'buyback': 10,
        'profit': 10,
        'returns': 10,
        'earnings': 10,
        'above': 5,
        'buying': 5,
        'green': 5,
        'power': 5,
        'up': 5,
        'beats': 5,
        'buy': 5,
        'overvalued': -10,
        "sell": -5,
        'down': -5,
        'downgraded': -5,
        'declined': -5,
        'drops': -10,
        'downtrend': -10,
    }
    for word, score, neg, score_neg in zip(positive.txt, positive.liked, negative.txt, negative.liked):
        token_words[word] = int(score)
        token_words[neg] = int(score_neg)

    clf_vader.lexicon.update(token_words)
    scores = []
    for i in range(len(tweets)):
        scores.append(round(clf_vader.polarity_scores(tweets[i])['compound']))

    return zip(tweets, scores)


def write_to_file(data, file):
    """
    writes data into csv file
    :param data: tuple of lists e.g: zip(list1, list2)
    :param file: str of file path/name
    """
    with open(file, 'w') as f:
        for line in data:
            f.write(line)
            f.write("\n")


def collect_data(iterations):
    """
    collect tweets from AAPL folder.
    :param iterations: number of tweets to collect.
    :return: collected tweets.
    """
    # data collected from start to end dates
    data_range = {'start': "2014-01-01", 'end': "2016-04-01"}
    current_date = data_range['start']
    data = []
    # the path to the data
    path_to_data = "Data/AAPL/"
    i = 1
    while (current_date != data_range['end']) and i < iterations:
        # read from file and assign the data to a panda's dataframe:
        in_file = path_to_data + current_date
        # update the current date
        current_date = convert_date(current_date, 1)
        try:
            with open(in_file, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
                    i = i + 1
        except FileNotFoundError:
            print(f"Error: file \"{in_file}\" not found ")
            continue
    data = set([clear_text(x['text']) for x in data])
    return list(data)
