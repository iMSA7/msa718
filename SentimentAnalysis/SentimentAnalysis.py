import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn import naive_bayes, ensemble, svm
from statistics import mode
import os


try:
    import SentimentAnalysis.DataManager as DataManager
    from SentimentAnalysis.ModelHandler import ModelHandler as MH
except Exception:
    import DataManager as DataManager
    from ModelHandler import ModelHandler as MH

# set of English stopwords
stopwords_set = set(stopwords.words('english'))
# vector transformer using 'Tf-idf'
vectorizor = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopwords_set)

# labelled tweets about apple shares (1981 tweets per class)
stocks = pd.read_csv("SentimentAnalysis/Data/stemmedBalanced",
                     sep='\t', names=['txt', 'liked'])


# concatenate more than one dataframe into one
frames = [stocks]
df = pd.concat(frames)

# labels [-1, 1]
y = df.liked
x = vectorizor.fit_transform(df.txt)

# classification voters: [Naive bays, SVM, Fores Trees]
clf_NB = naive_bayes.MultinomialNB()
clf_SVM = svm.SVC()
clf_FT = ensemble.RandomForestClassifier()


# load trained models
NB_path = 'SentimentAnalysis/NB.sav'
if os.path.isfile(NB_path):
    handler = MH(clf_NB, NB_path)
    clf_NB = handler.load()
else:
    clf_NB.fit(x, y)
    handler = MH(clf_NB, NB_path)
    handler.save()

SVM_path = 'SentimentAnalysis/SVM.sav'
if os.path.isfile(SVM_path):
    handler = MH(clf_SVM, SVM_path)
    clf_SVM = handler.load()
else:
    clf_SVM.fit(x, y)
    handler = MH(clf_SVM, SVM_path)
    handler.save()

FT_path = 'SentimentAnalysis/FT.sav'
if os.path.isfile(FT_path):
    handler = MH(clf_FT, FT_path)
    clf_FT = handler.load()
else:
    clf_FT.fit(x, y)
    handler = MH(clf_FT, FT_path)
    handler.save()

# nltk vader:
clf_vader = SentimentIntensityAnalyzer()

# updating vader's lexicon with the following trading related words
positive = pd.read_csv("SentimentAnalysis/Data/positive.txt",
                       sep='\t', names=['txt', 'liked'])

negative = pd.read_csv("SentimentAnalysis/Data/negative.txt",
                       sep='\t', names=['txt', 'liked'])

token_words = {
    'crushes': 10,
    'undervalued': 10,
    'rocket': 10,
    'launch': 10,
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


def main():
    # unlabelled and unbalanced tweets (around 400 negative, 1200 positive, and 1400 neutral tweets) for testing
    apple = pd.read_csv("SentimentAnalysis/Data/Apple.txt",
                        sep='\n', names=['txt'])

    # is the data uniformly distributed
    print(list(y).count(1), list(y).count(0), list(y).count(-1))

    # Apply
    tweets = apple.txt

    # simple example for the demo:

    # tweets = ["latest updates for ios show the great capabilities of apple and that it is trust worthy",
    #           "apple may failed us in the past but nowadays it is doing better",
    #           "apple price reaches highest since 2011, are we going to face a sellout soon?",
    #           "apple announced to postpone iphone 13",
    #           "apple released new system, ipados."]

    # vector transformation of the input
    stemmed_rev = [DataManager.stemmer_stop(DataManager.clear_text(rev)) for rev in tweets]
    rev_vector = vectorizor.transform(stemmed_rev)

    # predictions
    prediction_NB = clf_NB.predict(rev_vector)
    prediction_SVM = clf_SVM.predict(rev_vector)
    prediction_FT = clf_FT.predict(rev_vector)

    majority_vote = [mode((a, b, c)) for a, b, c, in zip(prediction_NB, prediction_SVM, prediction_FT)]
    scores = []
    for i in range(len(tweets)):
        scores.append(round(clf_vader.polarity_scores(tweets[i])['compound']))
        print(tweets[i])
        print('NB vote: ', prediction_NB[i], 'SVM vote: ', prediction_SVM[i], 'RFC vote: ', prediction_FT[i],
              'majority vote: ', majority_vote[i], '\t', "Vader: ", scores[i])

    # agreement among voters
    agreement = [nb == sv == ft for nb, sv, ft in zip(prediction_NB, prediction_SVM, prediction_FT)]

    results = f"NB Test Accuracy: {clf_NB.score(rev_vector, scores)}\nSVM Test Accuracy: {clf_SVM.score(rev_vector, scores)}" \
              f"\nRandom Forests Classifiers ACC: {clf_FT.score(rev_vector, scores)}\n" \
              f"Majority Vote ACC: {sum(1 for a, b in zip(majority_vote, scores) if a == b) / len(scores)}\n" \
              f"Number of Tweets: {len(agreement)}\nAgreement: {agreement.count(True)}\nAgreement Percent: {agreement.count(True) / len(agreement)}"
    with open("classifiers", 'w') as f:
        f.write(results)


if __name__ == '__main__':
    main()
