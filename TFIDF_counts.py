import pandas as pd
import csv
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud_generate import show_wordcloud


def frequent_phrases():
    # nltk.download('stopwords')
    # nltk.download('punkt')
    # nltk.download('stopwords')
    # nltk.download('punkt')

    txt1 = []
    df = pd.read_csv(
        'results/consolidated_date_formatted.csv', index_col=0)
    txt1 = df['tweet_lemmatized']

    stop_words = set(stopwords.words('english'))
    txt2 = []
    for i, line in enumerate(txt1):
        if line:
            line = str(line)
            txt2.append(
                ' '.join([x for x in line.split(' ') if (x not in stop_words)]))

    # for i in range(1, 9):
    for i in range(3, 4):
        #     vectorizer = CountVectorizer(ngram_range = (i, i), max_features=1000)
        #     X1 = vectorizer.fit_transform(txt2)
        #     features = (vectorizer.get_feature_names())

        #     sums = X1.sum(axis = 0)
        #     data1 = []
        #     for col, term in enumerate(features):
        #         data1.append( (term, sums[0, col] ))
        #     ranking = pd.DataFrame(data1, columns = ['term','rank'])
        #     words = (ranking.sort_values('rank', ascending = False))

        #     words.to_csv('./results/'+str(i)+'gram.csv', index=False)

        vectorizer = TfidfVectorizer(ngram_range=(i, i), max_features=10000)
        X2 = vectorizer.fit_transform(txt2)
        features = (vectorizer.get_feature_names())
        # scores = (X2.toarray())

        sums = X2.sum(axis=0)
        data1 = []
        for col, term in enumerate(features):
            data1.append((term, sums[0, col]))
        ranking = pd.DataFrame(data1, columns=['term', 'rank'])
        words = (ranking.sort_values('rank', ascending=False))

        words.to_csv('results/' + str(i)+'gram_tfidf.csv', index=False)

    show_wordcloud()


frequent_phrases()
