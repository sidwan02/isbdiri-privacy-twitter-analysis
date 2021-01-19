from wordcloud import WordCloud
import pandas as pd


def show_wordcloud():
    print('showing wordcloud')
    d = {}

    bag = pd.read_csv('results/3gram_tfidf.csv')

    for term, rank in bag.values:
        d[term] = rank

    wordcloud = WordCloud()
    wordcloud.generate_from_frequencies(frequencies=d)
    # plt.imshow(wordcloud, interpolation="bilinear")
    wordcloud.to_file("results/phrases_cloud.png")


# show_wordcloud()
