import wptools
import tweepy
import time
from datetime import date, timedelta
import json
import csv
import preprocessor as p
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import emoji
import itertools
import spacy
from twitter_scrape_keys import get_access_key, get_access_secret, get_consumer_key, get_consumer_secret
# from parallel_process import *
# https://medium.com/swlh/5-step-guide-to-parallel-processing-in-python-ac0ecdfcea09
import multiprocessing
import time
import concurrent.futures
import numpy as np


def privacy_analysis_multithreading():
    MAX_THREADS = 50

    # Twitter credentials
    consumer_key = get_consumer_key()
    consumer_secret = get_consumer_secret()
    access_key = get_access_key()
    access_secret = get_access_secret()

    # Creating the authentication object
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    # Setting access token and secret
    auth.set_access_token(access_key, access_secret)
    # Creating the API object while passing in auth information
    api = tweepy.API(auth)

    # serial no. for tweets (column 1)
    # global c

    def clean_method2(tweet):
        # remove mentions and URLs
        p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.HASHTAG, p.OPT.NUMBER)
        tweet = p.clean(tweet)
        # replace consecutive non-ASCII characters with a space
        tweet = re.sub(r'[^\x00-\x7F]+', ' ', tweet)
        # take care of contractions and stray quote marks
        tweet = re.sub(r'’', "'", tweet)
        # words = tweet.split()
        tweet = re.sub(r":", " ", tweet)
        tweet = re.sub(r"n't", " not", tweet)
        # fix spellings
        tweet = ''.join(''.join(s)[:2] for _, s in itertools.groupby(tweet))
        # emojis conversion
        tweet = emoji.demojize(tweet)
        tweet = ' '.join(tweet.split())
        return tweet

    def filter_method2(tweet):
        stop_words = set(stopwords.words('english'))
        stray_tokens = ['amp', '`', "``", "'", "''",
                        '"', "n't", "I", "i", ",00"]  # stray words
        punct = r'[{}]'.format(string.punctuation)
        tweet = re.sub(punct, ' ', tweet)
        tweet = re.sub(r'[0-9]', ' ', tweet)
        # correct spelling mistakes
        tweet = re.sub(
            r'privacy|privcy|privac|privasy|privasee|privarcy|priavcy|privsy', ' privacy ', tweet)
        tweet = re.sub(r'(^|\s)[a-z]($|\s)', ' ', tweet)
        tweet = re.sub(r'(^|\s)[a-z][a-z]($|\s)', ' ', tweet)
        word_tokens = word_tokenize(tweet)
        # filter using NLTK library append it to a string
        filtered_tweet = [w for w in word_tokens if not w in stop_words]
        filtered_tweet = []
        # looping through conditions
        for w in word_tokens:
            # check tokens against stopwords and punctuations
            if w not in stop_words and w not in string.punctuation and w not in stray_tokens:
                w = w.lower()
                filtered_tweet.append(w)
        tweet = ' '.join(filtered_tweet)
        # re-removing single characters
        tweet = re.sub(r'(^|\s)[a-z]($|\s)', ' ', tweet)
        tweet = re.sub(r'(^|\s)[a][a]($|\s)', ' ', tweet)  # fixing for privacy
        return tweet

    spice = spacy.load("en_core_web_sm")

    def lemma(tweet):
        doc = spice(tweet)
        wrong_class = ['pending', 'madras', 'data', 'media']
        wrong_words = ' '.join(wrong_class)
        wrong_doc = spice(wrong_words)
        lemma_tweet = []
        for token in doc:
            if token not in wrong_doc:
                token = token.lemma_
                if token != "-PRON-":
                    lemma_tweet.append(token)
        return ' '.join(lemma_tweet)

    def senti(tweet):
        analyser = SentimentIntensityAnalyzer()
        vader_polarity = analyser.polarity_scores(tweet)
        vp = vader_polarity['compound']
        return vp

    # https://towardsdatascience.com/named-entity-recognition-with-nltk-and-spacy-8c4a7d88e7da

    def named_entity_recognition(tweet_text):
        doc = spice(tweet_text)
        items = [(x.text, x.label_) for x in doc.ents]

        # print("items", items)
        org_tups = list(
            filter(lambda t: t[1] == 'ORG' or t[1] == 'PERSON', items))
        # get only the company names (and people names cuz for some reason orgs like gmail are considered ppl)
        orgs = np.array(list(map(lambda t: t[0], org_tups)))

        # print("orgs", orgs)
        return np.unique(orgs)

    def classify_all_cand_companies(companies):
        real_companies = []
        tags_to_orgs = {}
        orgs_to_tags = {}
        for company in companies:
            # print('tags_to_orgs', tags_to_orgs)
            # print('real_companies', real_companies)
            try:
                tags = get_company_classifier(company)

                real_companies.append(company)
                # print('company BOO', company)

                for tag in tags:
                    try:
                        prev_val = tags_to_orgs[tag]
                        # print('prev_val', prev_val)
                        # https://www.programiz.com/python-programming/methods/dictionary/update
                        tags_to_orgs.update(tag=prev_val.append(company))
                        # print('new_val', prev_val)
                    except:
                        tags_to_orgs[tag] = [company]

                orgs_to_tags[company] = tags
            except LookupError:
                # print('company ' + str(company) + ' does not exist')
                # tags_to_orgs[company] = []
                pass

        # https://gist.github.com/89465127/5776892
        # https://stackoverflow.com/questions/30418481/error-dict-object-has-no-attribute-iteritems
        # return {k: v for k, v in tags_to_orgs.items() if v != []}
        return [real_companies, tags_to_orgs, orgs_to_tags]

    def get_company_classifier(company):
        page = wptools.page(company, silent=True)
        wikidata = page.get_parse()
        # print(wikidata)
        # query = page.get_query()
        # print(query)
        infobox = wikidata.infobox
        # print(infobox)
        if infobox == None:
            if '(software)' in company:
                return []
            elif '(company)' in company:
                return get_company_classifier(company.replace('(company)', '') + ' (software)')
            else:
                return get_company_classifier(company + ' (company)')
        else:
            if 'industry' in infobox.keys():
                industries = infobox['industry']
                # print(industries)
                # https://stackoverflow.com/questions/2403122/regular-expression-to-extract-text-between-square-brackets
                cleaned_tags = list(map(lambda t: t.replace('[', ''), re.findall(
                    r'\[(.*?)\]', industries)))
                # print('cleaned_tags', cleaned_tags)
                return cleaned_tags
            if 'services' in infobox.keys():
                industries = infobox['services']
                # print(industries)
                # https://stackoverflow.com/questions/2403122/regular-expression-to-extract-text-between-square-brackets
                cleaned_tags = list(map(lambda t: t.replace('[', ''), re.findall(
                    r'\[(.*?)\]', industries)))
                # print('cleaned_tags', cleaned_tags)
                return cleaned_tags
            elif 'owner' in infobox.keys():
                owner = infobox['owner']
                # print('owner', owner)
                cleaned_owner = list(map(lambda t: t.replace('[', ''), re.findall(
                    r'\[(.*?)\]', owner)))[0]
                # print(cleaned_owner)
                # https://stackoverflow.com/questions/17778372/why-does-my-recursive-function-return-none
                return get_company_classifier(cleaned_owner)
            elif 'developer' in infobox.keys():
                developer = infobox['developer']
                # print('developer', developer)
                cleaned_developer = list(map(lambda t: t.replace('[', ''), re.findall(
                    r'\[(.*?)\]', developer)))[0]
                # print(cleaned_developer)
                # https://stackoverflow.com/questions/17778372/why-does-my-recursive-function-return-none
                return get_company_classifier(cleaned_developer)
            elif 'label' in infobox.keys():
                label = infobox['label']
                # print('label', label)
                cleaned_label = list(map(lambda t: t.replace('[', ''), re.findall(
                    r'\[(.*?)\]', label)))[0].split('|', 1)[0]
                # print(cleaned_label)
                # https://stackoverflow.com/questions/17778372/why-does-my-recursive-function-return-none
                return get_company_classifier(cleaned_label)
            else:
                if '(software)' in company:
                    return []
                elif '(company)' in company:
                    return get_company_classifier(company.replace('(company)', '') + ' (software)')
                else:
                    return get_company_classifier(company + ' (company)')

    # date-setting
    for nx in range(1, 2):  # edit to set number of days before current date
        # for nx in range(2, 32):
        today = date.today()
        end = today-timedelta(days=(nx-1))
        start = today-timedelta(days=nx)
        end_date = end.strftime('%Y-%m-%d')
        start_date = start.strftime('%Y-%m-%d')

        query = "privacy OR privcy OR privac OR private information OR privasy OR private info exclude:retweets"

        all_status = tweepy.Cursor(api.search,
                                   q=query,
                                   lang="en",
                                   since=start_date,
                                   until=end_date,
                                   result_type='recent',
                                   tweet_mode='extended',
                                   include_entities=True,
                                   monitor_rate_limit=True,
                                   wait_on_rate_limit=True).items(10000)

        print('all_status', all_status)

        all_tweets = []

        for status in all_status:

            tweet = status._json
            # print('tweet', tweet)
            all_tweets.append(tweet)

        # print('all_tweets len', len(all_tweets))

        # def process_randn__parallel(row_list, ncores=50, log_file=None):
        #     res = process_func_parallel(
        #         analyse_each_tweet, row_list, None, chunk_size=50, nthread=ncores, log_file=log_file)

        #     return res

        def analyse_each_tweet(inputs):
            tweet, c = inputs
            # print('tweet name', tweet['created_at'])
            print('c', c)

            # https://stackoverflow.com/questions/3348460/csv-file-written-with-python-has-blank-lines-between-each-row#:~:text=writer%20if%20you%20are%20getting,to%20whatever%20it%20should%20be.
            with open('./CSV/'+start_date+'.csv', mode='a', newline='', encoding="utf-8") as csv_out:
                writer = csv.writer(csv_out)  # create the csv writer object

                # print('writing data rows')

                if c == 0:
                    fields = ['s_no',
                              'date',
                              'created_at',
                              'id_str',
                              'screen_name',
                              'user_location',
                              'geo-tagged_place',
                              'tweet_text',
                              'tweet_tokenized',
                              'tweet_cleaned',
                              'tweet_filtered',
                              'tweet_lemmatized',
                              'tweet_mentioned_organizations',
                              'tweet_tags_classified',
                              'tweet_orgs_classified',
                              'retweet_count',
                              'fav_count',
                              'hashtags',
                              'urls',
                              'mentions',
                              'vader_polarity']
                    writer.writerow(fields)  # writes field
                else:
                    # for tweet in all_tweets:
                    tweet_original = tweet['full_text']
                    p.set_options(p.OPT.URL, p.OPT.HASHTAG, p.OPT.MENTION,
                                  p.OPT.EMOJI, p.OPT.NUMBER)
                    # https://pypi.org/project/tweet-preprocessor/
                    tweet_tokenized = p.tokenize(tweet_original)
                    tweet_cleaned = clean_method2(tweet_original)
                    tweet_filtered = filter_method2(tweet_cleaned)
                    tweet_lemmatized = lemma(tweet_filtered)

                    # keep only ASCII
                    # https://www.codegrepper.com/code-examples/python/python+remove+all+unicode+from+string
                    # tweet_mentions = [''.join([i if ord(i) < 128 else '' for i in mention['name']])
                    #                   for mention in tweet['entities']['user_mentions']]
                    # print('tweet_mentions', tweet_mentions)

                    mentions = [mention['name']
                                for mention in tweet['entities']['user_mentions']]
                    # mentions_for_ner = " ".join([mention['name']
                    #                              for mention in tweet['entities']['user_mentions']])

                    # retireval and classification of organization mentioned in tweet
                    full_ner_text = tweet_filtered.lower()  # + mentions

                    tweet_cand_orgs = named_entity_recognition(
                        full_ner_text)

                    [tweet_orgs, tweet_tags_classified,
                        tweet_orgs_classified] = classify_all_cand_companies(tweet_cand_orgs)

                    # [tweet_orgs_2, tweet_tags_classified_2,
                    #     tweet_orgs_classified_2] = classify_all_cand_companies(mentions)

                    # print(tweet_mentions)

                    vp = senti(tweet_cleaned)

                    if tweet['place']:
                        tweet_place = tweet['place']['full_name'] + \
                            ', '+tweet['place']['country_code']
                    else:
                        tweet_place = 'Not Geo-tagged'

                    # print('tweet', tweet)

                    hashtags = [hashtag_item['text']
                                for hashtag_item in tweet['entities']['hashtags']]
                    # hashtags = ", ".join([hashtag_item['text']
                    #                       for hashtag_item in tweet['entities']['hashtags']])
                    urls = ", ".join([url['expanded_url']
                                      for url in tweet['entities']['urls']])
                    # mentions = ", ".join([mention['screen_name']
                    #                       for mention in tweet['entities']['user_mentions']])

                    writer.writerow([c,
                                     start_date,
                                     tweet['created_at'],
                                     tweet['id_str'],
                                     tweet['user']['screen_name'],
                                     tweet['user']['location'],
                                     tweet_place,
                                     tweet_original.encode('utf-8'),
                                     tweet_tokenized,
                                     tweet_cleaned,
                                     tweet_filtered,
                                     tweet_lemmatized,
                                     tweet_orgs,
                                     tweet_tags_classified,
                                     tweet_orgs_classified,
                                     tweet['retweet_count'],
                                     tweet['favorite_count'],
                                     hashtags,
                                     urls,
                                     mentions,
                                     vp])

        # https://beckernick.github.io/faster-web-scraping-python/
        def analyse_all_tweets():
            all_tweets.insert(0, 0)
            all_c = range(0, len(all_tweets))
            print('in threading')
            inputs = zip(all_tweets, all_c)

            threads = min(MAX_THREADS, len(all_tweets))

            with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
                executor.map(analyse_each_tweet, inputs)

        analyse_all_tweets()

        # https://stackoverflow.com/questions/18204782/runtimeerror-on-windows-trying-python-multiprocessing
        # if __name__ == '__main__':
        #     # process_randn__parallel(all_tweets)
        #     analyse_each_tweet(tweet)

        # if __name__ == '__main__':
        #     # pool = multiprocessing.Pool()
        #     # pool = multiprocessing.Pool(processes=4)

        #     # https://stackoverflow.com/questions/16425046/how-do-i-parallelize-a-simple-python-def-with-multiple-argument
        #     # outputs = pool.map(analyse_each_tweet, inputs)

        #     print("Input: {}".format(inputs))
        #     print("Output: {}".format(outputs))

        #     # csv_out.close()


# privacy_analysis_multithreading()
