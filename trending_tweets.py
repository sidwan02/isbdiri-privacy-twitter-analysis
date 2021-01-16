import pandas as pd
import math
import numpy as np

# thresh = input(
#     "Enter threshold [Valid input: 'mean' | 'median' | <Any number> ]: ")

# df = pd.read_csv(
#     r'C:\Users\sidwa\OneDrive\OneDriveNew\Personal\Sid\Brown University\Internships\ISB\DIRI DS\privacy_data_sentiment_analysis\CSV\consolidated.csv', index_col=0)
# retweet = df['retweet_count']
# fav = df['fav_count']


def filter_by_col(col_name, thresh):
    df = pd.read_csv('results/consolidated.csv', index_col=0)
    # print('df1', df1)

    # retweet = df['retweet_count']
    # fav = df['fav_count']
    # print('thresh', thresh)

    col = df[col_name]
    # print('col', list(col))
    if thresh == 'mean':
        tot = sum(list(col))
        mean = tot/len(col)
        # print('mean', mean)

        df_sorted = df.sort_values(col_name, ascending=False)
        # print('df_sorted', df_sorted)
        return df_sorted.loc[df_sorted[col_name] >= mean]
    elif thresh == 'median':
        median = 0
        sorted = list(col)
        sorted.sort()
        med_pos = math.ceil(len(col) / 2)
        if len(col) % 2 == 0:
            median = (sorted[med_pos - 1] + sorted[med_pos]) / 2
        else:
            median = sorted[med_pos]
        print('median', median)

        df_sorted = df.sort_values(col_name, ascending=False)
        return df_sorted.loc[df_sorted[col_name] >= median]
    else:
        try:
            df_sorted = df.sort_values(col_name, ascending=False)
            return df_sorted.loc[df_sorted[col_name] >= int(thresh)]
        except ValueError:
            df_sorted = df.sort_values(col_name, ascending=False)
            return df_sorted.loc[df_sorted[col_name] >= 0]
        except TypeError:
            df_sorted = df.sort_values(col_name, ascending=False)
            return df_sorted.loc[df_sorted[col_name] >= 0]


# filtered_retweet = filter_by_col('retweet_count', thresh)
# print('filtered_retweet', filtered_retweet)
# filtered_fav = filter_by_col('fav_count', thresh)
# print('filtered_fav', filtered_fav)
