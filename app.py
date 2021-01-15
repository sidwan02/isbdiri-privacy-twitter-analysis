import functools
import math
import numpy as np
import plotly.graph_objs as go
import plotly.figure_factory as ff
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import State, Input, Output
from datetime import date, timedelta
import pandas as pd
import re
import dash_table
import plotly.express as px
# from merge_csv import *
from trending_tweets import *
# from privacy_history import *
# from privacy_history_multithreading import *
import ast
import base64


# analyse_over_all_dates()

# merge_all_csv()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
    'background': 'white',
    'text': 'black',
}

# https://github.com/plotly/dash/issues/71
image_filename = 'DIRI.png'  # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

server = app.server

app.layout = html.Div([
    html.Div([
        html.H1('Twitter Privacy Analysis'),
        html.H3('Organization References'),
        html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()))
    ], style={'width': '40%', 'height': '180px', 'align-items': 'center', 'justify-content': 'center', 'padding': '20px', 'background-color': 'white',   'border-radius': '5px', 'box-shadow': '2px 2px 2px lightgrey', 'position': 'absolute', 'right': '80px', 'top': '80px', 'color': 'black'
              }),
    html.Div([
        html.Div([
            dcc.DatePickerRange(
                id='date_picker',
                # min_date_allowed=date(2020, 8, 1),
                # max_date_allowed=date(2020, 12, 31),
                start_date=date(2020, 1, 1),
                end_date=date(2022, 1, 1),
                # style={"margin-top": "15px"}
            ),
            dcc.Dropdown(
                id='choice_consolidated_trending',
                options=[
                    {'label': 'Full Consolidated', 'value': 'full'},
                    {'label': 'Trending Retweets Consolidated',
                     'value': 'trending_retweets'},
                    {'label': 'Trending Favourites Consolidated',
                     'value': 'trending_favs'},
                ],
                value='full',
                clearable=False,
                style={"margin-top": "15px"}
            ),
            dcc.Input(
                id='choice_trending_thresh',
                placeholder='Enter a threshold',
                style={"margin-top": "15px"}
            ),
        ], style={'width': '40%', 'align-items': 'center', 'justify-content': 'center', 'padding': '20px', 'background-color': '#f7f7f7',   'border-radius': '5px', 'box-shadow': '2px 2px 2px lightgrey'
                  }),
        html.Div([
            html.Div([
                dcc.Dropdown(
                    id='choice_all_tweets',
                    options=[
                        {'label': 'All Tweets', 'value': 'tweets_all'},
                        {'label': 'Tweets Mentioning Organizations',
                         'value': 'tweets_orgs'},
                    ],
                    value='tweets_all',
                    clearable=False,
                ),
                dcc.Dropdown(
                    id='choice_orgs_selection',
                    multi=True,
                    style={"margin-top": "15px"}
                ),
                dcc.Dropdown(
                    id='choice_analysis',
                    options=[
                        {'label': 'Progressive', 'value': 'progressive'},
                        {'label': 'Overall', 'value': 'overall'},
                    ],
                    value='progressive',
                    clearable=False,
                    style={"margin-top": "15px"}
                ),
                dcc.Dropdown(
                    id='choice_org_centric',
                    options=[
                        {'label': 'Date Centric', 'value': 'date_centric'},
                        {'label': 'Organization Centric',
                         'value': 'org_centric'},
                    ],
                    value='date_centric',
                    clearable=False,
                    style={"margin-top": "15px"}
                ),
                dcc.Dropdown(
                    id='choice_tweet_property',
                    options=[
                        {'label': 'Retweets', 'value': 'retweets'},
                        {'label': 'Likes', 'value': 'favs'},
                        {'label': 'Sentiment', 'value': 'sentiment'},
                    ],
                    value='retweets',
                    clearable=False,
                    style={"margin-top": "15px"}

                ),

            ], style={'width': '40%', 'align-items': 'center', 'justify-content': 'center', 'padding': '20px', 'background-color': '#f7f7f7',   'border-radius': '5px', 'box-shadow': '2px 2px 2px lightgrey'
                      }),

            html.Div([
                dcc.Graph(
                    id='graph_central_tendency'
                ),
            ], style={'width': '97%', 'align-items': 'center', 'justify-content': 'center', 'padding': '20px', 'background-color': '#f7f7f7',   'border-radius': '5px', 'box-shadow': '2px 2px 2px lightgrey', 'margin-top': '10px'
                      }),
        ], style={'width': '97%', 'align-items': 'center', 'justify-content': 'center', 'padding': '20px', 'background-color': '#e6e6e6',   'border-radius': '5px', 'box-shadow': '2px 2px 2px lightgrey', 'margin-top': '10px'
                  }),
        html.Div([
            html.Div([
                dcc.Dropdown(
                    id='choice_organizations',
                    options=[
                        {'label': 'Organizations Mentioned',
                            'value': 'organizations'},
                        {'label': 'Industry', 'value': 'tags'},
                        {'label': 'Hashtags', 'value': 'hashtags'},
                    ],
                    value='organizations',
                    clearable=False
                ),
                dcc.Input(
                    id='choice_max_x',
                    type='number'
                ),

            ], style={'width': '40%', 'align-items': 'center', 'justify-content': 'center', 'padding': '20px', 'background-color': '#f7f7f7',   'border-radius': '5px', 'box-shadow': '2px 2px 2px lightgrey'
                      }),
            html.Div([
                dcc.Graph(
                    id='graph_organizations_and_tags'
                )
            ], style={'width': '97%', 'align-items': 'center', 'justify-content': 'center', 'padding': '20px', 'background-color': '#f7f7f7',   'border-radius': '5px', 'box-shadow': '2px 2px 2px lightgrey', 'margin-top': '10px'
                      }),
        ], style={'width': '97%', 'align-items': 'center', 'justify-content': 'center', 'padding': '20px', 'background-color': '#e6e6e6',   'border-radius': '5px', 'box-shadow': '2px 2px 2px lightgrey', 'margin-top': '10px'
                  }),
    ], style={'width': '97%', 'align-items': 'center', 'justify-content': 'center', 'padding': '20px', 'background-color': '#bababa',   'border-radius': '5px', 'box-shadow': '2px 2px 2px lightgrey', 'margin-top': '10px'
              }),
], style={
    'backgroundColor': 'white',
    'color': colors['text'],
})


@ app.callback(
    dash.dependencies.Output('graph_central_tendency', 'figure'),
    [dash.dependencies.Input('date_picker', 'start_date'),
     dash.dependencies.Input('date_picker', 'end_date'),
     dash.dependencies.Input('choice_consolidated_trending', 'value'),
     dash.dependencies.Input('choice_all_tweets', 'value'),
     dash.dependencies.Input('choice_tweet_property', 'value'),
     dash.dependencies.Input('choice_analysis', 'value'),
     dash.dependencies.Input('choice_trending_thresh', 'value'),
     dash.dependencies.Input('choice_org_centric', 'value'),
     dash.dependencies.Input('choice_orgs_selection', 'value')],
)
def update_graph_central_tendency(start_date, end_date, data_selection, tweets_selection, tendency_selection, analysis_selection, thresh, centric_selection, companies_selection):
    print('companies_selection', companies_selection)
    if companies_selection == None:
        companies_selection = []
    # print('yo')
    start_date_obj = date.fromisoformat(start_date)
    end_date_obj = date.fromisoformat(end_date)

    # https://stackoverflow.com/questions/59882714/python-generating-a-list-of-dates-between-two-dates
    # https://stackoverflow.com/questions/18684076/how-to-create-a-list-of-date-string-in-yyyymmdd-format-with-python-pandas
    date_range = [d.strftime('%Y-%m-%d') for d in pd.date_range(
        start_date_obj, end_date_obj-timedelta(days=1), freq='d')]

    if data_selection == 'full':
        df = pd.read_csv('./results/consolidated.csv')
    elif data_selection == 'trending_retweets':
        df = filter_by_col('retweet_count', thresh)
    elif data_selection == 'trending_favs':
        df = filter_by_col('fav_count', thresh)

    # # https://stackoverflow.com/questions/12096252/use-a-list-of-values-to-select-rows-from-a-pandas-dataframe
    # print('df_date', df['date'])

    df = df[df['date'].isin(date_range)]
    # print('date_range', date_range)

    if tweets_selection == 'tweets_all':
        print('do not change df')
    elif tweets_selection == 'tweets_orgs':
        def common_data(list1, list2):
            # print('list1', list1)
            # print('list2', list2)
            # traverse in the 1st list
            for x in list1:
                # print('x', x)

                # traverse in the 2nd list
                for y in list2:

                    # if one common
                    if x == y:
                        # print('found!')
                        return True
            # print('not found')
            return False

        mask = [common_data(ast.literal_eval(orgs), list(companies_selection))
                for orgs in df['tweet_mentioned_organizations']]

        df = df[mask]

    if analysis_selection == 'progressive':
        if (tendency_selection == 'retweets'):

            fig = px.strip(df, x="date", y="retweet_count",
                           )

            fig.update_layout(
                plot_bgcolor=colors['background'],
                paper_bgcolor=colors['background'],
                font_color=colors['text']
            )

            # https://www.codegrepper.com/code-examples/python/how+to+find+mean+of+one+column+based+on+another+column+in+python
            mean_sr = df.groupby('date')['retweet_count'].mean()
            mean_df = pd.DataFrame(
                {'date': mean_sr.index, 'retweet_count': mean_sr.values})

            median_sr = df.groupby('date')['retweet_count'].median()
            median_df = pd.DataFrame(
                {'date': median_sr.index, 'retweet_count': median_sr.values})

            # https://stackoverflow.com/questions/62122015/how-to-add-traces-in-plotly-express
            fig.add_trace(go.Scatter(
                x=mean_df['date'], y=mean_df['retweet_count'], name='mean'))
            fig.add_trace(go.Scatter(
                x=median_df['date'], y=median_df['retweet_count'], name='median'))

            return fig

        elif (tendency_selection == 'favs'):

            fig = px.strip(df, x="date", y="fav_count",
                           )

            fig.update_layout(
                plot_bgcolor=colors['background'],
                paper_bgcolor=colors['background'],
                font_color=colors['text']
            )

            mean_sr = df.groupby('date')['fav_count'].mean()
            mean_df = pd.DataFrame(
                {'date': mean_sr.index, 'fav_count': mean_sr.values})

            median_sr = df.groupby('date')['fav_count'].median()
            median_df = pd.DataFrame(
                {'date': median_sr.index, 'fav_count': median_sr.values})

            fig.add_trace(go.Scatter(
                x=mean_df['date'], y=mean_df['fav_count'], name='mean'))
            fig.add_trace(go.Scatter(
                x=median_df['date'], y=median_df['fav_count'], name='median'))

            return fig

        elif (tendency_selection == 'sentiment'):
            fig = px.strip(df, x="date", y="vader_polarity",
                           )

            fig.update_layout(
                plot_bgcolor=colors['background'],
                paper_bgcolor=colors['background'],
                font_color=colors['text']
            )

            mean_sr = df.groupby(
                'date')['vader_polarity'].mean()
            mean_df = pd.DataFrame(
                {'date': mean_sr.index, 'vader_polarity': mean_sr.values})

            median_sr = df.groupby('date')['vader_polarity'].median()
            median_df = pd.DataFrame(
                {'date': median_sr.index, 'vader_polarity': median_sr.values})

            fig.add_trace(go.Scatter(
                x=mean_df['date'], y=mean_df['vader_polarity'], name='mean'))
            fig.add_trace(go.Scatter(
                x=median_df['date'], y=median_df['vader_polarity'], name='median'))

            return fig

        elif analysis_selection == 'overall':
            count_arr = [1] * len(df.index)
            df['count'] = count_arr
            if (tendency_selection == 'retweets'):

                fig = px.histogram(df, x="retweet_count", y="count",
                                   color="date", marginal='rug', hover_data=['vader_polarity'], nbins=20, )

                fig.update_layout(
                    plot_bgcolor=colors['background'],
                    paper_bgcolor=colors['background'],
                    font_color=colors['text']
                )

                return fig

            elif (tendency_selection == 'favs'):

                fig = px.histogram(df, x="fav_count", y="count",
                                   color="date", marginal='rug', hover_data=['vader_polarity'], nbins=20, )

                fig.update_layout(
                    plot_bgcolor=colors['background'],
                    paper_bgcolor=colors['background'],
                    font_color=colors['text']
                )

                return fig

            elif (tendency_selection == 'sentiment'):
                fig = px.histogram(df, x="vader_polarity", y="count",
                                   color="date", marginal='rug', hover_data=['tweet_mentioned_organizations'], nbins=20, )

                fig.update_layout(
                    plot_bgcolor=colors['background'],
                    paper_bgcolor=colors['background'],
                    font_color=colors['text']
                )

                return fig
    elif analysis_selection == 'overall':
        if centric_selection == 'date_centric':

            # print('df', df)

            # fig = px.strip(df, x="date", y="retweet_count", )

            # return 'hi'
            if analysis_selection == 'progressive':
                if (tendency_selection == 'retweets'):

                    fig = px.strip(df, x="date", y="retweet_count",
                                   )

                    fig.update_layout(
                        plot_bgcolor=colors['background'],
                        paper_bgcolor=colors['background'],
                        font_color=colors['text']
                    )

                    # https://www.codegrepper.com/code-examples/python/how+to+find+mean+of+one+column+based+on+another+column+in+python
                    mean_sr = df.groupby('date')['retweet_count'].mean()
                    mean_df = pd.DataFrame(
                        {'date': mean_sr.index, 'retweet_count': mean_sr.values})

                    median_sr = df.groupby('date')['retweet_count'].median()
                    median_df = pd.DataFrame(
                        {'date': median_sr.index, 'retweet_count': median_sr.values})

                    # https://stackoverflow.com/questions/62122015/how-to-add-traces-in-plotly-express
                    fig.add_trace(go.Scatter(
                        x=mean_df['date'], y=mean_df['retweet_count'], name='mean'))
                    fig.add_trace(go.Scatter(
                        x=median_df['date'], y=median_df['retweet_count'], name='median'))

                    return fig

                elif (tendency_selection == 'favs'):

                    fig = px.strip(df, x="date", y="fav_count",
                                   )

                    fig.update_layout(
                        plot_bgcolor=colors['background'],
                        paper_bgcolor=colors['background'],
                        font_color=colors['text']
                    )

                    mean_sr = df.groupby('date')['fav_count'].mean()
                    mean_df = pd.DataFrame(
                        {'date': mean_sr.index, 'fav_count': mean_sr.values})

                    median_sr = df.groupby('date')['fav_count'].median()
                    median_df = pd.DataFrame(
                        {'date': median_sr.index, 'fav_count': median_sr.values})

                    fig.add_trace(go.Scatter(
                        x=mean_df['date'], y=mean_df['fav_count'], name='mean'))
                    fig.add_trace(go.Scatter(
                        x=median_df['date'], y=median_df['fav_count'], name='median'))

                    return fig

                elif (tendency_selection == 'sentiment'):
                    fig = px.strip(df, x="date", y="vader_polarity",
                                   )

                    fig.update_layout(
                        plot_bgcolor=colors['background'],
                        paper_bgcolor=colors['background'],
                        font_color=colors['text']
                    )

                    mean_sr = df.groupby(
                        'date')['vader_polarity'].mean()
                    mean_df = pd.DataFrame(
                        {'date': mean_sr.index, 'vader_polarity': mean_sr.values})

                    median_sr = df.groupby('date')['vader_polarity'].median()
                    median_df = pd.DataFrame(
                        {'date': median_sr.index, 'vader_polarity': median_sr.values})

                    fig.add_trace(go.Scatter(
                        x=mean_df['date'], y=mean_df['vader_polarity'], name='mean'))
                    fig.add_trace(go.Scatter(
                        x=median_df['date'], y=median_df['vader_polarity'], name='median'))

                    return fig
            elif analysis_selection == 'overall':
                count_arr = [1] * len(df.index)
                df['count'] = count_arr
                if (tendency_selection == 'retweets'):

                    fig = px.histogram(df, x="retweet_count", y="count",
                                       color="date", marginal='rug', hover_data=['vader_polarity'], nbins=20, )

                    fig.update_layout(
                        plot_bgcolor=colors['background'],
                        paper_bgcolor=colors['background'],
                        font_color=colors['text']
                    )

                    return fig

                elif (tendency_selection == 'favs'):

                    fig = px.histogram(df, x="fav_count", y="count",
                                       color="date", marginal='rug', hover_data=['vader_polarity'], nbins=20, )

                    fig.update_layout(
                        plot_bgcolor=colors['background'],
                        paper_bgcolor=colors['background'],
                        font_color=colors['text']
                    )

                    return fig

                elif (tendency_selection == 'sentiment'):
                    fig = px.histogram(df, x="vader_polarity", y="count",
                                       color="date", marginal='rug', hover_data=['tweet_mentioned_organizations'], nbins=20, )

                    fig.update_layout(
                        plot_bgcolor=colors['background'],
                        paper_bgcolor=colors['background'],
                        font_color=colors['text']
                    )

                    return fig
        elif centric_selection == 'org_centric':

            orgs_literal = list(map(lambda x: ast.literal_eval(
                x), df['tweet_mentioned_organizations'].to_numpy()))

            print(orgs_literal)

            df['display_orgs'] = list(map(
                lambda x: functools.reduce(lambda a, b: a + ', ' + b, x), orgs_literal))

            print('after filtering the companies selected', df)

            # print('df', df)

            # fig = px.strip(df, x="date", y="retweet_count", )

            # return 'hi'
            if analysis_selection == 'overall':
                count_arr = [1] * len(df.index)
                df['count'] = count_arr
                if (tendency_selection == 'retweets'):

                    fig = px.histogram(df, x="retweet_count", y="count",
                                       color="display_orgs", marginal='rug', hover_data=['date', 'fav_count'], nbins=20, )

                    fig.update_layout(
                        plot_bgcolor=colors['background'],
                        paper_bgcolor=colors['background'],
                        font_color=colors['text']
                    )

                    return fig

                elif (tendency_selection == 'favs'):

                    fig = px.histogram(df, x="fav_count", y="count",
                                       color="display_orgs", marginal='rug', hover_data=['date', 'retweet_count', 'vader_polarity'], nbins=20, )

                    fig.update_layout(
                        plot_bgcolor=colors['background'],
                        paper_bgcolor=colors['background'],
                        font_color=colors['text']
                    )

                    return fig

                elif (tendency_selection == 'sentiment'):
                    fig = px.histogram(df, x="vader_polarity", y="count",
                                       color="display_orgs", marginal='rug', hover_data=['date', 'retweet_count', 'fav_count'], nbins=20, )

                    fig.update_layout(
                        plot_bgcolor=colors['background'],
                        paper_bgcolor=colors['background'],
                        font_color=colors['text']
                    )

                    return fig


@ app.callback(
    dash.dependencies.Output('graph_organizations_and_tags', 'figure'),
    [dash.dependencies.Input('date_picker', 'start_date'),
     dash.dependencies.Input('date_picker', 'end_date'),
     dash.dependencies.Input('choice_tweet_property', 'value'),
     dash.dependencies.Input('choice_analysis', 'value'),
     dash.dependencies.Input('choice_organizations', 'value'),
     dash.dependencies.Input('choice_consolidated_trending', 'value'),
     dash.dependencies.Input('choice_trending_thresh', 'value'),
     dash.dependencies.Input('choice_max_x', 'value')])
def update_graph_organizations(start_date, end_date, mode_selection, analysis_selection, organizations_selection, data_selection, thresh, max_x):
    print('hi')
    start_date_obj = date.fromisoformat(start_date)
    end_date_obj = date.fromisoformat(end_date)

    # https://stackoverflow.com/questions/59882714/python-generating-a-list-of-dates-between-two-dates
    # https://stackoverflow.com/questions/18684076/how-to-create-a-list-of-date-string-in-yyyymmdd-format-with-python-pandas
    date_range = [d.strftime('%Y-%m-%d') for d in pd.date_range(
        start_date_obj, end_date_obj-timedelta(days=1), freq='d')]
    # print('date_range', date_range)

    # df = pd.read_csv('./results/consolidated.csv')
    if data_selection == 'full':
        df = pd.read_csv('./results/consolidated.csv')
    elif data_selection == 'trending_retweets':
        df = filter_by_col('retweet_count', thresh)
    elif data_selection == 'trending_favs':
        df = filter_by_col('fav_count', thresh)

    df = df[df['date'].isin(date_range)]

    # print('df', df)

    if organizations_selection == 'organizations':
        print('organizations!')

        # create a 2D array with Organization references having their own row
        indiv_org_ref_arr = []

        for _, row in df.iterrows():
            # https://stackoverflow.com/questions/23119472/in-pandas-python-reading-array-stored-as-string
            # to_csv makes arrays of strings string so need to extract the array back
            orgs_literal = ast.literal_eval(
                row['tweet_mentioned_organizations'])

            # print('orgs_literal', orgs_literal)
            for org in orgs_literal:
                # print('index', index)
                # print('date', row['date'])
                # print('date type', type(row['date']))
                # print('org', org)

                # print('org type', type(org))

                orgs_dict_literal = ast.literal_eval(
                    row['tweet_orgs_classified'])

                # print('orgs_dict_literal', orgs_dict_literal)
                # print('org_tag', orgs_dict_literal[org])

                # print('vader_polarity', row['vader_polarity'])

                indiv_org_ref_arr.append(
                    [row['date'], org, orgs_dict_literal[org], row['vader_polarity'], 1])
                # indiv_org_ref_arr.append(
                #     [row['date'], org, row['tweet_organization_tags'][org], row['vader_polarity'], 1])

        # convert the 2D array into a df
        indiv_org_ref_df = pd.DataFrame(
            indiv_org_ref_arr, columns=['date', 'organization', 'tags', 'sentiment', 'count'])

        # print('indiv_org_ref_df: ', indiv_org_ref_df)

        sum_sr = indiv_org_ref_df.groupby(
            ['date', 'organization'])['count'].sum()

        print(sum_sr)

        sum_df = pd.DataFrame()

        # print('date', sum_sr.loc['date'])
        # print('date', sum_sr.loc['organization'])
        # print('date', sum_sr.loc['count'])

        # print('shit', sum_sr.index[10])

        # print('date ting', sum_sr.index.unique(level='date'))
        # print('org ting', sum_sr.index.unique(level='organization'))
        # print('values', sum_sr.values)

        date_arr = []
        org_arr = []
        count_arr = []

        for i in range(0, sum_sr.size):
            index = sum_sr.index[i]
            value = sum_sr.values[i]

            date_arr.append(index[0])
            org_arr.append(index[1])
            count_arr.append(value)

        sum_df['date'] = date_arr
        sum_df['organization'] = org_arr
        sum_df['count'] = count_arr

        if (analysis_selection == 'overall'):
            fig = px.bar(sum_df, x="organization", y="count",
                         color="date", )

            fig.update_layout(
                plot_bgcolor=colors['background'],
                paper_bgcolor=colors['background'],
                font_color=colors['text'],
                xaxis={'categoryorder': 'total descending'}
            )

            if max_x != None:
                fig.update_xaxes(range=(-0.5, int(max_x) - 0.5))

            return fig
        elif (analysis_selection == 'progressive'):
            print('progressive!')

            # print('indiv_org_ref_df', indiv_org_ref_df)

            # indiv_org_ref_df.sort_values(by=['col1'])

            fig = px.bar(sum_df,
                         x='date', y='count', color="organization",
                         )

            # fig = px.bar(indiv_org_ref_df, x="organization", y="count",
            #              color="organization",
            #              hover_data=['tags', 'sentiment'], )

            fig.update_layout(
                plot_bgcolor=colors['background'],
                paper_bgcolor=colors['background'],
                font_color=colors['text'],
                xaxis={'categoryorder': 'total descending'}
            )

            return fig
    elif organizations_selection == 'tags':

        # create a 2D array with tags having their own row
        indiv_tags_arr = []

        for _, row in df.iterrows():
            tags_dict_literal = ast.literal_eval(
                row['tweet_tags_classified'])
            for tag in tags_dict_literal.keys():
                indiv_tags_arr.append(
                    [row['date'], tag, tags_dict_literal[tag], row['vader_polarity'], 1])

        # convert the 2D array into a df
        indiv_tags_df = pd.DataFrame(
            indiv_tags_arr, columns=['date', 'tag', 'companies', 'sentiment', 'count'])

        sum_sr = indiv_tags_df.groupby(
            ['date', 'tag'])['count'].sum()

        print(sum_sr)

        sum_df = pd.DataFrame()

        # print('date', sum_sr.loc['date'])
        # print('date', sum_sr.loc['organization'])
        # print('date', sum_sr.loc['count'])

        # print('shit', sum_sr.index[10])

        # print('date ting', sum_sr.index.unique(level='date'))
        # print('org ting', sum_sr.index.unique(level='organization'))
        # print('values', sum_sr.values)

        date_arr = []
        tag_arr = []
        count_arr = []

        for i in range(0, sum_sr.size):
            index = sum_sr.index[i]
            value = sum_sr.values[i]

            date_arr.append(index[0])
            tag_arr.append(index[1])
            count_arr.append(value)

        sum_df['date'] = date_arr
        sum_df['tag'] = tag_arr
        sum_df['count'] = count_arr

        if (analysis_selection == 'overall'):
            fig = px.bar(sum_df, x="tag", y="count",
                         color="date", )

            fig.update_layout(
                plot_bgcolor=colors['background'],
                paper_bgcolor=colors['background'],
                font_color=colors['text'],
                xaxis={'categoryorder': 'total descending'}
            )

            if max_x != None:
                fig.update_xaxes(range=(-0.5, int(max_x) - 0.5))

            return fig
        elif (analysis_selection == 'progressive'):
            fig = px.bar(sum_df, x="date", y="count",
                         color="tag", )

            fig.update_layout(
                plot_bgcolor=colors['background'],
                paper_bgcolor=colors['background'],
                font_color=colors['text'],
                xaxis={'categoryorder': 'total descending'}
            )

            return fig
    elif organizations_selection == 'hashtags':
        print('hashtags!')

        # create a 2D array with Organization references having their own row
        indiv_hashtag_ref_arr = []

        for _, row in df.iterrows():
            # https://stackoverflow.com/questions/23119472/in-pandas-python-reading-array-stored-as-string
            # to_csv makes arrays of strings string so need to extract the array back
            # print('row tings', row['hashtags'])
            # print('row tings type', type(row['hashtags']))
            hashtag_literal = ast.literal_eval(
                row['hashtags'])

            # print('hashtag_literal', hashtag_literal)
            for hashtag in hashtag_literal:
                # print('index', index)
                # print('date', row['date'])
                # print('date type', type(row['date']))
                # print('hashtag', hashtag)

                # print('hashtag type', type(hashtag))

                # hashtag_dict_literal = ast.literal_eval(
                #     row['hashtags'])

                # print('hashtag_dict_literal', hashtag_dict_literal)
                # print('hashtag_tag', hashtag)

                # print('vader_polarity', row['vader_polarity'])

                indiv_hashtag_ref_arr.append(
                    [row['date'], hashtag, row['vader_polarity'], 1])
                # indiv_hashtag_ref_arr.append(
                #     [row['date'], hashtag, row['tweet_hashtaganization_tags'][hashtag], row['vader_polarity'], 1])

        # convert the 2D array into a df
        indiv_hashtag_ref_df = pd.DataFrame(
            indiv_hashtag_ref_arr, columns=['date', 'hashtag', 'sentiment', 'count'])

        print('indiv_hashtag_ref_df: ', indiv_hashtag_ref_df)

        sum_sr = indiv_hashtag_ref_df.groupby(
            ['date', 'hashtag'])['count'].sum()

        print(sum_sr)

        sum_df = pd.DataFrame()

        # print('date', sum_sr.loc['date'])
        # print('date', sum_sr.loc['organization'])
        # print('date', sum_sr.loc['count'])

        # print('shit', sum_sr.index[10])

        # print('date ting', sum_sr.index.unique(level='date'))
        # print('org ting', sum_sr.index.unique(level='organization'))
        # print('values', sum_sr.values)

        date_arr = []
        hashtag_arr = []
        count_arr = []

        for i in range(0, sum_sr.size):
            index = sum_sr.index[i]
            value = sum_sr.values[i]

            date_arr.append(index[0])
            hashtag_arr.append(index[1])
            count_arr.append(value)

        sum_df['date'] = date_arr
        sum_df['hashtag'] = hashtag_arr
        sum_df['count'] = count_arr

        if (analysis_selection == 'overall'):
            ig = px.bar(sum_df, x="hashtag", y="count",
                        color="date", )

            fig.update_layout(
                plot_bgcolor=colors['background'],
                paper_bgcolor=colors['background'],
                font_color=colors['text'],
                xaxis={'categoryorder': 'total descending'}
            )

            if max_x != None:
                fig.update_xaxes(range=(-0.5, int(max_x) - 0.5))

            return fig
        elif (analysis_selection == 'progressive'):
            print('progressive!')
            fig = px.bar(sum_df, x="date", y="count",
                         color="hashtag", )

            fig.update_layout(
                plot_bgcolor=colors['background'],
                paper_bgcolor=colors['background'],
                font_color=colors['text'],
                xaxis={'categoryorder': 'total descending'}
            )

            return fig


@ app.callback(
    dash.dependencies.Output('choice_orgs_selection', 'style'),
    [dash.dependencies.Input('choice_all_tweets', 'value')])
def update_org_centric_input_visibility(choice_tweets):
    if choice_tweets == 'tweets_all':
        return {'display': 'none'}
    elif choice_tweets == 'tweets_orgs':
        return {'display': 'block'}


@ app.callback(
    dash.dependencies.Output('choice_orgs_selection', 'options'),
    [dash.dependencies.Input('choice_all_tweets', 'value')])
def update_org_selection_options(choice_tweets):
    print('updating!', choice_tweets)
    if choice_tweets == 'tweets_all':
        return []
    elif choice_tweets == 'tweets_orgs':
        df = pd.read_csv('./results/consolidated.csv')
        orgs_list = [ast.literal_eval(orgs)
                     for orgs in df['tweet_mentioned_organizations']]
        # print('orgs_list', orgs_list)

        orgs = np.unique(
            np.array(functools.reduce(lambda a, b: a + b, orgs_list)))
        print('orgs', orgs)
        return [{'label': org, 'value': org} for org in orgs]


@ app.callback(
    dash.dependencies.Output('choice_orgs_selection', 'value'),
    [dash.dependencies.Input('choice_all_tweets', 'value'),
     dash.dependencies.Input('choice_orgs_selection', 'options')])
def update_org_dafault_selection(choice_tweets, org_options):
    print('updating!', choice_tweets)
    if choice_tweets == 'tweets_all':
        return []
    elif choice_tweets == 'tweets_orgs':
        return [org['value'] for org in org_options]


# @ app.callback(
#     dash.dependencies.Output('choice_orgs_selection', 'style'),
#     [dash.dependencies.Input('choice_org_centric', 'value'),
#      dash.dependencies.Input('choice_analysis', 'value')])
# def update_org_selection_visibility(choice_org_centric, analysis_selection):
#     if (choice_org_centric == 'org_centric') & (analysis_selection == 'overall'):
#         return {'display': 'block'}
#     else:
#         return {'display': 'none'}


@ app.callback(
    dash.dependencies.Output('choice_max_x', 'style'),
    [dash.dependencies.Input('choice_analysis', 'value')])
def update_max_x_input_visibility(analysis_selection):
    if analysis_selection == 'progressive':
        return {'display': 'none'}
    elif analysis_selection == 'overall':
        return {'display': 'block'}


@ app.callback(
    dash.dependencies.Output('choice_max_x', 'placeholder'),
    [dash.dependencies.Input('choice_organizations', 'value')])
def update_placeholder_text(organizations_selection):
    if organizations_selection == 'organizations':
        return 'Enter Max Organizations'
    elif organizations_selection == 'tags':
        return 'Enter Max Tags'
    elif organizations_selection == 'hashtags':
        return 'Enter Max Hashtags'


@ app.callback(
    dash.dependencies.Output('choice_trending_thresh', 'style'),
    [dash.dependencies.Input('choice_consolidated_trending', 'value')])
def update_thresh_input_visibility(data_selection):
    if data_selection == 'full':
        return {'display': 'none'}
    elif data_selection == 'trending_retweets':
        return {'display': 'block'}
    elif data_selection == 'trending_favs':
        return {'display': 'block'}


@ app.callback(
    dash.dependencies.Output('choice_org_centric', 'style'),
    [dash.dependencies.Input('choice_analysis', 'value'),
     dash.dependencies.Input('choice_all_tweets', 'value')])
def update_analysis_drop_visibility(analysis_selection, tweets_selection):
    if (analysis_selection == 'overall') & (tweets_selection == 'tweets_orgs'):
        return {'display': 'block'}
    else:
        return {'display': 'none'}


if __name__ == '__main__':
    app.run_server(debug=True)
