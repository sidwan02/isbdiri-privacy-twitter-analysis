import functools
import math
import numpy as np
import plotly.graph_objs as go
import plotly.figure_factory as ff
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import State, Input, Output
from datetime import date, timedelta, datetime
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

app.title = 'Privacy Twitter Analysis'

app.layout = html.Div([
    html.Div([
        html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode())),
        html.H1('Twitter Privacy Analysis'),
        html.H3('Organization References'),
        html.Div('Siddharth Diwan, Anirudh Syal'),
    ], style={'width': '40%', 'height': '40%', 'align-items': 'center', 'justify-content': 'center', 'padding': '20px', 'background-color': 'white',   'border-radius': '5px', 'box-shadow': '2px 2px 2px lightgrey', 'position': 'absolute', 'right': '5%', 'top': '10%', 'color': 'black'
              }),

    html.Div([
        html.Div([
            dcc.DatePickerRange(
                id='date_picker',
                # min_date_allowed=date(2020, 8, 1),
                # max_date_allowed=date(2020, 12, 31),
                start_date=date(2021, 1, 8),
                end_date=date(2021, 2, 28),
                # style={"margin-top": "15px"}
            ),
            dcc.Dropdown(
                id='choice_consolidated_trending',
                options=[
                    # {'label': 'Full Consolidated', 'value': 'full'},
                    {'label': 'Trending Retweets',
                     'value': 'trending_retweets'},
                    {'label': 'Trending Favourites',
                     'value': 'trending_favs'},
                ],
                value='trending_retweets',
                clearable=False,
                style={"margin-top": "15px"}
            ),
            html.Div(children=[

            ], style={"margin-top": "15px"}),
            dcc.Slider(
                id='choice_trending_thresh_slider',
                # placeholder='Enter a threshold',
                min=0,
                max=1000,
                updatemode='drag',
                value=0
            ),
            html.Div(id='choice_trending_thresh', children=[

            ]),

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
                    id='choice_org_centric',
                    options=[
                        {'label': 'Date Centric', 'value': 'date_centric'},
                        {'label': 'Organization Centric',
                         'value': 'org_centric'},
                    ],
                    value='org_centric',
                    clearable=False,
                    style={"margin-top": "15px"}
                ),
                dcc.Dropdown(
                    id='choice_tweet_property',
                    options=[
                        {'label': 'Retweets', 'value': 'Retweets'},
                        {'label': 'Favourties', 'value': 'Favourites'},
                        {'label': 'Sentiment', 'value': 'Sentiment'},
                    ],
                    value='Retweets',
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

            html.Div([
                dcc.Graph(
                    id='graph_central_tendency_2'
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
                        {'label': 'Industries Mentioned', 'value': 'tags'},
                        # {'label': 'Hashtags', 'value': 'hashtags'},
                        {'label': 'Weighted Sentiment', 'value': 'weighted'}
                    ],
                    value='organizations',
                    clearable=False
                ),
                dcc.Input(
                    id='choice_min_count',
                    type='number',
                    value=0,
                    placeholder='Enter Min Frequency'
                ),

            ], style={'width': '40%', 'align-items': 'center', 'justify-content': 'center', 'padding': '20px', 'background-color': '#f7f7f7',   'border-radius': '5px', 'box-shadow': '2px 2px 2px lightgrey'
                      }),
            html.Div([
                dcc.Graph(
                    id='graph_organizations_and_tags'
                )
            ], style={'width': '97%', 'align-items': 'center', 'justify-content': 'center', 'padding': '20px', 'background-color': '#f7f7f7',   'border-radius': '5px', 'box-shadow': '2px 2px 2px lightgrey', 'margin-top': '10px'
                      }),

            html.Div([
                dcc.Dropdown(
                    id='choice_organizations_2',
                    options=[
                        {'label': 'Organizations Mentioned',
                            'value': 'organizations'},
                        {'label': 'Industries Mentioned', 'value': 'tags'},
                        # {'label': 'Top n Hashtags', 'value': 'hashtags'},
                        {'label': 'Frequent Phrases', 'value': 'phrases'}
                    ],
                    value='organizations',
                    clearable=False
                ),
                dcc.Input(
                    id='choice_max_x_2',
                    type='number',
                    value=10,
                    placeholder='Enter Max Displayed'
                ),
                dcc.Input(
                    id='choice_min_count_2',
                    type='number',
                    value=0,
                    placeholder='Enter Min Frequency'
                ),

            ], style={'width': '40%', 'align-items': 'center', 'justify-content': 'center', 'padding': '20px', 'background-color': '#f7f7f7',   'border-radius': '5px', 'box-shadow': '2px 2px 2px lightgrey'
                      }),

            html.Div([
                dcc.Graph(
                    id='graph_organizations_and_tags_2'
                )
            ], style={'width': '97%', 'align-items': 'center', 'justify-content': 'center', 'padding': '20px', 'background-color': '#f7f7f7',   'border-radius': '5px', 'box-shadow': '2px 2px 2px lightgrey', 'margin-top': '10px'
                      }),
        ], style={'width': '97%', 'align-items': 'center', 'justify-content': 'center', 'padding': '20px', 'background-color': '#e6e6e6',   'border-radius': '5px', 'box-shadow': '2px 2px 2px lightgrey', 'margin-top': '10px'
                  }),


        html.Div([
            "Siddharth Diwan ",
            html.A(
                'LinkedIn', href='https://www.linkedin.com/in/siddharth-diwan-10a4701b3/'),
            " | Anirudh Syal ",
            html.A('LinkedIn', href='https://www.linkedin.com/in/anirudhsyal/')

        ], style={'align-items': 'center', 'justify-content': 'center', 'text-align': 'center', 'background-color': 'white',   'border-radius': '5px', 'box-shadow': '2px 2px 2px lightgrey', 'color': 'black',  'margin-top': '10px'
                  }),
    ], style={'width': '97%', 'align-items': 'center', 'justify-content': 'center', 'padding': '20px', 'background-color': '#bababa',   'border-radius': '5px', 'box-shadow': '2px 2px 2px lightgrey', 'margin-top': '10px'
              }),

], style={
    'backgroundColor': 'white',
    'color': colors['text'],
})


def diverge_sentiment(df, color_col_name):
    if color_col_name == 'display_orgs & date':
        df['vader_polarity'] = np.where(
            df['vader_polarity'] < 0.7, -1, 1)

        df['count'] = [1] * len(df.index)

        sum_sr = df.groupby(
            ['date', 'display_orgs', 'vader_polarity'])['count'].sum()
        sum_df = pd.DataFrame()

        date_arr = []
        orgs_arr = []
        vader_arr = []
        count_arr = []

        for i in range(0, sum_sr.size):
            index = sum_sr.index[i]
            value = sum_sr.values[i]
            # #print('index', index)
            # #print('value', value)
            date_arr.append(index[0])
            orgs_arr.append(index[1])
            vader_arr.append(index[2])
            count_arr.append(value)

        sum_df['date'] = date_arr
        sum_df['display_orgs'] = orgs_arr
        sum_df['sentiment_score'] = vader_arr
        sum_df['count'] = count_arr
    else:
        df['vader_polarity'] = np.where(
            df['vader_polarity'] < 0.7, -1, 1)

        df['count'] = [1] * len(df.index)

        sum_sr = df.groupby(
            [color_col_name, 'vader_polarity'])['count'].sum()
        sum_df = pd.DataFrame()

        color_col_arr = []
        vader_arr = []
        count_arr = []

        for i in range(0, sum_sr.size):
            index = sum_sr.index[i]
            value = sum_sr.values[i]
            # #print('index', index)
            # #print('value', value)
            color_col_arr.append(index[0])
            vader_arr.append(index[1])
            count_arr.append(value)

        sum_df[color_col_name] = color_col_arr
        sum_df['sentiment_score'] = vader_arr
        sum_df['count'] = count_arr

    return sum_df


@ app.callback(
    dash.dependencies.Output('graph_central_tendency', 'figure'),
    [dash.dependencies.Input('date_picker', 'start_date'),
     dash.dependencies.Input('date_picker', 'end_date'),
     dash.dependencies.Input('choice_consolidated_trending', 'value'),
     dash.dependencies.Input('choice_all_tweets', 'value'),
     dash.dependencies.Input('choice_tweet_property', 'value'),
     dash.dependencies.Input('choice_trending_thresh_slider', 'value'),
     dash.dependencies.Input('choice_org_centric', 'value'),
     dash.dependencies.Input('choice_orgs_selection', 'value')],
)
def update_graph_central_tendency(start_date, end_date, data_selection, tweets_selection, tendency_selection, thresh, centric_selection, companies_selection):
    print('companies_selection', companies_selection)
    if companies_selection == None:
        companies_selection = []
    # #print('yo')
    start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')

    # https://stackoverflow.com/questions/59882714/python-generating-a-list-of-dates-between-two-dates
    # https://stackoverflow.com/questions/18684076/how-to-create-a-list-of-date-string-in-yyyymmdd-format-with-python-pandas
    date_range = [d.strftime('%Y-%m-%d') for d in pd.date_range(
        start_date_obj, end_date_obj-timedelta(days=1), freq='d')]

    if data_selection == 'full':
        df = pd.read_csv('results/consolidated_date_formatted.csv')
    elif data_selection == 'trending_retweets':
        df = filter_by_col('retweet_count', thresh)
    elif data_selection == 'trending_favs':
        df = filter_by_col('fav_count', thresh)

    # # https://stackoverflow.com/questions/12096252/use-a-list-of-values-to-select-rows-from-a-pandas-dataframe
    # #print('df_date', df['date'])

    df = df[df['date'].isin(date_range)]
    # #print('date_range', date_range)

    if tweets_selection == 'tweets_all':
        if (tendency_selection == 'Retweets'):

            fig = px.strip(df, x="date", y="retweet_count",
                           hover_data=["tweet_cleaned"])

            fig.update_layout(
                plot_bgcolor=colors['background'],
                paper_bgcolor=colors['background'],
                font_color=colors['text']
            )

            # # https://www.codegrepper.com/code-examples/python/how+to+find+mean+of+one+column+based+on+another+column+in+python
            # mean_sr = df.groupby('date')['retweet_count'].mean()
            # mean_df = pd.DataFrame(
            #     {'date': mean_sr.index, 'retweet_count': mean_sr.values})

            # median_sr = df.groupby('date')['retweet_count'].median()
            # median_df = pd.DataFrame(
            #     {'date': median_sr.index, 'retweet_count': median_sr.values})

            # # https://stackoverflow.com/questions/62122015/how-to-add-traces-in-plotly-express
            # fig.add_trace(go.Scatter(
            #     x=mean_df['date'], y=mean_df['retweet_count'], name='mean', visible="legendonly"))
            # fig.add_trace(go.Scatter(
            #     x=median_df['date'], y=median_df['retweet_count'], name='median', visible="legendonly"))

            return fig

        elif (tendency_selection == 'Favourites'):

            fig = px.strip(df, x="date", y="fav_count",
                           )

            fig.update_layout(
                plot_bgcolor=colors['background'],
                paper_bgcolor=colors['background'],
                font_color=colors['text']
            )

            # mean_sr = df.groupby('date')['fav_count'].mean()
            # mean_df = pd.DataFrame(
            #     {'date': mean_sr.index, 'fav_count': mean_sr.values})

            # median_sr = df.groupby('date')['fav_count'].median()
            # median_df = pd.DataFrame(
            #     {'date': median_sr.index, 'fav_count': median_sr.values})

            # fig.add_trace(go.Scatter(
            #     x=mean_df['date'], y=mean_df['fav_count'], name='mean', visible="legendonly"))
            # fig.add_trace(go.Scatter(
            #     x=median_df['date'], y=median_df['fav_count'], name='median', visible="legendonly"))

            return fig

        elif (tendency_selection == 'Sentiment'):
            df = diverge_sentiment(df, 'date')

            # #print('df gleeeeeeeeeeee', sum_df)

            fig = px.bar(df, x="date", y="count",
                         color="sentiment_score")

            fig.update_layout(
                plot_bgcolor=colors['background'],
                paper_bgcolor=colors['background'],
                font_color=colors['text']
            )

            # mean_sr = df.groupby(
            #     'date')['vader_polarity'].mean()
            # mean_df = pd.DataFrame(
            #     {'date': mean_sr.index, 'vader_polarity': mean_sr.values})

            # median_sr = df.groupby('date')['vader_polarity'].median()
            # median_df = pd.DataFrame(
            #     {'date': median_sr.index, 'vader_polarity': median_sr.values})

            # fig.add_trace(go.Scatter(
            #     x=mean_df['date'], y=mean_df['vader_polarity'], name='mean', visible="legendonly"))
            # fig.add_trace(go.Scatter(
            #     x=median_df['date'], y=median_df['vader_polarity'], name='median', visible="legendonly"))

            return fig
    elif tweets_selection == 'tweets_orgs':

        def common_data(list1, list2):
            # traverse in the 1st list
            for x in list1:
                # #print('x', x)

                # traverse in the 2nd list
                for y in list2:

                    # if one common
                    if x == y:
                        # #print('found!')
                        return True
            # #print('not found')
            return False

        # print('loblo', df)

        # print('1', df['tweet_mentioned_organizations'])

        mask = [common_data(ast.literal_eval(orgs), list(companies_selection))
                for orgs in df['tweet_mentioned_organizations']]

        df = df[mask]

        # print('Daily, orgs, centric selection', centric_selection)

        # print('2', df['tweet_mentioned_organizations'])

        if centric_selection == 'date_centric':
            if (tendency_selection == 'Retweets'):
                # print('hi retweets')

                fig = px.strip(df, x="date", y="retweet_count",
                               )

                fig.update_layout(
                    plot_bgcolor=colors['background'],
                    paper_bgcolor=colors['background'],
                    font_color=colors['text']
                )

                # https://www.codegrepper.com/code-examples/python/how+to+find+mean+of+one+column+based+on+another+column+in+python
                # mean_sr = df.groupby('date')['retweet_count'].mean()
                # mean_df = pd.DataFrame(
                #     {'date': mean_sr.index, 'retweet_count': mean_sr.values})

                # median_sr = df.groupby('date')['retweet_count'].median()
                # median_df = pd.DataFrame(
                #     {'date': median_sr.index, 'retweet_count': median_sr.values})

                # # https://stackoverflow.com/questions/62122015/how-to-add-traces-in-plotly-express
                # fig.add_trace(go.Scatter(
                #     x=mean_df['date'], y=mean_df['retweet_count'], name='mean', visible="legendonly"))
                # fig.add_trace(go.Scatter(
                #     x=median_df['date'], y=median_df['retweet_count'], name='median', visible="legendonly"))

                return fig

            elif (tendency_selection == 'Favourites'):

                fig = px.strip(df, x="date", y="fav_count",
                               )

                fig.update_layout(
                    plot_bgcolor=colors['background'],
                    paper_bgcolor=colors['background'],
                    font_color=colors['text']
                )

                # mean_sr = df.groupby('date')['fav_count'].mean()
                # mean_df = pd.DataFrame(
                #     {'date': mean_sr.index, 'fav_count': mean_sr.values})

                # median_sr = df.groupby('date')['fav_count'].median()
                # median_df = pd.DataFrame(
                #     {'date': median_sr.index, 'fav_count': median_sr.values})

                # fig.add_trace(go.Scatter(
                #     x=mean_df['date'], y=mean_df['fav_count'], name='mean', visible="legendonly"))
                # fig.add_trace(go.Scatter(
                #     x=median_df['date'], y=median_df['fav_count'], name='median', visible="legendonly"))

                return fig

            elif (tendency_selection == 'Sentiment'):
                df = diverge_sentiment(df, 'date')

                # #print('df gleeeeeeeeeeee', sum_df)

                fig = px.bar(df, x="date", y="count",
                             color="sentiment_score")

                fig.update_layout(
                    plot_bgcolor=colors['background'],
                    paper_bgcolor=colors['background'],
                    font_color=colors['text']
                )

                # mean_sr = df.groupby(
                #     'date')['vader_polarity'].mean()
                # mean_df = pd.DataFrame(
                #     {'date': mean_sr.index, 'vader_polarity': mean_sr.values})

                # median_sr = df.groupby('date')['vader_polarity'].median()
                # median_df = pd.DataFrame(
                #     {'date': median_sr.index, 'vader_polarity': median_sr.values})

                # fig.add_trace(go.Scatter(
                #     x=mean_df['date'], y=mean_df['vader_polarity'], name='mean', visible="legendonly"))
                # fig.add_trace(go.Scatter(
                #     x=median_df['date'], y=median_df['vader_polarity'], name='median', visible="legendonly"))

                return fig

            # elif analysis_selection == 'overall':
            #     count_arr = [1] * len(df.index)
            #     df['count'] = count_arr
            #     if (tendency_selection == 'Retweets'):

            #         fig = px.histogram(df, x="retweet_count", y="count",
            #                         color="date", marginal='rug', hover_data=['vader_polarity'], nbins=50, )

            #         fig.update_layout(
            #             plot_bgcolor=colors['background'],
            #             paper_bgcolor=colors['background'],
            #             font_color=colors['text']
            #         )

            #         return fig

            #     elif (tendency_selection == 'Favourites'):

            #         fig = px.histogram(df, x="fav_count", y="count",
            #                         color="date", marginal='rug', hover_data=['vader_polarity'], nbins=50, )

            #         fig.update_layout(
            #             plot_bgcolor=colors['background'],
            #             paper_bgcolor=colors['background'],
            #             font_color=colors['text']
            #         )

            #         return fig

            #     elif (tendency_selection == 'Sentiment'):

            #         df = diverge_sentiment(df, 'date')

            #         fig = px.bar(df, x="vader_polarity", y="count",
            #                         color="date")
            #         # fig = px.histogram(df, x="vader_polarity", y="count",
            #         #                    color="date", marginal='rug', hover_data=['tweet_mentioned_organizations'], nbins=50, )

            #         fig.update_layout(
            #             plot_bgcolor=colors['background'],
            #             paper_bgcolor=colors['background'],
            #             font_color=colors['text']
            #         )

            #         return fig
        elif centric_selection == 'org_centric':
            # print(df.columns)
            # print(df)
            orgs_literal = list(map(lambda x: ast.literal_eval(
                x), df['tweet_mentioned_organizations'].to_numpy()))

            # print('orgs_literal', orgs_literal)

            # #print(orgs_literal)

            df['display_orgs'] = list(map(
                lambda x: functools.reduce(lambda a, b: a + ', ' + b, x), orgs_literal))

            # print('FINALLYFINALLY', df['display_orgs'])

            if (tendency_selection == 'Retweets'):
                # print('hi retweets')

                fig = px.strip(df, x="date", y="retweet_count", color='display_orgs'
                               )

                fig.update_layout(
                    plot_bgcolor=colors['background'],
                    paper_bgcolor=colors['background'],
                    font_color=colors['text']
                )

                # https://www.codegrepper.com/code-examples/python/how+to+find+mean+of+one+column+based+on+another+column+in+python
                # mean_sr = df.groupby('date')['retweet_count'].mean()
                # mean_df = pd.DataFrame(
                #     {'date': mean_sr.index, 'retweet_count': mean_sr.values})

                # median_sr = df.groupby('date')['retweet_count'].median()
                # median_df = pd.DataFrame(
                #     {'date': median_sr.index, 'retweet_count': median_sr.values})

                # # https://stackoverflow.com/questions/62122015/how-to-add-traces-in-plotly-express
                # fig.add_trace(go.Scatter(
                #     x=mean_df['date'], y=mean_df['retweet_count'], name='mean', visible="legendonly"))
                # fig.add_trace(go.Scatter(
                #     x=median_df['date'], y=median_df['retweet_count'], name='median', visible="legendonly"))

                return fig

            elif (tendency_selection == 'Favourites'):

                fig = px.strip(df, x="date", y="fav_count", color='display_orgs'
                               )

                fig.update_layout(
                    plot_bgcolor=colors['background'],
                    paper_bgcolor=colors['background'],
                    font_color=colors['text']
                )

                # mean_sr = df.groupby('date')['fav_count'].mean()
                # mean_df = pd.DataFrame(
                #     {'date': mean_sr.index, 'fav_count': mean_sr.values})

                # median_sr = df.groupby('date')['fav_count'].median()
                # median_df = pd.DataFrame(
                #     {'date': median_sr.index, 'fav_count': median_sr.values})

                # fig.add_trace(go.Scatter(
                #     x=mean_df['date'], y=mean_df['fav_count'], name='mean', visible="legendonly"))
                # fig.add_trace(go.Scatter(
                #     x=median_df['date'], y=median_df['fav_count'], name='median', visible="legendonly"))

                return fig

            elif (tendency_selection == 'Sentiment'):
                df = diverge_sentiment(df, 'display_orgs & date')

                # #print('df gleeeeeeeeeeee', sum_df)

                fig = px.bar(df, x="date", y="count",
                             color="sentiment_score", text="display_orgs")

                fig.update_layout(
                    plot_bgcolor=colors['background'],
                    paper_bgcolor=colors['background'],
                    font_color=colors['text']
                )

                # mean_sr = df.groupby(
                #     'date')['vader_polarity'].mean()
                # mean_df = pd.DataFrame(
                #     {'date': mean_sr.index, 'vader_polarity': mean_sr.values})

                # median_sr = df.groupby('date')['vader_polarity'].median()
                # median_df = pd.DataFrame(
                #     {'date': median_sr.index, 'vader_polarity': median_sr.values})

                # fig.add_trace(go.Scatter(
                #     x=mean_df['date'], y=mean_df['vader_polarity'], name='mean', visible="legendonly"))
                # fig.add_trace(go.Scatter(
                #     x=median_df['date'], y=median_df['vader_polarity'], name='median', visible="legendonly"))

                return fig


@ app.callback(
    dash.dependencies.Output('graph_central_tendency_2', 'figure'),
    [dash.dependencies.Input('date_picker', 'start_date'),
     dash.dependencies.Input('date_picker', 'end_date'),
     dash.dependencies.Input('choice_consolidated_trending', 'value'),
     dash.dependencies.Input('choice_all_tweets', 'value'),
     dash.dependencies.Input('choice_tweet_property', 'value'),
     dash.dependencies.Input('choice_trending_thresh_slider', 'value'),
     dash.dependencies.Input('choice_org_centric', 'value'),
     dash.dependencies.Input('choice_orgs_selection', 'value')],
)
def update_graph_central_tendency_2(start_date, end_date, data_selection, tweets_selection, tendency_selection, thresh, centric_selection, companies_selection):
    print('companies_selection', companies_selection)
    if companies_selection == None:
        companies_selection = []
    # #print('yo')
    start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')

    # https://stackoverflow.com/questions/59882714/python-generating-a-list-of-dates-between-two-dates
    # https://stackoverflow.com/questions/18684076/how-to-create-a-list-of-date-string-in-yyyymmdd-format-with-python-pandas
    date_range = [d.strftime('%Y-%m-%d') for d in pd.date_range(
        start_date_obj, end_date_obj-timedelta(days=1), freq='d')]

    if data_selection == 'full':
        df = pd.read_csv('results/consolidated_date_formatted.csv')
    elif data_selection == 'trending_retweets':
        df = filter_by_col('retweet_count', thresh)
    elif data_selection == 'trending_favs':
        df = filter_by_col('fav_count', thresh)

    # # https://stackoverflow.com/questions/12096252/use-a-list-of-values-to-select-rows-from-a-pandas-dataframe
    # #print('df_date', df['date'])

    df = df[df['date'].isin(date_range)]
    # #print('date_range', date_range)

    # print('overall')
    if tweets_selection == 'tweets_all':
        count_arr = [1] * len(df.index)
        df['count'] = count_arr
        df['sentiment_score'] = df['vader_polarity']
        if (tendency_selection == 'Retweets'):

            fig = px.histogram(df, x="retweet_count", y="count",
                               color="date",
                               #    marginal='rug',
                               hover_data=['sentiment_score'], nbins=50, )

            fig.update_layout(
                plot_bgcolor=colors['background'],
                paper_bgcolor=colors['background'],
                font_color=colors['text']
            )

            return fig

        elif (tendency_selection == 'Favourites'):

            fig = px.histogram(df, x="fav_count", y="count",
                               color="date",
                               #    marginal='rug',
                               hover_data=['sentiment_score'], nbins=50,
                               labels={
                                   "date": "HAHA",
                                   "count": "HOHO",
                               })

            fig.update_layout(
                plot_bgcolor=colors['background'],
                paper_bgcolor=colors['background'],
                font_color=colors['text']
            )

            return fig

        elif (tendency_selection == 'Sentiment'):
            df = diverge_sentiment(df, 'date')
            fig = px.bar(df, x="sentiment_score", y="count",
                         color="date")
            # fig = px.histogram(df, x="vader_polarity", y="count",
            #                    color="date", marginal='rug', hover_data=['tweet_mentioned_organizations'], nbins=50, )

            fig.update_layout(
                plot_bgcolor=colors['background'],
                paper_bgcolor=colors['background'],
                font_color=colors['text']
            )

            return fig

    elif tweets_selection == 'tweets_orgs':
        # print('tweets_orgs')

        def common_data(list1, list2):
            # #print('list1', list1)
            # #print('list2', list2)
            # traverse in the 1st list
            for x in list1:
                # #print('x', x)

                # traverse in the 2nd list
                for y in list2:

                    # if one common
                    if x == y:
                        # #print('found!')
                        return True
            # #print('not found')
            return False

        mask = [common_data(ast.literal_eval(orgs), list(companies_selection))
                for orgs in df['tweet_mentioned_organizations']]

        df = df[mask]

        df['sentiment_score'] = df['vader_polarity']
        # print('Daily, orgs, centric selection', centric_selection)
        if centric_selection == 'date_centric':

            # #print('df', df)

            # fig = px.strip(df, x="date", y="retweet_count", )

            # return 'hi'

            count_arr = [1] * len(df.index)
            df['count'] = count_arr
            if (tendency_selection == 'Retweets'):

                fig = px.histogram(df, x="retweet_count", y="count",
                                   color="date",
                                   #    marginal='rug',
                                   hover_data=['sentiment_score'], nbins=50, )

                fig.update_layout(
                    plot_bgcolor=colors['background'],
                    paper_bgcolor=colors['background'],
                    font_color=colors['text']
                )

                return fig

            elif (tendency_selection == 'Favourites'):

                fig = px.histogram(df, x="fav_count", y="count",
                                   color="date",
                                   #    marginal='rug',
                                   hover_data=['sentiment_score'], nbins=50, )

                fig.update_layout(
                    plot_bgcolor=colors['background'],
                    paper_bgcolor=colors['background'],
                    font_color=colors['text']
                )

                return fig

            elif (tendency_selection == 'Sentiment'):
                df = diverge_sentiment(df, 'date')
                fig = px.bar(df, x="sentiment_score", y="count",
                             color="date")
                # fig = px.histogram(df, x="vader_polarity", y="count",
                #                    color="date", marginal='rug', hover_data=['tweet_mentioned_organizations'], nbins=50, )

                fig.update_layout(
                    plot_bgcolor=colors['background'],
                    paper_bgcolor=colors['background'],
                    font_color=colors['text']
                )

                return fig
        elif centric_selection == 'org_centric':
            # print('org_centric')

            orgs_literal = list(map(lambda x: ast.literal_eval(
                x), df['tweet_mentioned_organizations'].to_numpy()))

            # #print(orgs_literal)

            df['display_orgs'] = list(map(
                lambda x: functools.reduce(lambda a, b: a + ', ' + b, x, ), orgs_literal))
            # print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
            # print(df.columns)
            # #print('after filtering the companies selected', df)

            # #print('df', df)

            # fig = px.strip(df, x="date", y="retweet_count", )

            # return 'hi'

            count_arr = [1] * len(df.index)
            df['count'] = count_arr
            df['sentiment_score'] = df['vader_polarity']
            if (tendency_selection == 'Retweets'):

                fig = px.histogram(df, x="retweet_count", y="count",
                                   color="display_orgs", marginal='rug', hover_data=['date', 'fav_count'], nbins=50, )

                fig.update_layout(
                    plot_bgcolor=colors['background'],
                    paper_bgcolor=colors['background'],
                    font_color=colors['text']
                )

                return fig

            elif (tendency_selection == 'Favourites'):

                fig = px.histogram(df, x="fav_count", y="count",
                                   color="display_orgs", marginal='rug', hover_data=['date', 'retweet_count', 'sentiment_score'], nbins=50, )

                fig.update_layout(
                    plot_bgcolor=colors['background'],
                    paper_bgcolor=colors['background'],
                    font_color=colors['text']
                )

                return fig

            elif (tendency_selection == 'Sentiment'):
                df = diverge_sentiment(df, 'display_orgs')
                fig = px.bar(df, x="sentiment_score", y="count",
                             color="display_orgs")
                # fig = px.histogram(df, x="vader_polarity", y="count",
                #                    color="display_orgs", marginal='rug', hover_data=['date', 'retweet_count', 'fav_count'], nbins=50, )

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
     dash.dependencies.Input('choice_organizations', 'value'),
     dash.dependencies.Input('choice_consolidated_trending', 'value'),
     dash.dependencies.Input('choice_trending_thresh_slider', 'value'),
     dash.dependencies.Input('choice_min_count', 'value')])
def update_graph_organizations(start_date, end_date, mode_selection, organizations_selection, data_selection, thresh, min_count):
    # #print('hi')
    start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')

    # https://stackoverflow.com/questions/59882714/python-generating-a-list-of-dates-between-two-dates
    # https://stackoverflow.com/questions/18684076/how-to-create-a-list-of-date-string-in-yyyymmdd-format-with-python-pandas
    date_range = [d.strftime('%Y-%m-%d') for d in pd.date_range(
        start_date_obj, end_date_obj-timedelta(days=1), freq='d')]
    # #print('date_range', date_range)

    # df = pd.read_csv('results/consolidated_date_formatted.csv')
    if data_selection == 'full':
        df = pd.read_csv('results/consolidated_date_formatted.csv')
    elif data_selection == 'trending_retweets':
        df = filter_by_col('retweet_count', thresh)
    elif data_selection == 'trending_favs':
        df = filter_by_col('fav_count', thresh)

    df = df[df['date'].isin(date_range)]

    # #print('df', df)

    if organizations_selection == 'organizations':
        # #print('organizations!')

        # create a 2D array with Organization references having their own row
        indiv_org_ref_arr = []

        for _, row in df.iterrows():
            # https://stackoverflow.com/questions/23119472/in-pandas-python-reading-array-stored-as-string
            # to_csv makes arrays of strings string so need to extract the array back
            orgs_literal = ast.literal_eval(
                row['tweet_mentioned_organizations'])

            # #print('orgs_literal', orgs_literal)
            for org in orgs_literal:
                # #print('index', index)
                # #print('date', row['date'])
                # #print('date type', type(row['date']))
                # #print('org', org)

                # #print('org type', type(org))

                orgs_dict_literal = ast.literal_eval(
                    row['tweet_orgs_classified'])

                # #print('orgs_dict_literal', orgs_dict_literal)
                # #print('org_tag', orgs_dict_literal[org])

                # #print('vader_polarity', row['vader_polarity'])

                indiv_org_ref_arr.append(
                    [row['date'], org, orgs_dict_literal[org], row['vader_polarity'], 1])
                # indiv_org_ref_arr.append(
                #     [row['date'], org, row['tweet_organization_tags'][org], row['vader_polarity'], 1])

        # convert the 2D array into a df
        indiv_org_ref_df = pd.DataFrame(
            indiv_org_ref_arr, columns=['date', 'organization', 'tags', 'sentiment', 'count'])

        # #print('indiv_org_ref_df: ', indiv_org_ref_df)

        sum_sr = indiv_org_ref_df.groupby(
            ['date', 'organization'])['count'].sum()

        # #print(sum_sr)

        sum_df = pd.DataFrame()

        # #print('date', sum_sr.loc['date'])
        # #print('date', sum_sr.loc['organization'])
        # #print('date', sum_sr.loc['count'])

        # #print('shit', sum_sr.index[10])

        # #print('date ting', sum_sr.index.unique(level='date'))
        # #print('org ting', sum_sr.index.unique(level='organization'))
        # #print('values', sum_sr.values)

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

        sum_df = sum_df[sum_df['count'] > min_count]
        # print('sum_df', sum_df)

        # #print('Daily!')

        # #print('indiv_org_ref_df', indiv_org_ref_df)

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

        # #print(sum_sr)

        sum_df = pd.DataFrame()

        # #print('date', sum_sr.loc['date'])
        # #print('date', sum_sr.loc['organization'])
        # #print('date', sum_sr.loc['count'])

        # #print('shit', sum_sr.index[10])

        # #print('date ting', sum_sr.index.unique(level='date'))
        # #print('org ting', sum_sr.index.unique(level='organization'))
        # #print('values', sum_sr.values)

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

        sum_df = sum_df[sum_df['count'] > min_count]

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
        # #print('hashtags!')

        # create a 2D array with Organization references having their own row
        indiv_hashtag_ref_arr = []

        for index, row in df.iterrows():
            # print(index)
            # https://stackoverflow.com/questions/23119472/in-pandas-python-reading-array-stored-as-string
            # to_csv makes arrays of strings string so need to extract the array back
            # #print('row tings', row['hashtags'])
            # #print('row tings type', type(row['hashtags']))
            hashtag_literal = ast.literal_eval(
                row['hashtags'])

            # #print('hashtag_literal', hashtag_literal)
            for hashtag in hashtag_literal:
                # #print('index', index)
                # #print('date', row['date'])
                # #print('date type', type(row['date']))
                # #print('hashtag', hashtag)

                # #print('hashtag type', type(hashtag))

                # hashtag_dict_literal = ast.literal_eval(
                #     row['hashtags'])

                # #print('hashtag_dict_literal', hashtag_dict_literal)
                # #print('hashtag_tag', hashtag)

                # #print('vader_polarity', row['vader_polarity'])

                indiv_hashtag_ref_arr.append(
                    [row['date'], hashtag, row['vader_polarity'], 1])
                # indiv_hashtag_ref_arr.append(
                #     [row['date'], hashtag, row['tweet_hashtaganization_tags'][hashtag], row['vader_polarity'], 1])

        # convert the 2D array into a df
        indiv_hashtag_ref_df = pd.DataFrame(
            indiv_hashtag_ref_arr, columns=['date', 'hashtag', 'sentiment', 'count'])

        # #print('indiv_hashtag_ref_df: ', indiv_hashtag_ref_df)

        sum_sr = indiv_hashtag_ref_df.groupby(
            ['date', 'hashtag'])['count'].sum()
        # print('done with sr')
        # #print(sum_sr)

        sum_df = pd.DataFrame()

        # #print('date', sum_sr.loc['date'])
        # #print('date', sum_sr.loc['organization'])
        # #print('date', sum_sr.loc['count'])

        # #print('shit', sum_sr.index[10])

        # #print('date ting', sum_sr.index.unique(level='date'))
        # #print('org ting', sum_sr.index.unique(level='organization'))
        # #print('values', sum_sr.values)

        date_arr = []
        hashtag_arr = []
        count_arr = []

        for i in range(0, sum_sr.size):
            # print(i)
            index = sum_sr.index[i]
            value = sum_sr.values[i]

            date_arr.append(index[0])
            hashtag_arr.append(index[1])
            count_arr.append(value)

        sum_df['date'] = date_arr
        sum_df['hashtag'] = hashtag_arr
        sum_df['count'] = count_arr

        sum_df = sum_df[sum_df['count'] > min_count]

        # print('Daily!')

        fig = px.bar(sum_df, x="date", y="count",
                     color="hashtag", )

        fig.update_layout(
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font_color=colors['text'],
            xaxis={'categoryorder': 'total descending'}
        )

        return fig
    elif organizations_selection == 'phrases':
        fig = go.Figure()

        # fig.add_layout_image(
        #     dict(
        #         source="results/phrases_cloud.png",
        #         xref="x",
        #         yref="y",
        #         x=0,
        #         y=3,
        #         sizex=2,
        #         sizey=2,
        #         sizing="stretch",
        #         opacity=0.5,
        #         layer="below"
        #     )
        # )
        # Create figure
        fig = go.Figure()

        # Constants
        img_width = 1600
        img_height = 900
        scale_factor = 0.5

        # Add invisible scatter trace.
        # This trace is added to help the autoresize logic work.
        fig.add_trace(
            go.Scatter(
                x=[0, img_width * scale_factor],
                y=[0, img_height * scale_factor],
                mode="markers",
                marker_opacity=0
            )
        )

        # Configure axes
        fig.update_xaxes(
            visible=False,
            range=[0, img_width * scale_factor]
        )

        fig.update_yaxes(
            visible=False,
            range=[0, img_height * scale_factor],
            # the scaleanchor attribute ensures that the aspect ratio stays constant
            scaleanchor="x"
        )

        # Add image
        fig.add_layout_image(
            dict(
                x=0,
                sizex=img_width * scale_factor,
                y=img_height * scale_factor,
                sizey=img_height * scale_factor,
                xref="x",
                yref="y",
                opacity=1.0,
                layer="below",
                # sizing="stretch",
                source="https://raw.githubusercontent.com/sidwan02/isbdiri-privacy-twitter-analysis/main/results/phrases_cloud.png?token=AQ3LMOJZNGN2KT5CHVZ6PMTAB6D5W")
            # src="results/phrases_cloud.png")
        )

        # Configure other layout
        # fig.update_layout(
        #     width=img_width * scale_factor,
        #     height=img_height * scale_factor,
        #     margin={"l": 0, "r": 0, "t": 0, "b": 0},
        # )

        return fig
    elif organizations_selection == 'weighted':
        df['vader_polarity'] = np.where(
            df['vader_polarity'] < 0.7, -1, 1)
        df['virality'] = df['retweet_count'] + df['fav_count']
        df['consolidated_sentiment'] = df['virality'] * df['vader_polarity']

        sum_sr = df.groupby('date')['consolidated_sentiment'].sum()
        # print('sum_sr', sum_sr)
        sum_sr_2 = df.groupby('date')['virality'].sum()

        sum_df = pd.DataFrame()

        date_arr = []
        sentiment_sum_arr = []
        virality_arr = []

        for i in range(0, sum_sr.size):
            index = sum_sr.index[i]
            value = sum_sr.values[i]
            # #print('index', index)
            # #print('value', value)
            date_arr.append(index)
            sentiment_sum_arr.append(value)

        for i in range(0, sum_sr_2.size):
            # index = sum_sr.index[i]
            value = sum_sr_2.values[i]
            # #print('index', index)
            # #print('value', value)
            # .append(index)
            virality_arr.append(value)

        # print('sentiment_sum_arr', sentiment_sum_arr)
        # print('virality_arr', virality_arr)

        sum_df['date'] = date_arr
        sum_df['privacy_sentiment'] = np.array(
            sentiment_sum_arr) / np.array(virality_arr)
        # print('privacy_sentiment', sum_df['privacy_sentiment'])
        # sum_df['count'] = count_arr

        # sum_df['sentiment_sum'] = sum_df['sentiment_sum'] / total_count
        return px.line(sum_df, x='date', y='privacy_sentiment')


@ app.callback(
    dash.dependencies.Output('graph_organizations_and_tags_2', 'figure'),
    [dash.dependencies.Input('date_picker', 'start_date'),
     dash.dependencies.Input('date_picker', 'end_date'),
     dash.dependencies.Input('choice_tweet_property', 'value'),
     dash.dependencies.Input('choice_organizations_2', 'value'),
     dash.dependencies.Input('choice_consolidated_trending', 'value'),
     dash.dependencies.Input('choice_trending_thresh_slider', 'value'),
     dash.dependencies.Input('choice_max_x_2', 'value'),
     dash.dependencies.Input('choice_min_count_2', 'value')])
def update_graph_organizations_2(start_date, end_date, mode_selection, organizations_selection, data_selection, thresh, max_x, min_count):
    # #print('hi')
    start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')

    # https://stackoverflow.com/questions/59882714/python-generating-a-list-of-dates-between-two-dates
    # https://stackoverflow.com/questions/18684076/how-to-create-a-list-of-date-string-in-yyyymmdd-format-with-python-pandas
    date_range = [d.strftime('%Y-%m-%d') for d in pd.date_range(
        start_date_obj, end_date_obj-timedelta(days=1), freq='d')]
    # #print('date_range', date_range)

    # df = pd.read_csv('results/consolidated_date_formatted.csv')
    if data_selection == 'full':
        df = pd.read_csv('results/consolidated_date_formatted.csv')
    elif data_selection == 'trending_retweets':
        df = filter_by_col('retweet_count', thresh)
    elif data_selection == 'trending_favs':
        df = filter_by_col('fav_count', thresh)

    df = df[df['date'].isin(date_range)]

    # #print('df', df)

    if organizations_selection == 'organizations':
        # #print('organizations!')

        # create a 2D array with Organization references having their own row
        indiv_org_ref_arr = []

        for _, row in df.iterrows():
            # https://stackoverflow.com/questions/23119472/in-pandas-python-reading-array-stored-as-string
            # to_csv makes arrays of strings string so need to extract the array back
            orgs_literal = ast.literal_eval(
                row['tweet_mentioned_organizations'])

            # #print('orgs_literal', orgs_literal)
            for org in orgs_literal:
                # #print('index', index)
                # #print('date', row['date'])
                # #print('date type', type(row['date']))
                # #print('org', org)

                # #print('org type', type(org))

                orgs_dict_literal = ast.literal_eval(
                    row['tweet_orgs_classified'])

                # #print('orgs_dict_literal', orgs_dict_literal)
                # #print('org_tag', orgs_dict_literal[org])

                # #print('vader_polarity', row['vader_polarity'])

                indiv_org_ref_arr.append(
                    [row['date'], org, orgs_dict_literal[org], row['vader_polarity'], 1])
                # indiv_org_ref_arr.append(
                #     [row['date'], org, row['tweet_organization_tags'][org], row['vader_polarity'], 1])

        # convert the 2D array into a df
        indiv_org_ref_df = pd.DataFrame(
            indiv_org_ref_arr, columns=['date', 'organization', 'tags', 'sentiment', 'count'])

        # #print('indiv_org_ref_df: ', indiv_org_ref_df)

        sum_sr = indiv_org_ref_df.groupby(
            ['date', 'organization'])['count'].sum()

        # #print(sum_sr)

        sum_df = pd.DataFrame()

        # #print('date', sum_sr.loc['date'])
        # #print('date', sum_sr.loc['organization'])
        # #print('date', sum_sr.loc['count'])

        # #print('shit', sum_sr.index[10])

        # #print('date ting', sum_sr.index.unique(level='date'))
        # #print('org ting', sum_sr.index.unique(level='organization'))
        # #print('values', sum_sr.values)

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

        sum_df = sum_df[sum_df['count'] > min_count]
        # print('sum_df', sum_df)

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

        # #print(sum_sr)

        sum_df = pd.DataFrame()

        # #print('date', sum_sr.loc['date'])
        # #print('date', sum_sr.loc['organization'])
        # #print('date', sum_sr.loc['count'])

        # #print('shit', sum_sr.index[10])

        # #print('date ting', sum_sr.index.unique(level='date'))
        # #print('org ting', sum_sr.index.unique(level='organization'))
        # #print('values', sum_sr.values)

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

        sum_df = sum_df[sum_df['count'] > min_count]

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

    elif organizations_selection == 'hashtags':
        # #print('hashtags!')

        # create a 2D array with Organization references having their own row
        indiv_hashtag_ref_arr = []

        for index, row in df.iterrows():
            # print(index)
            # https://stackoverflow.com/questions/23119472/in-pandas-python-reading-array-stored-as-string
            # to_csv makes arrays of strings string so need to extract the array back
            # #print('row tings', row['hashtags'])
            # #print('row tings type', type(row['hashtags']))
            hashtag_literal = ast.literal_eval(
                row['hashtags'])

            # #print('hashtag_literal', hashtag_literal)
            for hashtag in hashtag_literal:
                # #print('index', index)
                # #print('date', row['date'])
                # #print('date type', type(row['date']))
                # #print('hashtag', hashtag)

                # #print('hashtag type', type(hashtag))

                # hashtag_dict_literal = ast.literal_eval(
                #     row['hashtags'])

                # #print('hashtag_dict_literal', hashtag_dict_literal)
                # #print('hashtag_tag', hashtag)

                # #print('vader_polarity', row['vader_polarity'])

                indiv_hashtag_ref_arr.append(
                    [row['date'], hashtag, row['vader_polarity'], 1])
                # indiv_hashtag_ref_arr.append(
                #     [row['date'], hashtag, row['tweet_hashtaganization_tags'][hashtag], row['vader_polarity'], 1])

        # convert the 2D array into a df
        indiv_hashtag_ref_df = pd.DataFrame(
            indiv_hashtag_ref_arr, columns=['date', 'hashtag', 'sentiment', 'count'])

        # #print('indiv_hashtag_ref_df: ', indiv_hashtag_ref_df)

        sum_sr = indiv_hashtag_ref_df.groupby(
            ['date', 'hashtag'])['count'].sum()
        # print('done with sr')
        # #print(sum_sr)

        sum_df = pd.DataFrame()

        # #print('date', sum_sr.loc['date'])
        # #print('date', sum_sr.loc['organization'])
        # #print('date', sum_sr.loc['count'])

        # #print('shit', sum_sr.index[10])

        # #print('date ting', sum_sr.index.unique(level='date'))
        # #print('org ting', sum_sr.index.unique(level='organization'))
        # #print('values', sum_sr.values)

        date_arr = []
        hashtag_arr = []
        count_arr = []

        for i in range(0, sum_sr.size):
            # print(i)
            index = sum_sr.index[i]
            value = sum_sr.values[i]

            date_arr.append(index[0])
            hashtag_arr.append(index[1])
            count_arr.append(value)

        sum_df['date'] = date_arr
        sum_df['hashtag'] = hashtag_arr
        sum_df['count'] = count_arr

        sum_df = sum_df[sum_df['count'] > min_count]

        fig = px.bar(sum_df, x="hashtag", y="count",
                     color="date")

        fig.update_layout(
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font_color=colors['text'],
            xaxis={'categoryorder': 'total descending'}
        )

        if max_x != None:
            fig.update_xaxes(range=(-0.5, int(max_x) - 0.5))

        return fig

    elif organizations_selection == 'phrases':
        fig = go.Figure()

        # fig.add_layout_image(
        #     dict(
        #         source="results/phrases_cloud.png",
        #         xref="x",
        #         yref="y",
        #         x=0,
        #         y=3,
        #         sizex=2,
        #         sizey=2,
        #         sizing="stretch",
        #         opacity=0.5,
        #         layer="below"
        #     )
        # )
        # Create figure
        fig = go.Figure()

        # Constants
        img_width = 1600
        img_height = 900
        scale_factor = 0.5

        # Add invisible scatter trace.
        # This trace is added to help the autoresize logic work.
        fig.add_trace(
            go.Scatter(
                x=[0, img_width * scale_factor],
                y=[0, img_height * scale_factor],
                mode="markers",
                marker_opacity=0
            )
        )

        # Configure axes
        fig.update_xaxes(
            visible=False,
            range=[0, img_width * scale_factor]
        )

        fig.update_yaxes(
            visible=False,
            range=[0, img_height * scale_factor],
            # the scaleanchor attribute ensures that the aspect ratio stays constant
            scaleanchor="x"
        )

        # Add image
        fig.add_layout_image(
            dict(
                x=0,
                sizex=img_width * scale_factor,
                y=img_height * scale_factor,
                sizey=img_height * scale_factor,
                xref="x",
                yref="y",
                opacity=1.0,
                layer="below",
                # sizing="stretch",
                source="https://raw.githubusercontent.com/sidwan02/isbdiri-privacy-twitter-analysis/main/results/phrases_cloud.png?token=AQ3LMOJZNGN2KT5CHVZ6PMTAB6D5W")
            # src="results/phrases_cloud.png")
        )

        # Configure other layout
        # fig.update_layout(
        #     width=img_width * scale_factor,
        #     height=img_height * scale_factor,
        #     margin={"l": 0, "r": 0, "t": 0, "b": 0},
        # )

        return fig
    elif organizations_selection == 'weighted':
        df['vader_polarity'] = np.where(
            df['vader_polarity'] < 0.7, -1, 1)
        df['virality'] = df['retweet_count'] + df['fav_count']
        df['consolidated_sentiment'] = df['virality'] * df['vader_polarity']

        sum_sr = df.groupby('date')['consolidated_sentiment'].sum()
        # print('sum_sr', sum_sr)
        sum_sr_2 = df.groupby('date')['virality'].sum()

        sum_df = pd.DataFrame()

        date_arr = []
        sentiment_sum_arr = []
        virality_arr = []

        for i in range(0, sum_sr.size):
            index = sum_sr.index[i]
            value = sum_sr.values[i]
            # #print('index', index)
            # #print('value', value)
            date_arr.append(index)
            sentiment_sum_arr.append(value)

        for i in range(0, sum_sr_2.size):
            # index = sum_sr.index[i]
            value = sum_sr_2.values[i]
            # #print('index', index)
            # #print('value', value)
            # .append(index)
            virality_arr.append(value)

        # print('sentiment_sum_arr', sentiment_sum_arr)
        # print('virality_arr', virality_arr)

        sum_df['date'] = date_arr
        sum_df['privacy_sentiment'] = np.array(
            sentiment_sum_arr) / np.array(virality_arr)
        # print('privacy_sentiment', sum_df['privacy_sentiment'])
        # sum_df['count'] = count_arr

        # sum_df['sentiment_sum'] = sum_df['sentiment_sum'] / total_count
        return px.line(sum_df, x='date', y='privacy_sentiment')


# @ app.callback(
#     dash.dependencies.Output('choice_organizations', 'options'),
#     [dash.dependencies.Input('choice_analysis', 'value')])
# def update_tweet_property_options(analysis_selection):
#     # print('updating!', analysis_selection)

#     if analysis_selection == 'Daily':
#         property_options = [{'label': 'Organizations Mentioned',
#                              'value': 'organizations'},
#                             {'label': 'Industry', 'value': 'tags'},
#                             # {'label': 'Hashtags', 'value': 'hashtags'},
#                             {'label': 'Privacy Sentiment', 'value': 'weighted'}]
#     elif analysis_selection == 'overall':
#         property_options = [{'label': 'Organizations Mentioned',
#                              'value': 'organizations'},
#                             {'label': 'Industry', 'value': 'tags'},
#                             # {'label': 'Hashtags', 'value': 'hashtags'},
#                             ]
#     return property_options


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
        print('working')
        df = pd.read_csv('results/consolidated_date_formatted.csv')
        # for orgs in df['tweet_mentioned_organizations']:
        #     print(ast.literal_eval(orgs))

        orgs_list = []

        try:
            for org_list in df['tweet_mentioned_organizations']:
                # print(ast.literal_eval(org_list))
                orgs_list.append(ast.literal_eval(org_list))
                # print(orgs_list)
        except:
            print("An exception occurred")

        # orgs_list = [ast.literal_eval(orgs)
        #              for orgs in df['tweet_mentioned_organizations']]
        # print('orgs_list', orgs_list)

        # orgs_list = df['tweet_mentioned_organizations']
        print('DONE!')

        # orgs = []
        orgs = orgs_list

        orgs = np.unique(
            np.array(functools.reduce(lambda a, b: a + b, orgs_list)))
        print('orgs', orgs)
        return [{'label': org, 'value': org} for org in orgs]


@ app.callback(
    dash.dependencies.Output('choice_orgs_selection', 'value'),
    [dash.dependencies.Input('choice_all_tweets', 'value'),
     dash.dependencies.Input('choice_orgs_selection', 'options')])
def update_org_dafault_selection(choice_tweets, org_options):
    # #print('updating!', choice_tweets)
    if choice_tweets == 'tweets_all':
        return []
    elif choice_tweets == 'tweets_orgs':
        return [org_options[0]['value']]


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
    dash.dependencies.Output('choice_max_x_2', 'style'),
    [dash.dependencies.Input('choice_organizations_2', 'value')])
def update_max_x_input_visibility_2(organizations_selection):

    if (organizations_selection != 'phrases') & (organizations_selection != 'weighted'):
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@ app.callback(
    dash.dependencies.Output('choice_min_count', 'style'),
    dash.dependencies.Input('choice_organizations', 'value'))
def update_choice_min_input_visibility(organizations_selection):
    if ((organizations_selection == 'phrases') | (organizations_selection == 'weighted')):
        return {'display': 'none'}
    else:
        return {'display': 'block'}


@ app.callback(
    dash.dependencies.Output('choice_min_count_2', 'style'),
    dash.dependencies.Input('choice_organizations_2', 'value'))
def update_choice_min_input_visibility_2(organizations_selection):
    if ((organizations_selection == 'phrases') | (organizations_selection == 'weighted')):
        return {'display': 'none'}
    else:
        return {'display': 'block'}


# @ app.callback(
#     dash.dependencies.Output('choice_max_x', 'placeholder'),
#     [dash.dependencies.Input('choice_organizations', 'value')])
# def update_max_x_placeholder_text(organizations_selection):
#     if organizations_selection == 'organizations':
#         return 'Enter Max Organizations'
#     elif organizations_selection == 'tags':
#         return 'Enter Max Tags'
#     elif organizations_selection == 'hashtags':
#         return 'Enter Max Hashtags'


# @ app.callback(
#     dash.dependencies.Output('choice_min_count', 'placeholder'),
#     [dash.dependencies.Input('choice_organizations', 'value')])
# def update_min_count_placeholder_text(organizations_selection):
#     if organizations_selection == 'organizations':
#         return 'Enter Min Organization Frequency'
#     elif organizations_selection == 'tags':
#         return 'Enter Min Tag Frequency'
#     elif organizations_selection == 'hashtags':
#         return 'Enter Min Hashtag Frequency'


# @ app.callback(
#     dash.dependencies.Output('choice_trending_thresh', 'style'),
#     [dash.dependencies.Input('choice_consolidated_trending', 'value')])
# def update_thresh_input_visibility(data_selection):
#     if data_selection == 'full':
#         return {'display': 'none'}
#     elif data_selection == 'trending_retweets':
#         return {'display': 'block'}
#     elif data_selection == 'trending_favs':
#         return {'display': 'block'}


@ app.callback(
    dash.dependencies.Output('choice_org_centric', 'style'),
    [dash.dependencies.Input('choice_all_tweets', 'value')])
def update_analysis_drop_visibility(tweets_selection):
    # if (analysis_selection == 'overall') & (tweets_selection == 'tweets_orgs'):
    if (tweets_selection == 'tweets_orgs'):
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@ app.callback(Output('choice_trending_thresh', 'children'),
               Input('choice_trending_thresh_slider', 'value'))
def display_value(value):
    return f'Threshold Chosen: {value}'


if __name__ == '__main__':
    app.run_server(debug=True)
