import dash
import json
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
from itertools import count, takewhile, permutations
import regex
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.preprocessing import LabelEncoder
from baseball_scraper import statcast
import base64
from sklearn.ensemble import RandomForestRegressor

app = dash.Dash(__name__)

df = pd.read_csv('savant_data.csv')

fig = go.Figure(
        data=[go.Scatter(
            x=[-.4, 0, .4, -.4, 0, .4, -.4, 0, .4],
            y=[3.25,3.25,3.25,2.5,2.5,2.5, 1.75,1.75, 1.75],
            mode='markers')])

fig.add_shape(type="line",
    x0=-.8, y0=1.5, x1=.8, y1=1.5,
    line=dict(color="Black",width=3)
)
fig.add_shape(type="line",
    x0=-.8, y0=3.5, x1=.8, y1=3.5,
    line=dict(color="Black",width=3)
)
fig.add_shape(type="line",
    x0=-.8, y0=1.5, x1=-.8, y1=3.5,
    line=dict(color="Black",width=3)
)
fig.add_shape(type="line",
    x0=.8, y0=1.5, x1=.8, y1=3.5,
    line=dict(color="Black",width=3)
)
fig.update_layout(xaxis_range=[-2,2])
fig.update_layout(yaxis_range=[0,5],
    margin={
    'l':0,
    'r':0,
    't':0,
    'b':0})
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)
fig.layout.xaxis.fixedrange = True
fig.layout.yaxis.fixedrange = True

### Data Cleaning ###
df1 = df[df['events'].notnull()]
df1 = df1[df1['launch_speed'].notnull()]

hits = ['single', 'double', 'triple', 'home_run']
df1['hit'] = 0

df1.loc[(df['events'] == 'single') |
 (df1['events'] == 'double') |
 (df1['events'] == 'triple') |
 (df1['events'] == 'home_run'), 'hit'] = 1


list_predictor_categorical = []
list_predictor_continuous = []



# Convert pd.serise to np.array
Rredictor_Categorical = df1.loc[:,'home_team'].values

# Integer Encoding for Categorical variable
enc_pred = LabelEncoder()
Rredictor_Categorical_en = enc_pred.fit_transform(Rredictor_Categorical)
df1['home_team_label'] = Rredictor_Categorical_en

features = ['launch_speed', 'launch_angle','plate_x', 'plate_z', 'home_team_label']
df1 = df1[df1['hc_x'].notnull()]
df1 = df1[df1['hc_y'].notnull()]
X = df1[features]
y = df1['hit']

#########################

### Model Creation and Best Model ###

clf = RandomForestClassifier(n_jobs=-1, random_state = 42, verbose = 2)
param_grid = {'n_estimators': [x for x in range (50,150,5)],
              'min_samples_split': [2, 3, 5, 8, 13, 20, 32, 50, 80],
              'criterion':['gini', 'entropy'],
              'max_depth':[3,6,9]}
grid_clf = GridSearchCV(estimator = clf, param_grid = param_grid, n_jobs = 4, cv=3, verbose = 2, return_train_score = True)
grid_clf.fit(X, y)
df_gridsearch = pd.DataFrame(grid_clf.cv_results_)
best_model = grid_clf.best_params_

print(best_model)
best_model = {'criterion': 'entropy', 'max_depth': 9, 'min_samples_split': 8, 'n_estimators': 135}
classifier = RandomForestClassifier(criterion = best_model['criterion'],
                                    n_estimators = best_model['n_estimators'],
                                    min_samples_split = best_model['min_samples_split'],
                                    max_depth = best_model['max_depth'],
                                    random_state = 42)

classifier.fit(X, y)
########################

### Regression Model ###

regression_features = ['launch_speed', 'launch_angle','plate_x', 'plate_z', 'hc_x']
X1 = df1[regression_features]
y1 = df1['hc_y']
'''
clf = RandomForestRegressor(random_state = 42, verbose = 2)
param_grid = {'n_estimators': [x for x in range (50,150,10)],
              'min_samples_split': [2,3,4,5]}

grid_clf = GridSearchCV(estimator = clf, param_grid = param_grid, n_jobs = 4, cv=3, verbose = 2, return_train_score = True)
grid_clf.fit(X1, y1)
best_model1 = grid_clf.best_params_
'''

best_model1 = {'min_samples_split': 5, 'n_estimators': 140}
regressor = RandomForestRegressor(n_estimators = best_model1['n_estimators'],
                                    min_samples_split = best_model1['min_samples_split'],
                                    random_state = 42)
regressor.fit(X1, y1)
########################################################################
ballpark_df = df1[['home_team', 'home_team_label']].drop_duplicates().sort_values(by='home_team')
ballpark_list = ballpark_df['home_team'].to_list()
encoded_list = ballpark_df['home_team_label'].to_list()


home_teams = []
options_dict = {}
for i in range(len(ballpark_list)):
    options_dict = {'label':ballpark_list[i], 'value':encoded_list[i]}
    home_teams.append(options_dict)
    options_dict = {}




#########################

### App Creation ###

app.layout = html.Div(id = 'entry-point',children=[
    html.Div(children=[
        html.Div(id='two_thirds_column', children=[
            html.Div(id='container', children=[
                html.Div(id = 'header', children=[
                    html.Div(id = 'header-title', children=[
                        html.H4('xBA Web Application', style={
                            'font-family':'serif',
                            'margin':['5px','0px','0px','20px'],
                            'font-size':'2.6rem',
                            'line-height':1.35,
                            'letter-spacing':'-.08rem'
                            }
                        )
                    ],
                    style={
                        'display':'flex',
                        'flex-direction':'row',
                        'align-items':'center',
                        'margin-top':'50px'
                        }
                    ),
                    html.Div(id = 'header-description',children=[
                        html.P('Expected Batting Average measures the likelihood that a batted ball will become a hit.',
                            style={
                                'padding':['10px','0px','10px','0px'],
                                'margin-bottom':'0.75rem',
                                'margin-top':0,
                                'display':'block',
                                'margin-block-start':'1em',
                                'margin-block-end': '1em',
                                'margin-inline-start': '0px',
                                'margin-inline-end': '0px',
                                }
                            )
                        ],
                        style={
                            'padding-bottom':'20px',
                        }
                    )
                ],
                style={
                    'display': 'flex',
                    'align-items': 'center',
                    'padding-bottom':'20px',
                    'flex-direction':'column',
                    }
                ),
                html.Div(id='xBA-container',children=[
                    html.Div(id='test-data')
                    ],
                    style={
                        'display':'flex',
                        'justify-content':'center',
                        'font-family':'serif',
                        'font-size':25
                    }
                ),
                html.Div(id='graph-container', children=[
                    dcc.Graph(id='park_graph',
                        config = {'displayModeBar': False}
                    )

                ],
                style={
                    'min-height':'50vh',
                    'margin':'auto',
                    'box-sizing':'border-box',
                    'justify-content':'center',
                    'border-color':'black',
                    'padding':'10px',
                    'width':'80%'
                    }
                )
            ],
            style={
                'width':'100%',
                'padding':0,
                'position':'relative',
                'max-width':'960px',
                'margin-top':'0px',
                'margin-right':'auto',
                'margin-bottom':'0px',
                'margin-left':'auto',
                'box-sizing':'border-box',
                'display':'block'
                }
            ),
        ],
        style={
            'width':'65.333%',
            'margin-left':0,
            'float':'left',
            'box-sizing':'border-box',
            'display':'block'
            }
        ),
        html.Div(id='one_third_column', children=[
            html.Div(id='home-team', children=[
                html.P('Ballpark',
                    style={
                        'font-size':25,
                        'font-weight':'bold'
                    }
                ),
                dcc.Dropdown(
                    id='ballpark-drop',
                    options=home_teams,
                    searchable=False,
                    value=1,
                    style={
                        'color':'black'
                    }
                ),
                html.Pre(id='ballpark-data',
                    style={
                        'display':'none'
                    }
                )
            ],
            style={
                'display': 'inline-block',
                'width': '80%',
                'margin-left':30,
                'padding-left':'20px',
                'padding-top':'20px',
                'padding-right':'20px',
                'text-align':'center'
                }
            ),
            html.Div(id='exit-velocity', children=[
                html.P('Exit Velocity',
                    style={
                        'font-size':25,
                        'font-weight':'bold'
                    }
                ),
                dcc.Slider(
                    id="ev-slider",
                    min=int(min(df1['launch_speed'])),
                    max=int(max(df1['launch_speed'])),
                    value=7,
                    step=1,
                    marks={
                        7:{'label': str(int(min(df1['launch_speed']))),
                            'style':{'color':'#0000FF', 'font-size':'26px'}},
                        121:{'label': str(int(max(df1['launch_speed']))),
                            'style':{'color':'#FF0000', 'font-size':'26px'}}
                    }
                ),
                html.Pre(id='slide-data-2')
                ],
                style={
                    'display': 'inline-block',
                    'width': '80%',
                    'margin-left':30,
                    'padding':'20px',
                    'text-align':'center'
                }
            ),
            html.Div(id='launch-angle', children=[
                html.P('Launch Angle',
                    style={
                        'font-size':25,
                        'font-weight':'bold'
                    }
                ),
                dcc.Slider(
                    id="la-slider",
                    min=min(df1['launch_angle']),
                    max=max(df1['launch_angle']),
                    value=-89,
                    step=1,
                    marks={
                        -89:{'label': str(int(min(df1['launch_angle']))) + '°',
                            'style':{'color':'#0000FF', 'font-size':'26px'}},
                        90:{'label': str(int(max(df1['launch_angle']))) + '°',
                            'style':{'color':'#FF0000', 'font-size':'26px'}}
                    }
                ),
                html.Pre(id='slide-data-1')
                ],
                style={
                    'display': 'inline-block',
                    'width': '80%',
                    'margin-left':30,
                    'padding':'20px',
                    'text-align':'center'
                }
            ),
            html.Div(id='x-coord', children=[
                html.P('Hit X Coordinate',
                    style={
                        'font-size':25,
                        'font-weight':'bold'
                    }
                ),
                dcc.Slider(
                    id='x-coord-slider',
                    min=min(df1['hc_x']),
                    max=max(df1['hc_x']),
                    value=2,
                    step=.1,
                    marks={
                        2:{'label': str(int(min(df1['hc_x']))),
                            'style':{'color':'#FFFFFF', 'font-size':'26px'}},
                        248:{'label': str(int(max(df1['hc_x']) + 1)),
                            'style':{'color':'#FFFFFF', 'font-size':'26px'}}
                    }
                ),
                html.Pre(id='x-data')
                ],
                style={
                    'display': 'inline-block',
                    'width': '80%',
                    'margin-left':30,
                    'padding':'20px',
                    'text-align':'center'
                }
            ),

            html.Div(id='pitch-location', children=[
                    html.P('Pitch Location',
                        style={
                            'font-size':25,
                            'font-weight':'bold',
                            'text-align':'center'
                        }
                    ),
                    dcc.Graph(
                        id="pitch-loc-graph",
                        figure = fig,
                        config = {'displayModeBar': False}
                    ),
                    html.Div(id='click-data',
                        style={
                            'padding-top':'10px'
                        }
                    )
                ],
                style={
                    'display': 'inline-block',
                    'width': '80%',
                    'margin-left':40,
                    'padding':'10px',
                    'text-align':'center'
                }
            )
            ],
            style={
                'width':'30.667%',
                'background-color':'#1D1D1D',
                'min-height':'200vh',
                'max-height':'200vh',
                'overflow-y':'scroll',
                'overflow':'scroll',
                'padding':'25px',
                'border-left-color':'black',
                'border-left-style':'solid',
                'border-left-width':'5px',
                'margin-left':'4%',
                'float':'right',
                'box-sizing':'border-box'
            }
        )
        ],
        style={
            'width':'100%',
            'float':'left',
            'box-sizing':'border-box',
            'background-color':'#141414',
            'color':'#F4F6F8',
            'font-family':'"Open Sans", sans-serif',
            'margin':0
        }
    )
    ]
)
#########################

### App Callbacks ###

@app.callback(
    Output('click-data', 'children'),
    [Input('pitch-loc-graph', 'clickData')]
)
def display_click_data(clickData):
    data = json.dumps(clickData, indent=2)
    x_coord = regex.findall('"x": [\d-?]?[\d\.]?[\d.]?[\d]', data)
    x_coord = ''.join(x_coord).strip('"x:" ')
    y_coord = regex.findall('"y": [\d-?]?[\d\.]?[\d.]?[\d]', data)
    y_coord = ''.join(y_coord).strip('"y:" ')
    coordinates = (float(x_coord), float(y_coord))
    return 'Pitch Location: {}'.format(coordinates)

@app.callback(
    Output('slide-data-2','children'),
    [Input('ev-slider','value')]
)
def display_exit_velocity(ev_value):
    return 'Exit Velocity: {}mph'.format(ev_value)

@app.callback(
    Output('slide-data-1', 'children'),
    [Input('la-slider','value')]
)
def display_launch_value(value):
    return 'Launch Angle: {}°'.format(value)

@app.callback(
    Output('x-data', 'children'),
    [Input('x-coord-slider','value')]
)
def display_x_coord(x_value):
    return 'X Coordinate: {}'.format(x_value)

@app.callback(
    Output('ballpark-data', 'children'),
    [Input('ballpark-drop', 'value')]
)

def display_ballpark(ballpark_value):
    return 'Ballpark: {}'.format(ballpark_value)

@app.callback(
    Output('test-data', 'children'),
    [Input('pitch-loc-graph', 'clickData'),
    Input('la-slider', 'value'),
    Input('ev-slider', 'value'),
    Input('ballpark-drop', 'value')]
)

def get_test_data(clickData, value, ev_value, ballpark_value):
    data = json.dumps(clickData, indent=2)
    x_coord = regex.findall('"x": [\d-?]?[\d\.]?[\d.]?[\d]', data)
    x_coord = ''.join(x_coord).strip('"x:" ')
    y_coord = regex.findall('"y": [\d-?]?[\d\.]?[\d.]?[\d]', data)
    y_coord = ''.join(y_coord).strip('"y:" ')


    X_test = pd.DataFrame(
        data = np.array([[int(ev_value), int(value), float(x_coord), float(y_coord), int(ballpark_value)]]),
        columns = ['launch_speed', 'launch_angle','plate_x', 'plate_z', 'home_team'])


    xBA = round(classifier.predict_proba(X_test)[0][1], 3)
    return "xBA = {}".format(xBA)

@app.callback(
    Output('park_graph','figure'),
    [Input('pitch-loc-graph', 'clickData'),
    Input('la-slider', 'value'),
    Input('ev-slider', 'value'),
    Input('ballpark-drop', 'value'),
    Input('x-coord-slider', 'value')]
)

def get_graph(clickData, value, ev_value, ballpark_value, x_value):

    data = json.dumps(clickData, indent=2)
    x_coord = regex.findall('"x": [\d-?]?[\d\.]?[\d.]?[\d]', data)
    x_coord = ''.join(x_coord).strip('"x:" ')
    y_coord = regex.findall('"y": [\d-?]?[\d\.]?[\d.]?[\d]', data)
    y_coord = ''.join(y_coord).strip('"y:" ')

    team_dict = {0:'ARI', 1:'ATL',2:'BAL',3:'BOS',4:'CHC',5:'CIN',6:'CLE',7:'COL',
        8:'CWS',9:'DET',10:'HOU', 11:'KC',12:'LAA',13:'LAD',14:'MIA',15:'MIL',16:'MIN',
        17:'NYM',18:'NYY',19:'OAK',20:'PHI',21:'PIT',22:'SD',23:'SEA',24:'SF',25:'STL',
        26:'TB',27:'TEX',28:'TOR',29:'WSH'}

    test_png = './ballparks/' + team_dict[ballpark_value] + '.png'
    test_base64 = base64.b64encode(open(test_png, 'rb').read()).decode('ascii')
    encoded_image = 'data:image/png;base64,{}'.format(test_base64)

    X_test1 = pd.DataFrame(
        data = np.array([[int(ev_value), int(value), float(x_coord), float(y_coord), float(x_value)]]),
        columns = ['launch_speed', 'launch_angle','plate_x', 'plate_z', 'hc_x'])

    y_value = regressor.predict(X_test1)

    print(y_value)

    fig1 = go.Figure(
        data=go.Scatter(
            x=[x_value],
            y=y_value,
            mode='markers',
            showlegend = False,
            hovertemplate=
            '<extra></extra>',
            marker=dict(
                color='#bf5700',
                size=15,
                line=dict(
                    color='black',
                    width=3
                )
            )
        )
    )

    # Constants
    img_width = 300
    img_height = 300
    scale_factor = 1

    # Add invisible scatter trace.
    # This trace is added to help the autoresize logic work.
    fig1.add_trace(
        go.Scatter(
            x=[0, img_width * scale_factor],
            y=[0, img_height * scale_factor],
            mode="markers",
            marker_opacity=0,
            showlegend = False

        )
    )

    # Configure axes
    fig1.update_xaxes(
        visible=False,
        range=[0, img_width * scale_factor]
    )

    fig1.update_yaxes(
        visible=False,
        range=[0, img_height * scale_factor],
        # the scaleanchor attribute ensures that the aspect ratio stays constant
        scaleanchor="x",
        autorange = 'reversed'
    )

    # Add image
    fig1.add_layout_image(
        dict(
            x=-30,
            sizex=img_width * scale_factor,
            y=-8,
            sizey=img_height * scale_factor,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            source=encoded_image)
    )

    # Configure other layout
    fig1.update_layout(
        width=img_width * 3 * scale_factor,
        height=img_height * 4 * scale_factor,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    fig1.layout.xaxis.fixedrange = True
    fig1.layout.yaxis.fixedrange = True

    return fig1

#########################

if __name__ == '__main__':
    app.run_server(debug=True)
