import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import plotly.graph_objs as go

path = 'https://raw.githubusercontent.com/andremforte/FinalProjectDV/master/'

df = pd.read_csv(path + 'df.csv')

country_options = [dict(label=country, value=country) for country in df['Country'].unique()]

indicator_options = ['Quality_Of_Life', 'Purchasing_Power']

indicator_names = [dict(label=indicator.replace('_', ' '), value=indicator) for indicator in indicator_options]

dropdown_country = dcc.Dropdown(
        id='country_drop',
        options=country_options,
        value=['Portugal'],
        multi=True)

#dropdown_indicator = dcc.Dropdown(
#        id='indicator_drop',\
#        options=indicator_options,
#        value=['Quality of Life Index'],
 #       multi=True)

slider_year = dcc.Slider(
        id='year_slider',
        min=df['Year'].min(),
        max=df['Year'].max(),
        marks={str(i): '{}'.format(str(i)) for i in
               [2017, 2019, 2021]},
        value=df['Year'].min(),
        step=1
    )

radio_lin_log = dcc.RadioItems(
        id='lin_log',
        options= indicator_names
        [dict(label='Quality_Of_Life', value='Quality_Of_Life'), dict(label='Purchasing_Power', value='Purchasing_Power')],
        value='Quality_Of_Life')
 #   )

#######################################################

app = dash.Dash(__name__)

server = app.server

app.layout = html.Div([
    html.Div([
        html.H1('Quality of Life Dashboard', style={"background-color": "#096484",
            'color' : 'White', 'text-align':'center'})
    ], id='1st row', style={'width': '60%','padding-left':'20%', 'padding-right':'20%'}, className='pretty_box'),
    html.Div([
        html.H4('Dashboard presenting the imports and exports of merchandise trade and '
                'commercial services trade of European Countries based on their GDP between 2000 and 2020',
                style={'text-align':'center',
                        'color' : 'Black',
                        "height": "10%",
                       'display': 'block',
                       'width': '80%',
                       'padding-left':'10%',
                       'padding-right':'10%'
                                                })
    ], id='2nd row', className='pretty_box'),
    html.Div([
        html.Div([
     #   html.Label('Choose your Indicator'),
      #  dropdown_indicator,
      #  html.Br(),
        html.Label('Choose the Year Range'),
        slider_year,
        html.Br(),
        #html.Label('Choose your Country'),
        #dropdown_country,
        #html.Br(),
        html.Label('Linear Log'),
        radio_lin_log,
        html.Br(),
        html.Button('Search', id='button' , style={'text-align':'center',
                       'display': 'center-block',
                       'background-color': '#008CBA',
                       'color' : 'White',
                       'text-align': 'center',
                       'display': 'inline-block',
                       'font-size': '13px',
                       'padding': '7px 20px',
                       'width': '20%'
                                                   })
        ], id='Iteraction', style={'width': '30%'}, className='pretty_box'),
    html.Div([
        html.Div([
        dcc.Graph(id='choropleth'),
        ], id='Map', className='pretty_box', style={'boxShadow': '#e3e3e3 4px 4px 2px',
                                                    'border-radius': '10px',
                                                    'backgroundColor': '#add8e6',
                                                    'padding':'.3rem',
                                                    'marginLeft':'10rem',
                                                    'marginRight':'10rem'}),
    ], id='Else', style={'width': '70%'}),
], id='3nd row', style={'display': 'flex'})
])


@app.callback(
    [
        Output("choropleth", "figure"),
    ],
    [
        Input("button", "n_clicks")
    ],
    [
        State("year_slider", "value"),
       # State("country_drop", "value"),
        State("lin_log", "value"),
        #State("indicator_drop", "value")
    ]
)

def plots(n_clicks, year, indicator):
    #############################################Second Choropleth######################################################

    df_choro = df.loc[df['Year'] == year]

    z = df_choro[indicator]

    data_choropleth = dict(type='choropleth',
                           locations=df_choro['CODE'],
                           # There are three ways to 'merge' your data with the data pre embedded in the map
                           z=z,
                           text=df_choro['Country'],
                           colorscale='Blues',
                           colorbar=dict(title='GDP'),
                          # hovertemplate='Country: %{text} <br>' ': %{z}',
                           name=''
                           )


    layout_choropleth = dict(geo=dict(scope='europe',  # default
                                      projection=dict(type='equirectangular'
                                                      ),
                                      # showland=True,   # default = True
                                      landcolor='black',
                                      lakecolor='white',
                                      showocean=True,  # default = False
                                      oceancolor='azure',
                                      bgcolor='#f9f9f9'
                                      ),

                             title=dict(
                                 text='European ' + 'GDP' + ' Choropleth Map on the year ' + str(
                                     year),
                                 x=.5  # Title relative position according to the xaxis, range (0,1)

                             ),
                             paper_bgcolor='rgba(0,0,0,0)',
                             plot_bgcolor='rgba(0,0,0,0)'
                             )

    fig_choro = go.Figure(data=data_choropleth, layout=layout_choropleth)

    return  [fig_choro]

if __name__ == '__main__':
    app.run_server(debug=True)
