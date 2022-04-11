import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import plotly.graph_objs as go

path = 'https://raw.githubusercontent.com/andremforte/FinalProjectDV/master/'

#Main dataset <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
df = pd.read_excel(path + 'df_final.xlsx')

#Income <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
income_total = pd.read_excel(path + 'Income_final.xlsx', sheet_name='Income_pergroup')

table = pd.pivot_table(income_total, index=['Group'], columns=['Country'], values=[2016, 2017, 2018, 2019, 2020])
table = table.T
table.reset_index(inplace=True)
table = table.rename(columns={"level_0": "Year"})

# Education <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
education = pd.read_excel(path + 'Education_final.xlsx')

education = pd.pivot_table(education, index =['Level_Education'], columns = ['Country', 'Year'], values = ['Educational_Attainment_Level'])
education = education.T
education.reset_index(inplace = True)
education.drop('level_0', axis = 1,  inplace = True)

#Preparation <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
country_options = [dict(label=country, value=country) for country in df['Country'].unique()]

indicator_options = ['Quality_Of_Life', 'Purchasing_Power', 'Safety', 'Health_Care', 'Cost_Of_Living', 'Property_Price_To_Income',
                     'Traffic_Commute_Time', 'Pollution', 'Climate']

indicator_names = [dict(label=indicator.replace('_', ' '), value=indicator) for indicator in indicator_options]

groups = ['Male', 'Female', 'Total']

group_options = [dict(label = group, value =group) for group in groups]

dropdown_country = dcc.Dropdown(
        id='country_drop',
        options=country_options,
        value=['Portugal', 'Spain'],
        multi=True
)
#
radio_group = dcc.RadioItems(
        id='radio_group',
        options= group_options,
        value='Total',
        labelStyle={'display': 'block'}
   )
#
slider_year = dcc.Slider(
        id='year_slider',
        min=df['Year'].min(),
        max=df['Year'].max(),
        marks={str(i): '{}'.format(str(i)) for i in
               [2016, 2017, 2018, 2019, 2020, 2021]},
       value=df['Year'].min(),
        step=1
    )
radio_indicator = dcc.RadioItems(
        id='radio_indicator',
        options= indicator_names,
        value='Quality_Of_Life',
        labelStyle={'display': 'block'}
   )

#######################################################

app = dash.Dash(__name__)

server = app.server

app.layout = html.Div([
    html.Div([
        html.H1('Quality of Life Dashboard', style={"background-color": "#096484",
            'color' : 'White', 'text-align':'center'})
    ], id='1st row', style={'width': '60%','padding-left':'20%', 'padding-right':'20%'}, className='pretty_box'),
    html.Div([
        html.H3('Dashboard presenting the European Life Quality Index and its indicators and subindicators between 2016 and 2020',
                style={'text-align':'center',
                        'border': '2px solid #add8e6',
                        'color' : 'black',
                        "height": "15%",
                       'display': 'block',
                       'width': '90%',
                       'padding-left':'5%',
                       'padding-right':'5%'
                                                })
    ], id='2nd row', className='pretty_box', style={'width': '84%','padding-left':'8%', 'padding-right':'8%'}),
    html.Div([
        html.Div([
     #   html.Label('Name'),
      #  dropdown_indicator,
      #  html.Br(),
        html.Label('Choose the Year Range'),
        slider_year,
        html.Br(),
        html.Label('Choose your Country'),
        dropdown_country,
        html.Br(),
        html.Label('Choose the Indicator', style = {'fontsize': '16'}),
        radio_indicator,
        html.Br(),
        html.Button('Search', id='button' , style={'text-align':'center',
                       'display': 'center-block',
                       'background-color': '#008CBA',
                       'color' : 'White',
                       'font-size': '13px',
                     #  'padding': '7px 20px',
                       'width': '20%'
                                                   })
        ], id='Iteraction', style={'width': '25%','boxShadow': '#e3e3e3 4px 4px 2px',
                                                        'border-radius': '10px',
                                                        'backgroundColor': '#add8e6',
                                                        'padding':'.1.5rem',
                                                        'marginLeft':'1rem',
                                                        'marginRight':'1rem',
                                                       'color': 'Black',
                                                       'padding-left': '2%',
                                                       'padding-right': '2%',
                                                       'padding-top': '2%'
                                                       }, className='pretty_box'),

    html.Div([
        html.Div([
        dcc.Graph(id='choropleth'),
        ], id='Map', className='pretty_box', style={'boxShadow': '#e3e3e3 4px 4px 2px',
                                                    'border-radius': '10px',
                                                    'backgroundColor': '#add8e6',
                                                    'padding':'.1.5rem',
                                                    'marginLeft':'1rem',
                                                    'marginRight':'1rem',
                                                    'color': 'Black',
                                                    'padding-left': '2%',
                                                    'padding-right': '2%',
                                                    'padding-top': '2%'
                                                    }),
    ], id='Else', style={'width': '75%'}),
], id='3nd row', style={'display': 'flex'}),

    html.Div([
        html.Div([
        html.Label('Choose your Group'),
        radio_group,
        html.Br(),
           ], id='Iteraction1', style={'width': '10%','boxShadow': '#e3e3e3 4px 4px 2px',
                                                        'border-radius': '10px',
                                                        'backgroundColor': '#add8e6',
                                                        'padding':'.1.5rem',
                                                        'marginLeft':'1rem',
                                                        'marginRight':'1rem',
                                                        'marginTop': '1rem',
                                                        'height': '40%',
                                                       'color': 'Black',
                                                       'padding-left': '1%',
                                                       'padding-right': '1%',
                                                       'padding-top': '2%'
                                                       }, className='pretty_box'),

        html.Div([
        html.Div([
        dcc.Graph(id='aggregate_graph', style={'boxShadow': '#e3e3e3 4px 4px 2px',
                                                    'border-radius': '10px',
                                                    'backgroundColor': '#add8e6',
                                                    'padding':'.1.5rem',
                                                    'marginLeft':'0.5rem',
                                                    'marginRight':'0.5rem',
                                                    'marginTop': '1rem',
                                                    'color': 'Black',
                                                    'padding-left': '2%',
                                                    'padding-right': '2%',
                                                    'padding-top': '2%'}),
            ], id='Map2', className='pretty_box'),
            ], id='Else1', style={'width': '45%'}),

        html.Div([
            html.Div([
            dcc.Graph(id='bar_graph', style={'boxShadow': '#e3e3e3 4px 4px 2px',
                                                'border-radius': '10px',
                                                'backgroundColor': '#add8e6',
                                                'padding':'.1.5rem',
                                                'marginLeft':'1rem',
                                                'marginRight':'1rem',
                                                'marginTop': '1rem',
                                                'color': 'Black',
                                                'padding-left': '2%',
                                                'padding-right': '2%',
                                                'padding-top': '2%'}),
            ], id='Map3', className='pretty_box'),
            ], id='Else2', style={'width': '45%'}),
        ], id='4th row', style={'display': 'flex'}),

    html.Div([
    html.Div([
           ], id='Iteraction2', style={'width': '10%','boxShadow': '#e3e3e3 4px 4px 2px',
                                                        'border-radius': '10px',
                                                        'backgroundColor': '#add8e6',
                                                        'padding':'.1.5rem',
                                                        'marginLeft':'0.5rem',
                                                        'marginRight':'0.5rem',
                                                        'marginTop': '1rem',
                                                        'height': '40%',
                                                       'color': 'Black',
                                                       'padding-left': '1%',
                                                       'padding-right': '1%',
                                                       'padding-top': '2%'
                                                       }, className='pretty_box'),
    html.Div([
            html.Div([
            dcc.Graph(id='pie_graph', style={'boxShadow': '#e3e3e3 4px 4px 2px',
                                        'border-radius': '10px',
                                        'backgroundColor': '#add8e6',
                                   #     'padding':'.1rem',
                                    #    'marginLeft':'1rem',
                                     #   'marginRight':'1rem',
                                        'marginTop': '1rem'
                                    }),
            ], id='Map4', className= 'pretty_box'),
                ], id='Else3', style={'width': '40%'}),

    html.Div([
                html.Div([
                dcc.Graph(id='graph_space2'),
                ], id='Map5', className='pretty_box',style={'boxShadow': '#e3e3e3 4px 4px 2px',
                                                        'border-radius': '10px',
                                                        'backgroundColor': '#add8e6',
                                                        'padding':'.1.5rem',
                                                        'marginLeft':'1rem',
                                                        'marginRight':'1rem',
                                                        'marginTop': '1rem'}),
                ], id='Else4', style={'width': '40%'}),
            html.Div([
            ], id='Iteraction3', style={'width': '10%', 'boxShadow': '#e3e3e3 4px 4px 2px',
                                    'border-radius': '10px',
                                    'backgroundColor': '#add8e6',
                                    'padding': '.1.5rem',
                                    'marginLeft': '0.5rem',
                                    'marginRight': '0.5rem',
                                    'marginTop': '1rem',
                                    'height': '40%',
                                    'color': 'Black',
                                    'padding-left': '1%',
                                    'padding-right': '1%',
                                    'padding-top': '2%'
                                    }, className='pretty_box'),
            ], id='5th row', style={'display': 'flex'}),
        ])

@app.callback(
    [
        Output("choropleth", "figure"),
        Output("bar_graph", "figure"),
        Output ("aggregate_graph", "figure"),
        Output ("pie_graph", "figure")
    ],
    [
        Input("button", "n_clicks")
    ],
    [
        Input("year_slider", "value"),
        Input("country_drop", "value"),
        Input("radio_indicator", "value"),
        Input("radio_group", "value"),
        #State("indicator_drop", "value")
    ]
)

def plots(n_clicks, year, countries, indicator, groups):
    ############################################First Plot - Bar##########################################################
    data_bar = []
    for country in countries:
        df_bar = df.loc[(df['Country'] == country)]

        x_bar = df_bar['Year']
        y_bar = df_bar[indicator]

        data_bar.append(dict(type='bar', x=x_bar, y=y_bar, name=country))

    layout_bar = dict(title=dict(text='Indicator ' + str(indicator.replace('_', ' ')) + ' between 2016 and 2020',
                                 x = .5),
                      yaxis=dict(title='Value'),
                      xaxis=dict(title='Year'),
                      paper_bgcolor='#add8e6'
                      )

    #############################################Second Plot - Choropleth######################################################

    df_choro = df.loc[df['Year'] == year]

    z = df_choro[indicator]

    data_choropleth = dict(type='choropleth',
                           locations=df_choro['CODE'],
                           # There are three ways to 'merge' your data with the data pre embedded in the map
                           z=z,
                           text=df_choro['Country'],
                           colorscale='Blues',
                           colorbar=dict(title='Index'),
                          # hovertemplate='Country: %{text} <br>' ': %{z}',
                           name=''
                           )


    layout_choropleth = dict(geo=dict(scope='europe',  # default
                                      projection=dict(type='equirectangular'
                                                      ),
                                      # showland=True,   # default = True
                                      landcolor='black',
                                 #     lakecolor='white',
                                      showocean=False,  # default = False
                              #        oceancolor='azure',
                                      bgcolor='#add8e6'

                                      ),

                             title=dict(
                                 text='European ' + str(indicator.replace('_', ' ')) + ' Index' + ' Choropleth Map on the year ' + str(year),
                                 x=.5  # Title relative position according to the xaxis, range (0,1)

                             ),
                             paper_bgcolor='#add8e6'
                             )

    fig_choro = go.Figure(data=data_choropleth, layout=layout_choropleth)


    fig_choro.update_layout(
        margin=dict(l=50, r=50, t=50, b=50, pad=4,
                autoexpand=True))

    ############################################Third Plot - Scatter ######################################################

    data_agg =[]


    for country in countries:
        df_line = table.loc[(table['Country'] == country)]

        x_line = df_line['Year']

        y_line = df_line[groups]

        data_agg.append(dict(type= 'scatter',
                                x=x_line,
                                y=y_line,
                                name=country,
                          #      labels = dict('Group ' + str(groups)),
                               # text = str(groups), #ADD MORE INFO HERE
                            mode='lines+markers',
                                     )
                                )

    layout_agg = dict(title=dict(text='Income per Country on year between 2016 and 2020'),
                      yaxis=dict(title='Income'),
                      xaxis=dict(title='Countries'),
                      paper_bgcolor='#add8e6'
                      )

############################################ Forth Plot - Pie #############################################################

    data_pie = []
    df_pie = education.loc[(education['Country'] == 'Austria')]

    labels = df_pie.columns[2:].tolist()
    values = df_pie.loc[0,:][2:].tolist()

    data_pie.append(dict(type='pie', labels = labels, values = values, hole = 0.5))

    layout_pie = dict(title=dict(text='Indicator ' + ' between 2016 and 2020',
                                 x=.5),
                      paper_bgcolor='#add8e6',
                      legend = {'x': 0.3, 'y': 0})

    fig_pie = go.Figure(data=data_pie, layout=layout_pie)

    fig_pie.update_layout(
        margin=dict(l=50, r=50, t=50, b=50)
                   )

#
    return  fig_choro, \
            go.Figure(data=data_bar, layout=layout_bar), \
            go.Figure(data=data_agg, layout=layout_agg), \
            fig_pie


if __name__ == '__main__':
    app.run_server(debug=True)
