import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import numpy as np
import math
import pandas as pd
import plotly.graph_objs as go

path = 'https://raw.githubusercontent.com/andremforte/FinalProjectDV/master/'

#Main dataset <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
df = pd.read_excel(path + 'df_final.xlsx')

#df.Year = df.Year.astype(str)

#Income <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
income_total = pd.read_excel(path + 'Income_final.xlsx', sheet_name='Income_pergroup')

table = pd.pivot_table(income_total, index=['Group'], columns=['Country'], values=[2016, 2017, 2018, 2019, 2020])
table = table.T
table.reset_index(inplace=True)
table = table.rename(columns={"level_0": "Year"})

# Education <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
education = pd.read_excel(path + 'Education_final.xlsx')

education = pd.pivot_table(education, index =['Level_Education', 'Country'], columns = ['Year'], values = ['Educational_Attainment_Level'])
education.reset_index(inplace = True)
education.set_index('Country', inplace = True)

country_list = education.index.unique().tolist()

# poverty_crime_pop <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PCP = pd.read_csv(path + 'poverty_crime_pop.csv')

# Crime and Poverty <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
crime = pd.read_excel(path + 'Crime.xlsx')

crime = (crime.set_index(["TIME"])
             .stack()
             .reset_index(name='Crime')
             .rename(columns={'level_1': 'Year', "TIME": "Countries"}))

poverty = pd.read_excel(path + 'Poverty.xlsx')

poverty = (poverty.set_index(["TIME"])
               .stack()
               .reset_index(name='Poverty')
               .rename(columns={'level_1': 'Year', "TIME": "Countries"}))

scatter_df = pd.merge(crime, poverty, on=["Year", "Countries"])

# Energy <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
df_heatmap = pd.read_excel(path + 'ShareEnergyRenewableData.xlsx')

df_heatmap.set_index('GEO/TIME', inplace=True)

#Preparation <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
country_options = [dict(label=country.replace('_', ' '), value=country) for country in df['Country'].unique()]

country_options_pie = [dict(label=country, value=country) for country in country_list]

indicator_names = ['Quality_Of_Life', 'Purchasing_Power', 'Safety', 'Health_Care', 'Cost_Of_Living', 'Property_Price_To_Income',
                     'Traffic_Commute_Time', 'Pollution', 'Climate']

indicator_options = [dict(label=indicator.replace('_', ' '), value=indicator) for indicator in indicator_names]

groups = ['Male', 'Female', 'Total']

group_options = [dict(label = group, value =group) for group in groups]

years = [2016, 2017, 2018, 2019, 2020]

year_options = [dict(label = year, value =year) for year in years]

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

dropdown_country = dcc.Dropdown(
        id='country_drop',
        options=country_options,
        value=['Portugal', 'Spain', 'France', 'Germany'],
        multi=True
)
#
radio_group = dcc.RadioItems(
        id='radio_group',
        options= group_options,
        value='Total',
        labelStyle={'display': 'block'},
   )
#

radio_year = dcc.RadioItems(
        id='radio_year',
        options=year_options,
        value=2016,
        labelStyle={'display': 'block'}

)

radio_indicator = dcc.RadioItems(
        id='radio_indicator',
        options= indicator_options,
        value='Quality_Of_Life',
        labelStyle={'display': 'block'}
   )

dropdown_country1 = dcc.Dropdown(
        id='country_drop1',
        options=country_options_pie,
        value='Austria',
        multi=False
)

dropdown_year = dcc.Dropdown(
        id='year_drop',
        options=year_options,
        value=2016,
        multi=False
)

dropdown_country2 = dcc.Dropdown(
        id='country_drop2',
        options=country_options_pie,
        value='Portugal',
        multi=False
)

dropdown_year1 = dcc.Dropdown(
        id='year_drop1',
        options=year_options,
        value=2016,
        multi=False
)

dropdown_country1 = dcc.Dropdown(
        id='country_drop1',
        options=country_options_pie,
        value='Austria',
        multi=False
)
slider_year = dcc.Slider(
        id='year_slider',
        min=df['Year'].min(),
        max=df['Year'].max(),
        marks={str(i): '{}'.format(str(i)) for i in
                [2016, 2017, 2018, 2019, 2020]},
        value=df['Year'].min(),
        step=1,
        included=False,
    )

dropdown_country3 = dcc.Dropdown(
        id='country_drop3',
        options=country_options,
        value=['Portugal', 'Spain', 'France', 'Germany'],
        multi=True
)

dropdown_indicator = dcc.Dropdown(
        id='indicator_drop',
        options=indicator_options,
        value='Purchasing_Power',
        multi=False
)

#dropdown_year2 = dcc.Dropdown(
 #       id='year_drop2',
  #      options=year_options,
   #     value=2016,
    #    multi=False
#)
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
        html.Label('Choose the Year'),
        radio_year,
        html.Br(),
        html.Label('Choose the Indicator', style = {'fontsize': '14'}),
        radio_indicator,
        html.Br(),
        ], id='Iteraction', style={'width': '20%','boxShadow': '#e3e3e3 4px 4px 2px',
                                                        'border-radius': '10px',
                                                        'backgroundColor': '#add8e6',
                                                        'padding':'1rem',
                                                        'marginLeft':'0.5rem',
                                                        'marginRight':'0.5rem',
                                                        'marginTop':'0.5rem',
                                                        "height": '80%',
                                                       'color': 'Black',
                                                       'padding-left': '1%',
                                                       'padding-right': '1%',
                                                       'padding-top': '1%',
                                                       'padding-bottom': '1%'
                                                       }, className='pretty_box'),

    html.Div([
        html.Div([
        dcc.Graph(id='choropleth'),
        ], id='Map', className='pretty_box', style={'boxShadow': '#e3e3e3 4px 4px 2px',
                                                    'border-radius': '10px',
                                                    'backgroundColor': '#add8e6',
                                                    'padding':'1rem',
                                                    'marginLeft':'0.5rem',
                                                    'marginRight':'0.5rem',
                                                    'marginTop':'0.5rem',
                                                    'color': 'Black',
                                                    'padding-left': '1%',
                                                    'padding-right': '1%',
                                                    'padding-top': '1%'
                                                    }),
    ], id='Else', style={'width': '60%'}),

    html.Div([
    html.Div([
    html.H3('Mean Value', style={"background-color": "#add8e6",
            'color' : '#096484', 'text-align':'center', 'marginLeft':'0.2rem',
                                                        'marginRight':'0.2rem',
                                                        'marginTop':'0.2rem',
                                                       'padding-left': '0.5%',
                                                       'padding-right': '0.5%',
                                                       'padding-top': '0.5%',
                                                       'padding-bottom': '0.5%'}),
    html.Tbody(id='kpi1', style={'text-align':'center'}, className='pretty_box')]),

    html.Div([
    html.H3('Change Rate', style={"background-color": "#add8e6",
            'color' : '#096484', 'text-align':'center'}),
    html.Tbody(id='kpi2', style={'text-align':'center', 'fontSize' : '16'}, className='pretty_box')]),

    html.Div([
    html.H3('Information', style={"background-color": "#add8e6",
            'color' : '#096484', 'text-align':'center'}),
    html.Tbody("Quality of Life Index (higher is better) is an estimation of overall quality of life "
            "by using an empirical formula which takes into account purchasing power (higher is better)," 
            "pollution (lower is better), house price to income (lower is better), cost of living " 
            "lower is better), safety (higher is better), health care (higher is better), "
            " traffic commute time (lower is better) and climate (higher is better) indices.",
             style={'text-align':'center', 'fontSize' : '10'}, className='pretty_box')]),

                      ], id='IteractionX', style={'width': '20%','boxShadow': '#e3e3e3 4px 4px 2px',
                                                        'border-radius': '10px',
                                                        'backgroundColor': '#add8e6',
                                                        'padding':'1rem',
                                                        'marginLeft':'0.5rem',
                                                        'marginRight':'0.5rem',
                                                        'marginTop':'0.5rem',
                                                    "height": "80%",
                                                       'color': 'Black',
                                                       'padding-left': '1%',
                                                       'padding-right': '1%',
                                                       'padding-top': '1%',
                                                       'padding-bottom': '1%'
                                                       }, className='pretty_box'),
], id='3nd row', style={'display': 'flex'}),

    html.Div([
        html.Div([
        html.Div([
            dcc.Graph(id='bar_graph', style={'boxShadow': '#e3e3e3 4px 4px 2px',
                                             'border-radius': '10px',
                                             'backgroundColor': '#add8e6',
                                             'padding': '.1rem',
                                             'marginLeft': '0.5rem',
                                             'marginRight': '0.5rem',
                                             'marginTop': '0.5rem',
                                             'color': 'Black',
                                             'padding-left': '2%',
                                             'padding-right': '2%',
                                             'padding-top': '2%'}),
            ], id='Map2', className='pretty_box'),
           ], id='Iteraction1', style={'width':'40%'}, className='pretty_box'),

        html.Div([
        html.Label('Choose your Country'),
        dropdown_country,
        html.Br(),
        html.Label('Choose the second Indicator'),
        dropdown_indicator,
        html.Br(),
                    ], id='Else1', style={'width': '20%',   'boxShadow': '#e3e3e3 4px 4px 2px',
                                                     'border-radius': '10px',
                                                     'backgroundColor': '#add8e6',
                                                     'padding': '.1 rem',
                                                     'marginLeft': '0.5rem',
                                                     'marginRight': '0.5rem',
                                                     'marginTop': '0.5rem',
                                                     'color': 'Black',
                                                     'padding-left': '2%',
                                                     'padding-right': '2%',
                                                     'padding-top': '2%'}),

        html.Div([
            html.Div([
            dcc.Graph(id='scatterplot_graph', style={'boxShadow': '#e3e3e3 4px 4px 2px',
                                                     'border-radius': '10px',
                                                     'backgroundColor': '#add8e6',
                                                     'padding': '.1 rem',
                                                     'marginLeft': '0.5rem',
                                                     'marginRight': '0.5rem',
                                                     'marginTop': '0.5rem',
                                                     'color': 'Black',
                                                     'padding-left': '2%',
                                                     'padding-right': '2%',
                                                     'padding-top': '2%'})
            ], id='Map3', className='pretty_box'),
            ], id='Else2', style={'width': '40%'}),
        ], id='4th row', style={'display': 'flex'}),

         html.Div([
        html.H1('Indicators related to Quality of Life', style={"background-color": "#096484",
                                            'color': 'White', 'text-align': 'center'})
    ], id='11th row', style={'width': '60%', 'padding-left': '20%', 'padding-right': '20%'},
        className='pretty_box'),

        html.Div([
        html.Div([
            html.Div([
                dcc.Graph(id='aggregate_graph', style={'boxShadow': '#e3e3e3 4px 4px 2px',
                                                       'border-radius': '10px',
                                                       'backgroundColor': '#add8e6',
                                                       'padding': '.1.5rem',
                                                       'marginLeft': '0.5rem',
                                                       'marginRight': '0.5rem',
                                                       'marginTop': '1rem',
                                                       'color': 'Black',
                                                       'padding-left': '2%',
                                                       'padding-right': '2%',
                                                       'padding-top': '2%'}),
            ], id='Map34', className= 'pretty_box'),
                ], id='Else98', style={'width': '40%'}),

            html.Div([
                html.Label('Choose your Group'),
                radio_group,
                html.Br(),
                html.Label('Choose your Country'),
                dropdown_country3,
                html.Br(),
                html.Label('Choose the Year'),
                slider_year,
                html.Br(),
            ], id='Iteraction54', style={'width': '20%', 'boxShadow': '#e3e3e3 4px 4px 2px',
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

        html.Div([
                html.Div([
                dcc.Graph(id='bubble_graph', className='pretty_box', style={'boxShadow': '#e3e3e3 4px 4px 2px',
                                                             'border-radius': '10px',
                                                             'backgroundColor': '#add8e6',
                                                             'padding': '.1.5rem',
                                                             'marginLeft': '1rem',
                                                             'marginRight': '1rem',
                                                             'marginTop': '1rem'}),
                ], id='Map1864', className='pretty_box', style={'boxShadow': '#e3e3e3 4px 4px 2px',
                                                             'border-radius': '10px',
                                                             'backgroundColor': '#add8e6',
                                                             'padding': '.1.5rem',
                                                             'marginLeft': '1rem',
                                                             'marginRight': '1rem',
                                                             'marginTop': '1rem'}),
                ], id='Else6347', style={'width': '40%'}),
            ], id='5th row', style={'display': 'flex'}),

        html.Div([
        html.Div([

           ], id='Iteraction34', style={'width': '5%',
                                        }, className='pretty_box'),
        html.Div([
            html.Div([
                dcc.Graph(id='heatmap', style={'boxShadow': '#e3e3e3 4px 4px 2px',
                                               'border-radius': '10px',
                                               'backgroundColor': '#add8e6',
                                               'padding': '.1.5rem',
                                               'marginLeft': '0.5rem',
                                               'marginRight': '0.5rem',
                                               'marginTop': '1rem',
                                               'color': 'Black',
                                               'padding-left': '2%',
                                               'padding-right': '2%',
                                               'padding-top': '2%'}),
            ], id='Map12', className= 'pretty_box'),
                ], id='Else48', style={'width': '90%'}),

        html.Div([

                ], id='Else854', style={'width': '5%'}),
            ], id='6th row', style={'display': 'flex'}),

    html.Div([
        html.H1('Exploratory Space', style={"background-color": "#096484",
                                                    'color': 'White', 'text-align': 'center'})
    ], id='7th row', style={'width': '60%', 'padding-left': '20%', 'padding-right': '20%'}, className='pretty_box'),

    html.Div([
        html.Div([
            html.Label('Choose your Country'),
            dropdown_country1,
            html.Br(),
            html.Label('Choose the Year'),
            dropdown_year,
            html.Br(),
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
                                                  #   'marginLeft':'1rem',
                                                  #  'marginRight':'1rem',
                                                     'marginTop': '1rem'
                                                 }),
            ], id='Map4', className= 'pretty_box'),
                ], id='Else3', style={'width': '40%'}),

        html.Div([
                html.Div([
                    dcc.Graph(id='pie_graph1', style={'boxShadow': '#e3e3e3 4px 4px 2px',
                                                     'border-radius': '10px',
                                                     'backgroundColor': '#add8e6',
                                                     #     'padding':'.1rem',
                                                     #    'marginLeft':'1rem',
                                                     #   'marginRight':'1rem',
                                                     'marginTop': '1rem'
                                                     }),
                ], id='Map5', className='pretty_box',style={'boxShadow': '#e3e3e3 4px 4px 2px',
                                                        'border-radius': '10px',
                                                        'backgroundColor': '#add8e6',
                                                        'padding':'.1.5rem',
                                                        'marginLeft':'1rem',
                                                        'marginRight':'1rem',
                                                        'marginTop': '1rem'}),
                ], id='Else4', style={'width': '40%'}),
        html.Div([
            html.Label('Choose your Country'),
            dropdown_country2,
            html.Br(),
            html.Label('Choose the Year'),
            dropdown_year1,
            html.Br(),
            ], id='Iteraction3', style={'width': '10%', 'boxShadow': '#d3d3d3 4px 4px 2px',
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
            ], id='8th row', style={'display': 'flex'}),

        ])

@app.callback(
    [
        Output("choropleth", "figure"),
        Output("bar_graph", "figure"),
        Output ("aggregate_graph", "figure"),
        Output("scatterplot_graph", "figure"),
        Output("heatmap", "figure"),
        Output("bubble_graph", "figure"),
        Output ("pie_graph", "figure"),
        Output ("pie_graph1", "figure"),
    ],
    [
        Input('radio_year', "value"),
        Input("radio_indicator", "value"),
        Input("country_drop", "value"),
        Input("radio_group", "value"),
        Input("country_drop1", "value"),
        Input("year_drop", "value"),
        Input("country_drop2", "value"),
        Input("year_drop1", "value"),
        Input("indicator_drop", "value"),
        Input("country_drop3", "value"),
        Input("year_slider", "value"),
        #State("indicator_drop", "value")
    ]
)

def plots( year, indicator, countries, groups,country1, year1, country2, year2, indicator2, country3, year4):
    ############################################First Plot - Bar##########################################################
    data_bar = []
    for country in countries:
        df_bar = df.loc[(df['Country'] == country)]

        x_bar = df_bar['Year']
        y_bar = df_bar[indicator]

        data_bar.append(dict(type='bar', x=x_bar, y=y_bar, name=country))

    layout_bar = dict(title=dict(text='Indicator ' + str(indicator.replace('_', ' ')),
                                 x = .5),
                      title_font_size=15,
                      yaxis=dict(title='Value'),
                      xaxis=dict(title='Year'),
                      paper_bgcolor='#add8e6',
                      plot_bgcolor='#add8e6',
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
                                      bgcolor='#add8e6',


                                      ),

                             title=dict(
                                 text='European ' + str(indicator.replace('_', ' ')) + ' Index' + ' Choropleth Map on the year ' + str(year),
                                 x=.5  # Title relative position according to the xaxis, range (0,1)

                             ),
                             title_font_size=15,
                             paper_bgcolor='#add8e6',


                             )

    fig_choro = go.Figure(data=data_choropleth, layout=layout_choropleth)


    fig_choro.update_layout(
        margin=dict(l=50, r=50, t=50, b=50, pad=4,
                autoexpand=True))

    ############################################Third Plot - Scatter ######################################################

    data_agg =[]

    for country in country3:
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
                      title_font_size = 15,
                      yaxis=dict(title='Income'),
                      xaxis=dict(title='Years'),
                      paper_bgcolor='#add8e6',
                      plot_bgcolor='#add8e6',

                      )

############################################ Forth Plot - Scatter (Poverty, Crime) ################################

    scatterplot_list = []


    for country in countries:
        df_scatter = df.loc[(df['Year'] == year)]
        df_scatter = df_scatter.loc[(df_scatter['Country'] == country)]

        x_line = df_scatter[indicator2]

        y_line = df_scatter[indicator]

        scatterplot_list.append(dict(type='scatter',
                             x=x_line,
                             y=y_line,
                             name=country,
                             #      labels = dict('Group ' + str(groups)),
                             # text = str(groups), #ADD MORE INFO HERE
                             mode='markers',
                             marker_size = 18,
                             marker_symbol = '219',
        )
                        )

    layout_ag = dict(title=dict(text= str(indicator).replace('_', ' ') + ' vs '+  str(indicator2).replace('_', ' ') + ' in ' + str(year),
                                x=.5),
                      title_font_size=15,
                      yaxis=dict(title=str(indicator).replace('_', ' ')),
                      xaxis=dict(title=str(indicator2).replace('_', ' ')),
                      paper_bgcolor='#add8e6',
                      plot_bgcolor='#add8e6',

                      )

    fig_scatterplot = go.Figure(data=scatterplot_list, layout=layout_ag)

    ############################################ Fifth Plot - Heatmap #############################################################

    df_heatmap_final = df_heatmap.loc[df_heatmap['Year'] == year4]

    df_heatmap_final1 = df_heatmap_final[df_heatmap_final.columns[1:]]

    df_heatmap_ = df_heatmap_final1.transpose()

    x1 = df_heatmap_.columns
    y1 = df_heatmap_.index
    z1 = df_heatmap_.values

    data_heat = dict(type='heatmap', x=x1, y=y1, z=z1, colorscale='blues')
    layout_heat = dict(title=dict(text='Share of Renewable Energy (Total and By Sector)',
                                  x =.5),
                       autosize=True,
                       paper_bgcolor='#add8e6',
                       plot_bgcolor='#add8e6',
                       yaxis=dict(title='Indicators',
                                  ),
                       xaxis = dict(title='Countries',
                                    tickangle =45,
                                    )
                       )

    fig_heat = go.Figure(data=data_heat, layout=layout_heat)

    ############################################ Sixth Plot - Bubble Chart #############################################################

    fig_bubb = go.Figure()

    df_bubbb = PCP

    hover_text = []
    bubble_size = []

    for index, row in df_bubbb.iterrows():
        hover_text.append(('Country: {Country}<br>' +
                           'Poverty: {Poverty}<br>' +
                           'Crime: {Crime}<br>' +
                           'Population: {Pop}<br>' +
                           'Year: {Year}').format(Country=row['Country'],
                                                  Poverty=row['Poverty'],
                                                  Crime=row['Crime'],
                                                  Pop=row['Population'],
                                                  Year=row['Year'])),
        bubble_size.append(math.sqrt(row['Population']))

    df_bubbb['text'] = hover_text
    df_bubbb['size'] = bubble_size
    sizeref = 2. * max(df_bubbb['size']) / (100 ** 2)

    for country in country3:
        df_bubb = df_bubbb.loc[(df_bubbb['Country'] == country) & (df_bubbb['Year'] == year4)]

        fig_bubb.add_trace(go.Scatter(
            x=df_bubb['Poverty'], y=df_bubb['Crime'],
            name=country,
            text=df_bubb['text'],
            marker_size=df_bubb['size'],
        )),

    # Tune marker appearance and layout
    fig_bubb.update_traces(mode='markers', marker=dict(sizemode='area', sizeref=sizeref,
                                                       line_width=2)),

    fig_bubb.update_layout(
        title= dict(text = 'Poverty vs. Crime',x=.5),
        title_font_size=15,
        xaxis=dict(
            title='Poverty (by year)',
            # gridcolor='white',
            # type='log',
            # gridwidth=2,
        ),

        yaxis=dict(
            title='% Crime (by years)',
            # gridcolor='white',
            # gridwidth=2,
        ),
        paper_bgcolor='#add8e6',
        plot_bgcolor='#add8e6',
    )
    ############################################ Seventh Plot - Pie #############################################################

    data_pie = []

    df_pie = education.loc[country1]

    labels = education['Level_Education'].unique().tolist()

    values = df_pie[('Educational_Attainment_Level', year1)].tolist()

    data_pie.append(dict(type='pie', labels = labels, values =values, hole = 0.5))

    layout_pie = dict(title=dict(text='Education in ' + str(country1) + ' in ' + str(year1),
                                 x=.5,
                                 ),
                      title_font_size=15,
                      paper_bgcolor='#add8e6',
                     plot_bgcolor = '#add8e6',
                      legend=dict(
                          orientation="h",
                          yanchor="bottom",
                          y=1.02,
                          xanchor="right",
                          x=1.1
                      ))

    fig_pie = go.Figure(data=data_pie, layout=layout_pie)

    ############################################ Eighth Plot - Pie #############################################################

    data_pie1 = []

    df_pie1 = education.loc[country2]

    labels1 = education['Level_Education'].unique().tolist()

    values1 = df_pie1[('Educational_Attainment_Level', year2)].tolist()

    data_pie1.append(dict(type='pie', labels=labels1, values=values1, hole=0.5))

    layout_pie1 = dict(title=dict(text='Education in ' + str(country2) + ' in ' + str(year2),
                                 x=.5,
                                 ),
                       title_font_size=15,
                      paper_bgcolor='#add8e6',
                      plot_bgcolor='#add8e6',
                       legend=dict(
                           orientation="h",
                           yanchor="bottom",
                           y=1.02,
                           xanchor="right",
                           x=1.1
                       ))

    fig_pie1 = go.Figure(data=data_pie1, layout=layout_pie1)


    return  fig_choro, \
            go.Figure(data=data_bar, layout=layout_bar), \
            go.Figure(data=data_agg, layout=layout_agg), \
            fig_scatterplot, \
            fig_heat, \
            fig_bubb, \
            fig_pie, \
            fig_pie1


@app.callback(
    [
        Output("kpi1", "children"),
        Output("kpi2", "children"),
      #  Output("gas_3", "children"),
      #  Output("gas_4", "children"),
      #  Output("gas_5", "children")
    ],
    [
        Input("radio_year", "value"),
        Input("radio_indicator", "value"),
    ]
)
def indicator(year, indicator):
    df_cards1 = df.loc[(df['Year'] == year)]
    past_year = year-1
    df_cards2 = df.loc[(df['Year'] == past_year)]

    value_1 = round(df_cards1[indicator].mean())

    if past_year == 2015:
        value_2 = str('There is no value!')
    else:
        current = df_cards1[indicator].mean()
        past = df_cards2[indicator].mean()
        value_2 = round(((current-past)/past), 4)


   # value_3 = round(df_loc.loc[df_loc['year'] == year][gas_names[2]].values[0], 2)
   # value_4 = round(df_loc.loc[df_loc['year'] == year][gas_names[3]].values[0], 2)
   # value_5 = round(df_loc.loc[df_loc['year'] == year][gas_names[4]].values[0], 2)

    return str(indicator).replace('_', ' ') + ': ' + str(value_1), \
           str(indicator).replace('_', ' ') + ': ' + str(value_2), \
          # str(gas_names[2]).replace('_', ' ') + ': ' + str(value_3), \
          # str(gas_names[3]).replace('_', ' ') + ': ' + str(value_4), \
          # str(gas_names[4]).replace('_', ' ') + ': ' + str(value_5),

if __name__ == '__main__':
    app.run_server(debug=True)
