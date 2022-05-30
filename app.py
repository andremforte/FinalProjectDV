
#pip install yahoo_fin
#pip install streamlit-option-menu
import streamlit as st 
import pandas as pd
import yfinance as yf
from yahoo_fin import stock_info
from datetime import date, datetime,timedelta
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image 
import holoviews as hv
import hvplot.pandas
from scipy import stats, signal 
from urllib.request import urlopen
from plotly.subplots import make_subplots
from streamlit_option_menu import option_menu


#Dashboard configuration <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

st.set_page_config(page_title="Invest4Some Financial dashboard",
			        page_icon="chart_with_upwards_trend",
					layout="wide")

header = st.container()

col5, col6, col7 = st.columns([1, 1, 1])

dataset =st.container()

col16, col17, col31, col18, col19, col32, col33, col34, col20 = st.columns([0.3, 0.1, 0.1, 0.3, 0.1, 0.15, 0.3, 0.1, 0.3])

explor_1 = st.container()

currencies_explor = st.container()
col44, col45, col46, col47, col48, col49 = st.columns([0.01, 0.45, 0.45, 0.1, 0.3, 0.3])
currencies_2 = st.container()

stocks = st.container()
col82, col43, col37, col38, col39, col41 = st.columns([0.01, 0.45, 0.45, 0.1, 0.3, 0.3])
stocks_2 = st.container()


selection  = st.container()

col1, col2= st.columns([2.75, 1.5])

currencies = st.container() 

col3, col4 = st.columns([2.75, 1.5])

col21, col22 = st.columns([3,1.5])

financial_indicators = st.container()

col23, col24 = st.columns([2,2])

col35, col36 = st.columns([2.5, 0.5])

financial_ind = st.container()

model_training = st.container()

col12, col13 = st.columns([3.5, 1]) 


#Side bar
options = st.sidebar.radio('Select which asset do you want to analyse', options= ['Cryptocurrencies', 'Currencies', 'Other Assets', 'Exploratory Space', 'Menu'], index = 4)

#Cryptocurrencies

today = date.today()
yesterday = date.today() - timedelta(days = 1)

if options == "Menu": 
	with header: 
		st.markdown("<h1 style='text-align: center; color: 	#191970;'>Invest4Some Financial Dashboard</h1>", unsafe_allow_html=True)
		st.markdown("""<hr style="height:10px;border:none;color:#191970;background-color:#191970;" /> """, unsafe_allow_html=True)
		st.markdown ("<h5 style='text-align: center; color: 	#000000;'>Welcome! In this dashboard, it is possible to analyse different financial assets in order to have more information to make better investment decisions. In the sidebar, it is possible to select which asset do you want to analyse, having indepedent analyses for each of them. Lastly, there is a section called 'Exploratory Space' where you can perform you personalized analysis, including different indicators in the same plot and extracting the information that you need.</h5>", unsafe_allow_html=True)
		st.markdown("""<hr style="height:10px;border:none;color:#191970;background-color:#191970;" /> """, unsafe_allow_html=True)

	with col5:
		st.markdown("<h3 style='text-align: center; color: 	#004AAD ;'>Cryptocurrencies</h3>", unsafe_allow_html=True)
		st.markdown ("<h5 style='text-align: center; color: #000000;'>In this chapter, there is information about Cryptocurrencies. You have access to the current price and the distribution of the prices from January 1st, 2021. Moreover, you can find information about financial indicators. Finally, using Machine Learning algorithms, it is possible to see the prediction for the next closing price for each cryptocurrency.</h5>", unsafe_allow_html=True)

	with col6: 
		st.markdown("<h3 style='text-align: center; color: 	#004AAD ;'>Currencies</h3>", unsafe_allow_html=True)
		st.markdown ("<h5 style='text-align: center; color: #000000;'>Here you can find information about Currencies. It is possible to analyse their prices’ distribution and some important financial indicators. </h5>", unsafe_allow_html=True)

	with col7:
		st.markdown("<h3 style='text-align: center; color: 	#004AAD ;'>Other Assets</h3>", unsafe_allow_html=True)
		st.markdown ("<h5 style='text-align: center; color: #000000;'>In “Other Assets”, it possible to analyse stocks prices. You can analyse financial indicators and extract interesting patterns from them.</h5>", unsafe_allow_html=True)

	with dataset: 
		st.markdown("""<hr style="height:10px;border:none;color:#191970;background-color:#191970;" /> """, unsafe_allow_html=True)


if options == 'Cryptocurrencies': 

	with header: 
		st.markdown("<h1 style='text-align: center; color: 	#191970;'>Invest4Some Financial Dashboard</h1>", unsafe_allow_html=True)
		st.markdown("""<hr style="height:10px;border:none;color:#191970;background-color:#191970;" /> """, unsafe_allow_html=True)
	
	crypto1 = st.sidebar.text_input("Write the crypto symbol (E.g.: BTC) ", 'BTC')
	categories = st.sidebar.selectbox('Select the price that you want to analyse', options = ['Close', 'Low', 'High', 'Open', 'Adj Close'], index = 0)
	start_date = st.sidebar.date_input('Start Date', date(2021,1,1))
	end_date = st.sidebar.date_input('End Date', date.today()-timedelta(days=1))
	st.sidebar.markdown("Source: https://finance.yahoo.com/")

	df = yf.download(str(crypto1) + "-USD",  start="2021-01-01",  end= today)

	with selection:
		st.markdown("<h2 style='text-align: center; color: 	#00000;'>Cryptocurrencies Analysis</h2>", unsafe_allow_html=True)

	def definition (df, crypto, selection, date1, date2): 
		c = df.loc[date1:date2+timedelta(days = 1)]
		st.subheader(str(crypto) + " Prices")

		fig = make_subplots(specs=[[{"secondary_y": True}]])

		fig.add_scatter(x=c.index, y=c[str(selection)], secondary_y=False, name = str(crypto) + " - " + str(selection))

		fig.add_bar(x=c.index, y=c.Volume, secondary_y=True, name = 'Volume')
		    
		fig.update_yaxes(title_text=str(selection), secondary_y=False, rangemode = 'tozero', showgrid= False)
		fig.update_yaxes(title_text="Volume", secondary_y=True, rangemode = 'tozero', showgrid= False)
		fig.update_xaxes(title_text = 'Date', showgrid =  False)

		fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)', title={'text': 'Distribution of the ' + str(selection) + ' Price and Volume of ' + str(crypto),
		                                                                                       'y':0.9,
		                                                                                       'x':0.5,
		                                                                                       'xanchor': 'center',
		                                                                                       'yanchor': 'top'}, 
		                                                                                       legend=dict(
																							    orientation="h",
																							    yanchor="bottom",
																							    y=1.02,
																							    xanchor="right",
																							    x=1
																							))

		st.plotly_chart(fig, use_container_width=True)

		
	with col1: 
		data = round(stock_info.get_live_price(crypto1 + "-USD"),2)

		a = df['Close'].tail(8)
		a = a.head(7)
		a = a.tail(1)
		variation = round(((data - float(a))/ float(a))*100,2)

		definition (df, crypto1, categories,start_date, end_date)

	with col2: 
		st.metric(label="Current Trading Price "f"(% difference from yesterday's closing price)", value=f"{data} USD", delta = f"{variation} %")
		 
	def definition2 (df, crypto, selection): 
		st.markdown ("<h5 style='text-align: center; color: #000000;'>Table with prices and volume of the last 7 days (USD)</h5>", unsafe_allow_html=True)
		df1 = round(df[[selection, 'Volume']],4).tail(7)
		df1= pd.DataFrame(df1)

		df1.reset_index(inplace = True)
		df1['Date']= df1['Date'].dt.strftime("%d %B, %Y")
		df1.set_index('Date', inplace = True)
		st.dataframe(df1)

	with col2:
		definition2(df, crypto1, categories)

#INDICATORS --------------

	def indicators (df):

			# Set the short window and long windows
		short_window = 50
		long_window = 100
			# Generate the short and long moving averages (50 and 100 days, respectively)

		df['SMA50'] = df['Close'].rolling(window=short_window).mean()
		df['SMA100'] = df['Close'].rolling(window=long_window).mean()
		df['ShortEMA'] = df['Close'].ewm(span=7, adjust=False).mean()

		df['LongEMA'] = df.Close.ewm(span=26, adjust=False).mean()
		    # Calculate the Moving Average Convergence/Divergence (MACD)
		df['MACD'] = df['ShortEMA'] - df['LongEMA']
		# Calcualte the signal line
		df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

		df['Signal'] = 0.0

		# Generate the trading signal 0 or 1,
		    # where 0 is when the SMA50 is under the SMA100, and
		    # where 1 is when the SMA50 is higher (or crosses over) the SMA100
		df['Signal'][short_window:] = np.where(
		    df['SMA50'][short_window:] > df['SMA100'][short_window:], 1.0, 0.0)
		# Calculate the points in time at which a position should be taken, 1 or -1

		df['Entry/Exit'] = df['Signal'].diff()

		df['Up Move'] = np.nan
		df['Down Move'] = np.nan
		df['Average Up'] = np.nan
		df['Average Down'] = np.nan
		# Relative Strength
		df['RS'] = np.nan
		# Relative Strength Index
		df['RSI'] = np.nan


		for x in range(1, len(df)):
		    df['Up Move'][x] = 0
		    df['Down Move'][x] = 0
		    
		    if df['Adj Close'][x] > df['Adj Close'][x-1]:
		        df['Up Move'][x] = df['Adj Close'][x] - df['Adj Close'][x-1]
		        
		    if df['Adj Close'][x] < df['Adj Close'][x-1]:
		        df['Down Move'][x] = abs(df['Adj Close'][x] - df['Adj Close'][x-1])  
		        
		## Calculate initial Average Up & Down, RS and RSI
		df['Average Up'][14] = df['Up Move'][1:15].mean()
		df['Average Down'][14] = df['Down Move'][1:15].mean()
		df['RS'][14] = df['Average Up'][14] / df['Average Down'][14]
		df['RSI'][14] = 100 - (100/(1+df['RS'][14]))



			## Calculate rest of Average Up, Average Down, RS, RSI
		for x in range(15, len(df)):
		    df['Average Up'][x] = (df['Average Up'][x-1]*13+df['Up Move'][x])/14
		    df['Average Down'][x] = (df['Average Down'][x-1]*13+df['Down Move'][x])/14
		    df['RS'][x] = df['Average Up'][x] / df['Average Down'][x]
		    df['RSI'][x] = 100 - (100/(1+df['RS'][x]))

		## Calculate the buy & sell signals
		## Initialize the columns that we need
		df['Long Tomorrow'] = np.nan
		df['Buy Signal'] = np.nan
		df['Sell Signal'] = np.nan
		df['Buy RSI'] = np.nan
		df['Sell RSI'] = np.nan
		df['Strategy'] = np.nan


			## Calculate the buy & sell signals
		for x in range(15, len(df)):

			# Calculate "Long Tomorrow" column
			if ((df['RSI'][x] <= 40) & (df['RSI'][x-1]>40) ):
			    df['Long Tomorrow'][x] = True
			elif ((df['Long Tomorrow'][x-1] == True) & (df['RSI'][x] <= 70)):
			    df['Long Tomorrow'][x] = True
			else:
			    df['Long Tomorrow'][x] = False
			    
			# Calculate "Buy Signal" column
			if ((df['Long Tomorrow'][x] == True) & (df['Long Tomorrow'][x-1] == False)):
			    df['Buy Signal'][x] = df['Adj Close'][x]
			    df['Buy RSI'][x] = df['RSI'][x]
			    
			# Calculate "Sell Signal" column
			if ((df['Long Tomorrow'][x] == False) & (df['Long Tomorrow'][x-1] == True)):
			    df['Sell Signal'][x] = df['Adj Close'][x]
			    df['Sell RSI'][x] = df['RSI'][x]

			## Calculate strategy performance

		df['Strategy'][15] = df['Adj Close'][15]

		for x in range(16, len(df)):
			if df['Long Tomorrow'][x-1] == True:
			    df['Strategy'][x] = df['Strategy'][x-1]* (df['Adj Close'][x] / df['Adj Close'][x-1])
			else:
			    df['Strategy'][x] = df['Strategy'][x-1]


		return df

	data = indicators(df)


	def indicators_plot(data, categories, crypto, date1, date2):
		
		df1= data.copy()

		df = df1.loc[date1:date2+timedelta(days = 1)]

			# Visualize exit position relative to close price
		exit = df[df['Entry/Exit'] == -1.0]['Close'].hvplot.scatter(
		    color='red',
		    legend=False,
		    ylabel='Price in $',
		    width=1000,
		    height=400, hover_cols=[str(crypto)]
		)
		# Visualize entry position relative to close price
		entry = df[df['Entry/Exit'] == 1.0]['Close'].hvplot.scatter(
		    color='green',
		    legend=False,
		    ylabel='Price in $',
		    width=1000,
		    height=400, hover_cols=[str(crypto)]
		)
		# Visualize close price for the investment
		security_close = df[['Close']].hvplot(
		    line_color='lightgray',
		    ylabel='Price in $',
		    width=1000,
		    height=400, hover_cols=[str(crypto)]
		)
		# Visualize moving averages
		moving_avgs = df[['SMA50', 'SMA100']].hvplot(
		    ylabel='Price in $',
		    width=1000,
		    height=400,hover_cols=[str(crypto)]
		)
		# Overlay plots for the SMA
		entry_exit_plot = security_close * moving_avgs * entry * exit
		fig=entry_exit_plot.opts(xaxis=None)
		#st.bokeh_chart(hv.render(fig, backend='bokeh'))


		# plot for MACD and Signal line
		fig1 = go.Figure()
		fig1.add_trace(go.Scatter(
		    name="MACD",
		    mode="lines", x=df.index, y=df["MACD"],
		))
		fig1.add_trace(go.Scatter(
		    name="Signal_Line",
		    mode="lines", x=df.index, y=df["Signal_Line"],
		    
		))
		
		# Updating layout
		fig1.update_layout(
		    title={'text':'MACD indicator',
		    'y':0.9,
	       'x':0.5,
	       'xanchor': 'center',
	       'yanchor': 'top'}, 
		    xaxis_title='Date',
		    template='plotly_white',
		    paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)'
		)


		fig1=fig1.update_xaxes(showgrid=True, ticklabelmode="period")
		st.plotly_chart(fig1, use_container_width=True)


	def indicators_plot2(data, categories, crypto, date1, date2):

		df1= data.copy()

		df = df1.loc[date1:date2+timedelta(days = 1)]

	# plot for RSI
		fig2 = go.Figure()
		fig2.add_trace(go.Scatter(
		    name="Buy",
		    mode="markers", x=df.index, y=df["Buy RSI"],
		))
		fig2.add_trace(go.Scatter(
		    name="Sell",
		    mode="markers", x=df.index, y=df["Sell RSI"],
		    
		))
		
		fig2.add_trace(go.Scatter(
		    name="RSI",
		    mode="lines", x=df.index, y=df["RSI"]
		    
		))
		
		# Updating layout
		fig2.update_layout(
		    title={'text':'RSI indicator',
		     'y':0.9,
		       'x':0.5,
		       'xanchor': 'center',
		       'yanchor': 'top'}, 
		    xaxis_title='Date',
		    template='plotly_white',
		    paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h",
																							    yanchor="bottom",
																							    y=1.02,
																							    xanchor="right",
																							    x=1
																							)
		)

	
		fig2=fig2.update_xaxes(showgrid=True, ticklabelmode="period")

		st.plotly_chart(fig2, use_container_width=True)


	def indicators_plot3(data, categories, crypto, date1, date2):

		df1= data.copy()

		df = df1.loc[date1:date2+timedelta(days = 1)]

		# plot for SMA in plotly
		fig = go.Figure()
		
		fig.add_trace(go.Scatter(
		    name="Exit",
		    mode="markers", x=df[df['Entry/Exit'] == -1.0]['Close'].index, y=df[df['Entry/Exit'] == -1.0]['Close'], marker_color='red'
		    
		))
		
		fig.add_trace(go.Scatter(
		    name="Entry",
		    mode="markers", x=df[df['Entry/Exit'] == 1.0]['Close'].index, y=df[df['Entry/Exit'] == 1.0]['Close'], marker_color='green'
		    
		))
		
		fig.add_trace(go.Scatter(
		    name="Close",
		    mode="lines", x=df.index, y=df["Close"], line_color='grey'
		))

		fig.add_trace(go.Scatter(
		    name="SMA50",
		    mode="lines", x=df.index, y=df["SMA50"], line_color='red'
		))

		fig.add_trace(go.Scatter(
		    name="SMA100",
		    mode="lines", x=df.index, y=df["SMA100"], line_color='gold'
		))

		# Updating layout
		fig.update_layout(
			title={'text': "Simple Moving Average indicator for " + str(crypto),
			       'y':0.9,
			       'x':0.5,
			       'xanchor': 'center',
			       'yanchor': 'top'},
		    xaxis_title='Date',
		    yaxis_title='Price',
		    template='plotly_white',
		    paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
		)

		fig=fig.update_xaxes(showgrid=True, ticklabelmode="period")
		st.plotly_chart(fig, use_container_width=True)


	def graph2 (df, crypto, window):
		data = df.copy()
		data['MA' + str(window)] = data.Close.rolling(window = window).mean()

		std = data.Close.rolling(window).std()
		bollinger_up = data['MA' + str(window)] + std * 2 # Calculate top band
		bollinger_down = data['MA' + str(window)] - std * 2 # Calculate bottom band

		data['bollinger_up'] = bollinger_up
		data['bollinger_down'] = bollinger_down

		fig = go.Figure(data = [go.Candlestick(x=data.index,
			                open=data['Open'],
			                high=data['High'],
			                low=data['Low'],
			                close=data['Close'], name = str(crypto) + ' prices'),
						go.Scatter(x=data.index, y=data['bollinger_up'], line=dict(color='orange', width=2), name = "Bollinger Up"),
						go.Scatter(x=data.index, y=data['bollinger_down'], line=dict(color='blue', width=2), name = "Bollinger Down")])

		# Add titles
		fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)', title={'text': "Bollinger Bands for " + str(crypto),
																						       'y':0.9,
																						       'x':0.5,
																						       'xanchor': 'center',
																						       'yanchor': 'top'}, 
																						        yaxis_title='Prices',
																						        xaxis_title = 'Date')

		return st.plotly_chart(fig, use_container_width=True)


	with financial_indicators: 
		st.header("Financial Indicators")


	with col24:
		indicators = st.selectbox('Select the indicator', options = ['Moving Average Convergence Divergence', 'RSI', 'Bollinger Bands', 'Moving Average'], index = 0)


	if indicators == 'Moving Average Convergence Divergence':
		with col35:
			indicators_plot(data, categories, crypto1, start_date, end_date)

		with col36:
			st.markdown ("<h5 color: 	#000000;'>Important Information: </h5>", unsafe_allow_html=True)
			st.markdown('>Traders may buy the security when the MACD crosses above its signal line and sell—or short—the security when the MACD crosses below the signal line.')

	if indicators == 'RSI':
		with col35:
			indicators_plot2(data, categories, crypto1, start_date, end_date)

		with col36:
			st.markdown ("<h5 color: 	#000000;'>Important Information: </h5>", unsafe_allow_html=True)
			st.markdown('> In Relative Strength Index (RSI), values of 70 or above indicate that a security is becoming overbought or overvalued and may be primed for a trend reversal or corrective pullback in price. An RSI reading of 30 or below indicates an oversold or undervalued condition.')

	if indicators == 'Moving Average': 
		with col35: 
			indicators_plot3(data, categories, crypto1, start_date, end_date)

		with col36:
			st.markdown ("<h5 color: 	#000000;'>Important Information: </h5>", unsafe_allow_html=True)
			st.markdown('>A death cross occurs when the 50-day SMA crosses below the 100-day SMA. This is considered a bearish signal, indicating that further losses are in store. The golden cross occurs when a short-term SMA breaks above a long-term SMA. Reinforced by high trading volumes, this can signal further gains are in store.')


	if indicators == 'Bollinger Bands':
		with col23:
			window = st.slider('Select the length of the window', 5, 100, 7)
		with col35:
			graph2(df, crypto1, window)

		with col36: 
			st.markdown ("<h5 color: 	#000000;'>Important Information: </h5>", unsafe_allow_html=True)
			st.markdown('>Bollinger Bands can be used to determine how strongly an asset is rising and when it is potentially reversing or losing strength. An uptrend that reaches the upper band indicates that the stock is pushing higher and traders can exploit the opportunity to make a buy decision. In a strong downtrend, the price will run along the lower band, and this shows that selling activity remains strong.')


	#4TH PART <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< PREDICTIONS <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
	with model_training: 
		st.header("Predictions")

	with col13: 
		window = st.selectbox('Select the sliding window (Days) for Modeling', options = np.arange(1, 91), index = 6)

	#Data Preparation for Modeling
	def preparation (crypto):
		g = yf.download(str(crypto) + "-USD",  start="2021-01-01",  end= today)
		g.dropna(inplace=True)
		return g

	df1 = preparation(crypto1)

	def feature_eng(df, crypto):
		#financial indicators for predictions
		df['SMA_7'] = df['Close'].rolling(window=7).mean() #Simple Moving Average - 7 days
		df['EMA_7'] = df['Close'].ewm(span=7, adjust=False).mean()
		df['14-high'] = df['High'].rolling(14).max()
		df['14-low'] = df['Low'].rolling(14).min()
		df['%K'] = (df['Close'] - df['14-low'])*100/(df['14-high'] - df['14-low'])
		df['%D'] = df['%K'].rolling(3).mean()
		df['spread'] =  df['High'] -  df['Low']
		df['volatility'] =  df['spread'] /  df['Open']
		df['close_off_high'] =  df['Close']- df['High']


	    #external data (EURUSD Change Rate)
		euro_dollar = yf.download("EURUSD=X", start="2021-01-01", end=yesterday)
		external_data = euro_dollar['Close']
		external_data = pd.DataFrame(external_data)
		external_data.rename(columns = {'Close': 'EUR/USD_close'}, inplace = True)
		external_data.reset_index(inplace = True)

		df.reset_index(inplace = True)

		#MERGE EXTERNAL DATA WITH ORIGINAL DF
		df2 = pd.merge(df, external_data, on ='Date', how = 'outer',sort=True) 
		merged_df = df2.merge(df, on ='Date', how = 'inner',sort=True, suffixes=('', '_y'))
		merged_df.drop(merged_df.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)
		merged_df['EUR/USD_close'].ffill(axis = 0, inplace = True) #filling the Nan's (weekend day's) from 'Close' with the previous value (Friday)
		merged_df.set_index('Date', inplace = True)

		#ADD COLUMN BTC
		if crypto1 != "BTC":
		    btc = yf.download("BTC-USD",  start="2021-01-01",  end= today)
		    btc.reset_index(inplace = True)
		    btc = btc[['Date', 'Close']]
		    btc.rename(columns={'Close': 'BTC_ClosePrice'}, inplace = True)        
		    merged= pd.merge(merged_df, btc, on = 'Date', how = 'inner')
		    merged.set_index('Date',inplace = True)
		    final_df = merged.copy()

		else: 
		    final_df = merged_df.copy()

		return final_df

	df2 = feature_eng(df1, crypto1)

	def normalization(df):
		scaler = MinMaxScaler().fit(df) 
		df1 = pd.DataFrame(scaler.transform(df), columns=df.columns).set_index(df.index)
		return df1

	df3 = normalization(df2)

	def correlation (df):
		cor_matrix = df.corr().abs()
		upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
		to_drop = [column for column in upper_tri.columns if any((upper_tri[column] > 0.85) | (upper_tri[column] < -0.85))] 

		if 'Close' in to_drop: 
		    to_drop.remove('Close')
		    to_drop.append('Open')

		df1 = df.drop(to_drop, axis = 1)

		return df1

	df4 = correlation(df3)

	df4 = df4.dropna()

	#Modeling - XGBoost
		#Train and test split
	training_size=int(len(df4)*0.70)
	test_size=len(df4)-training_size
	train_data,test_data=df4.iloc[:training_size,:],df4.iloc[training_size:,:]

		#Sliding window

	def create_dataset(dataset, time_step):
	    dataX, dataY = [], []
	    for i in range(len(dataset)-time_step-1):
	        a = dataset.iloc[i:(i+time_step), :]   
	        dataX.append(a)
	        dataY.append(dataset.iloc[i + time_step, 0]) 
	    return np.array(dataX), np.array(dataY)

	X_train, y_train = create_dataset(train_data, window)
	X_test, y_test = create_dataset(test_data, window)


	nsamples, nx, ny = X_train.shape
	X_train = X_train.reshape((nsamples,nx*ny))

	nsamples, nx, ny = X_test.shape
	X_test = X_test.reshape((nsamples,nx*ny))

	#Model
	my_model = XGBRegressor(n_estimators=1000, objective ='reg:linear', learning_rate = 0.05, max_depth = 1, alpha = 0.01, random_state = 1)
	my_model.fit(X_train, y_train, verbose = False)

	#Predictions - train and test datasets
	predictions = my_model.predict(X_test).reshape(-1,1)
	y_test = y_test.reshape(-1,1)


	predictions_train = my_model.predict(X_train).reshape(-1,1)
	y_train = y_train.reshape(-1,1)

	#Predictions for next day's closing price of the selected cryptocurrency

	df= preparation(crypto1)
	df = feature_eng(df,crypto1)
	df = normalization(df)
	df = correlation(df)
	predictions1 = df.tail(window)
	predictions1 = np.array(predictions1)

	x_input=predictions1[len(predictions1)-window:].reshape(1,-1)
	temp_input=list(x_input)
	temp_input=temp_input[0].tolist()

	output9=[]
	n_steps=window
	i=0
	pred_days = 1


	while(i<pred_days):

		if(len(temp_input)>window):        
		    x_input=np.array(temp_input[i:])
		    print("{} day input {}".format(i,x_input))
		    x_input=x_input.reshape(1,-1)
		    
		    yhat = my_model.predict(x_input)
		    print("{} day output {}".format(i,yhat))
		    temp_input.extend(yhat.tolist())
		    temp_input=temp_input[i:]
		   
		    output9.extend(yhat.tolist())
		    i=i+1
	    

	    #Final table with last X days and predictions for the next day

	a = df['Close'].tail(window)
	a = pd.DataFrame(a)
	a.reset_index(inplace = True)

	m = {'Date': today, 'Close': output9[0]}
	final_table = a.append(m, ignore_index = True)
	final_table.set_index('Date', inplace = True)
	final = pd.DataFrame(final_table)

	final_list = list(final.Close)

	with col12: 

		#Denormalize the final results		
		k = yf.download(str(crypto1) + "-USD",  start="2021-01-01",  end= today)
		k.dropna(inplace=True)

		empty = []

		for i,x in enumerate(final_list):
			final_result2 = ((max(k.Close)-min(k.Close))*(final_list[i]))+min(k.Close)
			empty.append(final_result2)

		final['DenormalizedClose'] = empty

		fig = go.Figure()
		fig.add_scattergl(x=final.index, y=final.DenormalizedClose, line={'color': 'blue'}, name = "last " + str(window) + " closing prices")
					
		if final.DenormalizedClose[window] >= final.DenormalizedClose[window-1]:
			fig.add_scattergl(x=final.index[window-1:], y=final.DenormalizedClose[window-1:], line={'color': 'green'}, name = "predicted closing price")
		else: 
			fig.add_scattergl(x=final.index[window-1:], y=final.DenormalizedClose[window-1:], line={'color': 'red'}, name = "predicted closing price")
		
		fig.update_yaxes(title_text = 'Prices', rangemode="tozero")
		fig.update_xaxes(title_text = 'Date')
		fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)', title={'text': 'Distribution of the Closing Price of ' + str(crypto1) + " in the last " + str(window) +" days and the predicted value",
																						       'y':0.9,
																						       'x':0.5,
																						       'xanchor': 'center',
																						       'yanchor': 'top'}, 
																						       )
		st.plotly_chart(fig, use_container_width=True)

	with col13:
		#Denormalize the final result
		j = yf.download(str(crypto1) + "-USD",  start="2021-01-01",  end= today)
		j.dropna(inplace=True)

		final_result = ((max(j.Close)-min(j.Close))*output9[0])+min(j.Close)
		last_result = j['Close'].tail(1)
		variation2 = round(((float(final_result) - float(last_result))/ float(last_result))*100, 2)

		#FINAL RESULT
		d2 = today.strftime("%B %d, %Y")
		#st.markdown("""<hr style="height:5px;border:none;color:#FFC300;background-color:#FFC300;" /> """, unsafe_allow_html=True)
		st.metric(label="Predicted closing price of " + str(crypto1) + " on " + f"{d2}", value= f"{round(final.DenormalizedClose[window],2)} USD", delta = f"{(round(variation2, 2))} %")
		st.markdown ("<h5 color: 	#000000;'>Important Information: </h5>", unsafe_allow_html=True)
		st.markdown('>Considering our prediction, the price of ' + str(crypto1) + " will vary " + f"{(round(variation2, 2))} % comparing to yesterday's price.")

#################################################################CURRENCIES####################################################################################

if options == 'Currencies':

	with header: 
		st.markdown("<h1 style='text-align: center; color: 	#191970;'>Invest4Some Financial Dashboard</h1>", unsafe_allow_html=True)
		st.markdown("""<hr style="height:10px;border:none;color:#191970;background-color:#191970;" /> """, unsafe_allow_html=True)

	with currencies:
		
		st.markdown("<h2 style='text-align: center; color: 	#00000;'>Currencies Analysis</h2>", unsafe_allow_html=True)
	curr = st.sidebar.text_input('Write the currency symbol (E.g. EURUSD)', 'EURUSD')
	categories = st.sidebar.selectbox('Select the price that you want to analyse', options = ['Close', 'Low', 'High', 'Open', 'Adj Close'], index = 0)
	start_date = st.sidebar.date_input('Start Date', date(2021,1,1))
	end_date = st.sidebar.date_input('End Date', date.today()-timedelta(days=1))
	st.sidebar.markdown("Source: https://finance.yahoo.com/")

	df_curr = yf.download(str(curr) + "=X",  start="2021-01-01",  end= today)

	def external_data(df, curr, selection, date1, date2): 
		d = df.loc[date1:date2+timedelta(days = 1)]

		fig = go.Figure()
		fig.add_scattergl(x=d.index, y=d[selection], line={'color': 'blue'}, name = str(curr) + " - " + str(selection))
		
		fig.update_yaxes(title = 'Prices', rangemode="tozero")
		fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)', title={'text': 'Distribution of the ' + str(selection) + ' Price and Volume of ' + str(curr),
																						       'y':0.9,
																						       'x':0.5,
																						       'xanchor': 'center',
																						       'yanchor': 'top'}, 
																						       legend=dict(
																							    orientation="h",
																							    yanchor="bottom",
																							    y=1.02,
																							    xanchor="right",
																							    x=1
																							))
		fig.update_xaxes(title = "Date")
		st.plotly_chart(fig, use_container_width=True)


	with col3: 
		real_time_curr = round(stock_info.get_live_price(curr + "=X"),2)

		df_var2 = yf.download(str(curr) + "=X",  start="2021-01-01",  end= yesterday)
		a2 = df_var2['Close'].tail(1)
		variation2 = round(((real_time_curr - float(a2))/ float(a2))*100,2)

		st.subheader('Currencies Prices')
		external_data(df_curr, curr, categories, start_date, end_date)

	def external_data2 (df, crypto, selection): 
		st.subheader('Table with prices of the last 7 days (' + str(curr[3:]) + ")")
		df3 = round(df[selection],4).tail(7)
		df3= pd.DataFrame(df3)
		df3.reset_index(inplace = True)
		df3['Date']= df3['Date'].dt.strftime("%d %B, %Y")
		df3.set_index('Date', inplace = True)
		st.dataframe(df3)

	with col4: 
		st.metric(label="Current Trading Price "f"(% difference from yesterday's closing price)", value=f"{real_time_curr} " + str(curr[3:]), delta = f"{variation2} %")
		external_data2(df_curr, curr, categories)


	with financial_indicators: 
		st.header("Financial Indicators")


	def indicators (df):

			# Set the short window and long windows
		short_window = 50
		long_window = 100
			# Generate the short and long moving averages (50 and 100 days, respectively)

		df['SMA50'] = df['Close'].rolling(window=short_window).mean()
		df['SMA100'] = df['Close'].rolling(window=long_window).mean()
		df['ShortEMA'] = df['Close'].ewm(span=7, adjust=False).mean()

		df['LongEMA'] = df.Close.ewm(span=26, adjust=False).mean()
		    # Calculate the Moving Average Convergence/Divergence (MACD)
		df['MACD'] = df['ShortEMA'] - df['LongEMA']
		# Calcualte the signal line
		df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

		df['Signal'] = 0.0

		# Generate the trading signal 0 or 1,
		    # where 0 is when the SMA50 is under the SMA100, and
		    # where 1 is when the SMA50 is higher (or crosses over) the SMA100
		df['Signal'][short_window:] = np.where(
		    df['SMA50'][short_window:] > df['SMA100'][short_window:], 1.0, 0.0)
		# Calculate the points in time at which a position should be taken, 1 or -1

		df['Entry/Exit'] = df['Signal'].diff()

		df['Up Move'] = np.nan
		df['Down Move'] = np.nan
		df['Average Up'] = np.nan
		df['Average Down'] = np.nan
		# Relative Strength
		df['RS'] = np.nan
		# Relative Strength Index
		df['RSI'] = np.nan


		for x in range(1, len(df)):
		    df['Up Move'][x] = 0
		    df['Down Move'][x] = 0
		    
		    if df['Adj Close'][x] > df['Adj Close'][x-1]:
		        df['Up Move'][x] = df['Adj Close'][x] - df['Adj Close'][x-1]
		        
		    if df['Adj Close'][x] < df['Adj Close'][x-1]:
		        df['Down Move'][x] = abs(df['Adj Close'][x] - df['Adj Close'][x-1])  
		        
		## Calculate initial Average Up & Down, RS and RSI
		df['Average Up'][14] = df['Up Move'][1:15].mean()
		df['Average Down'][14] = df['Down Move'][1:15].mean()
		df['RS'][14] = df['Average Up'][14] / df['Average Down'][14]
		df['RSI'][14] = 100 - (100/(1+df['RS'][14]))


			## Calculate rest of Average Up, Average Down, RS, RSI
		for x in range(15, len(df)):
		    df['Average Up'][x] = (df['Average Up'][x-1]*13+df['Up Move'][x])/14
		    df['Average Down'][x] = (df['Average Down'][x-1]*13+df['Down Move'][x])/14
		    df['RS'][x] = df['Average Up'][x] / df['Average Down'][x]
		    df['RSI'][x] = 100 - (100/(1+df['RS'][x]))

		## Calculate the buy & sell signals
		## Initialize the columns that we need
		df['Long Tomorrow'] = np.nan
		df['Buy Signal'] = np.nan
		df['Sell Signal'] = np.nan
		df['Buy RSI'] = np.nan
		df['Sell RSI'] = np.nan
		df['Strategy'] = np.nan


			## Calculate the buy & sell signals
		for x in range(15, len(df)):
		    
		    # Calculate "Long Tomorrow" column
		    if ((df['RSI'][x] <= 40) & (df['RSI'][x-1]>40) ):
		        df['Long Tomorrow'][x] = True
		    elif ((df['Long Tomorrow'][x-1] == True) & (df['RSI'][x] <= 70)):
		        df['Long Tomorrow'][x] = True
		    else:
		    	df['Long Tomorrow'][x] = False
		        
		    # Calculate "Buy Signal" column
		    if ((df['Long Tomorrow'][x] == True) & (df['Long Tomorrow'][x-1] == False)):
		        df['Buy Signal'][x] = df['Adj Close'][x]
		        df['Buy RSI'][x] = df['RSI'][x]
		        
		    # Calculate "Sell Signal" column
		    if ((df['Long Tomorrow'][x] == False) & (df['Long Tomorrow'][x-1] == True)):
		        df['Sell Signal'][x] = df['Adj Close'][x]
		        df['Sell RSI'][x] = df['RSI'][x]

			## Calculate strategy performance
		df['Strategy'][15] = df['Adj Close'][15]


		for x in range(16, len(df)):
		    if df['Long Tomorrow'][x-1] == True:
		        df['Strategy'][x] = df['Strategy'][x-1]* (df['Adj Close'][x] / df['Adj Close'][x-1])
		    else:
		        df['Strategy'][x] = df['Strategy'][x-1]


		return df

	data2 = indicators(df_curr)


	def indicators_plot(data, categories, crypto, date1, date2):
		
		df1= data.copy()

		df = df1.loc[date1:date2+timedelta(days = 1)]

			# Visualize exit position relative to close price
		exit = df[df['Entry/Exit'] == -1.0]['Close'].hvplot.scatter(
		    color='red',
		    legend=False,
		    ylabel='Price in $',
		    width=1000,
		    height=400, hover_cols=[str(crypto)]
		)
		# Visualize entry position relative to close price
		entry = df[df['Entry/Exit'] == 1.0]['Close'].hvplot.scatter(
		    color='green',
		    legend=False,
		    ylabel='Price in $',
		    width=1000,
		    height=400, hover_cols=[str(crypto)]
		)
		# Visualize close price for the investment
		security_close = df[['Close']].hvplot(
		    line_color='lightgray',
		    ylabel='Price in $',
		    width=1000,
		    height=400, hover_cols=[str(crypto)]
		)
		# Visualize moving averages
		moving_avgs = df[['SMA50', 'SMA100']].hvplot(
		    ylabel='Price in $',
		    width=1000,
		    height=400,hover_cols=[str(crypto)]
		)
		# Overlay plots for the SMA
		entry_exit_plot = security_close * moving_avgs * entry * exit
		fig=entry_exit_plot.opts(xaxis=None)
		#st.bokeh_chart(hv.render(fig, backend='bokeh'))


		# plot for MACD and Signal line
		fig1 = go.Figure()
		fig1.add_trace(go.Scatter(
		    name="MACD",
		    mode="lines", x=df.index, y=df["MACD"],
		))
		fig1.add_trace(go.Scatter(
		    name="Signal_Line",
		    mode="lines", x=df.index, y=df["Signal_Line"],
		    
		))
		
		# Updating layout
		fig1.update_layout(
		    title={'text':'MACD indicator',
		    'y':0.9,
	       'x':0.5,
	       'xanchor': 'center',
	       'yanchor': 'top'}, 
		    xaxis_title='Date',
		    template='plotly_white',
		    paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)'
		)


		fig1=fig1.update_xaxes(showgrid=True, ticklabelmode="period")
		st.plotly_chart(fig1, use_container_width=True)


	def indicators_plot2(data, categories, crypto, date1, date2):

		df1= data.copy()

		df = df1.loc[date1:date2+timedelta(days = 1)]

	# plot for RSI
		fig2 = go.Figure()
		fig2.add_trace(go.Scatter(
		    name="Buy",
		    mode="markers", x=df.index, y=df["Buy RSI"],
		))
		fig2.add_trace(go.Scatter(
		    name="Sell",
		    mode="markers", x=df.index, y=df["Sell RSI"],
		    
		))
		
		fig2.add_trace(go.Scatter(
		    name="RSI",
		    mode="lines", x=df.index, y=df["RSI"]
		    
		))
		
		# Updating layout
		fig2.update_layout(
		    title={'text':'RSI indicator',
		     'y':0.9,
		       'x':0.5,
		       'xanchor': 'center',
		       'yanchor': 'top'}, 
		    xaxis_title='Date',
		    template='plotly_white',
		    paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h",
																							    yanchor="bottom",
																							    y=1.02,
																							    xanchor="right",
																							    x=1
																							)
		)

	
		fig2=fig2.update_xaxes(showgrid=True, ticklabelmode="period")

		st.plotly_chart(fig2, use_container_width=True)


	def indicators_plot3(data, categories, crypto, date1, date2):

		df1= data.copy()

		df = df1.loc[date1:date2+timedelta(days = 1)]

		# plot for SMA in plotly
		fig = go.Figure()
		
		fig.add_trace(go.Scatter(
		    name="Exit",
		    mode="markers", x=df[df['Entry/Exit'] == -1.0]['Close'].index, y=df[df['Entry/Exit'] == -1.0]['Close'], marker_color='red'
		    
		))
		
		fig.add_trace(go.Scatter(
		    name="Entry",
		    mode="markers", x=df[df['Entry/Exit'] == 1.0]['Close'].index, y=df[df['Entry/Exit'] == 1.0]['Close'], marker_color='green'
		    
		))
		
		fig.add_trace(go.Scatter(
		    name="Close",
		    mode="lines", x=df.index, y=df["Close"], line_color='grey'
		))

		fig.add_trace(go.Scatter(
		    name="SMA50",
		    mode="lines", x=df.index, y=df["SMA50"], line_color='red'
		))

		fig.add_trace(go.Scatter(
		    name="SMA100",
		    mode="lines", x=df.index, y=df["SMA100"], line_color='gold'
		))

		# Updating layout
		fig.update_layout(
			title={'text': "Simple Moving Average indicator for " + str(crypto),
			       'y':0.9,
			       'x':0.5,
			       'xanchor': 'center',
			       'yanchor': 'top'},
		    xaxis_title='Date',
		    yaxis_title='Price',
		    template='plotly_white',
		    paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
		)

		fig=fig.update_xaxes(showgrid=True, ticklabelmode="period")
		st.plotly_chart(fig, use_container_width=True)


	def graph2 (df, crypto, window):
		data = df.copy()
		data['MA' + str(window)] = data.Close.rolling(window = window).mean()

		std = data.Close.rolling(window).std()
		bollinger_up = data['MA' + str(window)] + std * 2 # Calculate top band
		bollinger_down = data['MA' + str(window)] - std * 2 # Calculate bottom band

		data['bollinger_up'] = bollinger_up
		data['bollinger_down'] = bollinger_down

		fig = go.Figure(data = [go.Candlestick(x=data.index,
			                open=data['Open'],
			                high=data['High'],
			                low=data['Low'],
			                close=data['Close'], name = str(crypto) + ' prices'),
						go.Scatter(x=data.index, y=data['bollinger_up'], line=dict(color='orange', width=2), name = "Bollinger Up"),
						go.Scatter(x=data.index, y=data['bollinger_down'], line=dict(color='blue', width=2), name = "Bollinger Down")])

		# Add titles
		fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)', title={'text': "Bollinger Bands for " + str(crypto),
																						       'y':0.9,
																						       'x':0.5,
																						       'xanchor': 'center',
																						       'yanchor': 'top'}, 
																						        yaxis_title='Prices',
																						        xaxis_title = 'Date')

		return st.plotly_chart(fig, use_container_width=True)


	with col24:
		indicators = st.selectbox('Select the indicator', options = ['Moving Average Convergence Divergence', 'RSI', 'Bollinger Bands', 'Moving Average'], index = 0)


	if indicators == 'Moving Average Convergence Divergence':
		with col35:
			indicators_plot(data2, categories, curr, start_date, end_date)

		with col36:
			st.markdown ("<h5 color: 	#000000;'>Important Information: </h5>", unsafe_allow_html=True)
			st.markdown('>Traders may buy the security when the MACD crosses above its signal line and sell—or short—the security when the MACD crosses below the signal line.')

	if indicators == 'RSI':
		with col35:
			indicators_plot2(data2, categories, curr, start_date, end_date)

		with col36:
			st.markdown ("<h5 color: 	#000000;'>Important Information: </h5>", unsafe_allow_html=True)
			st.markdown('> In Relative Strength Index (RSI), values of 70 or above indicate that a security is becoming overbought or overvalued and may be primed for a trend reversal or corrective pullback in price. An RSI reading of 30 or below indicates an oversold or undervalued condition.')

	if indicators == 'Moving Average': 
		with col35: 
			indicators_plot3(data2, categories, curr, start_date, end_date)

		with col36:
			st.markdown ("<h5 color: 	#000000;'>Important Information: </h5>", unsafe_allow_html=True)
			st.markdown('>A death cross occurs when the 50-day SMA crosses below the 100-day SMA. This is considered a bearish signal, indicating that further losses are in store. The golden cross occurs when a short-term SMA breaks above a long-term SMA. Reinforced by high trading volumes, this can signal further gains are in store.')


	if indicators == 'Bollinger Bands':
		with col23:
			window = st.slider('Select the length of the window', 5, 100, 7)
		with col35:
			graph2(data2, curr, window)

		with col36: 
			st.markdown ("<h5 color: 	#000000;'>Important Information: </h5>", unsafe_allow_html=True)
			st.markdown('>Bollinger Bands can be used to determine how strongly an asset is rising and when it is potentially reversing or losing strength. An uptrend that reaches the upper band indicates that the stock is pushing higher and traders can exploit the opportunity to make a buy decision. In a strong downtrend, the price will run along the lower band, and this shows that selling activity remains strong.')



##################################################################STOCKS###############################################################################


if options == "Other Assets":

	with header: 
		st.markdown("<h1 style='text-align: center; color: 	#191970;'>Invest4Some Financial Dashboard</h1>", unsafe_allow_html=True)
		st.markdown("""<hr style="height:10px;border:none;color:#191970;background-color:#191970;" /> """, unsafe_allow_html=True)

	with dataset: 

		st.markdown("<h2 style='text-align: center; color: 	#00000;'>Stocks Analysis</h2>", unsafe_allow_html=True)
	stocks = st.sidebar.text_input("Write the index symbol (E.g.: AMZN for AMAZON) ", "AMZN")
	categories = st.sidebar.selectbox('Select the price that you want to analyse', options = ['Close', 'Low', 'High', 'Open', 'Adj Close'], index = 0)
	start_date = st.sidebar.date_input('Start Date', date(2021,1,1))
	end_date = st.sidebar.date_input('End Date', date.today()-timedelta(days=1))
	st.sidebar.markdown("Source: https://finance.yahoo.com/")

	df_stocks = yf.download(str(stocks),  start="2021-01-01",  end= today)

	def stocks_plot (df, stocks, selection, date1, date2): 
		c = df.loc[date1:date2+timedelta(days = 1)]
		st.subheader("Stock Prices")
		fig = make_subplots(specs=[[{"secondary_y": True}]])

		fig.add_scatter(x=c.index, y=c[str(selection)], secondary_y=False, name = "Price - " + str(selection))

		fig.add_bar(x=c.index, y=c.Volume, secondary_y=True,name = 'Volume')
		    
		fig.update_yaxes(title_text=str(selection), secondary_y=False, rangemode = 'tozero', showgrid= False)
		fig.update_yaxes(title_text="Volume", secondary_y=True, rangemode = 'tozero', showgrid= False)
		fig.update_xaxes(title = 'Date', showgrid =  False)

		fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)', title={'text': 'Distribution of the ' + str(selection) + ' Price and Volume',
		                                                                                       'y':0.9,
		                                                                                       'x':0.5,
		                                                                                       'xanchor': 'center',
		                                                                                       'yanchor': 'top'}, 
		                                                                                       legend=dict(
																							    orientation="h",
																							    yanchor="bottom",
																							    y=1.02,
																							    xanchor="right",
																							    x=1
																							))

		st.plotly_chart(fig, use_container_width=True)


	with col21: 
		real_time_stock = round(stock_info.get_live_price(stocks),2)

		df_var3 = df_stocks.copy()
		a3 = df_var3['Close'].iloc[-2]
		variation3 = round(((real_time_stock - float(a3))/ float(a3))*100,2)

		stocks_plot(df_stocks, stocks,categories,start_date, end_date)

	def stocks_table (df, stocks, selection): 
		st.markdown ("<h5 style='text-align: center; color: #000000;'>Table with prices and volume of the last 7 days (USD)</h5>", unsafe_allow_html=True)
		df1 = round(df[[selection, 'Volume']],4).tail(7)
		df1= pd.DataFrame(df1)
		df1.reset_index(inplace = True)
		df1['Date']= df1['Date'].dt.strftime("%d %B, %Y")
		df1.set_index('Date', inplace = True)
		st.dataframe(df1)

	with col22: 
		st.metric(label="Current Trading Price "f"(% difference from yesterday's closing price)", value=f"{real_time_stock} USD", delta = f"{variation3} %")
		stocks_table(df_stocks, stocks, categories)

	with financial_indicators: 
		st.header("Financial Indicators")


	def indicators (df):

			# Set the short window and long windows
		short_window = 50
		long_window = 100
			# Generate the short and long moving averages (50 and 100 days, respectively)

		df['SMA50'] = df['Close'].rolling(window=short_window).mean()
		df['SMA100'] = df['Close'].rolling(window=long_window).mean()
		df['ShortEMA'] = df['Close'].ewm(span=7, adjust=False).mean()

		df['LongEMA'] = df.Close.ewm(span=26, adjust=False).mean()
		    # Calculate the Moving Average Convergence/Divergence (MACD)
		df['MACD'] = df['ShortEMA'] - df['LongEMA']
		# Calcualte the signal line
		df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

		df['Signal'] = 0.0

		# Generate the trading signal 0 or 1,
		    # where 0 is when the SMA50 is under the SMA100, and
		    # where 1 is when the SMA50 is higher (or crosses over) the SMA100
		df['Signal'][short_window:] = np.where(
		    df['SMA50'][short_window:] > df['SMA100'][short_window:], 1.0, 0.0)
		# Calculate the points in time at which a position should be taken, 1 or -1

		df['Entry/Exit'] = df['Signal'].diff()

		df['Up Move'] = np.nan
		df['Down Move'] = np.nan
		df['Average Up'] = np.nan
		df['Average Down'] = np.nan
		# Relative Strength
		df['RS'] = np.nan
		# Relative Strength Index
		df['RSI'] = np.nan


		for x in range(1, len(df)):
		    df['Up Move'][x] = 0
		    df['Down Move'][x] = 0
		    
		    if df['Adj Close'][x] > df['Adj Close'][x-1]:
		        df['Up Move'][x] = df['Adj Close'][x] - df['Adj Close'][x-1]
		        
		    if df['Adj Close'][x] < df['Adj Close'][x-1]:
		        df['Down Move'][x] = abs(df['Adj Close'][x] - df['Adj Close'][x-1])  
		        
		## Calculate initial Average Up & Down, RS and RSI
		df['Average Up'][14] = df['Up Move'][1:15].mean()
		df['Average Down'][14] = df['Down Move'][1:15].mean()
		df['RS'][14] = df['Average Up'][14] / df['Average Down'][14]
		df['RSI'][14] = 100 - (100/(1+df['RS'][14]))



			## Calculate rest of Average Up, Average Down, RS, RSI
		for x in range(15, len(df)):
		    df['Average Up'][x] = (df['Average Up'][x-1]*13+df['Up Move'][x])/14
		    df['Average Down'][x] = (df['Average Down'][x-1]*13+df['Down Move'][x])/14
		    df['RS'][x] = df['Average Up'][x] / df['Average Down'][x]
		    df['RSI'][x] = 100 - (100/(1+df['RS'][x]))

		## Calculate the buy & sell signals
		## Initialize the columns that we need
		df['Long Tomorrow'] = np.nan
		df['Buy Signal'] = np.nan
		df['Sell Signal'] = np.nan
		df['Buy RSI'] = np.nan
		df['Sell RSI'] = np.nan
		df['Strategy'] = np.nan


			## Calculate the buy & sell signals
		for x in range(15, len(df)):
		    
		    # Calculate "Long Tomorrow" column
		    if ((df['RSI'][x] <= 40) & (df['RSI'][x-1]>40) ):
		        df['Long Tomorrow'][x] = True
		    elif ((df['Long Tomorrow'][x-1] == True) & (df['RSI'][x] <= 70)):
		        df['Long Tomorrow'][x] = True
		    else:
		        df['Long Tomorrow'][x] = False
		        
		    # Calculate "Buy Signal" column
		    if ((df['Long Tomorrow'][x] == True) & (df['Long Tomorrow'][x-1] == False)):
		        df['Buy Signal'][x] = df['Adj Close'][x]
		        df['Buy RSI'][x] = df['RSI'][x]
		        
		    # Calculate "Sell Signal" column
		    if ((df['Long Tomorrow'][x] == False) & (df['Long Tomorrow'][x-1] == True)):
		        df['Sell Signal'][x] = df['Adj Close'][x]
		        df['Sell RSI'][x] = df['RSI'][x]

			## Calculate strategy performance
		df['Strategy'][15] = df['Adj Close'][15]


		for x in range(16, len(df)):
		    if df['Long Tomorrow'][x-1] == True:
		        df['Strategy'][x] = df['Strategy'][x-1]* (df['Adj Close'][x] / df['Adj Close'][x-1])
		    else:
		        df['Strategy'][x] = df['Strategy'][x-1]


		return df

	data3 = indicators(df_stocks)


	def indicators_plot(data, categories, crypto, date1, date2):
		
		df1= data.copy()

		df = df1.loc[date1:date2+timedelta(days = 1)]

			# Visualize exit position relative to close price
		exit = df[df['Entry/Exit'] == -1.0]['Close'].hvplot.scatter(
		    color='red',
		    legend=False,
		    ylabel='Price in $',
		    width=1000,
		    height=400, hover_cols=[str(crypto)]
		)
		# Visualize entry position relative to close price
		entry = df[df['Entry/Exit'] == 1.0]['Close'].hvplot.scatter(
		    color='green',
		    legend=False,
		    ylabel='Price in $',
		    width=1000,
		    height=400, hover_cols=[str(crypto)]
		)
		# Visualize close price for the investment
		security_close = df[['Close']].hvplot(
		    line_color='lightgray',
		    ylabel='Price in $',
		    width=1000,
		    height=400, hover_cols=[str(crypto)]
		)
		# Visualize moving averages
		moving_avgs = df[['SMA50', 'SMA100']].hvplot(
		    ylabel='Price in $',
		    width=1000,
		    height=400,hover_cols=[str(crypto)]
		)
		# Overlay plots for the SMA
		entry_exit_plot = security_close * moving_avgs * entry * exit
		fig=entry_exit_plot.opts(xaxis=None)
		#st.bokeh_chart(hv.render(fig, backend='bokeh'))


		# plot for MACD and Signal line
		fig1 = go.Figure()
		fig1.add_trace(go.Scatter(
		    name="MACD",
		    mode="lines", x=df.index, y=df["MACD"],
		))
		fig1.add_trace(go.Scatter(
		    name="Signal_Line",
		    mode="lines", x=df.index, y=df["Signal_Line"],
		    
		))
		
		# Updating layout
		fig1.update_layout(
		    title={'text':'MACD indicator',
		    'y':0.9,
	       'x':0.5,
	       'xanchor': 'center',
	       'yanchor': 'top'}, 
		    xaxis_title='Date',
		    template='plotly_white',
		    paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)'
		)


		fig1=fig1.update_xaxes(showgrid=True, ticklabelmode="period")
		st.plotly_chart(fig1, use_container_width=True)


	def indicators_plot2(data, categories, crypto, date1, date2):

		df1= data.copy()

		df = df1.loc[date1:date2+timedelta(days = 1)]

	# plot for RSI
		fig2 = go.Figure()
		fig2.add_trace(go.Scatter(
		    name="Buy",
		    mode="markers", x=df.index, y=df["Buy RSI"],
		))
		fig2.add_trace(go.Scatter(
		    name="Sell",
		    mode="markers", x=df.index, y=df["Sell RSI"],
		    
		))
		
		fig2.add_trace(go.Scatter(
		    name="RSI",
		    mode="lines", x=df.index, y=df["RSI"]
		    
		))
		
		# Updating layout
		fig2.update_layout(
		    title={'text':'RSI indicator',
		     'y':0.9,
		       'x':0.5,
		       'xanchor': 'center',
		       'yanchor': 'top'}, 
		    xaxis_title='Date',
		    template='plotly_white',
		    paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',legend=dict(orientation="h",
																							    yanchor="bottom",
																							    y=1.02,
																							    xanchor="right",
																							    x=1
																							)
		)

	
		fig2=fig2.update_xaxes(showgrid=True, ticklabelmode="period")

		st.plotly_chart(fig2, use_container_width=True)


	def indicators_plot3(data, categories, crypto, date1, date2):

		df1= data.copy()

		df = df1.loc[date1:date2+timedelta(days = 1)]

		# plot for SMA in plotly
		fig = go.Figure()
		
		fig.add_trace(go.Scatter(
		    name="Exit",
		    mode="markers", x=df[df['Entry/Exit'] == -1.0]['Close'].index, y=df[df['Entry/Exit'] == -1.0]['Close'], marker_color='red'
		    
		))
		
		fig.add_trace(go.Scatter(
		    name="Entry",
		    mode="markers", x=df[df['Entry/Exit'] == 1.0]['Close'].index, y=df[df['Entry/Exit'] == 1.0]['Close'], marker_color='green'
		    
		))
		
		fig.add_trace(go.Scatter(
		    name="Close",
		    mode="lines", x=df.index, y=df["Close"], line_color='grey'
		))

		fig.add_trace(go.Scatter(
		    name="SMA50",
		    mode="lines", x=df.index, y=df["SMA50"], line_color='red'
		))

		fig.add_trace(go.Scatter(
		    name="SMA100",
		    mode="lines", x=df.index, y=df["SMA100"], line_color='gold'
		))

		# Updating layout
		fig.update_layout(
			title={'text': "Simple Moving Average indicator for " + str(crypto),
			       'y':0.9,
			       'x':0.5,
			       'xanchor': 'center',
			       'yanchor': 'top'},
		    xaxis_title='Date',
		    yaxis_title='Price',
		    template='plotly_white',
		    paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
		)

		fig=fig.update_xaxes(showgrid=True, ticklabelmode="period")
		st.plotly_chart(fig, use_container_width=True)


	def graph2 (df, crypto, window):
		data = df.copy()
		data['MA' + str(window)] = data.Close.rolling(window = window).mean()

		std = data.Close.rolling(window).std()
		bollinger_up = data['MA' + str(window)] + std * 2 # Calculate top band
		bollinger_down = data['MA' + str(window)] - std * 2 # Calculate bottom band

		data['bollinger_up'] = bollinger_up
		data['bollinger_down'] = bollinger_down

		fig = go.Figure(data = [go.Candlestick(x=data.index,
			                open=data['Open'],
			                high=data['High'],
			                low=data['Low'],
			                close=data['Close'], name = str(crypto) + ' prices'),
						go.Scatter(x=data.index, y=data['bollinger_up'], line=dict(color='orange', width=2), name = "Bollinger Up"),
						go.Scatter(x=data.index, y=data['bollinger_down'], line=dict(color='blue', width=2), name = "Bollinger Down")])

		# Add titles
		fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)', title={'text': "Bollinger Bands for " + str(crypto),
																						       'y':0.9,
																						       'x':0.5,
																						       'xanchor': 'center',
																						       'yanchor': 'top'}, 
																						        yaxis_title='Prices',
																						        xaxis_title = 'Date')

		return st.plotly_chart(fig, use_container_width=True)


	with col24:
		indicators = st.selectbox('Select the indicator', options = ['Moving Average Convergence Divergence', 'RSI', 'Bollinger Bands', 'Moving Average'], index = 0)


	if indicators == 'Moving Average Convergence Divergence':
		with col35:
			indicators_plot(data3, categories, stocks, start_date, end_date)

		with col36:
			st.markdown ("<h5 color: 	#000000;'>Important Information: </h5>", unsafe_allow_html=True)
			st.markdown('>Traders may buy the security when the MACD crosses above its signal line and sell—or short—the security when the MACD crosses below the signal line.')

	if indicators == 'RSI':
		with col35:
			indicators_plot2(data3, categories, stocks, start_date, end_date)

		with col36:
			st.markdown ("<h5 color: 	#000000;'>Important Information: </h5>", unsafe_allow_html=True)
			st.markdown('> In Relative Strength Index (RSI), values of 70 or above indicate that a security is becoming overbought or overvalued and may be primed for a trend reversal or corrective pullback in price. An RSI reading of 30 or below indicates an oversold or undervalued condition.')

	if indicators == 'Moving Average': 
		with col35: 
			indicators_plot3(data3, categories, stocks, start_date, end_date)

		with col36:
			st.markdown ("<h5 color: 	#000000;'>Important Information: </h5>", unsafe_allow_html=True)
			st.markdown('>A death cross occurs when the 50-day SMA crosses below the 100-day SMA. This is considered a bearish signal, indicating that further losses are in store. The golden cross occurs when a short-term SMA breaks above a long-term SMA. Reinforced by high trading volumes, this can signal further gains are in store.')


	if indicators == 'Bollinger Bands':
		with col23:
			window = st.slider('Select the length of the window', 5, 100, 7)
		with col35:
			graph2(data3, stocks, window)

		with col36: 
			st.markdown ("<h5 color: 	#000000;'>Important Information: </h5>", unsafe_allow_html=True)
			st.markdown('>Bollinger Bands can be used to determine how strongly an asset is rising and when it is potentially reversing or losing strength. An uptrend that reaches the upper band indicates that the stock is pushing higher and traders can exploit the opportunity to make a buy decision. In a strong downtrend, the price will run along the lower band, and this shows that selling activity remains strong.')


#5TH PART <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


if options == "Exploratory Space":
	st.sidebar.markdown("Source: https://finance.yahoo.com/")

	with header: 
		st.markdown("<h1 style='text-align: center; color: 	#191970;'>Invest4Some Financial Dashboard</h1>", unsafe_allow_html=True)
		st.markdown("""<hr style="height:10px;border:none;color:#191970;background-color:#191970;" /> """, unsafe_allow_html=True)
		st.markdown ("<h5 style='text-align: center; color: 	#000000;'>Welcome to Exploratory Space! In this page, it is possible to make extra analyses, comparing cryptocurrencies, currencies, and stocks. You may also select which chart do you want to analyse and which category of prices. Improve your analysis and extract useful information to make better decisions with this space!</h5>", unsafe_allow_html=True)
		st.markdown("""<hr style="height:10px;border:none;color:#191970;background-color:#191970;" /> """, unsafe_allow_html=True)

	with dataset: 

		selected = option_menu( 
			menu_title = None,
			options = ['Cryptocurrencies', 'Currencies','Stocks'],
			icons = ['cash-coin', 'currency-exchange', 'coin'],
			default_index= 0,		
			orientation = 'horizontal',
			styles={"icon": {"color": "black", "font-size": "25px"},
					"nav-link": {"font-size": "20px" },}		
		)

		if selected == 'Cryptocurrencies':
			

			def currencies_image(crypto):
				#for image in images:
				#	for name
				if crypto == 'BTC':
					imageBTC= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/1.png'))
					st.image(imageBTC)
				elif crypto == 'ETH':
					imageETH= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/1027.png'))
					st.image(imageETH)
				elif crypto == 'USDT':
					imageUSDT= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/825.png'))
					st.image(imageUSDT)
				elif crypto == 'USDC':
					imageUSDC= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/18852.png'))
					st.image(imageUSDC)
				elif crypto == 'BNB':
					imageBNB= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/1839.png'))
					st.image(imageBNB)
				elif crypto == 'XRP':
					imageXRP= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/52.png'))
					st.image(imageXRP)
				elif crypto == 'HEX':
					imageHEX= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/5015.png'))
					st.image(imageHEX)
				elif crypto == 'BUSD':
					imageBUSD= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/4687.png'))
					st.image(imageBUSD)
				elif crypto == 'ADA':
					imageADA= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/2010.png'))
					st.image(imageADA)
				elif crypto == 'SOL':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/16116.png'))
					st.image(image)
				elif crypto == 'DOGE':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/74.png'))
					st.image(image)
				elif crypto == 'DOT':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/6636.png'))
					st.image(image)
				elif crypto == 'WBTC':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/3717.png'))
					st.image(image)
				elif crypto == 'TRX':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/1958.png'))
					st.image(image)
				elif crypto == 'DAI':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/4943.png'))
					st.image(image)
				elif crypto == 'AVAX':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/5805.png'))
					st.image(image)
				elif crypto == 'WETH':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/2396.png'))
					st.image(image)
				elif crypto == 'FLEX':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/5190.png'))
					st.image(image)
				elif crypto == 'STETH':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/8085.png'))
					st.image(image)
				elif crypto == 'SHIB':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/5994.png'))
					st.image(image)
				elif crypto == 'MATIC':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/3890.png'))
					st.image(image)
				elif crypto == 'LEO':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/19469.png'))
					st.image(image)
				elif crypto == 'LTC':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/2.png'))
					st.image(image)
				elif crypto == 'CRO':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/3635.png'))
					st.image(image)
				elif crypto == 'YOUC':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/7321.png'))
					st.image(image)
				elif crypto == 'FTT':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/4195.png'))
					st.image(image)
				elif crypto == 'NEAR':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/6535.png'))
					st.image(image)
				elif crypto == 'BCH':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/1831.png'))
					st.image(image)
				elif crypto == 'XMR':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/328.png'))
					st.image(image)
				elif crypto == 'LINK':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/1975.png'))
					st.image(image)
				elif crypto == 'ETC':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/1321.png'))
					st.image(image)
				elif crypto == 'XLM':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/512.png'))
					st.image(image)
				elif crypto == 'BTCB':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/4023.png'))
					st.image(image)
				elif crypto == 'ATOM':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/3794.png'))
					st.image(image)
				elif crypto == 'WAVES':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/1274.png'))
					st.image(image)	
				elif crypto == 'ALGO':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/4030.png'))
					st.image(image)
				elif crypto == 'FLOW':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/4558.png'))
					st.image(image)
				elif crypto == 'VET':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/3077.png'))
					st.image(image)
				elif crypto == 'HBAR':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/4642.png'))
					st.image(image)
				elif crypto == 'MANA':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/12712.png'))
					st.image(image)
				elif crypto == 'XTZ':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/2011.png'))
					st.image(image)
				elif crypto == 'SAND':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/6210.png'))
					st.image(image)
				elif crypto == 'TRY':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/12225.png'))
					st.image(image)
				elif crypto == 'FIL':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/2280.png'))
					st.image(image)
				elif crypto == 'WBNB':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/7192.png'))
					st.image(image)
				elif crypto == 'EGLD':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/6892.png'))
					st.image(image)
				elif crypto == 'KCS':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/2087.png'))
					st.image(image)
				elif crypto == 'FRAX':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/6952.png'))
					st.image(image)
				elif crypto == 'AAVE':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/7278.png'))
					st.image(image)
				elif crypto == 'ZEC':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/1437.png'))
					st.image(image)
				elif crypto == 'BTT':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/16698.png'))
					st.image(image)
				elif crypto == 'EOS':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/1765.png'))
					st.image(image)
				elif crypto == 'TUSD':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/2563.png'))
					st.image(image)
				elif crypto == 'KLAY':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/4256.png'))
					st.image(image)
				elif crypto == 'MKR':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/1518.png'))
					st.image(image)
				elif crypto == 'THETA':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/2416.png'))
					st.image(image)
				elif crypto == 'HBTC':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/6941.png'))
					st.image(image)
				elif crypto == 'AXS':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/6783.png'))
					st.image(image)
				elif crypto == 'DFI':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/17360.png'))
					st.image(image)
				elif crypto == 'HT':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/2502.png'))
					st.image(image)
				elif crypto == 'BSV':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/3602.png'))
					st.image(image)
				elif crypto == 'USDP':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/8886.png'))
					st.image(image)
				elif crypto == 'FTM':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/3513.png'))
					st.image(image)
				elif crypto == 'XEC':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/10791.png'))
					st.image(image)
				elif crypto == 'RUNE':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/9905.png'))
					st.image(image)
				elif crypto == 'MIOTA':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/1720.png'))
					st.image(image)
				elif crypto == 'HNT':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/5665.png'))
					st.image(image)
				elif crypto == 'USDN':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/5068.png'))
					st.image(image)
				elif crypto == 'QNT':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/3155.png'))
					st.image(image)
				elif crypto == 'CAKE':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/16149.png'))
					st.image(image)
				elif crypto == 'NEO':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/1376.png'))
					st.image(image)
				elif crypto == 'LUSD':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/19714.png'))
					st.image(image)
				elif crypto == 'NEXO':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/2694.png'))
					st.image(image)
				elif crypto == 'OKB':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/3897.png'))
					st.image(image)
				elif crypto == 'STX':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/4847.png'))
					st.image(image)
				elif crypto == 'CHZ':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/4066.png'))
					st.image(image)
				elif crypto == 'LRC':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/1934.png'))
					st.image(image)
				elif crypto == 'LUNA1':
					image= Image.open(urlopen('https://s.yimg.com/uc/fin/img/reports-thumbnails/4172.png'))
					st.image(image)
				elif crypto == 'REP':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/1104.png'))
					st.image(image)
				elif crypto == 'QC':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/2319.png'))
					st.image(image)
				elif crypto == 'PAXG':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/4705.png'))
					st.image(image)
				elif crypto == 'DASH':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/131.png'))
					st.image(image)
				elif crypto == 'ZIL':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/2469.png'))
					st.image(image)
				elif crypto == 'KSM':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/5034.png'))
					st.image(image)
				elif crypto == 'CVX':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/9903.png'))
					st.image(image)
				elif crypto == 'CELO':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/5567.png'))
					st.image(image)
				elif crypto == 'BAT':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/1697.png'))
					st.image(image)
				elif crypto == 'ENJ':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/2130.png'))
					st.image(image)
				elif crypto == 'XDC':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/2634.png'))
					st.image(image)
				elif crypto == 'WEMIX':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/7548.png'))
					st.image(image)
				elif crypto == 'AMP':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/6945.png'))
					st.image(image)
				elif crypto == 'GNO':
					image= Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/1659.png'))
					st.image(image)
				
				


			def graph(crypto1, crypto2, crypto3, crypto4, selection, scale): 
				k = yf.download(str(crypto1) + '-USD', start="2021-01-01",  end= today)
				k2 = yf.download(str(crypto2) + '-USD', start="2021-01-01",  end= today)
				k3 = yf.download(str(crypto3) + '-USD', start="2021-01-01",  end= today)
				k4 = yf.download(str(crypto4) + '-USD', start="2021-01-01",  end= today)
				
				df1 = k[selection]
				df1 = pd.DataFrame(df1)
				df1.rename(columns = { 'Close':str(crypto1), 'Low':str(crypto1), 'High':str(crypto1), 'Open':str(crypto1), 'Adj Close':str(crypto1), 'Volume':str(crypto1)}, inplace = True)
				
				df2 = k2[selection]
				df2 = pd.DataFrame(df2)
				df2.rename(columns = { 'Close':' '+str(crypto2), 'Low':' '+str(crypto2), 'High':' '+str(crypto2), 'Open':' '+str(crypto2), 'Adj Close':' '+str(crypto2), 'Volume':' '+str(crypto2)}, inplace = True)
				kf = pd.concat([df1, df2], axis = 1)
				
				df3 = k3[selection]
				df3 = pd.DataFrame(df3)
				df3.rename(columns = { 'Close':'  '+str(crypto3), 'Low':'  '+str(crypto3), 'High':'  '+str(crypto3), 'Open':'  '+str(crypto3), 'Adj Close':'  '+str(crypto3), 'Volume':'  '+str(crypto3)}, inplace = True)
				kf = pd.concat([kf, df3], axis = 1)
				
				df4 = k4[selection]
				df4 = pd.DataFrame(df4)
				df4.rename(columns = { 'Close':'   '+str(crypto4), 'Low':'   '+str(crypto4), 'High':'   '+str(crypto4), 'Open':'   '+str(crypto4), 'Adj Close':'   '+str(crypto4), 'Volume':'   '+str(crypto4)}, inplace = True)
				kf = pd.concat([kf, df4], axis = 1)
				

				if scale == 'Normal':
					fig = px.line(kf, x=kf.index, y=kf.columns, labels={'variable':'Cryptocurrencies:'})
					fig.update_xaxes(showgrid=False, gridwidth=0.1, gridcolor='lightgrey')
					fig.update_yaxes(rangemode="tozero", showgrid=False, gridwidth=0.1, gridcolor='lightgrey')
					fig.update_layout(title='Variable '+str(selection)+' Over Time', title_x=0.5)
					fig.update_layout(xaxis=dict(rangeselector=dict(
								buttons=[ dict(count=7,
									label="7d",
				                    step="day",
				                    stepmode="backward"),
								dict(count=1,
				                    label="1m",
				                    step="month",
				                    stepmode="backward"),
				                dict(count=6,
				                    label="6m",
				                    step="month",
				                    stepmode="backward"),
				                dict(count=1,
				                    label="YTD",
				                    step="year",
				                    stepmode="todate"),
				                dict(count=1,
				                    label="1y",
				                    step="year",
				                    stepmode="backward"),
				                dict(step="all")]),type="date"), 
							xaxis_rangeslider_visible=True,  xaxis_type="date",
							 plot_bgcolor= 'white')
					fig.update_layout(
						updatemenus=[
						go.layout.Updatemenu(
							dict(buttons=list([
		            		dict(args=["type", "scatter"],
		            			label="Scatter Plot",
		            			method="restyle"),
		            		dict(args=["type", "bar"],
		            			label="Bar Chart",
		            			method="restyle"),
		            		]),direction = 'down')
		            		)
						])
					
					st.plotly_chart(fig, use_container_width=True)

				elif scale == 'Log':
					fig = px.line(kf, x=kf.index, y=kf.columns, labels={'variable':'Cryptocurrencies:'}, log_y=True)
					fig.update_xaxes(showgrid=False, gridwidth=0.1, gridcolor='lightgrey')
					fig.update_yaxes(rangemode="tozero", showgrid=False, gridwidth=0.1, gridcolor='lightgrey')
					fig.update_layout(title='Variable '+str(selection)+' Over Time', title_x=0.5)
					fig.update_layout(xaxis=dict(rangeselector=dict(
								buttons=[ dict(count=7,
									label="7d",
				                    step="day",
				                    stepmode="backward"),
								dict(count=1,
				                    label="1m",
				                    step="month",
				                    stepmode="backward"),
				                dict(count=6,
				                    label="6m",
				                    step="month",
				                    stepmode="backward"),
				                dict(count=1,
				                    label="YTD",
				                    step="year",
				                    stepmode="todate"),
				                dict(count=1,
				                    label="1y",
				                    step="year",
				                    stepmode="backward"),
				                dict(step="all")]),type="date"), 
							xaxis_rangeslider_visible=True,  xaxis_type="date",
							 plot_bgcolor= 'white')
					fig.update_layout(
						updatemenus=[
						go.layout.Updatemenu(
							dict(buttons=list([
		            		dict(args=["type", "scatter"],
		            			label="Scatter Plot",
		            			method="restyle"),
		            		dict(args=["type", "bar"],
		            			label="Bar Chart",
		            			method="restyle"),
		            		]),direction = 'down')
		            		)
						])
					
					st.plotly_chart(fig, use_container_width=True)
				

			
			with col16: 
				dropdown =  st.text_input('Write your 1º cryptocurrency:', 'BTC') 
				dropdown3 =  st.text_input('Write your 3º cryptocurrency:', '') 
				
				
			with col17:
				st.write('')
				st.write('')
				currencies_image(dropdown) 
				st.write('')
				currencies_image(dropdown3)
				
			with col18:
				dropdown2 =  st.text_input('Write your 2º cryptocurrency:', 'ETH')
				dropdown4 =  st.text_input('Write your 4º cryptocurrency:', '')


			with col19:
				st.write('')
				currencies_image(dropdown2)
				st.write('')
				currencies_image(dropdown4)

			with col33:
				catego = st.radio('Categories', options= ['Close', 'Low', 'High', 'Open', 'Adj Close', 'Volume'])

				
			with col20:
				scale = st.radio('Select scale type:', ('Normal', 'Log'))
				
			with explor_1:
				graph(dropdown, dropdown2, dropdown3, dropdown4, catego, scale)


	with currencies_explor:
		if selected == 'Currencies':
			def graph_stock(curr, curr2, curr3, curr4, selection, scale): 
				k = yf.download(str(curr) + "=X",  start="2021-01-01",  end= today)
				k2 = yf.download(str(curr2)+ "=X",  start="2021-01-01",  end= today)
				k3 = yf.download(str(curr3)+ "=X",  start="2021-01-01",  end= today)
				k4 = yf.download(str(curr4)+ "=X",  start="2021-01-01",  end= today)
					
				df1 = k[selection]
				df1 = pd.DataFrame(df1)
				df1.rename(columns = { 'Close':str(curr), 'Low':str(curr), 'High':str(curr), 'Open':str(curr), 'Adj Close':str(curr), 'Volume':str(curr)}, inplace = True)
				
				df2 = k2[selection]
				df2 = pd.DataFrame(df2)
				df2.rename(columns = { 'Close':' '+str(curr2), 'Low':' '+str(curr2), 'High':' '+str(curr2), 'Open':' '+str(curr2), 'Adj Close':' '+str(curr2), 'Volume':' '+str(curr2)}, inplace = True)
				kf = pd.concat([df1, df2], axis = 1)
					
				df3 = k3[selection]
				df3 = pd.DataFrame(df3)
				df3.rename(columns = { 'Close':'  '+str(curr3), 'Low':'  '+str(curr3), 'High':'  '+str(curr3), 'Open':'  '+str(curr3), 'Adj Close':'  '+str(curr3), 'Volume':'  '+str(curr3)}, inplace = True)
				kf = pd.concat([kf, df3], axis = 1)
				
				df4 = k4[selection]
				df4 = pd.DataFrame(df4)
				df4.rename(columns = { 'Close':'   '+str(curr4), 'Low':'   '+str(curr4), 'High':'   '+str(curr4), 'Open':'   '+str(curr4), 'Adj Close':'   '+str(curr4), 'Volume':'   '+str(curr4)}, inplace = True)
				kf = pd.concat([kf, df4], axis = 1)
					

				if scale == 'Normal':
					fig = px.line(kf, x=kf.index, y=kf.columns, labels={'variable':'Cryptocurrencies:'})
					fig.update_xaxes(showgrid=False, gridwidth=0.1, gridcolor='lightgrey')
					fig.update_yaxes(rangemode="tozero", showgrid=False, gridwidth=0.1, gridcolor='lightgrey')
					fig.update_layout(title='Variable '+str(selection)+' Over Time', title_x=0.5)
					fig.update_layout(xaxis=dict(rangeselector=dict(
								buttons=[ dict(count=7,
									label="7d",
				                    step="day",
				                    stepmode="backward"),
								dict(count=1,
				                    label="1m",
				                    step="month",
				                    stepmode="backward"),
				                dict(count=6,
				                    label="6m",
				                    step="month",
				                    stepmode="backward"),
				                dict(count=1,
				                    label="YTD",
				                    step="year",
				                    stepmode="todate"),
				                dict(count=1,
				                    label="1y",
				                    step="year",
				                    stepmode="backward"),
				                dict(step="all")]),type="date"), 
							xaxis_rangeslider_visible=True,  xaxis_type="date",
							 plot_bgcolor= 'white')
					fig.update_layout(
						updatemenus=[
						go.layout.Updatemenu(
							dict(buttons=list([
		            		dict(args=["type", "scatter"],
		            			label="Scatter Plot",
		            			method="restyle"),
		            		dict(args=["type", "bar"],
		            			label="Bar Chart",
		            			method="restyle"),
		            		]),direction = 'down')
		            		)
						])
					
					st.plotly_chart(fig, use_container_width=True)
				elif scale == 'Log':
					fig = px.line(kf, x=kf.index, y=kf.columns, labels={'variable':'Cryptocurrencies:'}, log_y=True)
					fig.update_xaxes(showgrid=False, gridwidth=0.1, gridcolor='lightgrey')
					fig.update_yaxes(rangemode="tozero", showgrid=False, gridwidth=0.1, gridcolor='lightgrey')
					fig.update_layout(title='Variable '+str(selection)+' Over Time', title_x=0.5)
					fig.update_layout(xaxis=dict(rangeselector=dict(
								buttons=[ dict(count=7,
									label="7d",
				                    step="day",
				                    stepmode="backward"),
								dict(count=1,
				                    label="1m",
				                    step="month",
				                    stepmode="backward"),
				                dict(count=6,
				                    label="6m",
				                    step="month",
				                    stepmode="backward"),
				                dict(count=1,
				                    label="YTD",
				                    step="year",
				                    stepmode="todate"),
				                dict(count=1,
				                    label="1y",
				                    step="year",
				                    stepmode="backward"),
				                dict(step="all")]),type="date"), 
							xaxis_rangeslider_visible=True,  xaxis_type="date",
							 plot_bgcolor= 'white')
					fig.update_layout(
						updatemenus=[
						go.layout.Updatemenu(
							dict(buttons=list([
		            		dict(args=["type", "scatter"],
		            			label="Scatter Plot",
		            			method="restyle"),
		            		dict(args=["type", "bar"],
		            			label="Bar Chart",
		            			method="restyle"),
		            		]),direction = 'down')
		            		)
						])
					
					st.plotly_chart(fig, use_container_width=True)
			
			with col44:
				st.write('')
			
			with col45:
				curr = st.text_input("Write the 1º currency symbol (E.g. EURUSD): ", "EURUSD") 
				curr3 = st.text_input("Write the 3º currency symbol: ", "")

			with col46:
				curr2 = st.text_input("Write the 2º currency symbol: ", "")
				curr4 = st.text_input("Write the 4º currency symbol: ", "")

			with col48:
				catego3 = st.radio('Categories', options= ['Close', 'Low', 'High', 'Open', 'Adj Close'])

			with col49:
				scale3 = st.radio('Select scale type:', ('Normal', 'Log'))

			with currencies_2:
				graph_stock(curr, curr2, curr3, curr4, catego3, scale3)



	with stocks:

		if selected == 'Stocks':


			def graph_stock(stocks, stocks2, stocks3, stocks4, selection, scale): 
				k = yf.download(str(stocks) ,  start="2021-01-01",  end= today)
				k2 = yf.download(str(stocks2),  start="2021-01-01",  end= today)
				k3 = yf.download(str(stocks3),  start="2021-01-01",  end= today)
				k4 = yf.download(str(stocks4),  start="2021-01-01",  end= today)
					
				df1 = k[selection]
				df1 = pd.DataFrame(df1)
				df1.rename(columns = { 'Close':str(stocks), 'Low':str(stocks), 'High':str(stocks), 'Open':str(stocks), 'Adj Close':str(stocks), 'Volume':str(stocks)}, inplace = True)
				
				df2 = k2[selection]
				df2 = pd.DataFrame(df2)
				df2.rename(columns = { 'Close':' '+str(stocks2), 'Low':' '+str(stocks2), 'High':' '+str(stocks2), 'Open':' '+str(stocks2), 'Adj Close':' '+str(stocks2), 'Volume':' '+str(stocks2)}, inplace = True)
				kf = pd.concat([df1, df2], axis = 1)
					
				df3 = k3[selection]
				df3 = pd.DataFrame(df3)
				df3.rename(columns = { 'Close':'  '+str(stocks3), 'Low':'  '+str(stocks3), 'High':'  '+str(stocks3), 'Open':'  '+str(stocks3), 'Adj Close':'  '+str(stocks3), 'Volume':'  '+str(stocks3)}, inplace = True)
				kf = pd.concat([kf, df3], axis = 1)
				
				df4 = k4[selection]
				df4 = pd.DataFrame(df4)
				df4.rename(columns = { 'Close':'   '+str(stocks4), 'Low':'   '+str(stocks4), 'High':'   '+str(stocks4), 'Open':'   '+str(stocks4), 'Adj Close':'   '+str(stocks4), 'Volume':'   '+str(stocks4)}, inplace = True)
				kf = pd.concat([kf, df4], axis = 1)
					

				if scale == 'Normal':
					fig = px.line(kf, x=kf.index, y=kf.columns, labels={'variable':'Cryptocurrencies:'})
					fig.update_xaxes(showgrid=False, gridwidth=0.1, gridcolor='lightgrey')
					fig.update_yaxes(rangemode="tozero", showgrid=False, gridwidth=0.1, gridcolor='lightgrey')
					fig.update_layout(title='Variable '+str(selection)+' Over Time', title_x=0.5)
					fig.update_layout(xaxis=dict(rangeselector=dict(
								buttons=[ dict(count=7,
									label="7d",
				                    step="day",
				                    stepmode="backward"),
								dict(count=1,
				                    label="1m",
				                    step="month",
				                    stepmode="backward"),
				                dict(count=6,
				                    label="6m",
				                    step="month",
				                    stepmode="backward"),
				                dict(count=1,
				                    label="YTD",
				                    step="year",
				                    stepmode="todate"),
				                dict(count=1,
				                    label="1y",
				                    step="year",
				                    stepmode="backward"),
				                dict(step="all")]),type="date"), 
							xaxis_rangeslider_visible=True,  xaxis_type="date",
							 plot_bgcolor= 'white')
					fig.update_layout(
						updatemenus=[
						go.layout.Updatemenu(
							dict(buttons=list([
		            		dict(args=["type", "scatter"],
		            			label="Scatter Plot",
		            			method="restyle"),
		            		dict(args=["type", "bar"],
		            			label="Bar Chart",
		            			method="restyle"),
		            		]),direction = 'down')
		            		)
						])
					
					st.plotly_chart(fig, use_container_width=True)
				elif scale == 'Log':
					fig = px.line(kf, x=kf.index, y=kf.columns, labels={'variable':'Cryptocurrencies:'}, log_y=True)
					fig.update_xaxes(showgrid=False, gridwidth=0.1, gridcolor='lightgrey')
					fig.update_yaxes(rangemode="tozero", showgrid=False, gridwidth=0.1, gridcolor='lightgrey')
					fig.update_layout(title='Variable '+str(selection)+' Over Time', title_x=0.5)
					fig.update_layout(xaxis=dict(rangeselector=dict(
								buttons=[ dict(count=7,
									label="7d",
				                    step="day",
				                    stepmode="backward"),
								dict(count=1,
				                    label="1m",
				                    step="month",
				                    stepmode="backward"),
				                dict(count=6,
				                    label="6m",
				                    step="month",
				                    stepmode="backward"),
				                dict(count=1,
				                    label="YTD",
				                    step="year",
				                    stepmode="todate"),
				                dict(count=1,
				                    label="1y",
				                    step="year",
				                    stepmode="backward"),
				                dict(step="all")]),type="date"), 
							xaxis_rangeslider_visible=True,  xaxis_type="date",
							 plot_bgcolor= 'white')
					fig.update_layout(
						updatemenus=[
						go.layout.Updatemenu(
							dict(buttons=list([
		            		dict(args=["type", "scatter"],
		            			label="Scatter Plot",
		            			method="restyle"),
		            		dict(args=["type", "bar"],
		            			label="Bar Chart",
		            			method="restyle"),
		            		]),direction = 'down')
		            		)
						])
					
					st.plotly_chart(fig, use_container_width=True)
			
			with col82:
				st.write('')
			
			with col43:
				stocks = st.text_input("Write the 1º index symbol (E.g.: AMZN for AMAZON): ", "AMZN") 
				stocks3 = st.text_input("Write the 3º index symbol: ", "Select")

			with col37:
				stocks2 = st.text_input("Write the 2º index symbol: ", "Select")
				stocks4 = st.text_input("Write the 4º index symbol: ", "Select")

			with col39:
				catego2 = st.radio('Categories', options= ['Close', 'Low', 'High', 'Open', 'Adj Close', 'Volume'])

			with col41:
				scale2 = st.radio('Select scale type:', ('Normal', 'Log'))

			with stocks_2:
				graph_stock(stocks, stocks2, stocks3, stocks4, catego2, scale2)

	
