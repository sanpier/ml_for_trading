import cufflinks as cf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema
cf.set_config_file(offline=True)

"""
# Indicators & Strategies

+ SMA: simple moving average
+ EMA: exponential moving average
+ SMA & EMA: together
+ MACD: moving average convergence & divergence
+ RSI: relative strength index
+ SO: stochastic oscillator
+ SOA: stochastic oscillator alternative
+ BB: bollinger bands
+ PP: pivot point
+ FR: fibonacci retracement
+ ROC: rate of change
+ WR: williams %r
+ CMF: chaikin money flow
+ OBV: on balance volume
+ ATR: average true range
"""

def sma_strategy(df_original, s, l):
    """ SMA: simple moving average """
    # copy data
    df = df_original.copy()
    # set short and long simple moving averages based on strategy
    df[f'SMA_Close_S'] = df["Close"].rolling(window=s).mean()
    df[f'SMA_Close_L'] = df["Close"].rolling(window=l).mean()
    # set position based on two periods
    df["SMA_strategy"] = np.where(df["SMA_Close_S"] > df["SMA_Close_L"], 1, -1)
    buy, sell = decide_buy_sell(df, "SMA")
    return df, buy, sell

def ema_strategy(df_original, s, l):
    """ EMA: exponential moving average """
    # copy data
    df = df_original.copy()
    # set short and long simple moving averages based on strategy
    df['EMA_Close_S'] = df["Close"].ewm(span=s, min_periods=s).mean()
    df['EMA_Close_L'] = df["Close"].ewm(span=l, min_periods=l).mean()
    # set position based on two periods
    df["EMA_strategy"] = np.where(df["EMA_Close_S"] > df["EMA_Close_L"], 1, -1)
    buy, sell = decide_buy_sell(df, "EMA")
    return df, buy, sell

def smaema_strategy(df_original, s):
    """ SMA & EMA: together """
    # copy data
    df = df_original.copy()
    # set short and long simple moving averages based on strategy
    df['SMAEMA_Close_S'] = df["Close"].ewm(span=s, min_periods=s).mean()
    df['SMAEMA_Close_L'] = df["Close"].rolling(window=s).mean()
    # set position based on two periods
    df["SMAEMA_strategy"] = np.where(df["SMAEMA_Close_S"] > df["SMAEMA_Close_L"], 1, -1)
    buy, sell = decide_buy_sell(df, "SMAEMA")
    return df, buy, sell

def macd_strategy(df_original, s, l, t):
    """ MACD: moving average convergence & divergence """
    # copy data
    df = df_original.copy()
    # set short and long simple moving averages based on strategy
    ema_s = df["Close"].ewm(span=s, min_periods=s).mean()
    ema_l = df["Close"].ewm(span=l, min_periods=l).mean()
    df['MACD_Close_S'] = ema_s - ema_l
    df['MACD_Close_L'] = df['MACD_Close_S'].ewm(span=t, min_periods=t).mean()
    # set position based on two periods
    df["MACD_strategy"] = np.where(df["MACD_Close_S"] > df["MACD_Close_L"], 1, -1)
    buy, sell = decide_buy_sell(df, "MACD")
    return df, buy, sell

def rsi_strategy(df_original, roll, threshold):
    """ RSI: relative strength index """
    # copy data
    df = df_original.copy()
    # set short and long simple moving averages based on strategy
    df['up'] = np.where(df["Close"].diff() > 0, df["Close"].diff(), 0)
    df['down'] = np.where(df["Close"].diff() < 0, -df["Close"].diff(), 0)
    df['SMA_up'] = df['up'].rolling(window=roll).mean()
    df['SMA_down'] = df['down'].rolling(window=roll).mean()
    df['RSI'] = df['SMA_up'] / (df['SMA_up'] + df['SMA_down']) * 100
    df.drop(columns=['up', 'down'], inplace=True)
    # set position based on rsi
    df["RSI_strategy"] = np.where(df["RSI"] > 100-threshold, -1, 0)
    df["RSI_strategy"] = np.where(df["RSI"] < threshold, 1, df["RSI_strategy"])
    buy, sell = decide_buy_sell(df, "RSI")
    return df, buy, sell

def stochastic_oscillator(df_original, roll, t):
    """ SO: stochastic oscillator """
    # copy data
    df = df_original.copy()
    # set short and long simple moving averages based on strategy
    roll_low = df["Low"].rolling(window=roll).min()
    roll_high = df["High"].rolling(window=roll).max()
    df['SO_Close_S'] = (df["Close"] - roll_low) / (roll_high - roll_low) * 100 # %K line = Fast Stochastic Indicator
    df['SO_Close_L'] = df['SO_Close_S'].rolling(window=t).mean() # %D line = Slow Stochastic Indicator
    # set position based on two periods
    df["SO_strategy"] = np.where(df["SO_Close_S"] > df["SO_Close_L"], 1, -1)
    buy, sell = decide_buy_sell(df, "SO")
    return df, buy, sell

def stochastic_oscillator_alternative(df_original, roll, threshold):
    """ SOA: stochastic oscillator alternative """
    # copy data
    df = df_original.copy()
    # set short and long simple moving averages based on strategy
    roll_low = df["Low"].rolling(window=roll).min()
    roll_high = df["High"].rolling(window=roll).max()
    df['K'] = (df["Close"] - roll_low) / (roll_high - roll_low) * 100 # %K line = Fast Stochastic Indicator    
    # set position based on so alternative 1
    df["SOA_strategy"] = np.where(df["K"] > 100-threshold, -1, 0)
    df["SOA_strategy"] = np.where(df["K"] < threshold, 1, df["SOA_strategy"])
    buy, sell = decide_buy_sell(df, "SOA")
    return df, buy, sell

def bollinger_bands(df_original, roll, std):
    """ BB: bollinger bands """
    # copy data
    df = df_original.copy()
    # set upper / lower bollinger bands
    df['SMA_Close'] = df["Close"].rolling(window=roll).mean()
    df['Upper_Band'] = df['SMA_Close'] + std*df["Close"].rolling(window=roll).std() # std e.g. 1.96
    df['Lower_Band'] = df['SMA_Close'] - std*df["Close"].rolling(window=roll).std()
    # set position by bollinger bands
    df["BB_strategy"] = np.where(df["Close"] < df["Lower_Band"], 1, 0)
    df["BB_strategy"] = np.where(df["Close"] > df["Upper_Band"], -1, df["BB_strategy"])
    buy, sell = decide_buy_sell(df, "BB")
    return df, buy, sell

def pivot_point(df_original, resistance_col):
    """ PP: pivot point """
    # copy data
    df = df_original.copy()
    if type(df.index) != pd.core.indexes.datetimes.DatetimeIndex:
        df.index = pd.to_datetime(df.index)
    # previous day: high / low / close
    previous_day = df.shift(1, freq='B').resample('D').agg({'High': 'max', 'Low': 'min', 'Open': 'first', 'Close': 'last'})
    previous_day.columns = ["High_d", "Low_d", "Open_d", "Close_d"]
    previous_day.dropna(inplace=True)
    # set day for merging
    df["day_date"] = df.index.floor('D')
    previous_day["day_date"] = previous_day.index
    # merge into data
    indexes = df.index.copy()
    df = df.merge(previous_day, on="day_date", how="left")
    df.index = indexes
    df.drop(columns=["day_date"], inplace=True)
    df.dropna(inplace=True)
    # calculate pivot point / resistance and support lines
    df["PP"] = (df["High_d"] + df["Low_d"] + df["Close_d"]) / 3
    df["S1"] = df["PP"] * 2 - df["High_d"]
    df["S2"] = df["PP"] - (df["High_d"] - df["Low_d"])
    df["R1"] = df["PP"] * 2 - df["Low_d"]
    df["R2"] = df["PP"] + (df["High_d"] - df["Low_d"])
    # set position by pivot point and resistance lines
    df["PP_strategy"] = np.where((df["Open"] > df["PP"]) & (df["Open"].shift(1) <= df["PP"].shift(1)), 1, 0)
    df["PP_strategy"] = np.where((df["Open"] <= df["PP"]) & (df["Open"].shift(1) > df["PP"].shift(1)), -1, df["PP_strategy"])
    df["PP_strategy"] = np.where(df["Open"] > df[resistance_col], -1, df["PP_strategy"]) # resistance_col == "R1" | "R2"
    buy, sell = decide_buy_sell(df, "PP")
    return df, buy, sell

def fibonacci_retracement(df_original, order):
    """ FR: fibonacci retracement """
    # copy data
    df = df_original.copy()
    # create relative columns
    df["local_high"] = np.nan
    df["local_high_date"] = np.nan
    df["local_low"] = np.nan
    df["local_low_date"] = np.nan
    # iterate over data chronologically
    for i in range(len(df)):
        # current date
        date = df.index[i]
        # determine local max
        high = df.iloc[:i+1].High
        local_max = argrelextrema(high.values, np.greater_equal, order=order)[0] # 420 ~ 3 months
        df.loc[date, "local_high"] = df.High.values[local_max][-1]
        df.loc[date, "local_high_date"] = df.index[local_max][-1]
        # determine local min
        low = df.iloc[:i+1].Low
        local_min = argrelextrema(low.values, np.less_equal, order=order)[0] # 420 ~ 3 months
        df.loc[date, "local_low"] = df.Low.values[local_min][-1]
        df.loc[date, "local_low_date"] = df.index[local_min][-1]
    # set trend
    df["trend"] = np.where(df["local_high_date"] > df["local_low_date"], "Up", "Down")
    df.drop(columns=["local_high_date", "local_low_date"], inplace=True)
    # set retracement levels
    df["R1_23.6"] = np.where(df.trend == "Up", df.local_high - (df.local_high - df.local_low) * 0.236, df.local_high - (df.local_high - df.local_low) * (1 - 0.236))
    df["R2_38.2"] = np.where(df.trend == "Up", df.local_high - (df.local_high - df.local_low) * 0.382, df.local_high - (df.local_high - df.local_low) * (1 - 0.382))
    df["R3_61.8"] = np.where(df.trend == "Up", df.local_high - (df.local_high - df.local_low) * 0.618, df.local_high - (df.local_high - df.local_low) * (1 - 0.618))
    # set position
    df["FR_strategy"] = np.where(df["trend"].shift(1) != df["trend"], -1, np.nan)
    df["FR_strategy"] = np.where((df["trend"] == "Down") & (df["Close"].shift(1) < df["R1_23.6"].shift(1)) & (df["Close"] >= df["R1_23.6"]), 1, df["FR_strategy"])
    df["FR_strategy"] = np.where((df["trend"] == "Down") & (df["Close"].shift(1) < df["R2_38.2"].shift(1)) & (df["Close"] >= df["R2_38.2"]), -1, df["FR_strategy"])
    df["FR_strategy"] = np.where((df["trend"] == "Down") & (df["Close"].shift(1) > df["local_low"].shift(1)) & (df["Close"] <= df["local_low"]), -1, df["FR_strategy"])
    df["FR_strategy"] = np.where((df["trend"] == "Up") & (df["Close"].shift(1) < df["R2_38.2"].shift(1)) & (df["Close"] >= df["R2_38.2"]), 1, df["FR_strategy"])
    df["FR_strategy"] = np.where((df["trend"] == "Up") & (df["Close"].shift(1) < df["R1_23.6"].shift(1)) & (df["Close"] >= df["R1_23.6"]), -1, df["FR_strategy"])
    df["FR_strategy"] = np.where((df["trend"] == "Up") & (df["Close"].shift(1) > df["R3_61.8"].shift(1)) & (df["Close"] <= df["R3_61.8"]), -1, df["FR_strategy"])
    df["FR_strategy"] = df["FR_strategy"].fillna(0)
    buy, sell = decide_buy_sell(df, "FR")
    return df, buy, sell

def rate_of_change(df_original, n):
    """ ROC: rate of change """
    # copy data
    df = df_original.copy()
    # set rate of change
    df[f"ROC_{n}"] = (df["Close"] / df["Close"].shift(n)) - 1
    # set position by rate of change
    df[f"ROC_{n}_strategy"] = np.where((df[f"ROC_{n}"] < 0) & (df[f"ROC_{n}"].shift(1) >= 0), 1, 0)
    df[f"ROC_{n}_strategy"] = np.where((df[f"ROC_{n}"] > 0) & (df[f"ROC_{n}"].shift(1) <= 0), -1, df[f"ROC_{n}_strategy"])
    buy, sell = decide_buy_sell(df, f"ROC_{n}")
    return df, buy, sell

def williams_r(df_original, n):
    """ WR: williams %r """
    # copy data
    df = df_original.copy()
    # set rate of change
    hh = df["High"].rolling(window=n).max()
    ll = df["Low"].rolling(window=n).min()
    df[f"WR_{n}"] = ((hh - df["Close"]) / (hh - ll)) * -100
    # set position by williams r strategy
    df[f"WR_{n}_strategy"] = np.where((df[f"WR_{n}"] < -80) & (df[f"WR_{n}"].shift(1) >= -80), 1, 0)
    df[f"WR_{n}_strategy"] = np.where((df[f"WR_{n}"] > -20) & (df[f"WR_{n}"].shift(1) <= -20), -1, df[f"WR_{n}_strategy"])
    buy, sell = decide_buy_sell(df, f"WR_{n}")
    return df, buy, sell

def chaikin_money_flow(df_original, n, signal):
    """ CMF: chaikin money flow """
    # copy data
    df = df_original.copy()
    # set money flow volume
    df["MF_multiplier"] = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"])
    df["MF_volume"] = (df["Volume"] * df["MF_multiplier"]).fillna(0)
    df["CMF"] = df["MF_volume"].rolling(window=n).sum() / df["Volume"].rolling(window=n).sum()
    # set position by chaikin money flow
    df["CMF_strategy"] = np.where((df["CMF"].shift(1) < signal) & (df["CMF"] >= signal), 1, 0)
    df["CMF_strategy"] = np.where((df["CMF"].shift(1) > -signal) & (df["CMF"] <= -signal), -1, df["CMF_strategy"])
    buy, sell = decide_buy_sell(df, "CMF")
    return df, buy, sell

def on_balance_volume(df_original):
    """ OBV: on-balance volume """
    # copy data
    df = df_original.copy()
    df["OBV"] = 0
    # iterate over data chronologically and calculate obv
    for i in range(1, len(df)):
        date = df.index[i]
        if df.iloc[i].Close > df.iloc[i-1].Close:
            df.loc[date, "OBV"] = df.iloc[i-1].OBV + df.iloc[i].Volume
        elif df.iloc[i].Close < df.iloc[i-1].Close:
            df.loc[date, "OBV"] = df.iloc[i-1].OBV - df.iloc[i].Volume
        else:
            df.loc[date, "OBV"] = df.iloc[i-1].OBV
    return df

def average_true_range(df_original, n):
    """ ATR: average true range """
    # copy data
    df = df_original.copy()
    # calculate true range
    df[f"TR_{n}"] = np.max(np.array([(df.High - df.Low), abs(df.High - df.Close.shift(1)), abs(df.Low - df.Close.shift(1))]), axis=0)
    df[f"ATR_{n}"] = 0
    # iterate over data chronologically and calculate atr
    for i in range(1, len(df)):
        date = df.index[i]
        df.loc[date, f"ATR_{n}"] = (df.iloc[i-1][f"ATR_{n}"] * (n-1) + df.iloc[i][f"TR_{n}"]) / n
    return df

def decide_buy_sell(df, strategy):
    """ find buy sell strategies given the strategy """
    # find buy & sell dates
    buy_mask = (df[f'{strategy}_strategy'].shift() != 1) & (df[f'{strategy}_strategy'] == 1)
    sell_mask = (df[f'{strategy}_strategy'].shift() != -1) & (df[f'{strategy}_strategy'] == -1)
    dates_of_buy = sorted(df.index[buy_mask])
    dates_of_sell = df.index[sell_mask]
    # set dates   
    buy, sell = [], []
    if (len(dates_of_buy) > 0) & (len(dates_of_sell) > 0):
        dates_of_sell = sorted([t for t in dates_of_sell if t > dates_of_buy[0]])
        if len(dates_of_sell) > 0:
            for buy_i in dates_of_buy:
                # find buy
                if len(sell) == 0:
                    buy.append(buy_i)
                    sell.append(dates_of_sell[0])
                elif buy_i > sell[-1]:
                    buy.append(buy_i)
                    # find corresponding sell
                    applicable_sells = [t for t in dates_of_sell if t > buy_i]
                    if len(applicable_sells) > 0:
                        sell.append(applicable_sells[0])
                    else:
                        break
    return buy, sell

def date_difference(df1, df2):
    return (pd.Series(sorted(list(set(df1.index).difference(set(df2.index))))).dt.strftime('%Y-%m-%d')).unique()

def check_day_data(df, day):
    return df[df.index.date == pd.to_datetime(day).date()]

def plot_generic(df, col1, col2, dates_of_buy, dates_of_sell):
    """# plot buy / sell dates
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

    # Add traces for short long
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode='lines', name="Close"))
    fig.add_trace(go.Scatter(x=df.index, y=df[col1], mode='lines', name=col1))
    fig.add_trace(go.Scatter(x=df.index, y=df[col2], mode='lines', name=col2))

    # Add vertical lines at specified dates
    for date in dates_of_buy:
        fig.add_shape(type="line", x0=date, x1=date, y0=0, y1=1, xref="x", yref="paper", line=dict(color="green", dash="dash"))
    for date in dates_of_sell:
        fig.add_shape(type="line", x0=date, x1=date, y0=0, y1=1, xref="x", yref="paper", line=dict(color="red", dash="dash"))

    # Update the layout and display the plot
    fig.update_layout(title_text=f"Stock Prices & Strategy", xaxis_title="Date", width=1200, height=600)
    fig.show()"""

    # Create subplots with 2 y-axes
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, specs=[[{'secondary_y': True}]])

    # Add traces for Close on the left y-axis
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode='lines', name="Close"))

    # Add traces for col1 and col2 on the right y-axis
    fig.add_trace(go.Scatter(x=df.index, y=df[col1], mode='lines', name=col1, yaxis='y2'))
    fig.add_trace(go.Scatter(x=df.index, y=df[col2], mode='lines', name=col2, yaxis='y2'))

    # Add vertical lines at specified dates
    for date in dates_of_buy:
        fig.add_shape(type="line", x0=date, x1=date, y0=0, y1=1, xref="x", yref="paper", line=dict(color="green", dash="dash"))
    for date in dates_of_sell:
        fig.add_shape(type="line", x0=date, x1=date, y0=0, y1=1, xref="x", yref="paper", line=dict(color="red", dash="dash"))

    # Update the layout and display the plot
    fig.update_layout(
        title_text=f"Stock Prices & Strategy",
        xaxis_title="Date",
        xaxis_rangeslider_visible=True,  # Add a range slider for zooming
        width=1200,
        height=600
    )

    # Set the y-axis titles
    fig.update_yaxes(title_text="Close", range=[df["Close"].min(), df["Close"].max()], row=1, col=1)
    fig.update_yaxes(title_text=col1, range=[df[col1].min(), df[col1].max()], secondary_y=True, row=1, col=1)
    fig.update_yaxes(title_text=col2, range=[df[col2].min(), df[col2].max()], secondary_y=True, row=1, col=1)

    fig.show()

def plot_strategy(df, dates_of_buy, dates_of_sell):
    # find strategy    
    strategy = [i for i in df.columns if "_Close_S" in i][0].split("_Close")[0]

    # plot buy / sell dates
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

    # Add traces for short long
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode='lines', name="Close"))
    fig.add_trace(go.Scatter(x=df.index, y=df[f"{strategy}_Close_S"], mode='lines', name=f'{strategy}_Close_S'))
    fig.add_trace(go.Scatter(x=df.index, y=df[f"{strategy}_Close_L"], mode='lines', name=f'{strategy}_Close_L'))

    # Add vertical lines at specified dates
    for date in dates_of_buy:
        fig.add_shape(type="line", x0=date, x1=date, y0=0, y1=1, xref="x", yref="paper", line=dict(color="green", dash="dash"))
    for date in dates_of_sell:
        fig.add_shape(type="line", x0=date, x1=date, y0=0, y1=1, xref="x", yref="paper", line=dict(color="red", dash="dash"))

    # Update the layout and display the plot
    fig.update_layout(title_text=f"Stock Prices & {strategy}", xaxis_title="Date", width=1200, height=600)
    fig.show()

def plot_with_threshold(df, col, dates_of_buy, dates_of_sell, threshold):
    # plot buy / sell dates
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

    # Add traces for short long
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode='lines', name="Close"))
    fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=col))
    fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=threshold, y1=threshold, line=dict(color="green"))
    fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=100-threshold, y1=100-threshold, line=dict(color="red"))

    # Add vertical lines at specified dates
    for date in dates_of_buy:
        fig.add_shape(type="line", x0=date, x1=date, y0=0, y1=1, xref="x", yref="paper", line=dict(color="green", dash="dash"))
    for date in dates_of_sell:
        fig.add_shape(type="line", x0=date, x1=date, y0=0, y1=1, xref="x", yref="paper", line=dict(color="red", dash="dash"))

    # Update the layout and display the plot
    fig.update_layout(title_text=f"Stock Prices & {col}", xaxis_title="Date", width=1200, height=600)
    fig.show()

def plot_bollinger_bands(df, dates_of_buy, dates_of_sell):
    # plot buy / sell dates
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

    # Add traces for short long
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode='lines', name="Close"))
    fig.add_trace(go.Scatter(x=df.index, y= df['Upper_Band'],line=dict(color='red', width=1.5), name = 'Upper Band (Sell)'))
    fig.add_trace(go.Scatter(x=df.index, y= df['Lower_Band'],line=dict(color='green', width=1.5), name = 'Lower Band (Buy)'))

    # Add vertical lines at specified dates
    for date in dates_of_buy:
        fig.add_shape(type="line", x0=date, x1=date, y0=0, y1=1, xref="x", yref="paper", line=dict(color="green", dash="dash"))
    for date in dates_of_sell:
        fig.add_shape(type="line", x0=date, x1=date, y0=0, y1=1, xref="x", yref="paper", line=dict(color="red", dash="dash"))

    # Update the layout and display the plot
    fig.update_layout(title_text=f"Stock Prices & Bollinger Bands", xaxis_title="Date", width=1200, height=600)
    fig.show()

def plot_pivot_point(df, dates_of_buy, dates_of_sell):
    # plot buy / sell dates
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
    
    # Add traces for short long
    fig.add_trace(go.Scatter(x=df.index, y=df["Open"], mode='lines', name="Open"))
    fig.add_trace(go.Scatter(x=df.index, y=df["PP"], mode='lines', name="Pivot Line"))
    fig.add_trace(go.Scatter(x=df.index, y=df["S1"], mode='lines', name="Support 1"))
    fig.add_trace(go.Scatter(x=df.index, y=df["S2"], mode='lines', name="Support 2"))
    fig.add_trace(go.Scatter(x=df.index, y=df["R1"], mode='lines', name="Resistance 1"))
    fig.add_trace(go.Scatter(x=df.index, y=df["R2"], mode='lines', name="Resistance 2"))

    # Add vertical lines at specified dates
    for date in dates_of_buy:
        fig.add_shape(type="line", x0=date, x1=date, y0=0, y1=1, xref="x", yref="paper", line=dict(color="green", dash="dash"))
    for date in dates_of_sell:
        fig.add_shape(type="line", x0=date, x1=date, y0=0, y1=1, xref="x", yref="paper", line=dict(color="red", dash="dash"))

    # Update the layout and display the plot
    fig.update_layout(title_text=f"Stock Prices & Pivot Strategy", xaxis_title="Date", width=1200, height=600)
    fig.show()

def plot_fibonacci_retracement(df, dates_of_buy, dates_of_sell):
    # plot buy / sell dates
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
    
    # Add traces for short long
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode='lines', name="Close"))
    fig.add_trace(go.Scatter(x=df.index, y=df["local_low"], mode='lines', text=df['trend'], textposition='top right', name="Local Low"))
    fig.add_trace(go.Scatter(x=df.index, y=df["local_high"], mode='lines', text=df['trend'], textposition='top right',  name="Local High"))
    fig.add_trace(go.Scatter(x=df.index, y=df["R1_23.6"], mode='lines', name="R1_23.6"))
    fig.add_trace(go.Scatter(x=df.index, y=df["R2_38.2"], mode='lines', name="R2_38.2"))
    fig.add_trace(go.Scatter(x=df.index, y=df["R3_61.8"], mode='lines', name="R3_61.8"))

    # Add vertical lines at specified dates
    for date in dates_of_buy:
        fig.add_shape(type="line", x0=date, x1=date, y0=0, y1=1, xref="x", yref="paper", line=dict(color="green", dash="dash"))
    for date in dates_of_sell:
        fig.add_shape(type="line", x0=date, x1=date, y0=0, y1=1, xref="x", yref="paper", line=dict(color="red", dash="dash"))

    # Update the layout and display the plot
    fig.update_layout(title_text=f"Stock Prices & Fibonacci Retracement Strategy", xaxis_title="Date", width=1200, height=600)
    fig.show()

def plot_rate_of_change(df, dates_of_buy, dates_of_sell):
    # plot buy / sell dates
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, specs=[[{"secondary_y": True}]])

    # Add traces for short long
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode='lines', name="Close"), secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index, y=df["ROC"], mode='lines', name="ROC"), secondary_y=True)

    # Add vertical lines at specified dates
    for date in dates_of_buy:
        fig.add_shape(type="line", x0=date, x1=date, y0=0, y1=1, xref="x", yref="paper", line=dict(color="green", dash="dash"))
    for date in dates_of_sell:
        fig.add_shape(type="line", x0=date, x1=date, y0=0, y1=1, xref="x", yref="paper", line=dict(color="red", dash="dash"))

    # Update the layout and display the plot
    fig.update_layout(title_text="Stock Prices & ROC", xaxis_title="Date", width=1200, height=600)
    fig.update_yaxes(title_text="Close", secondary_y=False)
    fig.update_yaxes(title_text="ROC", secondary_y=True)
    fig.show()

def plot_williams_r(df, dates_of_buy, dates_of_sell):
    # plot buy / sell dates
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, specs=[[{"secondary_y": True}]])

    # Add traces for short long
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode='lines', name="Close"), secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index, y=df["WR"], mode='lines', name="WR"), secondary_y=True)
    fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=-20, y1=-20, xref="x", yref="y2", line=dict(color="red", width=1.5))
    fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=-80, y1=-80, xref="x", yref="y2", line=dict(color="green", width=1.5))

    # Add vertical lines at specified dates
    for date in dates_of_buy:
        fig.add_shape(type="line", x0=date, x1=date, y0=0, y1=1, xref="x", yref="paper", line=dict(color="green", dash="dash"))
    for date in dates_of_sell:
        fig.add_shape(type="line", x0=date, x1=date, y0=0, y1=1, xref="x", yref="paper", line=dict(color="red", dash="dash"))

    # Update the layout and display the plot
    fig.update_layout(title_text="Stock Prices & WR", xaxis_title="Date", width=1200, height=600)
    fig.update_yaxes(title_text="Close", secondary_y=False)
    fig.update_yaxes(title_text="WR", secondary_y=True)
    fig.show()

def plot_chaikin_money_flow(df, dates_of_buy, dates_of_sell, signal):
    # plot buy / sell dates
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, specs=[[{"secondary_y": True}]])

    # Add traces for short long
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode='lines', name="Close"), secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index, y=df["CMF"], mode='lines', name="CMF"), secondary_y=True)
    fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=-signal, y1=-signal, xref="x", yref="y2", line=dict(color="red", width=1.5))
    fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=signal, y1=signal, xref="x", yref="y2", line=dict(color="green", width=1.5))

    # Add vertical lines at specified dates
    for date in dates_of_buy:
        fig.add_shape(type="line", x0=date, x1=date, y0=0, y1=1, xref="x", yref="paper", line=dict(color="green", dash="dash"))
    for date in dates_of_sell:
        fig.add_shape(type="line", x0=date, x1=date, y0=0, y1=1, xref="x", yref="paper", line=dict(color="red", dash="dash"))

    # Update the layout and display the plot
    fig.update_layout(title_text="Stock Prices & CMF", xaxis_title="Date", width=1200, height=600)
    fig.update_yaxes(title_text="Close", secondary_y=False)
    fig.update_yaxes(title_text="CMF", secondary_y=True)
    fig.show()

def backtest_strategy(df, buy, sell):
    # check return for invested 1$
    if (len(buy) > 0) & (len(sell) > 0):
        buy = buy[:len(sell)]
        return np.exp(sum([np.log(df["Close"].loc[sell[i]] / df["Close"].loc[buy[i]]) for i in range(len(buy))]))
    else:
        return 1