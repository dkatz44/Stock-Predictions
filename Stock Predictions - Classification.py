# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 16:56:55 2018

@author: dkatz44


#print available indicators
import talib
x = talib.get_function_groups()

categorylist = [y for y in x]

for z in categorylist:
    print(str(z) + ':' + str(x[z]))
    
# list of functions
#print(talib.get_functions())
    
"""

#%%

from talib.abstract import *
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

# Import unformatted data
fileName = '/Users/dkatz44/Desktop/AMD Base Data.csv'

#df_input = pd.read_excel(fileName, sheet_name='Price Data for Python', index_col ='Market_Date')
df_input = pd.read_csv(fileName, index_col = 'Market_Date')

df = df_input.rename(columns={
            'AMD_High':'high',
            'AMD_Low':'low',
            'AMD_Open':'open',
            'AMD_Close':'close',
            'AMD_Volume':'volume'})

df['volume'] = df['volume'].astype(float)
df['close'] = df['close'].astype(float)

df.index = pd.to_datetime(df.index)

targetColNameHigher = 'twoPercentIncrease'
targetColValueHigher = 0.02
targetColNameLower = 'twoPercentDecrease'
targetColValueLower = -0.02

#%%
import datetime
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader.data as web
import numpy as np

# Pull New Data
# Get Daily OCHLV Data For Stock List + Index + Tech Sector

#Set start and end dates
dfMaxDate = df.tail(1).index.values[0]
#date_1 = datetime.datetime.strptime(dfMaxDate, "%m/%d/%y")

start = dfMaxDate + np.timedelta64(1,'D') # Maxdate + 1
start = datetime.datetime.utcfromtimestamp(start.tolist()/1e9)
#start = date_1 + datetime.timedelta(days=1) # Max of data set date + 1
start = datetime.datetime.strftime(start, "%Y-%m-%d")
start = datetime.datetime.strptime(start, "%Y-%m-%d")

# today as datetime
end = datetime.datetime.today().strftime("%Y-%m-%d")
end = datetime.datetime.strptime(end, "%Y-%m-%d") 

print('Updated Query Start Date: ' + str(start))
print('Updated Query End Date: ' + str(end))

symbolList = ['AMD','SPY','VTI','VGT']

f = web.DataReader(symbolList, 'iex', start, end)

df_new_day = pd.DataFrame()

for x in symbolList:
    
    s = pd.DataFrame.from_dict(f[x])
    s = s.rename(columns={
        'open':x+'_Open',
        'close':x+'_Close',
        'high':x+'_High',
        'low':x+'_Low',
        'volume':x+'_Volume'
        })
        
    df_new_day = pd.concat([df_new_day,s],axis=1)

df_new_day = df_new_day.rename(columns={
    'AMD_Open':'open',
    'AMD_Close':'close',
    'AMD_High':'high',
    'AMD_Low':'low',
    'AMD_Volume':'volume'        
    })

#%%
   
df_new_day.head()

#%%
"""
#VIX
import quandl
quandl.ApiConfig.api_key = "CjQzF7bQmm17b-jgQgJ8"

vix = quandl.get("CBOE/VIX", start_date="2018-12-01", end_date="2018-12-09")
#vix = quandl.get("CBOE/VIX", start_date=start, end_date=end)
##data = quandl.get("FRED/GDP", start_date="2001-12-31", end_date="2005-12-31")

# Add VIX data to new day
df_new_day = pd.concat([df_new_day,vix],axis=1)
"""

# Add new day data to base dataset
df = df.append(df_new_day)

# Export Updated Base Data File
fileName = '/Users/dkatz44/Desktop/AMD Base Data.csv'

df.to_csv(fileName, index_label ='Market_Date')

#%%

"""

Previous & Next & Target

"""

df = df.rename(columns={
            'VIX High':'VIX_High',
            'VIX Low':'VIX_Low',
            'VIX Open':'VIX_Open',
            'VIX Close':'VIX_Close'})

#df = df.dropna(subset=['open', 'close', 'high', 'low', 'volume'])


timePeriodValues = [5,10,14,20,30,50]

for timeValue in timePeriodValues:
    
    timeAdjustString = '_' + str(timeValue)

    """
    
    Momentum Indicators
    
    """
        
    # Momentum
    Momentum = MOM(df, timeperiod=timeValue)
    
    df['Momentum' + timeAdjustString] = Momentum
    
    #ADX - Average Directional Movement Index
    # high, low, close
    df['ADX' + timeAdjustString] = ADX(df, timeperiod=timeValue)
      
    #ADXR - Average Directional Movement Index Rating
    # high, low, close
    df['ADXR' + timeAdjustString] = ADXR(df, timeperiod=timeValue)
     
    #APO - Absolute Price Oscillator
    # close
    df['APO'] = APO(df, fastperiod=timeValue, slowperiod=timeValue*2, matype=0)
     
    #AROON - Aroon
    # high, low
    df = df.join(
        AROON(df, timeperiod=timeValue).add_suffix('_AROON' + timeAdjustString)
        )
    
    #AROONOSC - Aroon Oscillator
    # high, low
    df['AROONOSC' + timeAdjustString] = AROONOSC(df, timeperiod=timeValue)
     
    #CCI - Commodity Channel Index
    # high, low, close
    df['CCI' + timeAdjustString] = CCI(df, timeperiod=timeValue)
     
    #CMO - Chande Momentum Oscillator
    # close
    df['CMO' + timeAdjustString] = CMO(df, timeperiod=timeValue)
    
    #DX - Directional Movement Index
    # high, low, close
    df['DX' + timeAdjustString] = DX(df, timeperiod=timeValue)
    
    #MACD - Moving Average Convergence/Divergence
    # close
    df = df.join(
        MACD(df, fastperiod=timeValue, slowperiod=timeValue*2, signalperiod=round(timeValue*.6,0)).add_suffix('_MACD' + timeAdjustString )
        ,rsuffix = '_MACD' + timeAdjustString )            
    
    #MACDEXT - MACD with controllable MA type
    # close
    df = df.join(
        MACDEXT(df, 
                fastperiod=timeValue, 
                fastmatype=0, 
                slowperiod=timeValue*2, 
                slowmatype=0, 
                signalperiod=round(timeValue*.6,0), 
                signalmatype=0).add_suffix('_MACDEXT' + timeAdjustString )
        ,rsuffix = '_MACDEXT' + timeAdjustString)
    
    #MACDFIX - Moving Average Convergence/Divergence Fix 12/26
    # close
    df = df.join(
        MACDFIX(df, signalperiod=round(timeValue*.6,0)).add_suffix('_MACDFIX' + timeAdjustString )
        ,rsuffix = '_MACDFIX' + timeAdjustString)
    
    #MFI - Money Flow Index
    # high, low, close, volume
    df['MFI' + timeAdjustString] = MFI(df, timeperiod=timeValue)
    
    #MINUS_DI - Minus Directional Indicator
    # high, low, close
    df['MINUS_DI' + timeAdjustString] = MINUS_DI(df, timeperiod=timeValue)
    
    #MINUS_DM - Minus Directional Movement
    # high, low
    df['MINUS_DM' + timeAdjustString] = MINUS_DM(df, timeperiod=timeValue)
     
    #PLUS_DI - Plus Directional Indicator
    # high, low, close
    df['PLUS_DI' + timeAdjustString] = PLUS_DI(df, timeperiod=timeValue)
    
    #PLUS_DM - Plus Directional Movement
    # high, low
    df['PLUS_DM' + timeAdjustString] = PLUS_DM(df, timeperiod=timeValue)
     
    #PPO - Percentage Price Oscillator
    # close
    df['PPO' + timeAdjustString] = PPO(df, fastperiod=timeValue, slowperiod=timeValue*2, matype=0)
     
    #ROC - Rate of change : ((price/prevPrice)-1)*100
    # close
    df['ROC' + timeAdjustString] = ROC(df, timeperiod=timeValue)
     
    #ROCP - Rate of change Percentage: (price-prevPrice)/prevPrice
    # close
    df['ROCP' + timeAdjustString] = ROCP(df, timeperiod=timeValue)
     
    #ROCR - Rate of change ratio: (price/prevPrice)
    # close
    df['ROCR' + timeAdjustString] = ROCR(df, timeperiod=timeValue)
    
    #ROCR100 - Rate of change ratio 100 scale: (price/prevPrice)*100
    # close
    df['ROCR100' + timeAdjustString] = ROCR100(df, timeperiod=timeValue)
     
    #RSI - Relative Strength Index
    # close
    df['RSI' + timeAdjustString] = RSI(df, timeperiod=timeValue)
     
    #STOCHF - Stochastic Fast
    # high, low, close
    df = df.join(
        STOCHF(df, fastk_period=timeValue, fastd_period=round(timeValue*.6,0), fastd_matype=0).add_suffix('_STOCHF' + timeAdjustString )
        ,rsuffix = '_STOCHF' + timeAdjustString)
    
    #STOCHRSI - Stochastic Relative Strength Index
    # close
    df = df.join(
        STOCHRSI(df, 
                 timeperiod=timeValue, 
                 fastk_period=round(timeValue*.4,0), 
                 fastd_period=round(timeValue*.2,0), 
                 fastd_matype=0).add_suffix('_STOCHRSI' + timeAdjustString )
        ,rsuffix = '_STOCHRSI' + timeAdjustString        
        )
    
    #TRIX - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
    # close
    df['TRIX' + timeAdjustString] = TRIX(df, timeperiod=timeValue)
     
    #ULTOSC - Ultimate Oscillator
    # high, low, close
    df['ULTOSC' + timeAdjustString] = ULTOSC(df, timeperiod1=timeValue, timeperiod2=timeValue*2, timeperiod3=timeValue*4)
     
    #WILLR - Williams %R
    # high, low, close
    df['WILLR' + timeAdjustString] = WILLR(df, timeperiod=timeValue)
    
    """
    
    Volatility Indicators
    
    """
    
    #ATR - Average True Range
    # high, low, close
    df['ATR' + timeAdjustString] = ATR(df, timeperiod=timeValue)
    
    #NATR - Normalized Average True Range
    # high, low, close
    df['NATR' + timeAdjustString] = NATR(df, timeperiod=timeValue)
        
    """
    
    Overlap Studies
    
    """
    
    # uses close prices (default)
    df['SMA_Close' + timeAdjustString] = SMA(df, timeperiod=timeValue, price='close')
    
    # uses open prices
    df['SMA_Open' + timeAdjustString] = SMA(df, timeperiod=timeValue, price='open')
    
    #DEMA - Double Exponential Moving Average
    # close
    df['DEMA' + timeAdjustString] = DEMA(df, timeperiod=timeValue)
    
    #EMA - Exponential Moving Average
    # close
    df['EMA' + timeAdjustString] = EMA(df, timeperiod=timeValue)
    
    #KAMA - Kaufman Adaptive Moving Average
    # close
    df['KAMA' + timeAdjustString] = KAMA(df, timeperiod=timeValue)
    
    #MA - Moving average
    # close
    df['MA' + timeAdjustString] = MA(df, timeperiod=timeValue, matype=0)
    
    """
    #MAMA - MESA Adaptive Moving Average
    # close
    
    df = df.join(
        MAMA(df, fastlimit=0, slowlimit=0).add_suffix('_MAMA')
        ,rsuffix = '_MAMA')
        
       
    #MAVP - Moving average with variable period
    # close
    df['MAVP'] = MAVP(df, periods=periods, minperiod=2, maxperiod=30, matype=0)
    """
    
    #MIDPOINT - MidPoint over period
    # close
    df['MIDPOINT' + timeAdjustString] = MIDPOINT(df, timeperiod=timeValue)
    
    #MIDPRICE - Midpoint Price over period
    # high, low
    df['MIDPRICE' + timeAdjustString] = MIDPRICE(df, timeperiod=timeValue)
    
    #T3 - Triple Exponential Moving Average (T3)
    # close
    df['T3' + timeAdjustString] = T3(df, timeperiod=timeValue, vfactor=0)
    
    #TEMA - Triple Exponential Moving Average
    # close
    df['TEMA' + timeAdjustString] = TEMA(df, timeperiod=timeValue)
    
    #TRIMA - Triangular Moving Average
    # close
    df['TRIMA' + timeAdjustString] = TRIMA(df, timeperiod=timeValue)
    
    #WMA - Weighted Moving Average
    # close
    df['WMA' + timeAdjustString] = WMA(df, timeperiod=timeValue)
    
    """
    
    Statistic Functions
    
    """
    
    #BETA - Beta
    # high, low
    df['BETA' + timeAdjustString] = BETA(df, timeperiod=timeValue)
    
    #CORREL - Pearson's Correlation Coefficient (r)
    # high, low
    df['CORREL' + timeAdjustString] = CORREL(df, timeperiod=timeValue)
    
    #LINEARREG - Linear Regression
    # close
    df['LINEARREG' + timeAdjustString] = LINEARREG(df, timeperiod=timeValue)
    
    #LINEARREG_ANGLE - Linear Regression Angle
    # close
    df['LINEARREG_ANGLE' + timeAdjustString] = LINEARREG_ANGLE(df, timeperiod=timeValue)
    
    #LINEARREG_INTERCEPT - Linear Regression Intercept
    # close
    df['LINEARREG_INTERCEPT' + timeAdjustString] = LINEARREG_INTERCEPT(df, timeperiod=timeValue)
    
    #LINEARREG_SLOPE - Linear Regression Slope
    # close
    df['LINEARREG_SLOPE' + timeAdjustString] = LINEARREG_SLOPE(df, timeperiod=timeValue)
    
    #STDDEV - Standard Deviation
    # close
    df['STDDEV' + timeAdjustString] = STDDEV(df, timeperiod=timeValue, nbdev=1)
    
    #TSF - Time Series Forecast
    # close
    df['TSF' + timeAdjustString] = TSF(df, timeperiod=timeValue)
    
    #VAR - Variance
    # close
    df['VAR' + timeAdjustString] = VAR(df, timeperiod=timeValue, nbdev=1)
    
"""

Cycle Indicators

"""

#HT_DCPERIOD - Hilbert Transform - Dominant Cycle Period
# close
df['HT_DCPERIOD'] = HT_DCPERIOD(df)

#HT_DCPHASE - Hilbert Transform - Dominant Cycle Phase
# close
df['HT_DCPHASE'] = HT_DCPHASE(df)

#HT_PHASOR - Hilbert Transform - Phasor Components
# close
df = df.join(
    HT_PHASOR(df).add_suffix('_HT_PHASOR')
    )

#HT_SINE - Hilbert Transform - SineWave
# close
df = df.join(
    HT_SINE(df).add_suffix('_HT_SINE')
    )

#HT_TRENDMODE - Hilbert Transform - Trend vs Cycle Mode
# close
df['HT_TRENDMODE'] = HT_TRENDMODE(df)

"""

Price Transform Functions

"""

#AVGPRICE - Average Price
# open, high, low, close
df['AVGPRICE'] = AVGPRICE(df)

#MEDPRICE - Median Price
# high, low
df['MEDPRICE'] = MEDPRICE(df)

#TYPPRICE - Typical Price
# high, low, close
df['TYPPRICE'] = TYPPRICE(df)

#WCLPRICE - Weighted Close Price
# high, low, close
df['WCLPRICE'] = WCLPRICE(df)

"""

Additional

"""

#SAR - Parabolic SAR
# high, low
df['SAR'] = SAR(df, acceleration=0, maximum=0)

#SAREXT - Parabolic SAR - Extended
# high, low
df['SAREXT'] = SAREXT(df, startvalue=0, offsetonreverse=0, accelerationinitlong=0, accelerationlong=0, accelerationmaxlong=0, accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0)
    
#HT_TRENDLINE - Hilbert Transform - Instantaneous Trendline
# close
df['HT_TRENDLINE'] = HT_TRENDLINE(df)

# uses close prices (default)
df = df.join(
    BBANDS(df, 20, 2, 2).add_suffix('_BBands')
    ,rsuffix='_BBands')
    
#TRANGE - True Range
# high, low, close
df['TRANGE'] = TRANGE(df)

#BOP - Balance Of Power
# open, high, low, close
df['BOP'] = BOP(df)
    
"""

Volume Indicators

"""

#AD - Chaikin A/D Line
# high, low, close, volume
df['AD'] = AD(df)

#ADOSC - Chaikin A/D Oscillator
# high, low, close, volume
df['ADOSC'] = ADOSC(df, fastperiod=3, slowperiod=10)

#OBV - On Balance Volume
# close, volume
df['OBV'] = OBV(df)
# RSI Overbought
# RSI Oversold
#n=14

# uses high, low, close (default)
df = df.join(
    STOCH(df, 5, 3, 0, 3, 0, prices=['high', 'low', 'close']).add_suffix('_STOCH_Close')
    ,rsuffix='_STOCH_Close')

# uses high, low, open instead
df = df.join(
    STOCH(df, 5, 3, 0, 3, 0, prices=['high', 'low', 'open']).add_suffix('_STOCH_Open')
    ,rsuffix = '_STOCH_Open')
    
#df['RSI' + str(n) + '_Diff'] = df['RSI_' + str(n)].diff()
#df['RSI' + str(n) + '_Shift'] = df['RSI_' + str(n)].shift(1)

df['RSI_Diff'] = df['RSI_10'].diff()
df['RSI_Shift'] = df['RSI_10'].shift(1)

df['RSI_Oversold_MoveIn'] = df.apply(lambda x: 1 if x['RSI_10'] <=30 and x['RSI_Diff'] < 0 and x['RSI_Shift'] > 30 else 0, axis=1)  
df['RSI_Oversold'] = df.apply(lambda x: 1 if x['RSI_10'] <= 30 else 0, axis=1)

df['RSI_Overbought_MoveIn'] = df.apply(lambda x: 1 if x['RSI_10'] >=70 and x['RSI_Diff'] > 0 and x['RSI_Shift'] < 70 else 0, axis=1)  
df['RSI_Overbought'] = df.apply(lambda x: 1 if x['RSI_10'] >=70 else 0, axis=1)   

df['CDL2CROWS'] = CDL2CROWS(df)
df['CDLXSIDEGAP3METHODS'] = CDLXSIDEGAP3METHODS(df)
df['CDLUPSIDEGAP2CROWS'] = CDLUPSIDEGAP2CROWS(df)
df['CDLUNIQUE3RIVER'] = CDLUNIQUE3RIVER(df)
df['CDLTRISTAR'] = CDLTRISTAR(df)
df['CDLTHRUSTING'] = CDLTHRUSTING(df)
df['CDLTASUKIGAP'] = CDLTASUKIGAP(df)
df['CDLTAKURI'] = CDLTAKURI(df)
df['CDLSTICKSANDWICH'] = CDLSTICKSANDWICH(df)
df['CDLSTALLEDPATTERN'] = CDLSTALLEDPATTERN(df)
df['CDLSPINNINGTOP'] = CDLSPINNINGTOP(df)
df['CDLSHORTLINE'] = CDLSHORTLINE(df)
df['CDLSHOOTINGSTAR'] = CDLSHOOTINGSTAR(df)
df['CDLSEPARATINGLINES'] = CDLSEPARATINGLINES(df)
df['CDLRISEFALL3METHODS'] = CDLRISEFALL3METHODS(df)
df['CDLRICKSHAWMAN'] = CDLRICKSHAWMAN(df)
df['CDLPIERCING'] = CDLPIERCING(df)
df['CDLONNECK'] = CDLONNECK(df)
df['CDLMORNINGSTAR'] = CDLMORNINGSTAR(df, penetration=0)
df['CDLMORNINGDOJISTAR'] = CDLMORNINGDOJISTAR(df, penetration=0)
df['CDLMATHOLD'] = CDLMATHOLD(df, penetration=0)
df['CDLMATCHINGLOW'] = CDLMATCHINGLOW(df)
df['CDLMARUBOZU'] = CDLMARUBOZU(df)
df['CDLLONGLINE'] = CDLLONGLINE(df)
df['CDLLONGLEGGEDDOJI'] = CDLLONGLEGGEDDOJI(df)
df['CDLLADDERBOTTOM'] = CDLLADDERBOTTOM(df)
df['CDLKICKINGBYLENGTH'] = CDLKICKINGBYLENGTH(df)
df['CDLKICKING'] = CDLKICKING(df)
df['CDLINVERTEDHAMMER'] = CDLINVERTEDHAMMER(df)
df['CDLINNECK'] = CDLINNECK(df)
df['CDLIDENTICAL3CROWS'] = CDLIDENTICAL3CROWS(df)
df['CDLHOMINGPIGEON'] = CDLHOMINGPIGEON(df)
df['CDLHIKKAKEMOD'] = CDLHIKKAKEMOD(df)
df['CDLHIKKAKE'] = CDLHIKKAKE(df)
df['CDLHIGHWAVE'] = CDLHIGHWAVE(df)
df['CDLHARAMICROSS'] = CDLHARAMICROSS(df)
df['CDLHARAMI'] = CDLHARAMI(df)
df['CDLHANGINGMAN'] = CDLHANGINGMAN(df)
df['CDLHAMMER'] = CDLHAMMER(df)
df['CDLGRAVESTONEDOJI'] = CDLGRAVESTONEDOJI(df)
df['CDLGAPSIDESIDEWHITE'] = CDLGAPSIDESIDEWHITE(df)
df['CDLEVENINGSTAR'] = CDLEVENINGSTAR(df, penetration=0)
df['CDLEVENINGDOJISTAR'] = CDLEVENINGDOJISTAR(df, penetration=0)
df['CDLENGULFING'] = CDLENGULFING(df)
df['CDLDRAGONFLYDOJI'] = CDLDRAGONFLYDOJI(df)
df['CDLDOJISTAR'] = CDLDOJISTAR(df)
df['CDLDOJI'] = CDLDOJI(df)
df['CDLDARKCLOUDCOVER'] = CDLDARKCLOUDCOVER(df, penetration=0)
df['CDLCOUNTERATTACK'] = CDLCOUNTERATTACK(df)
df['CDLCONCEALBABYSWALL'] = CDLCONCEALBABYSWALL(df)
df['CDLCLOSINGMARUBOZU'] = CDLCLOSINGMARUBOZU(df)
df['CDLBREAKAWAY'] = CDLBREAKAWAY(df)
df['CDLBELTHOLD'] = CDLBELTHOLD(df)
df['CDLADVANCEBLOCK'] = CDLADVANCEBLOCK(df)
df['CDLABANDONEDBABY'] = CDLABANDONEDBABY(df, penetration=0)
df['CDL3WHITESOLDIERS'] = CDL3WHITESOLDIERS(df)
df['CDL3STARSINSOUTH'] = CDL3STARSINSOUTH(df)
df['CDL3OUTSIDE'] = CDL3OUTSIDE(df)
df['CDL3LINESTRIKE'] = CDL3LINESTRIKE(df)
df['CDL3INSIDE'] = CDL3INSIDE(df)
df['CDL3BLACKCROWS'] = CDL3BLACKCROWS(df)

# Add in preivous day values and % change
def previousDayValue(df, colName):
    
    df = df.join(pd.Series(df[colName].shift(1),name='previous_' + colName))
    df['prevChange_' + colName] = (df[colName] / df['previous_' + colName]) - 1    
    
    return df  

for x in df.columns:

    df = previousDayValue(df, x)

df['Market_Q1'] = pd.Series(df.index.values, index = df.index).apply(lambda x: 1 if pd.Timestamp(x).quarter == 1 else 0)
df['Market_Q2'] = pd.Series(df.index.values, index = df.index).apply(lambda x: 1 if pd.Timestamp(x).quarter == 2 else 0)
df['Market_Q3'] = pd.Series(df.index.values, index = df.index).apply(lambda x: 1 if pd.Timestamp(x).quarter == 3 else 0)
df['Market_Q4'] = pd.Series(df.index.values, index = df.index).apply(lambda x: 1 if pd.Timestamp(x).quarter == 4 else 0)

def nextClose(df, targetNameParamHigher
                , targetValParamHigher
                , targetNameParamLower
                , targetValParamLower):
    
    nextClose = pd.Series(df['close'].shift(-1),name='nextClose',index=df.index)
    percentChange = pd.Series((nextClose/df['close'])-1,name='nextCloseChange',index=df.index)
    
    outcomeVarHigher = pd.Series(percentChange.map(
        lambda x: 1 if x >= targetValParamHigher else 0),name=targetNameParamHigher,index=df.index)
        
    outcomeVarLower = pd.Series(percentChange.map(
        lambda x: 1 if x <= targetValParamLower else 0),name=targetNameParamLower,index=df.index)
  
    df = df.join(nextClose)
    df = df.join(percentChange)
    df = df.join(outcomeVarHigher)
    df = df.join(outcomeVarLower)
    
    return df

df = nextClose(df,targetColNameHigher
                 ,targetColValueHigher
                 ,targetColNameLower
                 ,targetColValueLower) 
                 
# Export results 
fileName = '/Users/dkatz44/Desktop/AMD Formatted Data.csv'

df.to_csv(fileName, index_label='Market_Date')

#%%

"""

XGBoost Routine

"""

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, log_loss
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.cross_validation import *

def Create_Model (X_train,
                  X_test,
                  y_train,
                  y_test,
                  learning_rate,
                  n_estimators,
                  max_depth,
                  min_child_weight,
                  gamma,
                  subsample,
                  colsample_bytree,
                  reg_alpha,
                  eval_metric):
    
    ROCforest = XGBClassifier(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        gamma=gamma,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        objective= 'binary:logistic',
        nthread=4,
     #scale_pos_weight=1,
        seed=12)
    
    cv_folds=5
    #early_stopping_rounds=50
    
    eval_metric = eval_metric#'logloss'#'auc'
     
    xgb_param = ROCforest.get_xgb_params()
    xgtrain = xgb.DMatrix(X_train.values, label=y_train.values)
    cvresult = xgb.cv(
               xgb_param, 
               xgtrain, 
               num_boost_round=ROCforest.get_params()['n_estimators'], 
               nfold=cv_folds,
               metrics=eval_metric
               )
              # early_stopping_rounds=early_stopping_rounds) 
               # , show_progress=False)
    ROCforest.set_params(n_estimators=cvresult.shape[0])
    
    ROCforest.fit(X_train, y_train)
    
    return ROCforest
    
def Model_Results (model, X_test, y_test):
     # Determine the false positive and true positive rates
    y_pred = model.predict(X_test)
    y_pred_probs = model.predict_proba(X_test)[:,1]
    
    actualTrue = sum(y_test)
    actualFalse = y_test.tolist().count(0)
    overallPopulation = actualTrue + actualFalse
    
    # True Positives
    truePositives = sum([y for x,y in enumerate(y_pred) if y_test.tolist()[x] == 1 and y == 1])
    tprSum = truePositives / sum(y_test)
    
    # True Negatives
    trueNegatives = sum([1 for x,y in enumerate(y_pred) if y_test.tolist()[x] == 0 and y == 0])
    
    # False Positives
    falsePositives = sum([y for x,y in enumerate(y_pred) if y_test.tolist()[x] == 0 and y == 1])
    fprSum = falsePositives / y_test.tolist().count(0)
    
    # Accuracy
    accuracy = (truePositives + trueNegatives) / overallPopulation
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    log_loss_result = log_loss(y_true=y_test, y_pred=y_pred_probs)    
    
    print("Overall Population: ", overallPopulation)
    print("truePositives:",truePositives)
    print("Actual true:", sum(y_test))
    print("falsePositives:",falsePositives)
    print("Actual False:",trueNegatives)
    #print("FPR: %0.2f" % fprSum)
    print("True Positive Rate: %0.2f" % tprSum)
    print("True Negative Rate: %0.2f" % (1 - fprSum))
    print('ROC AUC: %0.2f' % roc_auc)
    print('ACCURACY: (%d + %d) / %d = %0.2f' % (truePositives, trueNegatives, overallPopulation, accuracy))
    print('Log Loss: %0.4f' % log_loss_result)
    print(log_loss_result)
    
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FP / Actual False)')
    plt.ylabel('True Positive Rate (TP / Actual True)')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
    
    pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
    
    # Add predictions to the test data
    finalDf = X_test
    
    finalDf['Preds'] = y_pred
    finalDf['Probs'] = y_pred_probs
    
    # Correlations
    #print(finalDf[finalDf.columns[0:]].corr()['Preds'][:-1].sort_values(ascending=False).dropna())
    
    finalDf['Target'] = y_test

    return finalDf

"""

Parameter Tuning

"""

def Parameter_Tuning(dataframe):

    train = formattedDf    
    
    predictors = [x for x in train.columns if x not in ['Target','nextClose','nextCloseChange']]
    
    parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
                  'objective':['binary:logistic'],
                  'learning_rate': [0.2,0.5,0.7,1], #so called `eta` value
                  'max_depth':[i for i in range(3,8)],
                  'min_child_weight':[i for i in range(3,8)],
                  'silent': [1],
                  'subsample':[i/10.0 for i in range(4,8)],
                  #'subsample':[0.6],
                  'colsample_bytree':[i/10.0 for i in range(5,9)],
                  'n_estimators': [100],  #number of trees, change it to 1000 for better results
                  'gamma': [i/10.0 for i in range(0,5)],
                  #'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100],
                  'reg_alpha':[0.1,0.5,1],
                  #'missing':['NAN'],
                  'seed': [1337]}
    
    xgb_model = xgb.XGBClassifier()
    
    clf = GridSearchCV(xgb_model, parameters, n_jobs=5, 
                       cv=StratifiedKFold(train["Target"], n_folds=3, shuffle=True), 
                       scoring='roc_auc',
                       #scoring = "neg_log_loss"
                       verbose=1, refit=True)
    
    clf.fit(train[predictors], train["Target"])
    
    #trust your CV!
    best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
    print('Raw AUC score:', score)
    for param_name in sorted(best_parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))

    return best_parameters


# Restrict on Data newer than 2015
# Rename target col as 'Target'
#formattedDf = df[df.index >= '2015-1-1'].rename(columns={targetColName: 'Target'})

formattedDf = df.rename(columns={targetColNameHigher: 'Target_Higher',
                                 targetColNameLower: 'Target_Lower'})

cols_to_drop = [
#'SPY_Close',	
#'SPY_High',
#'SPY_Low',	
#'SPY_Open',
#'SPY_Volume',
#'VGT_Close',
#'VGT_High',
#'VGT_Low',	
#'VGT_Open',
#'VGT_Volume',
#'VIX_Close',	
#'VIX_High',	
#'VIX_Low',	
#'VIX_Open',
#'VTI_Close',
#'VTI_High',	
#'VTI_Low',	
#'VTI_Open',
#'VTI_Volume'
]
  
# Separate out data from most recent day
newDayDf = formattedDf.tail(1).drop(['Target_Higher','Target_Lower','nextClose','nextCloseChange'], axis=1)
newDayDf = newDayDf.drop(cols_to_drop,axis=1)

formattedDf.index = pd.to_datetime(formattedDf.index)

# Remove most recent day from formattedDf
formattedDf = formattedDf.drop(formattedDf.index.max())

formattedDf = formattedDf.drop(cols_to_drop,axis=1)

# Split data into features + target
X_Higher = formattedDf.drop(['Target_Higher','Target_Lower','nextClose','nextCloseChange'], axis=1)
y_Higher = formattedDf['Target_Higher']

X_Lower = formattedDf.drop(['Target_Higher','Target_Lower','nextClose','nextCloseChange'], axis=1)
y_Lower = formattedDf['Target_Lower']

# shuffle and split training and test sets
X_train_Higher, X_test_Higher, y_train_Higher, y_test_Higher = train_test_split(X_Higher, y_Higher, test_size=.30, random_state=12) 
X_train_Lower, X_test_Lower, y_train_Lower, y_test_Lower = train_test_split(X_Lower, y_Lower, test_size=.30, random_state=12) 

# Set model params
learning_rate=0.01
n_estimators=2000
max_depth=5
min_child_weight=5
gamma=0.2
subsample=0.5
colsample_bytree=0.6
reg_alpha=1
eval_metric = 'logloss'#'auc'

ROCforest_Higher = Create_Model(X_train_Higher, X_test_Higher, y_train_Higher, y_test_Higher,
                         learning_rate=learning_rate,
                         n_estimators=n_estimators,
                         max_depth=max_depth,
                         min_child_weight=min_child_weight,
                         gamma=gamma,
                         subsample=subsample,
                         colsample_bytree=colsample_bytree,
                         reg_alpha=reg_alpha,
                         eval_metric = eval_metric)

ROCforest_Lower = Create_Model(X_train_Lower, X_test_Lower, y_train_Lower, y_test_Lower,
                         learning_rate=learning_rate,
                         n_estimators=n_estimators,
                         max_depth=max_depth,
                         min_child_weight=min_child_weight,
                         gamma=gamma,
                         subsample=subsample,
                         colsample_bytree=colsample_bytree,
                         reg_alpha=reg_alpha,
                         eval_metric = eval_metric)

#xgb.plot_tree(ROCforest,num_trees=0)
#plt.rcParams['figure.figsize'] = [100, 50]

plt.show()
xgb.plot_importance(ROCforest_Higher,max_num_features=20)
xgb.plot_importance(ROCforest_Lower,max_num_features=20)

finalDf_Higher = Model_Results(ROCforest_Higher, X_test_Higher, y_test_Higher)
finalDf_Lower = Model_Results(ROCforest_Lower, X_test_Lower, y_test_Lower)

finalDf = finalDf_Higher.join(df['nextCloseChange'])
finalDf = finalDf.rename(columns={'Preds': 'Preds_Higher',
                                  'Probs': 'Probs_Higher',
                                  'Target': 'Target_Higher'})
                                  
finalDf['Preds_Lower'] = finalDf_Lower['Preds']
finalDf['Probs_Lower'] = finalDf_Lower['Probs']
finalDf['Target_Lower'] = finalDf_Lower['Target']

# Predict the new day and add to full data
newDayPreds_Higher = ROCforest_Higher.predict(newDayDf)
newDayProbs_Higher = ROCforest_Higher.predict_proba(newDayDf)[:,1]

newDayPreds_Lower = ROCforest_Lower.predict(newDayDf)
newDayProbs_Lower = ROCforest_Lower.predict_proba(newDayDf)[:,1]

newDayDf['Preds_Higher'] = newDayPreds_Higher
newDayDf['Probs_Higher'] = newDayProbs_Higher

newDayDf['Preds_Lower'] = newDayPreds_Lower
newDayDf['Probs_Lower'] = newDayProbs_Lower

finalDf = finalDf.append(newDayDf)

# Export results 
fileName = '/Users/dkatz44/Desktop/AMD Preds CSV after Python Calcs.csv'

finalDf.to_csv(fileName, index_label = 'Market_Date')

predsDf = finalDf[['close',
                         'nextCloseChange',
                         'Preds_Higher',
                         'Preds_Lower',
                         'Probs_Higher',
                         'Probs_Lower',
                         'Target_Higher',
                         'Target_Lower']]

fileName = '/Users/dkatz44/Desktop/AMD Preds Relevant Cols.csv'

predsDf.to_csv(fileName, index_label = 'Market_Date')

#%%
"""

Feature Importance Testing

"""


feat_imp = pd.Series(ROCforest_Higher.get_booster().get_score(importance_type='weight')).sort_values(ascending=False)

#%%


fileName = '/Users/dkatz44/Desktop/feature importances.csv'

feat_imp.to_csv(fileName)
#%%

important_features = feat_imp[feat_imp > 10]

X_train, X_test, y_train, y_test = train_test_split(X[important_features.index.values], 
                                                    y, 
                                                    test_size=.20,
                                                    random_state=12)
#%%
ROCforest = Create_Model(X_train, X_test, y_train, y_test,
                         learning_rate=learning_rate,
                         n_estimators=n_estimators,
                         max_depth=max_depth,
                         min_child_weight=min_child_weight,
                         gamma=gamma,
                         subsample=subsample,
                         colsample_bytree=colsample_bytree,
                         reg_alpha=reg_alpha,
                         eval_metric = eval_metric)

#%%
finalDf = Model_Results(ROCforest)


"""

Feature Importance Threshold Testing

"""

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=12) 

#%%

# use feature importance for feature selection
from numpy import sort
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel

#%%
thresh = 10

selection = SelectFromModel(ROCforest, threshold=thresh, prefit=True)

#%%

selection.get_params
#%%

select_X_train = selection.transform(X_train)
#%%
# train model
selection_model = XGBClassifier()
selection_model.fit(select_X_train, y_train)
# eval model
select_X_test = selection.transform(X_test)
y_pred = selection_model.predict(select_X_test)

#%%

model = ROCforest

thresholds = sort(model.feature_importances_)
#%%

thresholds

#%%
for thresh in thresholds:
	# select features using threshold
	selection = SelectFromModel(model, threshold=thresh, prefit=True)
	select_X_train = selection.transform(X_train)
	# train model
	selection_model = XGBClassifier()
	selection_model.fit(select_X_train, y_train)
	# eval model
	select_X_test = selection.transform(X_test)
	y_pred = selection_model.predict(select_X_test)
	predictions = [round(value) for value in y_pred]
	accuracy = accuracy_score(y_test, predictions)
	print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))

#%%

import joblib

fileName = '/Users/dkatz44/Desktop/savedModelHigher.joblib.dat'

# save model to file
joblib.dump(ROCforest_Higher, fileName)

fileName = '/Users/dkatz44/Desktop/savedModelLower.joblib.dat'

# save model to file
joblib.dump(ROCforest_Lower, fileName)

#%%
# load models from file

import joblib

fileName = '/Users/dkatz44/Desktop/savedModelHigher.joblib.dat'

# save model to file
loaded_model_higher = joblib.load(fileName)

fileName = '/Users/dkatz44/Desktop/savedModelLower.joblib.dat'

# save model to file
loaded_model_lower = joblib.load(fileName)

formattedDf = df.rename(columns={targetColNameHigher: 'Target_Higher',
                                 targetColNameLower: 'Target_Lower'})

cols_to_drop = [
#'SPY_Close',	
#'SPY_High',
#'SPY_Low',	
#'SPY_Open',
#'SPY_Volume',
#'VGT_Close',
#'VGT_High',
#'VGT_Low',	
#'VGT_Open',
#'VGT_Volume',
#'VIX_Close',	
#'VIX_High',	
#'VIX_Low',	
#'VIX_Open',
#'VTI_Close',
#'VTI_High',	
#'VTI_Low',	
#'VTI_Open',
#'VTI_Volume'
]
  
# Separate out data from most recent day
newDayDf = formattedDf.tail(1).drop(['Target_Higher','Target_Lower','nextClose','nextCloseChange'], axis=1)
newDayDf = newDayDf.drop(cols_to_drop,axis=1)

# Predict the new day and add to full data
newDayPreds_Higher = loaded_model_higher.predict(newDayDf)
newDayProbs_Higher = loaded_model_higher.predict_proba(newDayDf)[:,1]

newDayPreds_Lower = loaded_model_lower.predict(newDayDf)
newDayProbs_Lower = loaded_model_lower.predict_proba(newDayDf)[:,1]

newDayDf['Preds_Higher'] = newDayPreds_Higher
newDayDf['Probs_Higher'] = newDayProbs_Higher

newDayDf['Preds_Lower'] = newDayPreds_Lower
newDayDf['Probs_Lower'] = newDayProbs_Lower

# Import unformatted data
fileName = '/Users/dkatz44/Desktop/AMD Preds Relevant Cols.csv'

#df_input = pd.read_excel(fileName, sheet_name='Price Data for Python', index_col ='Market_Date')
dfSavedResults = pd.read_csv(fileName, index_col = 'Market_Date')

predsDf = newDayDf[['close',
                         'Preds_Higher',
                         'Preds_Lower',
                         'Probs_Higher',
                         'Probs_Lower']]

dfSavedResults = dfSavedResults.append(predsDf)

dfSavedResults.tail()

fileName = '/Users/dkatz44/Desktop/AMD Preds Relevant Cols.csv'

dfSavedResults.to_csv(fileName, index_label = 'Market_Date')





