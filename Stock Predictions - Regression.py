# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 09:38:14 2019

@author: dkatz44

XGBOOST Regression

"""

"""

Step 1: Build the data set

"""

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

"""
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
"""

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

def nextClose(df):
    
    nextClose = pd.Series(df['close'].shift(-1),name='nextClose',index=df.index)
   # percentChange = pd.Series((nextClose/df['close'])-1,name='nextCloseChange',index=df.index)
    
  #  outcomeVarHigher = pd.Series(percentChange.map(
  #      lambda x: 1 if x >= targetValParamHigher else 0),name=targetNameParamHigher,index=df.index)
        
  #  outcomeVarLower = pd.Series(percentChange.map(
  #      lambda x: 1 if x <= targetValParamLower else 0),name=targetNameParamLower,index=df.index)
  
    df = df.join(nextClose)
    #df = df.join(percentChange)
    #df = df.join(outcomeVarHigher)
    #df = df.join(outcomeVarLower)
    
    return df

df = nextClose(df) 

#%%                 
"""

Step 2: XGBoost Routine

"""

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score

cols_to_drop=[]

formattedDf = df
                                 
# Separate out data from most recent day
newDayDf = formattedDf.tail(1).drop(['nextClose'], axis=1)
newDayDf = newDayDf.drop(cols_to_drop,axis=1)

formattedDf.index = pd.to_datetime(formattedDf.index)

# Remove most recent day from formattedDf
formattedDf = formattedDf.drop(formattedDf.index.max())
formattedDf = formattedDf.drop(cols_to_drop,axis=1)


# Split data into features + target
X_df = formattedDf.drop(['nextClose'], axis=1)
y_df = formattedDf['nextClose']


# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.3, random_state=12)

xgb_model = xgb.XGBRegressor(
                    learning_rate=0.01,
                    n_estimators=2000,
                    max_depth=5,
                    min_child_weight=5,
                    gamma=0.2,
                    subsample=0.5,
                    colsample_bytree=0.6,
                    reg_alpha=1
)

#%%                           
xgb_model.fit(X_train,y_train)   

#%%

predictions = xgb_model.predict(X_test)
#%%
print(explained_variance_score(predictions,y_test))  

#%%

y_test.head()

#%%
dfSavedResults = X_test

dfSavedResults['nextClose'] = y_test

dfSavedResults['nextClosePreds'] = predictions

dfSavedResults = dfSavedResults [['close','nextClose','nextClosePreds']]

#%%

dfSavedResults.head()

#%%

# Predict the new day and add to full data
newDayPreds = xgb_model.predict(newDayDf)

#%%

newDayWithPreds =  newDayDf[['close']]
newDayWithPreds['nextClosePreds'] = newDayPreds

#%%

dfSavedResults = dfSavedResults.append(newDayWithPreds)

#%%
fileName = '/Users/dkatz44/Desktop/AMD Regression Preds.csv'

dfSavedResults.to_csv(fileName, index_label = 'Market_Date')
                      