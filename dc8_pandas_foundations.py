import numpy as np
#AAPL.iloc[::3,-1] = np.nan #Here, :: means every 3 rows

"""getting numerical entries of a series in a dataframe"""
#AAPL['low'].values

"""using dicts as df"""
data = {'weekday':['sun','sun','mon'],
		'city':['austin','dallas','austin']
		}
print(pd.DataFrame(data)) #keys are used as columns

"""building from series instead"""
days = ['sun','sun','mon']
cities = ['austin','dallas','austin']
list_labels = ['days','cities']
list_cols = [days,cities]
zipped = list(zip(list_labels,list_cols))
data = dict(zipped)
print(pd.DataFrame(data))

"""changing column labels"""
#data.columns = ['a','b']

"""no headers, set column names, set na values, parse columns as dates"""
#data = pd.read_csv(filepath, header=None, names=[...,...], na_values={'column':['-1']},
#parse_dates=[[0,1,2]], index_col='')

"""set index"""
#data.index=data['yearmthday']
#data.index.name='date'

"""write to csv"""
#data.to_csv("filename.csv")
#data.to_excel("filename.xls")

"""different ways to plot"""
#data.column.plot()
#plt.plot(data.column)
#plt.plot(data.column)
#plt.plot(data)
#data.plot()
#data.plot(x=..,y=..,kind='scatter')
#data.plot(y=...,kind='box')
#data.plot(y=...,kind='hist',bins=..,range=(..,..),normed=..,cumulative=..)

#data.plot(kind='hist')
#data.plt.hist()
#data.hist()

"""VERY USEUFL"""
#data.plot(subplots=True) 

#data.column.plot(color='b',style='.-', legend=True)
#plt.axis(('2001','2002',0,100))

#plt.yscale('log')

"""saving figure"""
plt.savefig("___.png/jpg/pdf")

"""other configs"""
plt.title("_")
plt.xlabel('')
plt.ylabel('')

"""subplots: """
#fig, axes = plt.subplots(nrows=2, ncols=1)
#df.fraction.plot(ax=axes[0], kind='hist', normed=True, bins=30, range=(0,.3)) #specifying ax=axes[0] to use the subplots

"""data exploration beyond describe() and info()"""
#data.column.count() #applied to series
#data.count() #applied to df, returns a series of the counts
#count(),mean(),median(),std(),quantile(0.X) or quantile([0.X,0.X]), min(), max()
#data.column.unique()

#indices = data['species'] == 'setosa'
#data[indices,:]

"""computing errors"""
error_setosa = 100 * np.abs(setosa.describe()-data.describe())
error_setosa = error_setosa/setosa.describe()

#using datetime as indices are very common techniques
#you can provide the date and a subset of those rows will be returned
#data.loc['2015-2-5'] #partial datetime string selection
#data.loc['2015-2':'2016-2'] #partial selection by month range
#pd.to_datetime([...,...])
#data.reindex([...,...],method='ffill') #forward fill using preceding entries, or use bfill

#manual way without date index
#res = df1['Date'].apply(lambda x:x[0:12]=='2010-Aug-01')
#df1[df1['Date'].apply(lambda x:x[0:8]=='20100801')]
#df3.loc['2010-Aug-01']

#runtime converting datetime 
#pd.to_datetime(series, format='%Y-%m-%d %H:%M')

"""reindexing"""
#ts3 = ts2.reindex(ts1.index)

"""resampling, downsampling (daily to weekly), upsampling (daily to hourly) """
#daily_mean = sales.resample('D').mean()/.sum()/.max()/.count()/.var()/.std()/.ffill() #D stands for daily
#min/T, H, D, B, W, M, Q, A

"""rolling mean"""
#df['Temperature']['2010-Aug-01':'2010-Aug-15'].rolling(window=24).mean()
#if you use resampling and reindex to do it, looks 'digital'
#test = unsmoothed.resample('D').mean()
#test = test.reindex(unsmoothed.index, method='ffill')

#another way is to use interpolate
df.resample('A').first().interpolate('linear')

"""use str/contains to do string based search/etc"""
#df.str.upper()/contains().sum()/strip()
#df.dt.hour #this extracts the hour from the time
#df.dt.tz_localize('US/Central') #localize time ones is not the same as tz_convert()

#df.columns = df.columns.str.strip()

"""plotting time series data"""
#correcting data
#pd.to_numeric(errors='coerce')/to_date/to_str

"""pearson's correlation"""
#df.corr() #on a df with 2 columns
