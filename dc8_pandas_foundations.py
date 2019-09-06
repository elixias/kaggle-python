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
#data = pd.read_csv(filepath, header=None, name=[...,...], na_values={'column':['-1']},
#parse_dates=[[0,1,2]]) 

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