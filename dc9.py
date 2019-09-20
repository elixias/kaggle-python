"""Getting the index of the rows and columns"""
#df.index/columns.get_loc('...')

#you can use .loc this way too:
df.loc[:, 'salt':'pepper'] #gets all columns between salt and pepper

#with loc and iloc, you can use : or [] as selection criterias

"""reverse selections"""
p_counties_rev = election.loc['Potter':'Perry':-1,:]

"""filters"""
df[df.salt>60 & df.eggs<200] #can use &, |, !

"""transforming data - numpy universal functions"""
#convert to dozen units 
df.floordiv(12)
np.floor_divide(df, 12)
def dozens(n):
	return n/12
df.apply(dozens)
df.apply(lambda x:x//12)

"""creating a new column"""
#df[<new column>] = ..

"""str accessor for vectorized values"""
#df.index=df.index.str.upper()
#df.index.map(str.lower()) #has no apply

"""Dictionary mapping"""
red_vs_blue = {'Obama':'blue', 'Romney':'red'}
election['color'] = election['winner'].map(red_vs_blue) #if winner is Obama it is now mapped to blue

"""vectorized functions, UFuncs perform better than apply and map"""
from scipy.stats import zscore
election['zscore'] = zscore(election['turnout'])

del df['column_name']

"""you can use a tuple as an index / multiindex / hierachical index"""
stocks = stocks.set_index(['Symbol','Date'])
stocks = stocks.sort_index()
stocks.index.names #names not name
print(stocks)
stocks.loc[('CSCO','2016-10-04')] #accessing a multilevel index
stocks.loc[(['AAPL','MSFT'], '2016-10-05'),:] #fancy indexing
stocks.loc[(slice(None), slice('2016-10-03','2016-10-04')),:] #slicing w tuples to filter by innerlevels of index

"""pivoting"""
#if not all columns are used, result in multiple pivot tables for the remaining columns
visitors_pivot = users.pivot(index='weekday',columns='city',values='visitors')
"""pivot table"""
#used when there are duplicate entries. uses reduction (usually average) to determine new values
#use the aggfunc attribute i.e: aggfunc='count' or len
#you can use margins as well
signups_and_visitors_total = users.pivot_table(index='weekday',aggfunc=sum, margins=True) #adds a total 'All' to the end

"""unstacking"""
#unstacking multilevel index (more columns, fewer rows)
trials.unstack(level='gender') #gender in multiindex becomes one of the columns or unstack(level=1) <- this is zero based
"""stacking"""
#opposite of stack, turns a column into an index
trials.stack(level='gender')
"""swapping levels"""
swapped = stacked.swaplevel(0,1)

"""melting (revise)"""
#melting converts wide data into a long shape. id_vars specifies which columns to retain, value_vars which ones to convert into values
#value_name sets the name
visitors_by_city_weekday = visitors_by_city_weekday.reset_index() #take out index
visitors = pd.melt(visitors_by_city_weekday, id_vars=['weekday'], value_name='visitors')

"""groupby"""
sales.groupby('weekday').count() #ends with aggregation
sales.groupby(['city','weekday'])[['bread','butter']].mean() #advanced
#if multilevel index, specify the level parameter
sales.groupby(level=['city','weekday'])

life = pd.read_csv(life_fname, index_col='Country')
regions = pd.read_csv(regions_fname, index_col='Country')
life_by_region = life.groupby(regions['region']) #region is aligned to life first ***
print(life_by_region['2010'].mean())

"""group by with agg """
#.agg(['max','sum'])
#.agg({'bread':'sum','butter':custom_function})

"""<a group after groupby>.transform()"""
#use a .transform() method after grouping to apply a function to groups of data

by_sex_class = titanic.groupby(['sex','pclass'])
def impute_median(series):
    return series.fillna(series.median())
titanic.age = by_sex_class.age.transform(impute_median)
print(titanic.tail(10))	

groupby().groups #group object has a groups attribute that is a dict, use .keys() on it to get the index

#groupby using dictionary comprehension
#see https://campus.datacamp.com/courses/manipulating-dataframes-with-pandas/grouping-data?ex=13

#boolean groupby
auto.groupby(['yr',auto['name'].str.contains('chevrolet')])['mpg'].mean()

"""idxmax() idxmin"""
#returns row/col (axis='columns) with highest/lowest values

""""categorical conversion"""
medals.Medal = pd.Categorical(values=medals.Medal, categories=['Bronze','Silver','Gold'], ordered=True)
