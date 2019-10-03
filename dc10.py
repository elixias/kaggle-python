"""reading multiple files into dataframe"""
#read_csv(), html, json, excel

"""Glob"""
for glob import glob
filenames = glob('sale*.csv')
#instead of 
filenames = [...,...]

dataframes = []
for f in filenames:
	dataframes.append(pd.read_csv(f))
#Shorter!
dataframes = [pd.read_csv(f) for f in filenames]

ordered = ['Jan','Feb',...]
df = df.reindex(ordered) #to reorder to a given set of indices
df = df.reindex(anotherdf.index) #reindex fills with NaN if original is not found
#weather2 = weather1.reindex(year).ffill()
#common_names = names_1981.reindex(names_1881.index).dropna()
#or
df.sort_index(ascending=False)
df.sort_values('Max TemperatureF')

"""arithemetic between different dataframe"""
#you can perform ops on them df1 + df2 but behavior depends
week1_range.divide(week1_mean, axis='rows')
week1_mean.pct_change() * 10
bronze.add(silver, fill_value=0)

"""appending and concat"""
s1.append(s2).append(s3).reset_index(dropna=True) #rows of s2 stacked under s1, works for dataframes and series
#!!! does not make changes to index so may have duplicate index
pd.concat([s1,s2,s3],ignore_index=True) #accepts a list of, vertically or horizontally

pd.concat([[pd1,pd2], keys=[k1,k2], axis=0) #concat with multilvl index
#you can use dict instead of list, the keys of the dict will be used 

#pd.IndexSlice
idx = pd.IndexSlice
print(medals_sorted.loc[idx[:,'United Kingdom'],:])
#same as
idx = (slice(None), 'United Kingdom')
print(medals_sorted.loc[idx,:])
#slice_2_8 = february.loc['2015-02-02':'2015-02-08', idx[:, 'Company']]
#in summary, use pd.IndexSlice[:,X] where : is the multiindex to 'ignore'

"""np.hstack()/np.concatenate([B,A],axis=1) requires same number of rows"""
#stacks side by side (left n right)
"""np.vstack([A,C]) or np.concatenate([A,C],axis=0) requires same number of columns"""

"""outer joins""" """uses index to join"""
#union of index sets without label repetition, missing fields = NaN
pd.concat([pd1,pd2], axis=1, join='outer')
"""inner joins"""
#intersections only
pd.concat([pd1,pd2], axis=1, join='inner')

"""pd.merge uses data within columns common to both dataframes as a merge criteria"""
pd.merge(df1,df2) #results in columns of both dataframes
pd.merge(bronze, gold, on=['NOC','Country'], suffixes=['_bronze','_gold'])
pd.merge(counties, cities, left_on='CITY NAME', right_on='City', how='left') #if columns on both dfs are different names_1881
#how = 'left' retains info in left df
#left, right, outer, inner
pd.merge_ordered(hardware, software) #default is outer join, accepts on, suffixes, fill_method='ffil'

"""pd.merge_asof()"""
#merge up to highest column value in left df


population.join(unemployment) #does left join by default
population.join(unemployment, how='right') 
