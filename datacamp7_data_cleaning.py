"""common data problems"""
#inconsistent column names, outliers, missing data, duplicate rows, unexpected data values
import pandas as pd
df = pd.read_csv("", 'r')
df.head()
df.tail()
df.columns
df.shape
df.info()
df.describe() #numeric columns only
df.dtypes

"""using frequency count to quickly diagnose problems within column data"""
#Uses: too many/little counts, wrong datatype, missing values
#Good for categorical data
#frequency count
df['column'].value_counts(dropna=False) #dropna will count missing vaues too if any
#you can apply .head() to the above to show the top X

"""data visualization: also good for identifying outliers"""
#bar plots for discrete data, histogram for continuous data
df['column'].plot('hist')#other arguments: rot/logx/logy <- you may need logx or logy if you find min and max values are very far apart
import matplotlib.pyplot as plt
plt.show()
#then do filtering
#df[df.column>1000000]

"""box plots"""
df.boxplot(column="population",by="continent") #column is the value to plot, by is group
plt.show()

"""scatter plots"""
#rs between two numeric variables
#you can find potential errors that you may notfind by 1 variable alone
#df.plot(kind='scatter', x='initial_cost', y='total_est_fee', rot=70)

"""melting"""
#id_vars > keep
#value_vars > melt those columns into rows
pd.melt(frame=df,id_vars="name", value_vars=['treatment a','treatment b'], var_name='treatment', value_name="result")

"""opposite of melting, pivoting > unique values into columns"""
#df.pivoting(index="date",columns="element",values="value")
"""pivot table - how to aggregate values in the event of duplicates"""
#df.pivot_table(index="date",columns="element",values="value",aggfunc=np.mean) #np.mean is default so need not indicate
#df.index
#df_reset = df.reset_index()
#ebola_melt['str_split'] = ebola_melt['type_country'.str.split("_") #accessing the value

"""pandas concat"""
conc = pd.concat([df1,df2], ignore_index=True, axis=0) #if duplicated indices use Ignore Index with concat
#0 is row wise concat, 1 is column wise concat
print conc

#glob function to look for files
import glob
csv_files = glob.glob('*.csv') #returns an series of filenames

"""merging: joins tables"""
#pd.merge(left=dfleft, right=dfright, on=None, left_on="state", right_on="name") #on if both tables are same

"""converting types of a column"""
df['treatment_b'] = df['treatment_b'].astype(str)
df['treatment_b'] = df['treatment_b'].astype('category')
df['treatment_b'] = pd.to_numeric(df['treatment_b'],errors="coerce")

"""regex matching"""
import re
pattern = re.compile('^\$\d*\.\d{2}$')
result = bool(pattern.match('$17.89'))
#bool(re.match(pattern='[A-Z]\w*', string='Australia'))
#re.match(pattern='',string='')
#findall function
matches = re.findall('/d+', '\d is the pattern required to find digits. This should be followed with a + so that the previous element is matched one or more times. This ensures that 10 is viewed as one number and not as 1 and 0.')
#"".contains('regex')

"""apply functions for data cleaning across the df"""
df.apply(np.mean, axis=0) #apply across columns
df.apply(np.mean, axis=1) #apply across rows
#def diff_money(row, pattern):
#	icost = row['Initial Cost']
#	...
#Note: Applying to a single column aka series you do not require axis

# Write the lambda function using replace
tips['total_dollar_replace'] = tips.total_dollar.apply(lambda x: x.replace('$', ''))
# Write the lambda function using regular expressions
tips['total_dollar_re'] = tips.total_dollar.apply(lambda x: re.findall('\d+\.\d+', x)[0])

"""drop duplicates"""
df = df.drop_duplicates()

"""missing data"""
#leave , drop or fill
newdf = df.dropna()
df2 = df.fillna('value')
df2[['column1','column2']] = df2[['column1','column2']].fillna(0)

"""using assert"""
assert 1 == 1
assert 1 == 2 #error
assert df.notnull().all()
assert df['col'].dtypes == np.int64 #np.object, np.float64 etc
#assert pd.notnull(ebola).all().all()
#assert (ebola >= 0).all().all()