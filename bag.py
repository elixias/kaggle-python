import pandas as pd

data = pd.read_csv('bag.csv', header=None)
data.columns = ['Intent','Example']
data.set_index('Intent', inplace=True)

#make the dictionary
comb = ' '.join(data['Example']).lower().split()
dlist = sorted(list(set(comb)))

word_to_int = { c:i for i, c in enumerate(dlist) }

"""Number of Examples"""
example_count = data.groupby('Intent').Example.count().sort_values(ascending=False)


data.Example = data.Example.str.lower()+" "
res = data.Example.str.split(expand=True).stack()
res2= res.swaplevel(0,1).reset_index().drop(labels=['level_0'],axis=1)
res2.columns = ['Intent','Word']
res2['Count'] = 1
"""Word count and intents that contribute to it"""
res3=res2.groupby(['Word','Intent']).Count.sum()


res4=res2.groupby('Word').Count.sum()

"""save to excel"""
writer = pd.ExcelWriter('report.xlsx')
example_count.to_excel(writer, sheet_name="Number of Examples")
res3.to_excel(writer, sheet_name="Word Count (by Intent)")
res4.to_excel(writer, sheet_name="Word Count")
writer.save()
"""
#plot
import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x='Word', hue='Intent', data=res3)
sns.stripplot(x="Word", y="Intent", data=res2, jitter=True)


#res2.groupby(['Word','Intent']).Count.sum()
#reset_indeX()

res2.set_index('Word').plot(kind='bar',stacked=True)
res2['Word'].value_counts()
plt.hist(res2, bins=256, range=(0,256), normed=True, cumulative=True)

res = pd.melt(res.unstack(), id_vars=['Example'], value_name='Words')
res2=res.stack().swaplevel(0,1)
res2.columns = ['Num','Intent','Word']
res2.drop('Num', axis=1)
res3=res2.groupby(['Intent','Word']).sum()
res3.unstack()

res2.fillna(value=np.nan, inplace=True)
sns.heatmap(data=res2,x='Intent',y='Word')
 
res.pivot(index='Intent', columns=)
"""