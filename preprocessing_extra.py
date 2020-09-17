
import matplotlib.pyplot as plt
df['column'].value_counts(dropna=False) #counts include missing/NA items
df['column'].replace(0, np.nan, inplace=True)
df['column'].plot('hist')
"""for histograms use square root rule"""
plt.show()
#bivariate
#box plots
df.boxplot(column="population",by="continent") #column is the value to plot, by is group
plt.show()
#scatter plots
df.plot(kind='scatter', x='initial_cost', y='total_est_fee', rot=70)

#box plot, strip plot, swarm plot
#pairplot, joint plot (can see distributin of both easier), heatmap

_ = plt.hist(new_pixels, bins=64, range=(0,256),
               cumulative=True, normed=True,
               color='blue', alpha=0.4)
			   
#changing column names
df.index.map(str.lower())
df.columns.map(str.lower())
data.columns = ['a','b']

#set index column
data.index=data['yearmthday']



#split df into numeric/string/categorical columns

#category
medals.Medal = pd.Categorical(values=medals.Medal, categories=['Bronze','Silver','Gold'], ordered=True)
#df_region = df.get_dummies(drop_first=True)
#get_dummy()
#or sklearn.OneHotEncoder





#hyperparameter tuning with random/gridsearchcv
#fed pipeline into gridsearchcv

#import, create the classifier or predictor
#import model
#create model
#train/fit model
#test the score: model.score(X_test, y_test)
