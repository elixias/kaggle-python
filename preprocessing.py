"""
Import data
describe()
visual exploration - hist
setting nulls, cleaning nulls, clean numerical columns, convert categorical, remove outliers
set the LABELS
use a FunctionTransformer as a selector to go into Pipelines

#1) NLP: Use ranged ngrams and punctuation tokenization
#2) Stats: Interaction terms
#3) Computational: Hashing
#Other Strategies
#NLP: Stemming, stop word removal, 
#Model: RandomForest, k-NN, Naive Bayes
#Numerical Preprocessing: Imputation strategies
#Optimization: Grid Search over pipeline

"""

import pandas as pd

#if multiple files
#import glob
#filenames = glob('filename_*.csv')
#dataframes = [pd.read_csv(f) for f in filenames]
df = pd.read_csv("filename.csv", sep=',', header=0, names=['column names',''], usecols=[], na_values={'column':['-1']}, parse_dates=[[0,1,2]], index_col='column to use as row label')
df.index.map(str.lower())
df.columns.map(str.lower())

df.columns
df.head()
df.dtypes

#force types
df[] = df[].astype('str')

#exploratory
df.shape
df.info()
df.describe()
quantile([0.X,0.X])
unique()
count()

#pairplot
import seaborn as sns
df_sns = sns.load_dataset(df)
sns.pairplot(df_sns, hue='feature')

#remove outliers and na
df.isna().sum()
df.dropna(axis=0) #if not a lot of nas, 0 for rows, 1 for columns
#df = df.fillna('value')
df = df.drop_duplicates()

df.describe()

"""checkpoint - inconsistent column names, outliers, missing data, duplicate rows, unexpected data values"""

#for splitting features for different pipelines
#NUMER = ['',...]
#NONNUMER = [c for c in df.columns if c not in NUMER]
#X_numer = pd.get_dummies(df[NUMER])
#X_nonnumer = pd.get_dummies(df[NONNUMER])

df = df.get_dummies(drop_first=True, prefix=['',''])
#if ordinal
#df.column = pd.Categorical(values=df.column, categories=[<ascending order>], ordered=True)

y = df[:,-1] #target
x_features = ['', '',...]
X = df[x]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=404, stratify=y)

#build a pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScalar #scale/normalize
from sklearn. import
pipeline=[('imputation', SimpleImputer()), ('scalar', StandardScalar()), ('estimator',estimator)] #tuple (name_to_give_the_step, estimator)

#feed pipeline into a randomsearchcv
from sklearn.model_selection import RandomizedSearchCV
parameters = {estimator__n_neighors: np.arange(1,50)}
cv = RandomizedSearchCV(pipeline, param_grid=parameters, cv=5)
cv.fit(X_train, y_train)
cv.best_params_
cv.score

#feed best parameters into pipeline
estimator = ()
pipeline=[('imputation', SimpleImputer()), ('scalar', StandardScalar()), ('estimator',estimator)]
pipeline.predict(X_test)

"""checkpt"""
