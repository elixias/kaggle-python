#run through method1
#run through method2

import pandas as pd
documents = pd.read_csv("remedy_all_items.csv",header=0)
documents = documents[["trans_id","incident_title","incident_description","incident_resolution_method","incident_resolution"]]
doc = documents
SELECTED_COL = 'incident_title'
#doc = documents.loc[:,"incident_title"] #column to select

#######################
#Preliminary findings
#######################
#'incident_opscat_tier1', 'incident_opscat_tier2', 'incident_opscat_tier3', 'incident_prodcat_tier1','incident_prodcat_tier2', 'incident_prodcat_tier3',
len(documents['incident_opscat_tier1'].unique())
#39, tier2 is 77, same for tier 3
len(documents['incident_prodcat_tier1'].unique())
#11,27
grouped = documents.groupby(['incident_opscat_tier1', 'incident_opscat_tier2','incident_opscat_tier3','incident_prodcat_tier1','incident_prodcat_tier2','incident_prodcat_tier3'])
#grouped.count()
#[2201 rows x 68 columns]
(grouped["trans_id"].count())
(grouped["trans_id"].count()).sum()
grouped["trans_id"].count().sort_values(ascending=False)

#you can see majority is in account management, application
cond1=documents["incident_opscat_tier1"]=="Account Management"
cond2=documents["incident_prodcat_tier1"]=="Application"
documents[cond1 & cond2]

#####################################
#Stopwords removal
#####################################

import re
def decontracted(phrase):
  # specific
  phrase = re.sub(r"won't", "will not", phrase)
  phrase = re.sub(r"can\'t", "can not", phrase)
  # general
  phrase = re.sub(r"n\'t", " not", phrase)
  phrase = re.sub(r"\'re", " are", phrase)
  phrase = re.sub(r"\'s", " is", phrase)
  phrase = re.sub(r"\'d", " would", phrase)
  phrase = re.sub(r"\'ll", " will", phrase)
  phrase = re.sub(r"\'t", " not", phrase)
  phrase = re.sub(r"\'ve", " have", phrase)
  phrase = re.sub(r"\'m", " am", phrase)
  return phrase

def clean(cell):
  stopwords= set(['[walk-in]','br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',"you're", "you've","you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their','theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after','above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further','then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more','most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',"hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',"mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"])
  cell = decontracted(cell.strip().lower())
  #sentence = re.sub("\S*\d\S*", "", sentence).strip() #remove digits
  cell = re.sub('[^A-Za-z]+', ' ', cell)
  return ' '.join(e for e in cell.split() if e not in stopwords)

doc[SELECTED_COL] = doc[SELECTED_COL].apply(clean)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer() 
csr_mat = tfidf.fit_transform(doc[SELECTED_COL])
print(csr_mat.toarray())
words = tfidf.get_feature_names()
print(words)

###method1
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt


###finding optimal number of ocmponents
# Program to find the optimal number of components for Truncated SVD
n_comp = [4,10,15,20,50,100,150,200,500,700,800,900,1000,1500,2000,2500,3000,3500] # list containing different values of components
explained = [] # explained variance ratio for each component of Truncated SVD
for x in n_comp:
  svd = TruncatedSVD(n_components=x)
  svd.fit(csr_mat)
  explained.append(svd.explained_variance_ratio_.sum())
  print("Number of components = %r and explained variance = %r"%(x,svd.explained_variance_ratio_.sum()))
plt.plot(n_comp, explained)
plt.xlabel('Number of components')
plt.ylabel("Explained Variance")
plt.title("Plot of Number of components v/s explained variance")
plt.show()

svd = TruncatedSVD(n_components=3000)

####testing individual component###############
svdfit=svd.fit(csr_mat)
#svdfit.predict(csr_mat)
components_df = pd.DataFrame(svdfit.components_, columns=words)
####to print the words that make up each component
component = components_df.iloc[2]
print(component.nlargest())
###############################################

#need to do 
kmeans = KMeans(n_clusters=1000)
pipeline = make_pipeline(svd, kmeans)
pipeline.fit(csr_mat)
labels = pipeline.predict(csr_mat)

#finding the best number of clusters
bestcluster = []
for i in [1500,2000,3500]:
  kmeans = KMeans(n_clusters=i)
  pipeline = make_pipeline(svd, kmeans)
  pipeline.fit(csr_mat)
  bestcluster.append(kmeans.inertia_)
  print(i+" clusters inertia:"+kmeans.inertia_)
print(bestcluster)
#import joblib
#joblib.dump(pipeline.best_estimator_, 'fit_csr_mat_2500.pkl')

###
#Plotting
plt.plot([25,50,100,200,300,500,1000,1500,3000],
[35170.320197956185, 32371.097371796037, 28757.422040493544, 25189.982953250383, 23020.15175248972, 20448.578401665967, 16982.621445478384, 15002.791323188709, 11686.781740348306])
plt.xlabel('Clusters')
plt.ylabel("Inertia")
plt.title("Plot of Number of Clusters and Inertia scoring")
plt.show()
#[35170.320197956185, 32371.097371796037, 28757.422040493544, 25189.982953250383, 23020.15175248972, 20448.578401665967, 16982.621445478384, 11686.781740348306]
#clusters x components


#plotting the labels vs components
#y:records x:components_df 2500
#import plotly.express as px
#df = pd.concat([labels, csr_mat], axis=1)
#df.head()
#df = pd.DataFrame({'label': labels, 'article': doc})
#fig = px.parallel_coordinates(df, color="label", color_continuous_scale=px.colors.diverging.Tealrose)
#fig.show()

df = pd.DataFrame({'label': labels, 'article': doc})
#df = pd.concat([labels, doc], axis=1)
# Display df sorted by cluster label
print(df.sort_values('label'))

###method2
from sklearn.decomposition import NMF
model = NMF() #n_components=15 create 27 components, which comprises of a mix of keywords
model.fit(csr_mat) #fit our documents
nmf_features = model.transform(csr_mat)
print(nmf_features) #gives weightage

df = pd.DataFrame(nmf_features, index=doc)

components_df = pd.DataFrame(model.components_, columns=words)
print(components_df.shape)
component = components_df.iloc[2]
print(component.nlargest())

###evaluating

components_df
import seaborn as sns;
ax = sns.heatmap(component_df)
component = components_df.iloc[16]
print(component.nlargest())

for i in [0,1,23,2]:
  print(components_df.iloc[i].nlargest())
###get the highest component for df and that will be the "label" for the article
df.idxmax(axis=1)













#GENERATING EXAMPLES
#dff.groupby('label')['incident_title'].agg(['unique','nunique'])
dff.groupby('label')['incident_title'].agg(['unique','nunique']).sort_values('nunique', ascending=False)
#dff.groupby('label')['incident_title'].agg(['unique','nunique']).sort_values('nunique', ascending=False)
df1 = pd.read_csv("count_3000w_1250c.csv")
df2 = pd.read_csv("unique_val_3000w_1250c.csv")
df1 = df1.iloc[:,:2]
df1.columns = ['label','count']
df2 = df2.iloc[:,[0,2]]
df3=df1.merge(df2,on='label')
import seaborn as sns
import matplotlib.pyplot as plt
sns.scatterplot(data=df3[df3['count']<300],x='count',y='nunique')
plt.show()

filter=(df3['nunique']<df3['count']*0.5) & (df3['nunique']>1)
df3['hue'] = filter
sns.scatterplot(data=df3[df3['count']<300],x='count',y='nunique',hue='hue')
plt.show()
#getting only those with 50% number of unique labels
end=df3[df3['hue']]
end.describe()
end.to_csv("labels_to_use.csv")

#generate list to convert to KB

df = pd.read_csv("labels_to_use.csv")
df.iloc[:,1:4]
df['ratio']=df['nunique']/df['count']
df=df.sort_values('ratio',ascending=True)

od = pd.read_csv("final_labels_3000w_1250c.csv")
unique_title=od.groupby('label')['incident_title'].agg(['unique'])
unique_res=od.groupby('label')['incident_resolution'].agg(['unique'])
unique_title.merge(unique_res, on="label", how="left")
kb=unique_title.merge(unique_res, on="label", how="left")
final_kb=df.merge(kb, on="label", how="left")
final_kb = final_kb.iloc[:,1:]
final_kb.to_csv("final_kb_3000w_1250c.csv")

#generate KB from this final list

import pandas as pd
final_kb = pd.read_csv("final_kb_3000w_1250c.csv")
final_kb = final_kb.iloc[:,1:]

import json
def convertNDArray(x):
  #return np.fromstring(x).tolist()
  return x.tolist()
  #jsstr="\",\"".join(x)
  #jsstr="{\""+jsstr+"\"}"

#y=json.load(final_kb['unique_x'])
#res=[convertToJSON(i) for i in final_kb['unique_x']]


def convertToJSON(kbslice,min,max):
  template = {"intents":[],"actions": [],"stories": []}
  EX_LIMIT=10
  WORD_LIMIT=250#actually letter limit
  print(len(kbslice))
  for i in range(0,len(kbslice)):
    kbexample = convertNDArray(kbslice.iloc[i,4][:EX_LIMIT])
    kbname = kbexample[0]
    kbres = "Incident Title:<b>"+kbname+"</b><br/><br/>- "+"<br/>- ".join(final_kb.iloc[i,5].astype(str).tolist())#convertNDArray(kbslice.iloc[i,5])
    if len(kbres)>50:
      kbres=kbres[:WORD_LIMIT]+"..."
    itemplate={"intent":kbname,"entities":[],"texts":kbexample}
    atemplate={"name":kbname,"allActions": [[{"type": "TEXT","text": kbres}]]}
    stemplate={"name":kbname,"wait_checkpoint": "","intent": [{"intent": kbname,"intentConditions": [],"actions": [kbname]}],"return_checkpoint": "","defStory": False}
    template["intents"].append(itemplate)
    template["actions"].append(atemplate)
    template["stories"].append(stemplate)
  f = open(f"{min}_{max}.txt", "w")
  f.write(json.dumps(template))
  f.close()

convertToJSON(final_kb.iloc[0:50,:],0,50)

INC_VAL = 50
for x in range(0, len(final_kb), INC_VAL):
  #print(x)
  convertToJSON(final_kb.iloc[x:(x+INC_VAL),:],x,x+INC_VAL)
