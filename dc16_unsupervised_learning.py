from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
model.fit(samples)

labels = model.predict(samples) #still need to assign labels to know how it is being clustered
print(labels)

#using scatterplt to visualise clusters

import matplotlib.pyplot as plt
xs = samples[:,0] #x and y coordinates of each sample, 0 and 2 are columns (sepal length/width)
ys = samples[:,2] 
plt.scatter(xs, ys, c=labels)
plt.show()

"""Scatter plot with centroids"""
# Import pyplot
import matplotlib.pyplot as plt
# Assign the columns of new_points: xs and ys
xs = new_points[:,0]
ys = new_points[:,1]
# Make a scatter plot of xs and ys, using labels to define the colors
plt.scatter(xs,ys, c=labels, alpha=0.5)
# Assign the cluster centers: centroids
centroids = model.cluster_centers_
# Assign the columns of centroids: centroids_x, centroids_y
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]
# Make a scatter plot of centroids_x and centroids_y
plt.scatter(centroids_x,centroids_y,marker='D',s=50)
plt.show()

"""use of cross tabulation: clustered labels vs ***known*** clusters and check each row/column"""
import pandas as pd
df = pd.DataFrame({'labels':labels, 'species':species})
ct = pd.crosstab(df['labels'], df['species'])
print(ct)
#looking for tight clusters
#spread/inertia - lower spread the better
print(model.inertia_)

#elbow
ks = range(1, 6)
inertias = []
for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    # Fit model to samples
    model.fit(samples)
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)    
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

#with the labels, do a cross tabulation
# Create a KMeans model with 3 clusters: model
model = KMeans(n_clusters=3)
# Use fit_predict to fit model and obtain cluster labels: labels
labels = model.fit_predict(samples)
# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': varieties})
# Create crosstab: ct
ct = pd.crosstab(df['labels'],df['varieties'])
# Display ct
print(ct)

"""clustering may be affected by features having different variances"""
"""use StandardScalar"""
# Perform the necessary imports
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
# Create scaler: scaler
scaler = StandardScaler()
# Create KMeans instance: kmeans
kmeans = KMeans(n_clusters=4)
# Create pipeline: pipeline
pipeline = make_pipeline(scaler,kmeans)
print(pipeline.fit_predict(samples))
# Import pandas
import pandas as pd
# Fit the pipeline to samples
pipeline.fit(samples)
# Calculate the cluster labels: labels
labels = pipeline.predict(samples)
# Create a DataFrame with labels and species as columns: df
df = pd.DataFrame({'labels':labels, 'species':species})
# Create crosstab: ct
ct = pd.crosstab(df.labels, df.species)
# Display ct
print(ct)

"""While StandardScaler() standardizes features (such as the features of the fish data from the previous exercise) by removing the mean and scaling to unit variance, Normalizer() rescales each sample - here, each company's stock price - independently of the other."""

# Import Normalizer
from sklearn.preprocessing import Normalizer
# Create a normalizer: normalizer
normalizer = Normalizer()
# Create a KMeans model with 10 clusters: kmeans
kmeans = KMeans(n_clusters=10)
# Make a pipeline chaining normalizer and kmeans: pipeline
pipeline = make_pipeline(normalizer,kmeans)
# Fit pipeline to the daily price movements
pipeline.fit(movements)

"""Visualization: t-SNE, Hierachical clustering / dendrogram / agglomerative clustering"""
import matplotlib.pyplot as plt
from scipy.cluster.hierachy import linkage, dendrogram
mergings= linkage(samples, method='complete')
dendrogram(mergings, labels=country_names, leaf_rotation=20, leaf_font_size=56_
plt.show()


# Import normalize
from sklearn.preprocessing import normalize
# Normalize the movements: normalized_movements
normalized_movements = normalize(movements)
# Calculate the linkage: mergings
mergings = linkage(normalized_movements, method='complete')
# Plot the dendrogram
dendrogram(mergings, labels=companies, leaf_rotation=90, leaf_font_size=6)
plt.show()

#intermediate clustering
#selecting a height determines the extent of the clusters
#height shows distance between merging clusters
#linkage(normalized_movements, method='complete') <-- we are using maximum/complete distance between clusters to determine dendrogram
#single is distance of closest points of clusters
from scipy.cluster.hierachy import fcluster
labels=fcluster(mergings, 15, criterion='distance')
pairs = pd.DataFrame({'labels':labels, 'countries':country_names})
print(pairs.sort_values('labels'))
# Perform the necessary imports
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
# Calculate the linkage: mergings
mergings = linkage(samples, method='single')
# Plot the dendrogram
dendrogram(mergings, labels=country_names, leaf_rotation=90, leaf_font_size=6)
plt.show()

# Perform the necessary imports
import pandas as pd
from scipy.cluster.hierarchy import fcluster
# Use fcluster to extract labels: labels
labels = fcluster(mergings, 6, criterion='distance')
# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': varieties})
# Create crosstab: ct
ct = pd.crosstab(df['labels'],df['varieties'])
# Display ct
print(ct)

"""t-SNE"""
#t-distributed stochastic neighbor embedding
# Import TSNE
from sklearn.manifold import TSNE
# Create a TSNE instance: model
model = TSNE(learning_rate=200)
# Apply fit_transform to samples: tsne_features
tsne_features = model.fit_transform(samples)
# Select the 0th feature: xs
xs = tsne_features[:,0]
# Select the 1st feature: ys

ys = tsne_features[:,1]
# Scatter plot, coloring by variety_numbers
plt.scatter(xs,ys,c=variety_numbers)
plt.show()

"""dimension reduction"""
#PCA - Principal Component Analysis
#rotates samples to align w axis, shifts mean to 0 -> decorrelation
# Perform the necessary imports
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
# Assign the 0th column of grains: width
width = grains[:,0]
# Assign the 1st column of grains: length
length = grains[:,1]
# Scatter plot width vs length
plt.scatter(width, length)
plt.axis('equal')
plt.show()
# Calculate the Pearson correlation
correlation, pvalue = pearsonr(width, length)
# Display the correlation
print(correlation)

# Import PCA
from sklearn.decomposition import PCA
# Create PCA instance: model
model = PCA()
# Apply the fit_transform method of model to grains: pca_features
pca_features = model.fit_transform(grains)
# Assign 0th column of pca_features: xs
xs = pca_features[:,0]
# Assign 1st column of pca_features: ys
ys = pca_features[:,1]
# Scatter plot xs vs ys
plt.scatter(xs, ys)
plt.axis('equal')
plt.show()
# Calculate the Pearson correlation of xs and ys
correlation, pvalue = pearsonr(xs, ys)
# Display the correlation
print(correlation)

"""intrinsic dimension is num of features needed to approximate the dataset -> 2d flight path can be approximated with just 1d i.e: displacement from path since path is a straight line"""
#pca realigns samples with descending variance and picks the features with high variance
# Make a scatter plot of the untransformed points
plt.scatter(grains[:,0], grains[:,1])
# Create a PCA instance: model
model = PCA()
# Fit model to points
model.fit(grains)
# Get the mean of the grain samples: mean
mean = model.mean_
# Get the first principal component: first_pc
first_pc = model.components_[0,:]
# Plot first_pc as an arrow, starting at mean
plt.arrow(mean[0], mean[1], first_pc[0], first_pc[1], color='red', width=0.01)
# Keep axes on same scale
plt.axis('equal')
plt.show()

# Perform the necessary imports
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
# Create scaler: scaler
scaler = StandardScaler()
# Create a PCA instance: pca
pca = PCA()
# Create pipeline: pipeline
pipeline = make_pipeline(scaler,pca)
# Fit the pipeline to 'samples'
pipeline.fit(samples)
# Plot the explained variances
features = range(pca.n_components_)
print(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()

AI Notes:
# Import PCA
from sklearn.decomposition import PCA
# Create a PCA model with 2 components: pca
pca = PCA(n_components=2)
# Fit the PCA instance to the scaled samples
pca.fit(scaled_samples)
# Transform the scaled samples: pca_features
pca_features = pca.transform(scaled_samples)
# Print the shape of pca_features
print(pca_features.shape)

#text classification
# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# Create a TfidfVectorizer: tfidf
tfidf = TfidfVectorizer() 
# Apply fit_transform to document: csr_mat
csr_mat = tfidf.fit_transform(documents)
# Print result of toarray() method
print(csr_mat.toarray())
# Get the words: words
words = tfidf.get_feature_names()
# Print words
print(words)

#clustering articles with TruncatedSVD
# Perform the necessary imports
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
# Create a TruncatedSVD instance: svd
svd = TruncatedSVD(n_components=50)
# Create a KMeans instance: kmeans
kmeans = KMeans(n_clusters=6)
# Create a pipeline: pipeline
pipeline = make_pipeline(svd, kmeans)

# Import pandas
import pandas as pd
# Fit the pipeline to articles
pipeline.fit(articles) “””these articles corresponds to the words””"
# Calculate the cluster labels: labels
labels = pipeline.predict(articles)
# Create a DataFrame aligning labels and titles: df
df = pd.DataFrame({'label': labels, 'article': titles})
# Display df sorted by cluster label
print(df.sort_values('label'))


#Non negative matrix factorization like PCA but at interpetable
#all sample features must be non-negative
#works with numpy and csr_matrix
#nmf components multiply with feature values to get close sample and adding it up
# Import NMF
from sklearn.decomposition import NMF
# Create an NMF instance: model
model = NMF(n_components=6)
# Fit the model to articles
model.fit(articles)
# Transform the articles: nmf_features
nmf_features = model.transform(articles)
# Print the NMF features
print(nmf_features)

# Import pandas
import pandas as pd
# Create a pandas DataFrame: df
df = pd.DataFrame(nmf_features, index=titles)
# Print the row for 'Anne Hathaway'
print(df.loc['Anne Hathaway'])
# Print the row for 'Denzel Washington'
print(df.loc['Denzel Washington'])

#components are like... weights
#features are like the additional stuffs to multiply by the weights to get your scoring
"""NMF is applied to documents, the components (like u choose dimensions of 6 n_components, so it picks the 6 important things to look out for) correspond to topics of documents, and the NMF features reconstruct the documents from the topics."""
# Import pandas
import pandas as pd
# Create a DataFrame: components_df
components_df = pd.DataFrame(model.components_, columns=words)
# Print the shape of the DataFrame
print(components_df.shape)
# Select row 3: component
component = components_df.iloc[3]
# Print result of nlargest
print(component.nlargest())

"""So fitting the models, you select Anne Hathaway and found that the feature has a high score for component 3. Then now you check component 3 and find that the words constitute:
    film       0.627877
    award      0.253131
    starred    0.245284
    role       0.211451
    actress    0.186398
"""

#using it on images
# Import pyplot
from matplotlib import pyplot as plt
# Select the 0th row: digit
digit = samples[0]
# Print digit
print(digit)
# Reshape digit to a 13x8 array: bitmap
bitmap = digit.reshape(13,8)
# Print bitmap
print(bitmap)
# Use plt.imshow to display bitmap
plt.imshow(bitmap, cmap='gray', interpolation='nearest')
plt.colorbar()
plt.show()

# Import NMF
from sklearn.decomposition import NMF
# Create an NMF model: model
model = NMF(n_components=7)
# Apply fit_transform to samples: features
features = model.fit_transform(samples)
# Call show_as_image on each component
for component in model.components_:
    show_as_image(component)
# Assign the 0th row of features: digit_features
digit_features = features[0]
# Print digit_features
print(digit_features)

"""replacing with PCA, you can see that PCA does not extract meaningful parts from the samples"""

# Import PCA
from sklearn.decomposition import PCA
# Create a PCA instance: model
model = PCA(n_components=7)
# Apply fit_transform to samples: features
features = model.fit_transform(samples)
# Call show_as_image on each component
for component in model.components_:
    show_as_image(component)

"""comparing features for a recommender system"""
#using weights of features not a good indicator -> strong vs weak wording
#instead, uses cosine similarity to determine
#higher values = more similar
# Perform the necessary imports
import pandas as pd
from sklearn.preprocessing import normalize
# Normalize the NMF features: norm_features
norm_features = normalize(nmf_features)
# Create a DataFrame: df
df = pd.DataFrame(norm_features, index=titles)
# Select the row corresponding to 'Cristiano Ronaldo': article
article = df.loc['Cristiano Ronaldo']
# Compute the dot products: similarities
similarities = df.dot(article)
# Display those with the largest cosine similarity
print(similarities.nlargest())

"""Reocmmender: Music based on users"""
"""Not sure why they used a scaler and then a normaliser:"""
#The first step in the pipeline, MaxAbsScaler, transforms the data so that all users have the same influence on the model, regardless of how many different artists they've listened to.

# Perform the necessary imports
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer, MaxAbsScaler
from sklearn.pipeline import make_pipeline
# Create a MaxAbsScaler: scaler
scaler = MaxAbsScaler()
# Create an NMF model: nmf
nmf = NMF(n_components=20)
# Create a Normalizer: normalizer
normalizer = Normalizer()
# Create a pipeline: pipeline
pipeline = make_pipeline(scaler, nmf, normalizer)
# Apply fit_transform to artists: norm_features
norm_features = pipeline.fit_transform(artists)

# Import pandas
import pandas as pd
# Create a DataFrame: df
df = pd.DataFrame(norm_features, index=artist_names)
# Select row of 'Bruce Springsteen': artist
artist = df.loc['Bruce Springsteen']
# Compute cosine similarities: similarities
similarities = df.dot(artist)
# Display those with highest cosine similarity
print(similarities.nlargest())
