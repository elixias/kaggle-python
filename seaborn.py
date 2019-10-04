import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""One difference between seaborn and regular matplotlib plotting is that you can pass pandas DataFrames directly to the plot and refer to each column by name."""
tips = sns.load_dataset('tips')
sns.lmplot(x='total_bill', y='tip', data=tips, hue='sex', palette='Set1')
sns.lmplot(x='total_bill', y='tip', data=tips, col='sex') #or row
sns.residplot(x='age',y='fare',data=df,color='indianred') #residual plot

plt.scatter(...)
sns.regplot(x='weight', y='mpg', data=auto, label='order 1', color='blue', scatter=None, order=1)
#scatter none prevents replotting

plt.show()

"""univariate strip swarm violin"""
sns.stripplot(x='<category variable>', y='tip', data=tips, size=4, jitter=True)
sns.swarmplot(x='<category variable>', y='tip', data=tips, hue='<another category>', orient='h')
#box and violins
sns.boxplot(x='',y='',data=tips)
sns.violinplot(x='',y='',data=tips)
#combining
sns.violinplot(x='',y='',data=tips, inner=None,color='lightgray')
sns.stripplot(x='<category variable>', y='tip', data=tips, size=4, jitter=True) #by default, overlays

"""ultivariate"""
#joint plot
sns.jointplot(x='total_bill',y='tip',data=tips, kind='kde') #kernel density estimation
#possible joint plots of the variables
sns.pairplot(tips, hue='sex')
#heatmap
sns.heatmap(calc_covariance)

"""time series"""
#labels = dates.strftime('%b %d')
plt.xticks(dates, labels, rotation=60)
#inset views
#after plotting a diagram, specify an axes
plt.axes([0.25,0.5,0.35,0.35])
#and do another plot

"""histogram equalization"""
# Generate a cumulative histogram
cdf, bins, patches = plt.hist(pixels, bins=256, range=(0,256), normed=True, cumulative=True)
new_pixels = np.interp(pixels, bins[:-1], cdf*255)
# Load the image into an array: image
image = plt.imread('640px-Unequalized_Hawkes_Bay_NZ.jpg')

# Flatten the image into 1 dimension: pixels
pixels = image.flatten()

# Generate a cumulative histogram
cdf, bins, patches = plt.hist(pixels, bins=256, range=(0,256), normed=True, cumulative=True)
new_pixels = np.interp(pixels, bins[:-1], cdf*255)

# Reshape new_pixels as a 2-D array: new_image
new_image = new_pixels.reshape(image.shape)

# Display the new image with 'gray' color map
plt.subplot(2,1,1)
plt.title('Equalized image')
plt.axis('off')
plt.imshow(new_image,cmap='gray')

# Generate a histogram of the new pixels
plt.subplot(2,1,2)
pdf = plt.hist(new_pixels, bins=64, range=(0,256), normed=False,
               color='red', alpha=0.4)
plt.grid('off')

# Use plt.twinx() to overlay the CDF in the bottom subplot
plt.twinx()
plt.xlim((0,256))
plt.grid('off')

# Add title
plt.title('PDF & CDF (equalized image)')

# Generate a cumulative histogram of the new pixels
cdf = plt.hist(new_pixels, bins=64, range=(0,256),
               cumulative=True, normed=True,
               color='blue', alpha=0.4)
plt.show()
