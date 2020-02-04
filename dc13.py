import seaborn as sns
sns.set()

""" The "square root rule" is a commonly-used rule of thumb for choosing number of bins: choose the number of bins to be the square root of the number of samples. """

#binning bias
#you may have diff interpretations based on different number of bins
#hence, use
#bee swarm plot
#sometimes, the data pts are too many
#use ecdf
#first sort the x axis
x=np.sort(df[''])
y=np.arange(1,len(x)+1)/len(x)
plt.plot(x,y,marker='.',linestyle='none')
plt.margins(0.02) #2% margin around plt

np.percentiles(df,[25,50,75])

#interquartile range is 25-75%
#whiskers extend 1.5 times the IQR or end of data

_ = plt.plot(ptiles_vers, percentiles/100, marker='D', color='red',
         linestyle='none')

#variance, average of square distances from mean (of every point from mean)
#std sqrt(variance)

#pseudo random
np.random.seed(42)
np.random.random(size=4) #generates 4 random

np.var()
np.std()

#covariance is the product differences of point from mean of two different features
#if x is high and y is high -> positive covariance
#negative covariance if x and y are positive/negative
#divide the covariance by std of x * (std of y) 
#variability due to codependence/independent variability
np.cov(x,y) #returns covariance matrix for x and y
np.corrcoef()

#statistical inference probabilistic logic 

#probability mass function
#binomial distribution / bernoulli trials
np.random.binomial(40,0.5,1000) #40 bernoulli trials with success of 0.5 and number is 1000
#binomial cdf - use edcf

#poisson - erratic events
#timing of next event is independent of the next
#poisson - freq in an interval
#it is the limit of the binomial distribution for low probability of sucess and large number of trials/rare events
np.random.poisson(6, size=10000) #10000 multiple samples
"""Poisson distribution with arrival rate equal to np approximates a Binomial distribution for n Bernoulli trials with probability p of success (with n large and p small)."""

# Draw 10,000 samples out of Poisson distribution: samples_poisson
samples_poisson = np.random.poisson(10,size=10000)
# Print the mean and standard deviation
print('Poisson:     ', np.mean(samples_poisson),
                       np.std(samples_poisson))
# Specify values of n and p to consider for Binomial: n, p
n = [20,100,1000]
p = [0.5,0.1,0.01]
# Draw 10,000 samples for each n,p pair: samples_binomial
for i in range(3):
    samples_binomial = np.random.binomial(n[i],p[i],10000)
    # Print results
    print('n =', n[i], 'Binom:', np.mean(samples_binomial),
                                 np.std(samples_binomial))
"""
<script.py> output:
    Poisson:      10.0186 3.144813832327758
    n = 20 Binom: 9.9637 2.2163443572694206
    n = 100 Binom: 9.9947 3.0135812433050484
    n = 1000 Binom: 9.9985 3.139378561116833
"""
#The standard deviation of the Binomial distribution gets closer and closer to that of the Poisson distribution as the probability p gets lower and lower!

#probability for continuous variables
#follows normal distribution ALSO CALLED Gaussian Distribution
#PDF probability density function (continuous version of PMF)
#check the area under the distribution curve
#similarly, it has the cumulative density function CDF
np.random.normal(mean,std,size=10000)
samples_std1=np.random.normal(20,10,100000)
plt.hist(samples_std1,normed=True,histtype='step')

#time between poisson events are exponentially distributed
samples = np.random.exponential(mean,size=10000)

#time for 2 successive poisson events
size=10000
res=np.random.exponential(tau1, size)+np.random.exponential(tau2, size)