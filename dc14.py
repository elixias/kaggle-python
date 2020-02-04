#optimal parameters for generating a normal distribution are the mean and std of the given data
#scipy.stats, statsmodels
#are both good packages for statistical modelling
#τ tau

#in linear regression the difference between line and path is residual
#least squares method
slope, intercept = np.polyfit(x_data,y_data,1)  #3rd arg is the degree of polynomial function. 1st degree here.
#pearson_r
def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat
    corr_mat=np.corrcoef(x,y)
    # Return entry [0,1]
    return corr_mat[0,1]
	
_ = plt.plot(illiteracy, fertility, marker='.', linestyle='none')
plt.margins(0.02)
_ = plt.xlabel('percent illiterate')
_ = plt.ylabel('fertility')
a, b = np.polyfit(illiteracy,fertility,1)
print('slope =', a, 'children per woman / percent illiterate')
print('intercept =', b, 'children per woman')
x = np.array([0,100])
y = a * x + b
_ = plt.plot(x, y)
plt.show()

#eda: anscombe's quartet
#plotting linear
a, b = np.polyfit(x,y,1)
print(a,b)
x_theor = np.array([3, 15])
y_theor = a * x_theor + b
_ = plt.plot(x,y, marker='.',linestyle='none')
_ = plt.plot(x_theor,y_theor)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

"""using sampling with replacement (as hacker statistics) to generate multiple scenarios"""
"""taking a sample from a list and then replacing that taken sample, repeating the process"""
"""this is called bootstrapping"""
#each resampled array is called a bootstrap sample
bs_sample = np.random.choice([...,...], size=X, replace=True) #true is default, if set to false it just reorders the array

"""Bootstrap replicate function"""
def bootstrap_replicate(data, func): #where func is the function to perform
	bs_sample = np.random.choice(data, len(data)) # generate the sample
	return func(bs_sample) #return the replicate
def draw_bs_reps(data, func, size=1):
    bs_replicates = np.empty(size)
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data,func)
    return bs_replicates
bs_replicates = draw_bs_reps(data, func, size=1000)
_ = plt.hist(bs_replicates, bins=30, normed=True)
###now to do the same without having to generate a graphical plot
#by computing the 95% confidence interval of the mean
#repeat such that p% of observed values are within p% confidence interval
conf_int = np.percentile(bs_replicates, [2.5,97.5])

#the standard deviation of this distribution is called standard error of mean or SEM
sem = np.std(data) / np.sqrt(len(data)) """SEM is calculated from the original data, not from bs_replicates"""
#so you're comparing the SEM of the data and the STD of the bs_replicates, should be similar

#performing bs replicates for variance of rainfall dataset
bs_replicates = draw_bs_reps(rainfall, np.var, 10000)

"""so far, non parametric inference - only based on data, no assumptions of model / distribution probability underlying data"""
# pairs bootstrap ; makes least assumptions
#because np.choice can only pick from an array, we generate an index array and pick from there
inds = np.arange(len(data))
bs_inds = np.random.choice(inds, len(inds))
bs_res_pair1 = data[bs_inds]
bs_res_pair2 = data2[bs_inds]

bs_slope, bs_intercept = np.polyfit(bs_res_pair1,bs_res_pair2,1)

"""generating more of these ie np.polyfit(bs_res_pair1,bs_res_pair2,10000) lets you see how the slope and intercept can vary"""
def draw_bs_pairs_linreg(x, y, size=1):
    """Perform pairs bootstrap for linear regression."""
    inds = np.arange(0,len(x))
    bs_slope_reps = np.empty(size)
    bs_intercept_reps = np.empty(size)
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x,bs_y,1)
    return bs_slope_reps, bs_intercept_reps

bs_slope_reps, bs_intercept_reps = draw_bs_pairs_linreg(illiteracy, fertility, 1000)
print(np.percentile(bs_slope_reps, [2.5,97.5]))

x = np.array([0,100])
for i in range(100):
    _ = plt.plot(x, 
                 bs_slope_reps[i]*x + bs_intercept_reps[i],
                 linewidth=0.5, alpha=0.2, color='red')
_ = plt.scatter(illiteracy,fertility, marker='.',linestyle='none')
_ = plt.xlabel('illiteracy')
_ = plt.ylabel('fertility')
plt.margins(0.02)
plt.show()

"""using permutation to perform null hypothesis"""
#scrambles the data from two different sets (which we assume are identically distributed) and 'redistributing' them
#this is permutation sampling
def permutation_sample(df1,df2):
	both = np.concatenate((df1,df2))
	both_perm = np.random.permutation(both)
	perm_sample_1 = both_perm[:len(df1)]
	perm_sample_2 = both_perm[len(df1):] #df1 take note
	return perm_sample_1,perm_sample_2

for _ in range(50): #this results in a hazy ecdf range if rainfalls are identically distributed
    perm_sample_1, perm_sample_2 = permutation_sample(rain_june,rain_november)
    x_1, y_1 = ecdf(perm_sample_1)
    x_2, y_2 = ecdf(perm_sample_2)
    _ = plt.plot(x_1, y_1, marker='.', linestyle='none',
                 color='red', alpha=0.02)
    _ = plt.plot(x_2, x_2, marker='.', linestyle='none',
                 color='blue', alpha=0.02)
x_1, y_1 = ecdf(rain_june)
x_2, y_2 = ecdf(rain_november)
_ = plt.plot(x_1, y_1, marker='.', linestyle='none', color='red') #plotting individual distributions
_ = plt.plot(x_2, y_2, marker='.', linestyle='none', color='blue')
plt.margins(0.02)
_ = plt.xlabel('monthly rainfall (mm)')
_ = plt.ylabel('ECDF')
plt.show() #results: none of the observed data fall in the ecdf of the permutation samples

"""hypothesis testing"""
#assessing how reasonable the observed data are assuming hypothesis is true
#test statistic - compute from observed and simulated data
#i.e: difference in mean *of permutated sample* that was generated (this is a permutated replicate)	
#in this distribution, anything above the observed mean difference is used as the p-value
#the probability of getting at least as extreme as your test statistic (assume null hypothesis is true)
#it is not the probability of null hypothesis being true

#if p-value is small, statistically insignificant
#another name we use: NHST - null hypothesis significance testing

#function to draw permutation replicates (from your 
def draw_perm_reps(data_1, data_2, func, size=1):
    """Generate multiple permutation replicates."""
    perm_replicates = np.empty(size)
    for i in range(size):
        perm_sample_1, perm_sample_2 = permutation_sample(data_1,data_2)
        perm_replicates[i] = func(perm_sample_1,perm_sample_2)
    return perm_replicates
	
"""The average strike force of Frog A was 0.71 Newtons (N), and that of Frog B was 0.42 N for a difference of 0.29 N. It is possible the frogs strike with the same force and this observed difference was by chance. You will compute the probability of getting at least a 0.29 N difference in mean strike force under the hypothesis that the distributions of strike forces for the two frogs are identical. We use a permutation test with a test statistic of the difference of means to test this hypothesis."""

def diff_of_means(data_1, data_2):
    diff = np.mean(data_1)-np.mean(data_2)
    return diff
#from experiment: empirical_diff_means
empirical_diff_means = diff_of_means(force_a,force_b)
perm_replicates = draw_perm_reps(force_a, force_b,
                                 diff_of_means, size=10000)
p = np.sum(perm_replicates >= empirical_diff_means) / len(perm_replicates)
print('p-value =', p)

#p-value = 0.0063
"""The p-value tells you that there is about a 0.6% chance that you would get the difference of means observed in the experiment if frogs were exactly the same. A p-value below 0.01 is typically said to be "statistically significant," but: warning! warning! warning! You have computed a p-value; it is a number. I encourage you not to distill it to a yes-or-no phrase. p = 0.006 and p = 0.000000006 are both said to be "statistically significant," but they are definitely not the same!"""

"""
Premise 1
Frogs A has force of 0.29N more than Frogs B
Premise 2
Assume there is no distinction, Frogs A and Frog B hypothetically has no force difference
Premise 3
To achieve force difference of 0.29N is unlikely if there is originally no force difference between frogs A and Frogs B
Result
There should be a force difference between A and B to achieve the 0.29N difference since the no difference case cant achieve 0.29N
[4:09 PM, 10/25/2019] 霖 Lix: so we have two frogs. frog a is hitting harder. but we want to check it's not just by chance. so we reshuffled the values and check for the probability, which is low. so it's can't be by chance.
[4:09 PM, 10/25/2019] 霖 Lix: on the other hand, if they frogs were the same,
[4:10 PM, 10/25/2019] 霖 Lix: force A-ForceB <= 0.29 is actually high
[4:10 PM, 10/25/2019] 霖 Lix: they hit w same impact
[4:10 PM, 10/25/2019] 霖 Lix: but that's not the case so there is a difference between frog a and b
"""

#state null hypothesis
#define test statistic
#generate many permutation sets assuming null hypothesis is true
#compute test statistics for each set
#p value is fraction for those sets where the test statistic is at least that of the real data

"""for newcomb's vs michelson's light experiments, we hypothesize that michelson's mean is the same as newcomb's;
thus, need to perform shifting first of michelson's values
michelson_shifted = michelson_speed_light - np.mean(michelson_speed_light) + newcomb_mean
then, generate permutated samples from the new dataset
"""


#is a one sample test <-

# Make an array of translated impact forces: translated_force_b
translated_force_b = force_b + 0.55 - np.mean(force_b)
# Take bootstrap replicates of Frog B's translated impact forces: bs_replicates
bs_replicates = draw_bs_reps(translated_force_b, np.mean, 10000)
# Compute fraction of replicates that are less than the observed Frog B force: p
p = np.sum(bs_replicates <= np.mean(force_b)) / 10000
# Print the p-value
print('p = ', p)

def draw_bs_pairs(x, y, func, size=1):
    """Perform pairs bootstrap for a single statistic."""
    # Set up array of indices to sample from: inds
    inds = np.arange(len(x))
    # Initialize replicates: bs_replicates
    bs_replicates = np.empty(size)
    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, len(inds))
        bs_x, bs_y = x[i],y[i]
        bs_replicates[i] = func(bs_x,bs_y)
    return bs_replicates