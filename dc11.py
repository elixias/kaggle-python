"""Crosstab"""
#immediately gives the values for each category
pd.crosstab(ri.driver_race, ri.driver_gender)

# Calculate the mean 'stop_minutes' for each value in 'violation_raw'
print(ri.groupby('violation_raw').stop_minutes.mean())
# Save the resulting Series as 'stop_length'
stop_length = ri.groupby('violation_raw').stop_minutes.mean()
# Sort 'stop_length' by its values and create a horizontal bar plot
stop_length.sort_values().plot(kind='barh')
# Display the plot
plt.show()

# Copy 'WT01' through 'WT22' to a new DataFrame
WT = weather.loc[:,'WT01':'WT22']
# Calculate the sum of each row in 'WT'
weather['bad_conditions'] = WT.sum(axis='columns') #sum all the columns from WT01 to WT22
 Replace missing values in 'bad_conditions' with '0'
weather['bad_conditions'] = weather.bad_conditions.fillna(0).astype('int')
# Create a histogram to visualize 'bad_conditions'
weather['bad_conditions'].plot(kind='hist')
# Display the plot
plt.show()

# Change the data type of 'rating' to category
weather['rating'] = weather.rating.astype('category', categories=cats, ordered=True) #if not in cats, autofill with nan