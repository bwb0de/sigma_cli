import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison

# Create four random groups of data with a mean difference of 1

mu, sigma = 10, 3 # mean and standard deviation
#group1 = np.random.normal(mu, sigma, 50)
group1 = np.array([6,7,8,6,4])

mu, sigma = 11, 3 # mean and standard deviation
#group2 = np.random.normal(mu, sigma, 50)
group2 = np.array([2,5,4,3,5])

mu, sigma = 12, 3 # mean and standard deviation
#group3 = np.random.normal(mu, sigma, 50)
group3 = np.array([3,2,4,4,3])

mu, sigma = 13, 3 # mean and standard deviation
#group4 = np.random.normal(mu, sigma, 50)

# Show the results for Anova

F_statistic, pVal = stats.f_oneway(group1, group2, group3)#, group4)

print(group1)

print ('P value:')
print (pVal)

#For the multicomparison tests we will put the data into a dataframe. And then reshape it to a stacked dataframe

# Put into dataframe

df = pd.DataFrame()
df['treatment1'] = group1
df['treatment2'] = group2
df['treatment3'] = group3
#df['treatment4'] = group4

# Stack the data (and rename columns):

stacked_data = df.stack().reset_index()
stacked_data = stacked_data.rename(columns={'level_0': 'id',
                                            'level_1': 'treatment',
                                            0:'result'})
# Show the first 8 rows:

print(stacked_data.head(8))

# This method tests at P<0.05 (correcting for the fact that multiple comparisons are being made which would normally increase the probability of a significant difference being identified). A results of ’reject = True’ means that a significant difference has been observed.

# Set up the data for comparison (creates a specialised object)
MultiComp = MultiComparison(stacked_data['result'], stacked_data['treatment'])

# Show all pair-wise comparisons:
  # Print the comparisons

print(MultiComp.tukeyhsd().summary())


def group_harmonic_mean(iterator):
	arr = np.array(iterator, dtype='uint32')
	k = arr.size
	d = (1 / arr).sum()
	return k/d

