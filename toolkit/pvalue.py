#coding:utf8
'''
如果是检验问题，p值反映的是样本数据支持原假设的证据，p值越大，证据越强。

p值就是在原假设下，该总体出现现有数据的概率，或者说在现有数据下，原假设成立的一种合理性，p值越大，原假设成立的可能性就越大。 
p值越少，就说明原假设成立的可能性越小。通常当p值小于0.05时，就认为原假设不成立。

Refer:https://medium.com/@peilee_98185/t-%E6%AA%A2%E5%AE%9A-with-python-443c2364b071
'''
import numpy as np
from scipy import stats

def t_test(group1, group2):
	mean1 = np.mean(group1)
	mean2 = np.mean(group2)
	std1 = np.std(group1)
	std2 = np.std(group2)
	nobs1 = len(group1)
	nobs2 = len(group2)
	
	modified_std1 = np.sqrt(np.float32(nobs1)/
					np.float32(nobs1-1)) * std1
	modified_std2 = np.sqrt(np.float32(nobs2)/
					np.float32(nobs2-1)) * std2
	(statistic, pvalue) = stats.ttest_ind_from_stats( 
			   mean1=mean1, std1=modified_std1, nobs1=nobs1,   
			   mean2=mean2, std2=modified_std2, nobs2=nobs2 )
	return statistic, pvalue

## Define 2 random distributions
#Sample Size
N= 10
#Gaussian distributed data with mean = 2 and var = 1
a= np.random.randn(N)+ 2
#Gaussian distributed data with with mean = 0 and var = 1
b= np.random.randn(N)

s,p = t_test(a, b)
print("s={0},p={1}\n".format(s,p))
## Calculate the Standard Deviation
#Calculate the variance to get the standard deviation

#For unbiased max likelihood estimate we have to divide the var by N-1, and therefore the parameter ddof = 1
var_a= a.var(ddof=1)
var_b= b.var(ddof=1)

#std deviation
s= np.sqrt((var_a+ var_b)/2)

## Calculate the t-statistics
t= (a.mean()- b.mean())/(s*np.sqrt(2.0/N))



## Compare with the critical t-value
#Degrees of freedom
df= 2*N- 2

#p-value after comparison with the t
p= 1.0 - stats.t.cdf(t,df=df)


print("t = " + str(t))
print("p = " + str(2*p))
#Note that we multiply the p value by 2 because its a twp tail t-test
### You can see that after comparing the t statistic with the critical t value (computed internally) we get a good p value of 0.0005 and thus we reject the null hypothesis and thus it proves that the mean of the two distributions are different and statistically significant.


## Cross Checking with the internal scipy function
t2, p2= stats.ttest_ind(a,b)
print("t = " + str(t2))
print("p = " + str(p2))
