


# Core Libraries - Data manipulation and analysis
import pandas as pd
import numpy as np
import math
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
from scipy.cluster.vq import kmeans,vq
from numpy import vstack,array

# Core Libraries - Machine Learning
import sklearn
 
# Importing Clustering - Modelling
from sklearn.cluster import KMeans


# Warnings Library - Ignore warnings
import warnings
warnings.filterwarnings('ignore')





get_ipython().run_line_magic('matplotlib', 'inline')


# Load Data




stock_data = pd.read_csv('data_stocks.csv')
stock_data.head()


# Understand the Dataset and Data




stock_data.shape





stock_data.columns





stock_data.head()





stock_data.tail()





stock_data.info()





stock_data.get_dtype_counts()


# Clean the data
# 




print(stock_data.columns.values)


# Since we are unsure of the names of the companies and the stocks column names associated with them, it would be better to refrain from changing the column names. However, I guess we could replace the '.' in the column names with '_'. But we won't be doing that.

#  Clean Numerical Columns
# 

# Null values




# Check if there are any null values. Since there are many columns, total sum of rows or elements which are null 
# across all columns tells us whether we need to consider to impute or remove those rows
stock_data.isna().sum().sum()


# There are no null values

# 0s 




# Check if there are any 0 values. Since there are many columns, total sum of rows or elements which have '0' values 
# across all columns tells us whether we need to consider to impute or clean/correct the data in those rows or elements
stock_data[stock_data==0].any().sum()
       # OR
# data.loc[(data==0).any(axis=1),:].sum().sum()


# There are no rows or elements with '0' values 

# Nonsensical values




# Check if there are any -ve values. Since there are many columns, total sum of rows or elements which have '-ve' values 
# across all columns tells us whether we need to consider to impute or clean/correct the data in those rows or elements
stock_data[stock_data<0].any().sum()
       # OR
# stock_data.loc[(data<0).any(axis=1),:].sum().sum()


# There are no rows or elements with '-ve' values


# Get Basic Statistical Information




stock_data.describe()


# Explore Data - Non-Visual and Visual Analysis

#  Uni-variate




stock_data.iloc[:,1:11].hist(bins=100, figsize=(20,20), layout=(5,2)) # Plotting the first 10 columns excluding the date


# The distribution of values of stocks of the 10 companies, do not follow any specfic distribution,like normal distribution, multinomial distribution etc.

#  Time Series plot for 1st 10 stock columns in the dataset by date

# Time Series Plot using Epoch time




plt.figure(figsize= (20,20))
for i,col in enumerate(stock_data.iloc[:,0:11].columns.values):
    if i==0:
        continue
    else:
        plt.figure(figsize= (30,5))
        plt.title(col + ' stock')
        plt.xlabel('Date')
        plt.ylabel('Stock value ' + str(col))
        plt.xticks(rotation=90)
        sns.lineplot(stock_data.iloc[:,0],stock_data.iloc[:,i])


# Time Series Plot using Date Only




import datetime as dt
plt.figure(figsize= (20,20))
for i,col in enumerate(stock_data.iloc[:,0:10].columns.values):
    if i==0:
        continue
    else:
        plt.figure(figsize= (30,5))
        plt.title(col + ' stock')
        plt.xlabel('Date')
        plt.ylabel('Stock value ' + str(col))
        plt.xticks(rotation=90)
        sns.lineplot(stock_data.iloc[:,0].apply(lambda x:  dt.datetime.fromtimestamp(x).strftime('%Y-%m-%d')),
                     stock_data.iloc[:,i],)





date_df = pd.DataFrame()
date_df['date'] = stock_data.iloc[:,0].apply(lambda x:  dt.datetime.fromtimestamp(x).strftime('%Y-%m-%d'))
date_df['time'] = stock_data.iloc[:,0].apply(lambda x:  dt.datetime.fromtimestamp(x).strftime('%X'))
date_df.head()





date_df.groupby(by='date').count().shape # No. of days the data was taken





stock_prices = stock_data.drop('DATE', axis =1)


# In order to find out the optimal number of clusters upon which we can train our data, we need to find the least number of clusters where the change in Sum of Squared Errors (SSE) starts flatlining(becomes asymptotic) when we plot the SSE errors for different number of clusters according to the elbow method




# Calculate average annual percentage return and volatilities over the time period of the data in the dataset
performance = stock_prices.pct_change().mean() * 128
performance = pd.DataFrame(performance)
performance.columns = ['Performance']
performance['Volatility'] = stock_prices.pct_change().std() * sqrt(128)

#format the data as a numpy array to feed into the K-Means algorithm
data = np.asarray([np.asarray(performance['Performance']),np.asarray(performance['Volatility'])]).T

X = data
wc_sse = []
for k in range(2, 20):
    k_means = KMeans(n_clusters=k)
    k_means.fit(X)
    wc_sse.append(k_means.inertia_)

fig = plt.figure(figsize=(15, 5))
plt.plot(range(2, 20), wc_sse)
plt.grid(True)
plt.title('Elbow curve')





# computing K-Means with K = 6 (6 clusters)
centroids,_ = kmeans(data,6)
# assign each sample to a cluster
idx,_ = vq(data,centroids)

# some plotting using numpy's logical indexing
plt.plot(data[idx==0,0],data[idx==0,1],'ob',
     data[idx==1,0],data[idx==1,1],'oy',
     data[idx==2,0],data[idx==2,1],'or',
     data[idx==3,0],data[idx==3,1],'og',
     data[idx==4,0],data[idx==4,1],'om',
     data[idx==5,0],data[idx==5,1],'oc')

plt.plot(centroids[:,0],centroids[:,1],'sg',markersize=8)





#identify the outlier
print(performance.idxmax())





#drop the relevant stock from our data
performance.drop('NYSE.XRX',inplace=True)

#recreate data to feed into the algorithm
data = np.asarray([np.asarray(performance['Performance']),np.asarray(performance['Volatility'])]).T





# computing K-Means with K = 6(6 clusters)
centroids,_ = kmeans(data,6)
# assign each sample to a cluster
idx,_ = vq(data,centroids)

# some plotting using numpy's logical indexing
plt.plot(data[idx==0,0],data[idx==0,1],'ob',
     data[idx==1,0],data[idx==1,1],'oy',
     data[idx==2,0],data[idx==2,1],'or',
     data[idx==3,0],data[idx==3,1],'og',
     data[idx==4,0],data[idx==4,1],'om',
     data[idx==5,0],data[idx==5,1],'oc')

plt.plot(centroids[:,0],centroids[:,1],'sg',markersize=8)


#  Problem 1:   There are various stocks for which we have collected a data set, which all stocks are apparently similar in performance?




similar_performance = [(name,cluster) for name, cluster in zip(performance.index.values,idx)]
similar_performance





plt.figure(figsize = (20,200))
plt.title('Company Stocks Vs Clusters\n')
plt.ylim(-2,len(performance.index.values)+1)
plt.xlabel('Cluster Number')
plt.ylabel('Company Stocks')
plt.gca().invert_yaxis()
plt.grid(axis='both', alpha= 0.1)

sns.scatterplot(x=idx, y= performance.index.values, hue = idx,palette= 'Dark2', legend= False)


# Problem 2: How many Unique patterns that exist in the historical stock data set, based on fluctuations in price?

# Unique patterns that exist in the historical stock data set = 6

# Problem 3: Identify which all stocks are moving together and which all stocks are different from each other?




moving_together = [(name,cluster) for name, cluster in zip(performance.index.values,idx)]
moving_together

