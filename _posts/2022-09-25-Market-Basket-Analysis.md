---
layout: post
title: Market Bakset Analysis
---
Market Basket Analysis (MBA) is the process to identify customers buying habits by finding associations between the different items that customers place in their “shopping baskets”. This analysis is helpful for  retailers or E-Commerce to develop marketing strategies by gaining insight into which items are frequently bought together by customers. ![config.yml](/images/header.jpg)


* TOC
{:toc}



## Introduction

Market Basket Analysis (MBA) is a data mining technique that is use to identify customers buying habits by finding associations between the different items that customers place in their “shopping baskets”. When used appropriately, MBA can be an effective tool for business in understanding consumer behavior better and influence it. 

For example, if customers are buying cookies, how probably are they to also buy milk in the same transaction. This information may lead to increase sales by helping the business by doing **product placement, shelf arrangements, up-sell,cross-sell, and bundling opportunities.**

There are multiple algorithms that can be used in MBA to predict the probability of items that are bought together. 

- AIS
- SETM Algorithm
- Apriori Algorithm
- FP Growth

In this project I would be exploring the Apriori Algorithm.

## The Dataset

The groceries dataset was published in 2020 and can be download in [Kraggle](https://www.kaggle.com/datasets/heeraldedhia/groceries-dataset?resource=download).The dataset has 38765 rows of the purchase orders of people from the grocery stores. These orders can be analysed and association rules can be generated using Market Basket Analysis by algorithms like Apriori Algorithm.


## EDA

In this section we will be reading the dataset and doing some exploratory data analysis


```python
# Import Libaries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import re
# you might need to pip install mlxtend to import the libraries below
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from datetime import datetime, timedelta
from dateutil.parser import parse


```


```python
#import and read the dataset
df_basket = pd.read_csv("Groceries_dataset.csv")
df_basket.head(10)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Member_number</th>
      <th>Date</th>
      <th>itemDescription</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1808</td>
      <td>21-07-2015</td>
      <td>tropical fruit</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2552</td>
      <td>05-01-2015</td>
      <td>whole milk</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2300</td>
      <td>19-09-2015</td>
      <td>pip fruit</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1187</td>
      <td>12-12-2015</td>
      <td>other vegetables</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3037</td>
      <td>01-02-2015</td>
      <td>whole milk</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4941</td>
      <td>14-02-2015</td>
      <td>rolls/buns</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4501</td>
      <td>08-05-2015</td>
      <td>other vegetables</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3803</td>
      <td>23-12-2015</td>
      <td>pot plants</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2762</td>
      <td>20-03-2015</td>
      <td>whole milk</td>
    </tr>
    <tr>
      <th>9</th>
      <td>4119</td>
      <td>12-02-2015</td>
      <td>tropical fruit</td>
    </tr>
  </tbody>
</table>
</div>



Before applaying any algoriths or machine learning techniques, it is important to understand our dataset. 
- Check the shape of the dataset
- Check the data type in each column
- Check for any null values
- Check for duplicate entries
- Plot insight related to our problem



```python
#Check the shape of the dataset
df_basket.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 38765 entries, 0 to 38764
    Data columns (total 3 columns):
     #   Column           Non-Null Count  Dtype 
    ---  ------           --------------  ----- 
     0   Member_number    38765 non-null  int64 
     1   Date             38765 non-null  object
     2   itemDescription  38765 non-null  object
    dtypes: int64(1), object(2)
    memory usage: 908.7+ KB
    

The `.info` function is really useful to getting a quick overview of the dataset. This answer the following questions for our EDA: 
- The dataset contains 38764 rows and 3 columns
- We have two columns with intergers data type and one with object
- we have **Zero** null values 
- The `date` column have is in objective we should change this to Date format. 


```python
#convert to colunm date o date format
df_basket.Date = pd.to_datetime(df_basket.Date)
df_basket.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 38765 entries, 0 to 38764
    Data columns (total 3 columns):
     #   Column           Non-Null Count  Dtype         
    ---  ------           --------------  -----         
     0   Member_number    38765 non-null  int64         
     1   Date             38765 non-null  datetime64[ns]
     2   itemDescription  38765 non-null  object        
    dtypes: datetime64[ns](1), int64(1), object(1)
    memory usage: 908.7+ KB
    


```python
#Double Checking for Null Values 
df_basket.isnull().sum()
```




    Member_number      0
    Date               0
    itemDescription    0
    dtype: int64




```python
#Let's take a look at unique values
df_basket.nunique()
```




    Member_number      3898
    Date                728
    itemDescription     167
    dtype: int64



We can observe the following:
- we have 167 unique items 
- total of 3898 unique customers 


```python
#let check for duplicate rows
df_basket.duplicated().sum()
```




    759



The dataset have a total of 759 duplicate rows. Lets drop this rows. 



```python
# drop duplicates
df_basket = df_basket.drop_duplicates()
#check the shape of the dataset
df_basket.shape

```




    (38006, 3)



This Dataset was mostly clean and it didn't need to much cleaning. Now we can do some plotting to get some insights. 


```python
""""For plotting purpuses,I'm going to create use datetime series and creat colunms of for day of week, days, month, year"""
# copy data frame, using copy would allow us to refer to the orginal dataset later. 
df_time = df_basket.copy()
# create index for time
df_time.index = df_time["Date"]
#add colunms
df_time['date']=df_time["Date"]
df_time['day']=df_time.index.day
df_time['Week']=df_time.index.week
df_time['Month']=df_time.index.month
df_time['Year']=df_time.index.year
#drop Date colunm
df_time = df_time.drop("Date", axis=1)
#check the data set
df_time


```
df_time['Week']=df_time.index.week
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Member_number</th>
      <th>itemDescription</th>
      <th>date</th>
      <th>day</th>
      <th>Week</th>
      <th>Month</th>
      <th>Year</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-07-21</th>
      <td>1808</td>
      <td>tropical fruit</td>
      <td>2015-07-21</td>
      <td>21</td>
      <td>30</td>
      <td>7</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>2015-05-01</th>
      <td>2552</td>
      <td>whole milk</td>
      <td>2015-05-01</td>
      <td>1</td>
      <td>18</td>
      <td>5</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>2015-09-19</th>
      <td>2300</td>
      <td>pip fruit</td>
      <td>2015-09-19</td>
      <td>19</td>
      <td>38</td>
      <td>9</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>2015-12-12</th>
      <td>1187</td>
      <td>other vegetables</td>
      <td>2015-12-12</td>
      <td>12</td>
      <td>50</td>
      <td>12</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>2015-01-02</th>
      <td>3037</td>
      <td>whole milk</td>
      <td>2015-01-02</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2014-08-10</th>
      <td>4471</td>
      <td>sliced cheese</td>
      <td>2014-08-10</td>
      <td>10</td>
      <td>32</td>
      <td>8</td>
      <td>2014</td>
    </tr>
    <tr>
      <th>2014-02-23</th>
      <td>2022</td>
      <td>candy</td>
      <td>2014-02-23</td>
      <td>23</td>
      <td>8</td>
      <td>2</td>
      <td>2014</td>
    </tr>
    <tr>
      <th>2014-04-16</th>
      <td>1097</td>
      <td>cake bar</td>
      <td>2014-04-16</td>
      <td>16</td>
      <td>16</td>
      <td>4</td>
      <td>2014</td>
    </tr>
    <tr>
      <th>2014-03-12</th>
      <td>1510</td>
      <td>fruit/vegetable juice</td>
      <td>2014-03-12</td>
      <td>12</td>
      <td>11</td>
      <td>3</td>
      <td>2014</td>
    </tr>
    <tr>
      <th>2014-12-26</th>
      <td>1521</td>
      <td>cat food</td>
      <td>2014-12-26</td>
      <td>26</td>
      <td>52</td>
      <td>12</td>
      <td>2014</td>
    </tr>
  </tbody>
</table>
<p>38006 rows × 7 columns</p>
</div>




```python
#get days of the week
df_time['weekday'] = df_time['date'].apply(lambda x: parse(str(x)).strftime("%A"))
df_time
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Member_number</th>
      <th>itemDescription</th>
      <th>date</th>
      <th>day</th>
      <th>Week</th>
      <th>Month</th>
      <th>Year</th>
      <th>weekday</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-07-21</th>
      <td>1808</td>
      <td>tropical fruit</td>
      <td>2015-07-21</td>
      <td>21</td>
      <td>30</td>
      <td>7</td>
      <td>2015</td>
      <td>Tuesday</td>
    </tr>
    <tr>
      <th>2015-05-01</th>
      <td>2552</td>
      <td>whole milk</td>
      <td>2015-05-01</td>
      <td>1</td>
      <td>18</td>
      <td>5</td>
      <td>2015</td>
      <td>Friday</td>
    </tr>
    <tr>
      <th>2015-09-19</th>
      <td>2300</td>
      <td>pip fruit</td>
      <td>2015-09-19</td>
      <td>19</td>
      <td>38</td>
      <td>9</td>
      <td>2015</td>
      <td>Saturday</td>
    </tr>
    <tr>
      <th>2015-12-12</th>
      <td>1187</td>
      <td>other vegetables</td>
      <td>2015-12-12</td>
      <td>12</td>
      <td>50</td>
      <td>12</td>
      <td>2015</td>
      <td>Saturday</td>
    </tr>
    <tr>
      <th>2015-01-02</th>
      <td>3037</td>
      <td>whole milk</td>
      <td>2015-01-02</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2015</td>
      <td>Friday</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2014-08-10</th>
      <td>4471</td>
      <td>sliced cheese</td>
      <td>2014-08-10</td>
      <td>10</td>
      <td>32</td>
      <td>8</td>
      <td>2014</td>
      <td>Sunday</td>
    </tr>
    <tr>
      <th>2014-02-23</th>
      <td>2022</td>
      <td>candy</td>
      <td>2014-02-23</td>
      <td>23</td>
      <td>8</td>
      <td>2</td>
      <td>2014</td>
      <td>Sunday</td>
    </tr>
    <tr>
      <th>2014-04-16</th>
      <td>1097</td>
      <td>cake bar</td>
      <td>2014-04-16</td>
      <td>16</td>
      <td>16</td>
      <td>4</td>
      <td>2014</td>
      <td>Wednesday</td>
    </tr>
    <tr>
      <th>2014-03-12</th>
      <td>1510</td>
      <td>fruit/vegetable juice</td>
      <td>2014-03-12</td>
      <td>12</td>
      <td>11</td>
      <td>3</td>
      <td>2014</td>
      <td>Wednesday</td>
    </tr>
    <tr>
      <th>2014-12-26</th>
      <td>1521</td>
      <td>cat food</td>
      <td>2014-12-26</td>
      <td>26</td>
      <td>52</td>
      <td>12</td>
      <td>2014</td>
      <td>Friday</td>
    </tr>
  </tbody>
</table>
<p>38006 rows × 8 columns</p>
</div>




```python
#get the number average of transaction ped month
df_time[['date',"itemDescription"]].groupby('date').count().resample('M').mean().plot()
plt.xlabel("Year", fontsize=10)
plt.ylabel("Count of Transactions", fontsize=10)
plt.title("Average of Transactions per Year by Month", fontsize=14);



```


    
![png](/images/Images_MBA/output_20_0.png)
    


We can observe that business is doing well as their is a trend of transactions been increase over time. |The graph above can help us gain some insight in seasonality. for example, it looks like October trend to be a slower month. 


```python
#plot counts of transaction per year
fig,ax = plt.subplots()
sns.countplot(data=df_time,x="Year")
ax.set(xlabel='Year',title="Number of Transactions per Year");
```


    
![png](/images/Images_MBA/output_22_0.png)
    


2015 have better sales that 2014. 


```python
#plot counts of transaction for day of the Week
fig,ax = plt.subplots()
fig.set_size_inches(10,5)
sns.countplot(data=df_time,x="weekday")
ax.set(xlabel='Day',title="Transactions per day of the week");
```


    
![png](/images/Images_MBA//output_24_0.png)
    


It looks like Wednesday, Thursday, and Sundays are the busiest days. It is important for the business to be well stock in inventory and have full staff during these days. 


```python
#plot top 10 products
fig,ax = plt.subplots()
fig.set_size_inches(15,5)
sns.countplot(data=df_time,x="itemDescription",order=df_time.itemDescription.value_counts().iloc[:10].index)
ax.set(xlabel='Products',title="Top 10 products");
```


    
![png](/images/Images_MBA//output_26_0.png)
    


Understanding the top 10 sellers is beneficial to make sure the business is well stock of this products. In addition, this can be main drivers for people to walk in to a store so they can use this to their advantage to cross sell with other products, improve product placement, and bundling opportunities.


```python
#plot 10 low selling products
fig,ax = plt.subplots()
fig.set_size_inches(18,5)
sns.countplot(data=df_time,x="itemDescription",order=df_time.itemDescription.value_counts().iloc[-10:].index)
ax.set(xlabel='Products',title="Worst 10 products");
```


    
![png](/images/Images_MBA/output_28_0.png)
    


It is important to understand low selling products. We might be able to increase the sales of this product by doing a proper basket analysis. In addition, Business can investigate further if is profitable to carrier these products. 


```python
#plot top 20 customers
fig,ax = plt.subplots()
fig.set_size_inches(18,5)
sns.countplot(data=df_time,x="Member_number",order=df_time.Member_number.value_counts().iloc[:20].index)
ax.set(xlabel='Customer ID',title="Top 20 Customers");
```


    
![png](/images/Images_MBA/output_30_0.png)
    


From the plot above we can observe that top costumers tend two have a total of 35 -30 transaction from 2014 to 2015. 

## Apriori Algorithm

To do the Market Basket analysis i would use the Apriori Algorithm. I won't get into any detail for the math behind the algorithm.[Wikipedia has the exact details on how it works](https://en.wikipedia.org/wiki/Apriori_algorithm).

In the dataset, we can observe that each transaction item is record separately. For example, when a customer buy Whole Milk and Cookies it is recorded on the dataset as two rows.   

In order to do a market basket analysis it is important to group all the items that where purchase on the same transaction together. The best way to do this is by grouping the items by customer number and date.**I would go back to my original Data Frame (df_basket)**



```python
items = df_basket.groupby(['Member_number', 'Date']).agg({'itemDescription': lambda x: x.ravel().tolist()}).reset_index()
items.head()
items.shape
```




    (14963, 3)




```python
# Import the transaction encoder function from mlxtend
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

# Instantiate transaction encoder and identify unique items in transactions
transactions = items["itemDescription"]
encoder = TransactionEncoder().fit(transactions)

# One-hot encode transactions
onehot = encoder.transform(transactions)

# Convert one-hot encoded data to DataFrame
onehot_basket = pd.DataFrame(onehot, columns = encoder.columns_)

# Print the one-hot encoded transaction dataset
onehot_basket.head()
onehot_basket.shape
```




    (14963, 167)



## Determining Rules
Once our data is in the format above, we can begin to determine association rules.

Here, we calculate several metrics to analyse the rules. These are calculated automatically by the package, but we will take time to understand them.

First, all of our groups are designated as 'antecedents' and 'consequents'. This allows us to say: 'given this group of antecedents, we see this group of consequents with frequency x'. We will designate antecedents as  𝑋  and consequents as  𝑌  below.

Let's make some rules for illustration of these measures:



```python
from mlxtend.frequent_patterns import association_rules

"""" we don't have a large enough dataset, so i only used .02 for support.
     For simplicity I used only a max_len of 2, if you want to see more than two
     items you can channge this rule"""

x = apriori(onehot_basket, min_support=.001,max_len=2,use_colnames=True)

#take a look at the help for ways we can use this function
df_rules = association_rules(x, metric="lift", min_threshold=1)
#take a look
df_rules
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>antecedents</th>
      <th>consequents</th>
      <th>antecedent support</th>
      <th>consequent support</th>
      <th>support</th>
      <th>confidence</th>
      <th>lift</th>
      <th>leverage</th>
      <th>conviction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(tropical fruit)</td>
      <td>(UHT-milk)</td>
      <td>0.067767</td>
      <td>0.021386</td>
      <td>0.001537</td>
      <td>0.022682</td>
      <td>1.060617</td>
      <td>8.785064e-05</td>
      <td>1.001326</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(UHT-milk)</td>
      <td>(tropical fruit)</td>
      <td>0.021386</td>
      <td>0.067767</td>
      <td>0.001537</td>
      <td>0.071875</td>
      <td>1.060617</td>
      <td>8.785064e-05</td>
      <td>1.004426</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(brown bread)</td>
      <td>(beef)</td>
      <td>0.037626</td>
      <td>0.033950</td>
      <td>0.001537</td>
      <td>0.040853</td>
      <td>1.203301</td>
      <td>2.597018e-04</td>
      <td>1.007196</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(beef)</td>
      <td>(brown bread)</td>
      <td>0.033950</td>
      <td>0.037626</td>
      <td>0.001537</td>
      <td>0.045276</td>
      <td>1.203301</td>
      <td>2.597018e-04</td>
      <td>1.008012</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(beef)</td>
      <td>(citrus fruit)</td>
      <td>0.033950</td>
      <td>0.053131</td>
      <td>0.001804</td>
      <td>0.053150</td>
      <td>1.000349</td>
      <td>6.297697e-07</td>
      <td>1.000020</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>205</th>
      <td>(yogurt)</td>
      <td>(soft cheese)</td>
      <td>0.085879</td>
      <td>0.010025</td>
      <td>0.001270</td>
      <td>0.014786</td>
      <td>1.474952</td>
      <td>4.088903e-04</td>
      <td>1.004833</td>
    </tr>
    <tr>
      <th>206</th>
      <td>(specialty bar)</td>
      <td>(yogurt)</td>
      <td>0.013968</td>
      <td>0.085879</td>
      <td>0.001203</td>
      <td>0.086124</td>
      <td>1.002863</td>
      <td>3.434701e-06</td>
      <td>1.000269</td>
    </tr>
    <tr>
      <th>207</th>
      <td>(yogurt)</td>
      <td>(specialty bar)</td>
      <td>0.085879</td>
      <td>0.013968</td>
      <td>0.001203</td>
      <td>0.014008</td>
      <td>1.002863</td>
      <td>3.434701e-06</td>
      <td>1.000041</td>
    </tr>
    <tr>
      <th>208</th>
      <td>(specialty chocolate)</td>
      <td>(tropical fruit)</td>
      <td>0.015973</td>
      <td>0.067767</td>
      <td>0.001337</td>
      <td>0.083682</td>
      <td>1.234846</td>
      <td>2.542036e-04</td>
      <td>1.017368</td>
    </tr>
    <tr>
      <th>209</th>
      <td>(tropical fruit)</td>
      <td>(specialty chocolate)</td>
      <td>0.067767</td>
      <td>0.015973</td>
      <td>0.001337</td>
      <td>0.019724</td>
      <td>1.234846</td>
      <td>2.542036e-04</td>
      <td>1.003827</td>
    </tr>
  </tbody>
</table>
<p>210 rows × 9 columns</p>
</div>



## Interpreting Metrics

We have a lot of of metrics in the data frame above and is important to understand this metrics and how to get insight from it.


**Support** allows us to see how often the basket occurs. We don't want to waste our time promoting strong links between items if only a few people buy them.

**Confidence** allows us to see the strength of the rule. What proportion of transactions with our first item also contain the other item (or items)? For example, how true are both items (beef and brown bread) occurred in a transaction together

**Lift** can be interpreted a measure of how much we potentially drive up the sales of the consequent by the relationship? In theory it can be seen as proportional to the increase of sales of the antecedent. For any value higher than 1, lift shows that there is actually an association **Higher Values has generally stronger association**

---
Additional Association Rules: Leverage and Conviction are less common options for assessing the strength of the co-occurrence relationship.

**Leverage** computes the difference between the observed frequency of X and Y appearing together and the frequency that would be expected if X and Y were independent. A leverage value of 0 indicates independence.

The rationale in a sales setting is to find out how many more units (items X and Y together) are sold than expected from the independent sales.

**Conviction** looks at the ratio of the expected frequency that the rule makes an incorrect prediction if X and Y were independent, divided by the observed frequency of incorrect predictions.This is how strongly consequents depend on antecedent. For example, if a customer does not buy beef, they will not buy brown bread.  



Let's take a look at some insight by products by sorting the top 5 items that are shop together


```python
#sort the rules by support
df_rules.sort_values(by='support',ascending = False).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>antecedents</th>
      <th>consequents</th>
      <th>antecedent support</th>
      <th>consequent support</th>
      <th>support</th>
      <th>confidence</th>
      <th>lift</th>
      <th>leverage</th>
      <th>conviction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>198</th>
      <td>(sausage)</td>
      <td>(soda)</td>
      <td>0.060349</td>
      <td>0.097106</td>
      <td>0.005948</td>
      <td>0.098560</td>
      <td>1.014975</td>
      <td>0.000088</td>
      <td>1.001613</td>
    </tr>
    <tr>
      <th>199</th>
      <td>(soda)</td>
      <td>(sausage)</td>
      <td>0.097106</td>
      <td>0.060349</td>
      <td>0.005948</td>
      <td>0.061253</td>
      <td>1.014975</td>
      <td>0.000088</td>
      <td>1.000963</td>
    </tr>
    <tr>
      <th>201</th>
      <td>(yogurt)</td>
      <td>(sausage)</td>
      <td>0.085879</td>
      <td>0.060349</td>
      <td>0.005748</td>
      <td>0.066926</td>
      <td>1.108986</td>
      <td>0.000565</td>
      <td>1.007049</td>
    </tr>
    <tr>
      <th>200</th>
      <td>(sausage)</td>
      <td>(yogurt)</td>
      <td>0.060349</td>
      <td>0.085879</td>
      <td>0.005748</td>
      <td>0.095238</td>
      <td>1.108986</td>
      <td>0.000565</td>
      <td>1.010345</td>
    </tr>
    <tr>
      <th>128</th>
      <td>(frankfurter)</td>
      <td>(other vegetables)</td>
      <td>0.037760</td>
      <td>0.122101</td>
      <td>0.005146</td>
      <td>0.136283</td>
      <td>1.116150</td>
      <td>0.000536</td>
      <td>1.016420</td>
    </tr>
  </tbody>
</table>
</div>



The product sausage an soda are the items with the higher support. Lets take a look **Sausage** to gain more  this insight can be applied to any products but for simplicity I will only do one. 


```python
#Sort the dataset
sausage_insight = df_rules[df_rules['consequents'].astype(str).str.contains('sausage')]
sausage_insight = milk_insight.sort_values(by=['lift'],ascending = [False]).reset_index(drop = True)

sausage_insight.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>antecedents</th>
      <th>consequents</th>
      <th>antecedent support</th>
      <th>consequent support</th>
      <th>support</th>
      <th>confidence</th>
      <th>lift</th>
      <th>leverage</th>
      <th>conviction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(beverages)</td>
      <td>(sausage)</td>
      <td>0.016574</td>
      <td>0.060349</td>
      <td>0.001537</td>
      <td>0.092742</td>
      <td>1.536764</td>
      <td>0.000537</td>
      <td>1.035704</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(curd)</td>
      <td>(sausage)</td>
      <td>0.033683</td>
      <td>0.060349</td>
      <td>0.002941</td>
      <td>0.087302</td>
      <td>1.446615</td>
      <td>0.000908</td>
      <td>1.029531</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(frozen vegetables)</td>
      <td>(sausage)</td>
      <td>0.028002</td>
      <td>0.060349</td>
      <td>0.002072</td>
      <td>0.073986</td>
      <td>1.225966</td>
      <td>0.000382</td>
      <td>1.014726</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(bottled beer)</td>
      <td>(sausage)</td>
      <td>0.045312</td>
      <td>0.060349</td>
      <td>0.003342</td>
      <td>0.073746</td>
      <td>1.222000</td>
      <td>0.000607</td>
      <td>1.014464</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(yogurt)</td>
      <td>(sausage)</td>
      <td>0.085879</td>
      <td>0.060349</td>
      <td>0.005748</td>
      <td>0.066926</td>
      <td>1.108986</td>
      <td>0.000565</td>
      <td>1.007049</td>
    </tr>
  </tbody>
</table>
</div>



We can observe that beverage,curd,frozen vegetables,bottled beer, and yogurt drive the sales of sausage. Running promos and discount on these items can increase sales for sausages. This can be good insight if sausages are at the end of life and the stores want to get rid of them. 


```python
df 
df_rules.sort_values(by=['lift'],ascending = [False]).reset_index(drop = True).head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>antecedents</th>
      <th>consequents</th>
      <th>antecedent support</th>
      <th>consequent support</th>
      <th>support</th>
      <th>confidence</th>
      <th>lift</th>
      <th>leverage</th>
      <th>conviction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(specialty chocolate)</td>
      <td>(citrus fruit)</td>
      <td>0.015973</td>
      <td>0.053131</td>
      <td>0.001403</td>
      <td>0.087866</td>
      <td>1.653762</td>
      <td>0.000555</td>
      <td>1.038081</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(citrus fruit)</td>
      <td>(specialty chocolate)</td>
      <td>0.053131</td>
      <td>0.015973</td>
      <td>0.001403</td>
      <td>0.026415</td>
      <td>1.653762</td>
      <td>0.000555</td>
      <td>1.010726</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(tropical fruit)</td>
      <td>(flour)</td>
      <td>0.067767</td>
      <td>0.009757</td>
      <td>0.001069</td>
      <td>0.015779</td>
      <td>1.617141</td>
      <td>0.000408</td>
      <td>1.006118</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(flour)</td>
      <td>(tropical fruit)</td>
      <td>0.009757</td>
      <td>0.067767</td>
      <td>0.001069</td>
      <td>0.109589</td>
      <td>1.617141</td>
      <td>0.000408</td>
      <td>1.046969</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(beverages)</td>
      <td>(sausage)</td>
      <td>0.016574</td>
      <td>0.060349</td>
      <td>0.001537</td>
      <td>0.092742</td>
      <td>1.536764</td>
      <td>0.000537</td>
      <td>1.035704</td>
    </tr>
    <tr>
      <th>5</th>
      <td>(sausage)</td>
      <td>(beverages)</td>
      <td>0.060349</td>
      <td>0.016574</td>
      <td>0.001537</td>
      <td>0.025471</td>
      <td>1.536764</td>
      <td>0.000537</td>
      <td>1.009129</td>
    </tr>
    <tr>
      <th>6</th>
      <td>(napkins)</td>
      <td>(pastry)</td>
      <td>0.022121</td>
      <td>0.051728</td>
      <td>0.001738</td>
      <td>0.078550</td>
      <td>1.518529</td>
      <td>0.000593</td>
      <td>1.029109</td>
    </tr>
    <tr>
      <th>7</th>
      <td>(pastry)</td>
      <td>(napkins)</td>
      <td>0.051728</td>
      <td>0.022121</td>
      <td>0.001738</td>
      <td>0.033592</td>
      <td>1.518529</td>
      <td>0.000593</td>
      <td>1.011869</td>
    </tr>
    <tr>
      <th>8</th>
      <td>(processed cheese)</td>
      <td>(root vegetables)</td>
      <td>0.010158</td>
      <td>0.069572</td>
      <td>0.001069</td>
      <td>0.105263</td>
      <td>1.513019</td>
      <td>0.000363</td>
      <td>1.039891</td>
    </tr>
    <tr>
      <th>9</th>
      <td>(root vegetables)</td>
      <td>(processed cheese)</td>
      <td>0.069572</td>
      <td>0.010158</td>
      <td>0.001069</td>
      <td>0.015370</td>
      <td>1.513019</td>
      <td>0.000363</td>
      <td>1.005293</td>
    </tr>
  </tbody>
</table>
</div>



Lastly, I want to get insight of the products that have a high confidence and the highest lift scores. 



```python
# lets check some basic stats of our rules. 
df_rules.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>antecedent support</th>
      <th>consequent support</th>
      <th>support</th>
      <th>confidence</th>
      <th>lift</th>
      <th>leverage</th>
      <th>conviction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>210.000000</td>
      <td>210.000000</td>
      <td>210.000000</td>
      <td>210.000000</td>
      <td>210.000000</td>
      <td>2.100000e+02</td>
      <td>210.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.045238</td>
      <td>0.045238</td>
      <td>0.001666</td>
      <td>0.052092</td>
      <td>1.165016</td>
      <td>1.937097e-04</td>
      <td>1.007440</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.031016</td>
      <td>0.031016</td>
      <td>0.000928</td>
      <td>0.035145</td>
      <td>0.151203</td>
      <td>1.700523e-04</td>
      <td>0.009483</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.007084</td>
      <td>0.007084</td>
      <td>0.001002</td>
      <td>0.008210</td>
      <td>1.000136</td>
      <td>5.091755e-07</td>
      <td>1.000005</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.023592</td>
      <td>0.023592</td>
      <td>0.001136</td>
      <td>0.026415</td>
      <td>1.044134</td>
      <td>6.499581e-05</td>
      <td>1.001688</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.037359</td>
      <td>0.037359</td>
      <td>0.001403</td>
      <td>0.041638</td>
      <td>1.116675</td>
      <td>1.418098e-04</td>
      <td>1.003983</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.060349</td>
      <td>0.060349</td>
      <td>0.001671</td>
      <td>0.069275</td>
      <td>1.246844</td>
      <td>2.692913e-04</td>
      <td>1.009114</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.157923</td>
      <td>0.157923</td>
      <td>0.005948</td>
      <td>0.176056</td>
      <td>1.653762</td>
      <td>9.078510e-04</td>
      <td>1.046969</td>
    </tr>
  </tbody>
</table>
</div>




```python
#sort by confidence and then lift
df_rules.sort_values(['confidence', 'lift'], ascending=[False,False ], inplace=True)
df_rules.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>antecedents</th>
      <th>consequents</th>
      <th>antecedent support</th>
      <th>consequent support</th>
      <th>support</th>
      <th>confidence</th>
      <th>lift</th>
      <th>leverage</th>
      <th>conviction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>203</th>
      <td>(semi-finished bread)</td>
      <td>(whole milk)</td>
      <td>0.009490</td>
      <td>0.157923</td>
      <td>0.001671</td>
      <td>0.176056</td>
      <td>1.114825</td>
      <td>0.000172</td>
      <td>1.022008</td>
    </tr>
    <tr>
      <th>113</th>
      <td>(detergent)</td>
      <td>(whole milk)</td>
      <td>0.008621</td>
      <td>0.157923</td>
      <td>0.001403</td>
      <td>0.162791</td>
      <td>1.030824</td>
      <td>0.000042</td>
      <td>1.005814</td>
    </tr>
    <tr>
      <th>146</th>
      <td>(ham)</td>
      <td>(whole milk)</td>
      <td>0.017109</td>
      <td>0.157923</td>
      <td>0.002740</td>
      <td>0.160156</td>
      <td>1.014142</td>
      <td>0.000038</td>
      <td>1.002659</td>
    </tr>
    <tr>
      <th>180</th>
      <td>(processed cheese)</td>
      <td>(rolls/buns)</td>
      <td>0.010158</td>
      <td>0.110005</td>
      <td>0.001470</td>
      <td>0.144737</td>
      <td>1.315734</td>
      <td>0.000353</td>
      <td>1.040610</td>
    </tr>
    <tr>
      <th>176</th>
      <td>(packaged fruit/vegetables)</td>
      <td>(rolls/buns)</td>
      <td>0.008488</td>
      <td>0.110005</td>
      <td>0.001203</td>
      <td>0.141732</td>
      <td>1.288421</td>
      <td>0.000269</td>
      <td>1.036967</td>
    </tr>
    <tr>
      <th>186</th>
      <td>(seasonal products)</td>
      <td>(rolls/buns)</td>
      <td>0.007084</td>
      <td>0.110005</td>
      <td>0.001002</td>
      <td>0.141509</td>
      <td>1.286395</td>
      <td>0.000223</td>
      <td>1.036698</td>
    </tr>
    <tr>
      <th>128</th>
      <td>(frankfurter)</td>
      <td>(other vegetables)</td>
      <td>0.037760</td>
      <td>0.122101</td>
      <td>0.005146</td>
      <td>0.136283</td>
      <td>1.116150</td>
      <td>0.000536</td>
      <td>1.016420</td>
    </tr>
    <tr>
      <th>175</th>
      <td>(pot plants)</td>
      <td>(other vegetables)</td>
      <td>0.007819</td>
      <td>0.122101</td>
      <td>0.001002</td>
      <td>0.128205</td>
      <td>1.049991</td>
      <td>0.000048</td>
      <td>1.007002</td>
    </tr>
    <tr>
      <th>130</th>
      <td>(frozen meals)</td>
      <td>(other vegetables)</td>
      <td>0.016775</td>
      <td>0.122101</td>
      <td>0.002139</td>
      <td>0.127490</td>
      <td>1.044134</td>
      <td>0.000090</td>
      <td>1.006176</td>
    </tr>
    <tr>
      <th>185</th>
      <td>(red/blush wine)</td>
      <td>(rolls/buns)</td>
      <td>0.010493</td>
      <td>0.110005</td>
      <td>0.001337</td>
      <td>0.127389</td>
      <td>1.158028</td>
      <td>0.000182</td>
      <td>1.019922</td>
    </tr>
  </tbody>
</table>
</div>



## Findings and Conclusions

From the above we can observe the following: 

- Whole milk, rolls/buns and other vegetables tend to be frequently add on items. 
- We can place consequent items if possible next to the antecedent items to drive sales. 
- For items that can be place next to each other , like detergent and whole milk we can make sure that the layout of the stores are close to each others. . 

Using Apriori algorithm is a very useful technique to find associations between items. In addition, They are easy to implement and explain. However for more complex insights, such as the ones been used by Amazon, Google, Netflix we can use recommendation systems. 
