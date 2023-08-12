#!/usr/bin/env python
# coding: utf-8

# # About Dataset

# - The data refers to Daily prices of various commodities in India of all the states and districts. It has the wholesale maximum price, minimum price and modal price on daily basis. This dataset is generated through the AGMARKNET Portal (http://agmarknet.gov.in), which disseminates daily market information of various commodities.
# 
# According to the AGMARKNET Portal, the prices in the dataset refer to the wholesale prices of various commodities per quintal (100 kg) in Indian rupees. The wholesale price is the price at which goods are sold in large quantities to retailers or distributors.
# 
# - Features of the dataset include:
# 
# - State: The state in India where the market is located.
# - District: The district in India where the market is located.
# - Market: The name of the market.
# - Commodity: The name of the commodity.
# - Variety: The variety of the commodity.
# - Grade: The grade or quality of the commodity.
# - Min Price: (INR) The minimum wholesale price of the commodity on a given day, per quintal (100 kg).
# - Max Price: (INR) The maximum wholesale price of the commodity on a given day, per quintal (100 kg).
# - Modal Price: (INR) The most common or representative wholesale price of the commodity on a given day, per quintal (100 kg).
# - 1 INR = 0.012 USD (as on 7 July, 2023)

# # Market analysis: 
# You can use this dataset to analyze trends and patterns in the wholesale prices of various commodities across different markets in India. This can help you understand factors that affect prices, such as supply and demand, seasonality, and market conditions.
# # Commodity recommendation:
# Develop recommender systems that suggest the best markets or commodities for farmers or traders to sell or buy based on their location, preferences, and market conditions.
# 
# 

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[2]:


#load dataset
data=pd.read_csv('daily_price.csv')


# In[3]:


data.head(10)


# In[4]:


#information about the dataset
data.info()


# In[5]:


# how many rows and columns
data.shape


# In[6]:


#about the dataset variables
data.describe()


# In[7]:


#find null values sum
data.isnull().sum()



# In[8]:


# find unique value of each variables
data.nunique()


# # Visualiztion

# In[9]:


#imort some more libraries
import seaborn as sns
import matplotlib.pyplot as plt


# # Distribution of Data points by State

# In[10]:


#distribution of data points by state
sns.countplot(data=data, x='State')
plt.xlabel('State')
plt.ylabel('Count')
plt.title('Distribution of Data Points by State')
plt.xticks(rotation=90)
plt.show()


# # INSIGHT
# - It shows that State 'Uttar Pradesh' has the highest number of market. 'Kerala' and 'Punjab' are second and third respectively.

# # Distribution of Data Points by Commodity

# In[11]:


df=sns.set(font_scale=0.5)
df=plt.figure(figsize=(20, 10))  
df=sns.countplot(data=data, x='Commodity')
df=plt.xlabel('Commodity')
df=plt.ylabel('Count')
df=plt.title('Distribution of Data Points by Commodity')
df=plt.xticks(rotation=90)
df=plt.show()


# In[12]:


data['Commodity'].value_counts()


# # Insight
# - It shows that Potato is highest commodity in demand and supply. Onion is the second one, Tomato and Green Chilli forth and fifth respectively.

# In[13]:


#distribution of data by district
df=sns.set(font_scale=0.5)
df=plt.figure(figsize=(20, 10))  
df=sns.countplot(data=data, x='District')
df=plt.xlabel('Commodity')
df=plt.ylabel('Count')
df=plt.title('Distribution of Data Points by District')
df=plt.xticks(rotation=90)
df=plt.show()


# In[14]:


data['District'].value_counts()


# In[ ]:





# # Insight
# - It shows that Palakad (Kerala) district has more markets than others, Kangra (Himachal Pradesh) placed at second in terms of markets Bulandshahar (Uttar Pradesh) is third one.

# In[15]:


#famous market for commodity
data['Market'].value_counts()


# # Insight
# - Palakkad district os highest number of Market.

# In[16]:


data['Variety'].value_counts()


# # Insight
# - Green Chilly is highest among of all variety.

# # Price Distribution

# In[17]:


#distribution of min price, max price and modal price through pie chart
from seaborn._core.properties import FontSize
labels='Min Price', 'Max Price', 'Modal Price'
size =[215,130, 245]
colors = ['gold', 'yellowgreen', 'lightcoral']
textprops={"fontsize":18}
explode= (0.1, 0, 0)
#plot
plt.pie(size, explode= explode, labels=labels, colors=colors, shadow= True, autopct='%1.1f%%', textprops=textprops )
plt.axis('equal')
plt.show()


# # Insight
# - It shows that Modal Price has the highest percentage (The most common or representative wholesale price of the commodity on a given day, per quintal (100 kg)). Max Price (22%) is the lowest wholesale price of commodity. overall shows that mainly wholesale price fluctuate between 'Modal Price' or 'Min Price', less preferable of 'Max Price'.

# In[18]:


#find outliers using boxplot
sns.set(font_scale=1)
price_data =data[['Min Price','Max Price','Modal Price']]
sns.boxplot(data=price_data)
plt.xlabel('Price Type')
plt.ylabel('Price')
plt.title('Price Distribution')
plt.show()


# # Insight
# - It show that all three prices have outliers which are located above the maximum thus can affect the overall observation

# # Scatter plot of Min and Max Price

# In[19]:


plt.figure(figsize = (10,6))
sns.scatterplot(data =data, x='Min Price', y='Max Price', color='blue')
plt.xlabel('Min Price')
plt.ylabel('Max Price')
plt.title('Scatter plot of Min and Max Price')
plt.show()


# # Insight
# - scatterplot shows the distribution of data between two numerical value there is more clustter of both price lies below 40000 in both cases i.e 'Max Price' and 'Min Price'

# # Model Price Distribution Vs Frequency

# In[20]:


plt.figure(figsize=(8,5))
sns.histplot(data=data, x='Modal Price', bins=15)
plt.xlabel('Modal Price')
plt.ylabel('Frequency')
plt.title('Model Price Distribution vs Frequency')
plt.show()


# # Insight
# - It show that the modal price distribution is mostly below 20000 and has highest frequency above 3500.

# # Highest and Lowest Modal Price

# In[21]:


data=top_10_highest = data.nlargest(10,'Modal Price')
data=top_10_lowest = data.nsmallest(10,'Modal Price')



# In[22]:


print(top_10_highest)
print(top_10_lowest)


# # Highest and Lowest Prices State - Wise

# In[23]:


plt.figure(figsize=(10,6))
plt.subplot(1,2,1)
sns.barplot(data=top_10_highest, y='Modal Price', x= 'State')
plt.xlabel('State')
plt.ylabel('Modal Price')
plt.title('Top 10 Highest Prices State - Wise')


# # Insight
# - state Kerala has highest 'Modal Price',whereas Gujrat, NCT of Delhi and Rajasthan have almost eqaual distribution of 'Modal Price'.

# In[24]:


plt.figure(figsize=(10,6))
plt.subplot(1,2,1)
sns.barplot(data=top_10_lowest, y='Modal Price', x= 'State')
plt.xlabel('State')
plt.ylabel('Modal Price')
plt.title('Top 10 Lowest Prices State - Wise')


# # Insight
# - It shows that Kattappana (Kerala) market contribute the highest number of modal price whereas rest market contribution are almost equal.

# # Highest and lowest Prices by Market

# In[25]:


plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
sns.barplot(data=top_10_highest, x='Modal Price', y='Market')
plt.xlabel('Modal Price')
plt.ylabel('Market')
plt.title('Top 10 Highest Prices by Market')

plt.subplot(1, 2, 2)
sns.barplot(data=top_10_lowest, x='Modal Price', y='Market')
plt.xlabel('Modal Price')
plt.ylabel('Market')
plt.title('Top 10 Lowest Prices by Market')

plt.tight_layout()
plt.show()


# # Highest and Lowest Price by Variety

# In[26]:


plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.barplot(data=top_10_highest, x='Modal Price',y ='Variety', hue='Commodity')
plt.xlabel('Modal Price')
plt.ylabel('Variety')
plt.title('Top 10 Highest Prices by Variety')


# In[27]:


plt.figure(figsize=(10,5))
plt.subplot(1,2,2)
sns.barplot(data=top_10_lowest, x='Modal Price',y ='Variety', hue='Commodity')
plt.xlabel('Modal Price')
plt.ylabel('Variety')
plt.title('Top 10 Lowest Prices by Variety')


# # Highest and Lowest Prices by Commodity

# In[28]:


plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.barplot(data=top_10_highest, x='Modal Price',y='Commodity')
plt.xlabel('Modal Price')
plt.ylabel('Commodity')
plt.title('Top 10 Highest Prices by Commodity')

plt.subplot(1,2,2)
sns.barplot(data=top_10_lowest, x='Modal Price',y ='Commodity')
plt.xlabel('Modal Price')
plt.ylabel('Commodity')
plt.title('Top 10 Highest Prices by Commodity')

plt.tight_layout()
plt.show()


# # Modal Price Distribution

# In[29]:


plt.figure(figsize=(8,5))
sns.violinplot(data=data, y='Modal Price')
plt.ylabel('Modal Price')
plt.title('Modal Price Distribution')
plt.tight_layout()
plt.show()


# # Pairwise Scatter Plot
# - A Scatter plot is a diagram where each value in the dataset is repesented by dot.

# In[30]:


price_columns=['Min Price', 'Max Price','Modal Price']
plt.figure(figsize=(10,6))
sns.pairplot(data = data, vars=price_columns)
plt.suptitle('Pairwise Scatter Plot', y=1.01)
plt.tight_layout()
plt.show()


# # Heat Map of Price Corelations
# 
# - A Heat map is a graphical representation of multivariate data that is structured as a matrix of columns and rows.
# 
# - Heat maps are very useful in describing correlation among several numerical variables, visualizing patterns and anomalies.

# In[31]:


price_correlations= data[price_columns].corr()
plt.figure(figsize=(7,5))
sns.heatmap(price_correlations, annot= True, cmap='coolwarm',square= True)
plt.title('Price Correlations')
plt.tight_layout()
plt.show()


# - There are corelation between min price, max price and modal price.

# # BUSINESS OBJECTIVE¶
# 1. Understand price trends to forecast future movements and make informed purchasing and pricing decisions.
# 
# 2. Compare commodity prices across regions to identify market opportunities and optimize supply chain management.
# 
# 3. Analyze seasonal patterns to forecast demand fluctuations and adjust production and inventory strategies.
# 
# 4. Gain market intelligence to anticipate changes, mitigate risks, and make proactive decisions.
# 
# 5. Conduct competitive analysis to evaluate pricing advantages and develop competitive strategies.
# 
# 6. Optimize prices based on historical data and market conditions to maximize revenue.
# 
# 7. Manage risks associated with price fluctuations by developing risk management strategies.
# 
# 8. Evaluate suppliers based on historical price performance to make informed selection decisions.
# 
# 9. Understand customer behavior and preferences related to commodity prices to tailor marketing strategies.
# 
# 10. Identify new business opportunities and growth potential in underserved regions or commodities.
# 
# - These objectives aim to support businesses in making data-driven decisions, optimizing operations, minimizing risks, and       achieving a competitive advantage in the commodity market in India.
# 
# # Conclusion
# - Uttar Pradesh has the highest number of market located and Kerala has second highest number of market.
# 
# - But Palakkad (Kerala) has 162 number of market, and Kangra (Himachal Pradesh) has 138 market and Bulandshahar (Uttar Pradesh)   has 115 markets.
# 
# - Potato, onion, tomato and green chilly are the commodity has the highest demand and supply but in decending order.
# 
# - 'Other' variety are highest, while the variety of green chilly is the highest demanding variety among all.
# 
# - There is no idea of quality of commodity (Grade) is highest represent as FAQ.
# 
# * PRICE DISTRIBUTIONS~
# - Overview- Min Price (36.4%), Max Price (22.0%) and Modal Price (41.5%).
# 
# - we will choose Modal Price for further evaluation, Modal Price represent the most common wholesale price of the commodity on   given day, per quintal (100 kg).
# 
# - Find out the top 10 highest modal price.
# 
# - Kerala contribute the highest Modal Price whereas Gujrat, NCT Delhi and Rajasthan have almost equal contribution.
# 
# - Kattappana (Kerala) market has highest modal price.
# 
# - Cardamoms commodity has highest modal price, whereas cloves, jeera and fish have also modal price.
# 
# - Other variety has highest Modal Price while jeera's variety has second.
# 
# - Almost all the Modal Price falls under 20,000.Modal Price is correlated with Max Price and Min Price.
# 
# - This was the complete EDA project on Agricultre Market.

# # Pridiction of Modal Price

# In[32]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score


# In[33]:


grade_mapping={'low':1,'medium':2,'high':3, 'FAQ':4}
data['Grade']=data['Grade'].map(grade_mapping)

X= data[['Min Price','Max Price','Grade']]
y= data['Modal Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




# In[34]:


imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)


# # Linear Regression
# - Linear Regression a data plot that graphs the Linear Relationship between an independent and a dependent varibles. It is typically used to visually show the Strength of relationship, the dispersion of results.

# In[35]:


linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_pred = linear_model.predict(X_test)
linear_mse = mean_squared_error(y_test, linear_pred)
linear_r2 = r2_score(y_test, linear_pred)


# # Decision Tree
# 
# - A decision tree is a non-parametric supervised learning algorithm, which is utilized for both classification and regression tasks. It has a hierarchical, tree structure, which consists of a root node, branches, internal nodes and leaf nodes.

# In[36]:


tree_model = DecisionTreeRegressor()
tree_model.fit(X_train, y_train)
tree_pred = tree_model.predict(X_test)
tree_mse = mean_squared_error(y_test, tree_pred)
tree_r2 = r2_score(y_test, tree_pred)


# # Random Forest
# 
# - A random forest is a machine learning technique that's used to solve regression and classification problems. It utilizes ensemble learning, which is a technique that combines many classifiers to provide solutions to complex problems. A random forest algorithm consists of many decision trees.

# In[37]:


forest_model = RandomForestRegressor()
forest_model.fit(X_train, y_train)
forest_pred = forest_model.predict(X_test)
forest_mse = mean_squared_error(y_test, forest_pred)
forest_r2 = r2_score(y_test, forest_pred)


# # Gradient Boosting
# 
# - Gradient Boosting is a powerful boosting algorithm that combines several weak learners into strong learners, in which each new model is trained to minimize the loss function such as mean squared error or cross-entropy of the previous model using gradient descent.

# In[38]:


gb_model = GradientBoostingRegressor()
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
gb_mse = mean_squared_error(y_test, gb_pred)
gb_r2 = r2_score(y_test, gb_pred)


# # Support Vector Regression
# 
# - A support vector machine (SVM) is a type of deep learning algorithm that performs supervised learning for classification or regression of data groups.

# In[39]:


svr_model = SVR()
svr_model.fit(X_train, y_train)
svr_pred = svr_model.predict(X_test)
svr_mse = mean_squared_error(y_test, svr_pred)
svr_r2 = r2_score(y_test, svr_pred)


# # R2 Score and Plot

# In[40]:


print('Linear Regression - R-squared:',linear_r2)
print('DecisionTreeRegressor - R-squared:',tree_r2)
print('Random Forest Regressor - R-squared:',forest_r2)
print('Gradient Boosting Regressor- R-squared:', gb_r2)
print('Support Vector Regressor - R -squared:',svr_r2)


# # BUSINESS OBJECTIVE
# - Understand price trends to forecast future movements and make informed purchasing and pricing decisions.
# 
# - Compare commodity prices across regions to identify market opportunities and optimize supply chain management.
# 
# - Analyze seasonal patterns to forecast demand fluctuations and adjust production and inventory strategies.
# 
# - Gain market intelligence to anticipate changes, mitigate risks, and make proactive decisions.
# 
# - Conduct competitive analysis to evaluate pricing advantages and develop competitive strategies.
# 
# - Optimize prices based on historical data and market conditions to maximize revenue.
# 
# - Manage risks associated with price fluctuations by developing risk management strategies.
# 
# - Evaluate suppliers based on historical price performance to make informed selection decisions.
# 
# - Understand customer behavior and preferences related to commodity prices to tailor marketing strategies.
# 
# - Identify new business opportunities and growth potential in underserved regions or commodities.
# 
# - These objectives aim to support businesses in making data-driven decisions, optimizing operations, minimizing risks, and   achieving a competitive advantage in the commodity market in India.
# 
# # Conclusion
# - Uttar Pradesh has the highest number of market located and Kerala has second highest number of market.
# 
# - But Palakkad (Kerala) has 162 number of market, and Kangra (Himachal Pradesh) has 138 market and Bulandshahar (Uttar Pradesh) has 115 markets.
# 
# - Potato, onion, tomato and green chilly are the commodity has the highest demand and supply but in decending order.
# 
# - 'Other' variety are highest, while the variety of green chilly is the highest demanding variety among all.
# 
# - There is no idea of quality of commodity (Grade) is highest represent as FAQ.
# 
# 
# PRICE DISTRIBUTIONS~
# - Overview- Min Price (36.4%), Max Price (22.0%) and Modal Price (41.5%).
# 
# - we will choose Modal Price for further evaluation, Modal Price represent the most common wholesale price of the commodity on given day, per quintal (100 kg).
# 
# - Find out the top 10 highest modal price.
# 
# - Kerala contribute the highest Modal Price whereas Gujrat, NCT Delhi and Rajasthan have almost equal contribution.
# 
# - Kattappana (Kerala) market has highest modal price.
# 
# - Cardamoms commodity has highest modal price, whereas cloves, jeera and fish have also modal price.
# 
# - Other variety has highest Modal Price while jeera's variety has second.
# 
# - Almost all the Modal Price falls under 20,000.Modal Price is correlated with Max Price and Min Price.
# 
# - This was the complete EDA project on Agricultre Market.
