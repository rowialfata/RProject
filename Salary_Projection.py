#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd


# In[31]:


df = pd.read_csv("Salary_Data.csv")


# In[32]:


df.head()


# In[33]:


X = df.iloc[:, :-1]
y = df.iloc[:, 1]


# In[36]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)


# In[38]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train, y_train)


# In[39]:


y_pred = regressor.predict(X_test)


# In[57]:


import matplotlib.pyplot as plt
plt.scatter(X_train, y_train, color="green")
plt.plot(X_train, regressor.predict(X_train), color="red")
plt.title("Years Experience VS Salary")
plt.xlabel("Years Experience")
plt.ylabel("Salary")
plt.show()


# In[59]:


plt.scatter(X_test, y_test, color="green")
plt.plot(X_train, regressor.predict(X_train), color="red")
plt.title("Years Experience VS Salary")
plt.xlabel("Years Experience")
plt.ylabel("Salary")
plt.show()


# ## MAKE NEW PREDICTION

# In[61]:


salary_pred = regressor.predict([[12]])
print("The salary for that amount of years experience is: ", salary_pred)

