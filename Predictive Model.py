#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[2]:


cwd = os.getcwd()
cst_data = pd.read_csv(cwd + '/customer_booking.csv', encoding='ISO-8859-1')


# ## Exploratory Data Analysis

# In[3]:


cst_data.head()


# In[4]:


cst_data.shape


# In[5]:


cst_data.describe()


# In[6]:


cst_data.info()


# In[7]:


rndtrp = round(cst_data.trip_type.value_counts().values[0]/ cst_data.trip_type.count() *100, 2)
oneway = round(cst_data.trip_type.value_counts().values[1]/ cst_data.trip_type.count() *100, 2)
circle = round(cst_data.trip_type.value_counts().values[2]/ cst_data.trip_type.count() *100, 2)
print(f"Percentage of round trips: {rndtrp}%")
print(f"Percentage of One way trips: {oneway}%")
print(f"Percentage of circle trips: {circle}%")


# In[8]:


internet = round(cst_data.sales_channel.value_counts().values[0]  / cst_data.sales_channel.count() *100, 2)
mob = round(cst_data.sales_channel.value_counts().values[1]  / cst_data.sales_channel.count() *100, 2)
print(f"Number of bookings done through internet: {internet}%")
print(f"Number of bookings done through phone call: {mob}%")


# In[9]:


plt.figure(figsize=(15,6))
sns.boxplot(x=cst_data['purchase_lead'], palette='deep')


# As we cannot remove all the data shown as outliers in the boxplot, we can use a Histogram KDE plot to have a clearer view.

# In[10]:


plt.figure(figsize=(15,6))
sns.histplot(data=cst_data, x="purchase_lead", binwidth=25,kde=True, palette='deep')


# In[11]:


(cst_data.purchase_lead >550).value_counts()


# In[12]:


cst_data = cst_data[cst_data.purchase_lead <550 ]


# In[13]:


plt.figure(figsize=(15,6))
sns.boxplot(x=cst_data['length_of_stay'], palette='deep')


# As we cannot remove all the data shown as outliers in the boxplot, again, we can use a Histogram KDE plot to have a clearer view.

# In[14]:


plt.figure(figsize=(15,5))
sns.histplot(data=cst_data, x="length_of_stay", binwidth=15,kde=True, palette='deep')


# In[15]:


(cst_data.length_of_stay> 100).value_counts()


# In[16]:


cst_data[cst_data.length_of_stay > 500].booking_complete.value_counts()


# In[17]:


cst_data = cst_data[cst_data.purchase_lead < 500 ]


# In[18]:


cst_data.flight_day.value_counts()


# In[19]:


plt.figure(figsize=(15,5))
ax = cst_data.booking_origin.value_counts()[:15].plot(kind="bar")
ax.set_xlabel("Countries")
ax.set_ylabel("Number of bookings")


# In[20]:


plt.figure(figsize=(15,5))
ax = cst_data[cst_data.booking_complete == 1].booking_origin.value_counts()[:15].plot(kind="bar")
ax.set_xlabel("Countries")
ax.set_ylabel("Number of complete bookings")


# In[21]:


succ_booking = cst_data.booking_complete.value_counts().values[1] / len(cst_data) * 100


# In[22]:


print(f"Out of 50000 booking entries only {round(succ_booking,2)}% bookings were successfull or complete.")


# ## Predictive Model

# In[23]:


cst_data = cst_data.reset_index(drop=True)


# In[24]:


cst_data


# In[25]:


cst_data.info()


# In[26]:


encoder = OneHotEncoder(handle_unknown = 'ignore')

encoder_df = pd.DataFrame(encoder.fit_transform(cst_data[['sales_channel']]).toarray())
encoder_df = encoder_df.rename(columns={0:'Internet', 1:'Mobile'})
cst_data = cst_data.join(encoder_df)

encoder_df = pd.DataFrame(encoder.fit_transform(cst_data[['trip_type']]).toarray())
encoder_df = encoder_df.rename(columns={0:'RoundTrip', 1:'OneWayTrip', 2:'CircleTrip'})
cst_data = cst_data.join(encoder_df)


# In[27]:


cst_data.drop(['sales_channel', 'trip_type', 'booking_origin', 'route'], axis=1, inplace=True)


# In[28]:


cst_data.info()


# In[29]:


mapping = {"Mon" : 1, "Tue" : 2, "Wed" : 3, "Thu" : 4, "Fri" : 5, "Sat" : 6, "Sun" : 7}

cst_data.flight_day = cst_data.flight_day.map(mapping)


# In[30]:


cst_data.info()


# In[31]:


label = cst_data['booking_complete']


# In[32]:


cst_data.to_csv(cwd + "/filtered_customer_booking.csv")


# In[33]:


cst_data = cst_data.drop('booking_complete', axis=1)


# In[34]:


cst_data


# In[35]:


scaler = StandardScaler()
scaled_df = scaler.fit_transform(cst_data)


# In[36]:


scaled_df


# In[37]:


scaled_df = pd.DataFrame(scaled_df, columns= cst_data.columns)


# In[38]:


scaled_df['label']  = label


# In[39]:


scaled_df


# In[40]:


corr = scaled_df.corr()

plt.figure(figsize=(10,7))
sns.heatmap(corr)


# In[41]:


X = scaled_df.iloc[:,:-1]
y = scaled_df['label']

X_train, X_test, Y_train, Y_test = train_test_split(X.to_numpy(), y.to_numpy(), test_size=0.2, random_state=1)


# In[42]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.inspection import permutation_importance

from yellowbrick.classifier import ConfusionMatrix
from sklearn.model_selection import GridSearchCV,  RepeatedStratifiedKFold


# In[43]:


def model_fit_predict(model, X, Y, X_predict):
    model.fit(X, Y)
    return model.predict(X_predict)

def acc_score(Y_true, Y_pred):
    return accuracy_score(Y_true, Y_pred)

def pre_score(Y_true, Y_pred):
    return precision_score(Y_true, Y_pred)

def f_score(Y_true, Y_pred):
    return f1_score(Y_true, Y_pred)


# In[44]:


clf_rf = RandomForestClassifier(max_depth =50 , min_samples_split=5,random_state=0)


# In[45]:


Y_pred_train = model_fit_predict(clf_rf, X_train, Y_train, X_train)
set(Y_pred_train)

f1 = round(f1_score(Y_train, Y_pred_train),2) 
acc = round(accuracy_score(Y_train, Y_pred_train),2) 
pre = round(precision_score(Y_train, Y_pred_train),2) 

print(f"TRAINING DATA SCORES:\nAccuracy-Score: {acc},\nPrecision-Score: {pre},\nF1-score: {f1}")


# In[46]:


cm = ConfusionMatrix(clf_rf, classes=[0,1])
cm.fit(X_train, Y_train)

cm.score(X_train, Y_train)


# In[47]:


clf_rf = RandomForestClassifier(max_depth =50 , min_samples_split=5,random_state=1)


# In[48]:


Y_pred_test = model_fit_predict(clf_rf, X_train, Y_train, X_test)


# In[49]:


f1 = round(f1_score(Y_test, Y_pred_test),2) 
acc = round(accuracy_score(Y_test, Y_pred_test),2) 
pre = round(precision_score(Y_test, Y_pred_test),2) 

print(f"TEST DATA SCORES:\nAccuracy-Score: {acc},\nPrecision-Score: {pre},\nF1-score: {f1}")


# In[50]:


cm = ConfusionMatrix(clf_rf, classes=[0,1])
cm.fit(X_train, Y_train)

cm.score(X_test, Y_test)


# In[51]:


plt.figure(figsize=(10,5))
sorted_idx = clf_rf.feature_importances_.argsort()
plt.barh(scaled_df.iloc[:,:-1].columns[sorted_idx], clf_rf.feature_importances_[sorted_idx])
plt.xlabel("Random Forest Feature Importance")


# In[52]:


scaled_df.label.value_counts()


# In[53]:


scaled_df_0 = scaled_df[scaled_df.label ==0].sample(n=8000)


# In[54]:


scaled_df_new = pd.concat([scaled_df[scaled_df.label==1], scaled_df_0], ignore_index=True)


# In[55]:


scaled_df_new = scaled_df_new.sample(frac = 1).reset_index(drop=True)


# In[56]:


scaled_df_new


# In[57]:


X = scaled_df_new.iloc[:,:-1]
y = scaled_df_new['label']

X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(), y.to_numpy(), test_size=0.20, random_state=42)


# In[58]:


clf_rf = RandomForestClassifier(n_estimators=50,max_depth =50 , min_samples_split=5,random_state=0)


# In[62]:


y_pred_test = model_fit_predict(clf_rf, X_train, y_train, X_test)

f1 = round(f1_score(y_test, y_pred_test),2) 
acc = round(accuracy_score(y_test, y_pred_test),2) 
pre = round(precision_score(y_test, y_pred_test),2) 

print(f"TEST DATA SCORES:\nAccuracy-Score: {acc},\nPrecision-Score: {pre},\nF1-score: {f1}")


# In[60]:


cm = ConfusionMatrix(clf_rf, classes=[0,1])
cm.fit(X_train, y_train)

cm.score(X_test, y_test)


# In[61]:


plt.figure(figsize=(10,8))
sorted_idx = clf_rf.feature_importances_.argsort()
plt.barh(scaled_df.iloc[:,:-1].columns[sorted_idx], clf_rf.feature_importances_[sorted_idx])
plt.xlabel("Random Forest Feature Importance")


# In[ ]:




