#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
plt.style.use('dark_background')


# In[2]:


df = pd.read_csv('E:\\Zomato\\Zomato.csv')


# In[3]:


df.head()


# In[4]:


df.shape       #Rows = 51717  #Columns = 17


# In[5]:


df.columns


# In[7]:


df = df.drop(['url','address','phone','menu_item','dish_liked','reviews_list'],axis=1)
df.head()


# In[8]:


df.info()


# In[10]:


df.drop_duplicates(inplace = True)
df.shape


# In[11]:


df['rate'].unique()


# In[18]:


import numpy as np 

def handlerate(value):
    if value in ['NEW','-']:
        return np.nan
    else:
        value = str(value).split('/')
        return float(value[0])
    # Applying the handlerate function to the 'rate' column
df['rate'] = df['rate'].apply(handlerate)


# In[19]:


df['rate'].fillna(df['rate'].mean(), inplace = True)      #Filling NA Values 
df['rate'].isnull().sum()


# In[20]:


df.info()


# In[21]:


df.dropna(inplace = True)
df.head()


# In[22]:


df.rename(columns ={'approx_cost(for two people)'
:'Cost2plates','listed_in(type)':'Type'},inplace = True)
df.head()


# In[23]:


df['location'].unique()


# In[24]:


df['listed_in(city)'].unique()


# In[25]:


df = df.drop(['listed_in(city)'],axis=1)


# In[26]:


df


# In[27]:


df['Cost2plates'].unique()


# In[29]:


def handlecomma(value):
    value = str(value)
    if ',' in value:
        value = value.replace(',','')
    else:
        return float(value)
    df['Cost2plates'] = df['Cost2plates'].apply(handlecomma)
    df['Cost2plates'].unique()
            


# In[30]:


df['rest_type'].value_counts()


# In[31]:


rest_types = df['rest_type'].value_counts(ascending = True)
rest_types


# In[32]:


rest_types_lessthan1000 = rest_types[rest_types<1000]
rest_types_lessthan1000


# In[33]:


def handle_rest_type(value):
    if(value in rest_types_lessthan1000):
        return 'others'
    else:
        return value 
df['rest_type'] = df['rest_type'].apply(handle_rest_type)
df['rest_type'].value_counts()


# In[34]:


df['location'].value_counts()


# In[35]:


location = df['location'].value_counts(ascending = False)
locationlessthan300 = location[location<300]

def handle_location(value):
    if (value in location_lessthan300):
        return 'others'
    else:
        return value 
    df['location'] = df['location'].apply(handle_location)
    df['location'].value_counts()


# In[36]:


df['location'].value_counts()


# In[37]:


df['cuisines'].value_counts()


# In[38]:


cuisines = df['cuisines'].value_counts(ascending = False)
cuisines_lessthan100 = cuisines[cuisines<100]

def handle_cuisines(value):
    if (value in cuisines_lessthan100):
        return 'others'
    else:
        return value


# In[39]:


df['cuisines'] = df['cuisines'].apply(handle_cuisines)
df['cuisines'].value_counts()


# In[40]:


df['Type'].value_counts()


# In[41]:


#visuals 


# In[52]:


plt.figure(figsize=(16,10))
sns.countplot(x='location', data=df)
plt.xticks(rotation=90)
plt.show()


# In[55]:


top_locations = df['location'].value_counts().head(10).index
df_top10 = df[df['location'].isin(top_locations)]
plt.figure(figsize=(16, 10))
sns.countplot(x='location', data=df_top10, order=top_locations)
plt.xticks(rotation=90)
plt.show()


# In[58]:


plt.figure(figsize=(16,10))
sns.countplot(x='online_order', data=df,palette='inferno')
plt.xticks(rotation=90)
plt.show()


# In[59]:


plt.figure(figsize=(16, 10))

# Create a count plot
ax = sns.countplot(x='online_order', data=df, palette='inferno')

# Rotate x-axis labels for better readability
plt.xticks(rotation=90)

# Add count data labels on top of the bars
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=10)

# Show the plot
plt.show()


# In[62]:


plt.figure(figsize=(6,6))
sns.countplot(x='book_table', data=df,palette='rainbow')
plt.xticks(rotation=90)
plt.show()


# In[61]:


df.head()


# In[63]:


plt.figure(figsize=(6,6))
sns.boxplot(x='online_order',y='rate', data=df)
plt.xticks(rotation=90)
plt.show()


# In[64]:


plt.figure(figsize=(6,6))
sns.boxplot(x='book_table',y='rate', data=df)
plt.xticks(rotation=90)
plt.show()


# In[66]:


df1 = df.groupby(['location','online_order'])['name'].count()
df1.to_csv('location_online.csv')
df1=pd.read_csv('location_online.csv')
df1=pd.pivot_table(df1,values=None, index = ['location'],columns=['online_order'],fill_value=0,aggfunc=np.sum)
df1


# In[67]:


df1.plot(kind='bar',figsize=(15,8))


# In[69]:


plt.figure(figsize=(14,8))
sns.boxplot(x='Type',y='rate',data=df,palette = 'inferno')


# In[75]:


df3=df.groupby(['location','Type'])['name'].count()
df3.to_csv('location_Type.csv')
df3=pd.read_csv('location_Type.csv')
df3=pd.pivot_table(df3,values=None,index=['location'],columns=['Type'],fill_value=0,aggfunc=np.sum)
df3


# In[79]:


df3.plot(kind='bar',figsize =(40,5))


# In[81]:


df4=df[['location','votes']]
df4.drop_duplicates()
df5=df4.groupby(['location'])['votes'].sum()
df5=df5.to_frame()
df5=df5.sort_values('votes',ascending=False)
df5.head()


# In[87]:


plt.figure(figsize=(15, 8))
sns.barplot(x=df5.index, y=df5['votes'])
plt.xticks(rotation=90)
plt.show()


# In[88]:


df_top10 = df5.sort_values(by='votes', ascending=False).head(10)   #TOP 10 
plt.figure(figsize=(15, 8))
sns.barplot(x=df_top10.index, y=df_top10['votes'])
plt.xticks(rotation=90)
plt.show()


# In[89]:


df.head(5)


# In[91]:


df6 = df[['cuisines','votes']]
df6.drop_duplicates()
df7=df6.groupby(['cuisines'])['votes'].sum()
df7=df7.to_frame()
df7=df7.sort_values('votes',ascending=False)
df7.head()


# In[92]:


df7=df7.iloc[1:,:]
df7.head()


# In[99]:


plt.figure(figsize=(15, 8))
sns.barplot(x=df7.index, y=df7['votes'])
plt.xticks(rotation=90)
plt.show()


# In[97]:


plt.figure(figsize=(15, 8))
sns.barplot(x=df7.index, y=df7['votes'])
plt.xticks(rotation=90)
plt.show()

