#!/usr/bin/env python
# coding: utf-8

# In[1]:


#IMPORTING LIBRARIES
import pandas as pd 
import numpy as np 
import nltk
import re
import string
from wordcloud import WordCloud,STOPWORDS
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import sklearn 
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import BernoulliNB 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn import preprocessing 
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt 
from matplotlib import rcParams
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from collections import Counter
from imblearn.over_sampling import SMOTE


# In[2]:


import pandas as pd 
data = pd.read_csv("C://Users//Administrator//Documents//Tweets.csv")
df= data.copy()
data.head()


# In[3]:


data.shape


# In[4]:


data.info()


# In[5]:


data.describe()


# In[6]:


df=data.copy()
df.isnull().sum()


# In[7]:


#MISSING VALUES


# In[8]:


df['negativereason']=df['negativereason'].fillna('Missing')
df['negativereason_confidence']=df['negativereason_confidence'].fillna('Missing')
df['tweet_location']=df['tweet_location'].fillna('Missing')
df['user_timezone']=df['user_timezone'].fillna('Missing')
df.head()


# In[9]:


df['negativereason'].value_counts()


# In[10]:


df['airline_sentiment'].value_counts()


# In[11]:


def f(row):
    
    '''This function returns sentiment value based on the overall ratings from the user'''
    
    if row['negativereason'] == 3:
        val = 'Neutral'
    elif row['negativereason'] == 1 or row['negativereason'] == 2:
        val = 'Negative'
    elif row['negativereason'] == 4 or row['negativereason'] == 5:
        val = 'Positive'
    else:
        val = -1
    return val


# In[12]:


#APPLY FUNCTIONS
df['sentiment'] = df.apply(f, axis=1)
df.head()


# In[13]:


df['sentiment'].value_counts()


# In[14]:


#FILTERING


# In[15]:


stop_words= ['yourselves', 'between', 'whom', 'itself', 'is', "she's", 'up', 'herself', 'here', 'your', 'each', 
             'we', 'he', 'my', "you've", 'having', 'in', 'both', 'for', 'themselves', 'are', 'them', 'other',
             'and', 'an', 'during', 'their', 'can', 'yourself', 'she', 'until', 'so', 'these', 'ours', 'above', 
             'what', 'while', 'have', 're', 'more', 'only', "needn't", 'when', 'just', 'that', 'were', "don't", 
             'very', 'should', 'any', 'y', 'isn', 'who',  'a', 'they', 'to', 'too', "should've", 'has', 'before',
             'into', 'yours', "it's", 'do', 'against', 'on',  'now', 'her', 've', 'd', 'by', 'am', 'from', 
             'about', 'further', "that'll", "you'd", 'you', 'as', 'how', 'been', 'the', 'or', 'doing', 'such',
             'his', 'himself', 'ourselves',  'was', 'through', 'out', 'below', 'own', 'myself', 'theirs', 
             'me', 'why', 'once',  'him', 'than', 'be', 'most', "you'll", 'same', 'some', 'with', 'few', 'it',
             'at', 'after', 'its', 'which', 'there','our', 'this', 'hers', 'being', 'did', 'of', 'had', 'under',
             'over','again', 'where', 'those', 'then', "you're", 'i', 'because', 'does', 'all']
df['text'] =df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
df.head()


# In[16]:


pd.DataFrame(df.groupby('airline_sentiment')['airline_sentiment_confidence'].mean())


# In[17]:


import seaborn as sns
import matplotlib.pyplot as plt 
from matplotlib import rcParams
plt.rcParams.update({'font.size': 18})
rcParams['figure.figsize'] = 16,9
senti_help= pd.DataFrame(df, columns = ['airline_sentiment', 'airline_sentiment_confidence'])
senti_help = senti_help[senti_help['airline_sentiment_confidence'] != 0.00] 
sns.violinplot( x=senti_help["airline_sentiment"], y=senti_help["airline_sentiment_confidence"])
plt.title('Sentiment vs Confidence')
plt.xlabel('Sentiment')
plt.ylabel('Confidence')
plt.show()


# In[18]:


label_encoder = preprocessing.LabelEncoder() 
df['airline_sentimentt']= label_encoder.fit_transform(df['airline_sentiment']) 
  
df['airline_sentiment'].unique() 


# In[19]:


df['airline_sentiment'].value_counts()


# In[20]:


#lowercaseing
df["text_lower"] = df["text"].str.lower()
df.head()


# In[21]:


#Punctuaion removal
df.drop(["text_lower"], axis=1, inplace=True)

PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

df["text_wo_punct"] = df["text"].apply(lambda text: remove_punctuation(text))
df.head()


# In[22]:


#stemming
stemmer = PorterStemmer()
def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])

df["text_stemmed"] = df["text_wo_punct"].apply(lambda text: stem_words(text))
df.head()


# In[23]:


#Lemmatization
lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])
df["text"] = df["text_stemmed"].apply(lambda text: lemmatize_words(text))
df.head()


# In[24]:


review_features=df.copy()
review_features=df[['text']].reset_index(drop=True)
review_features.head()


# In[25]:


#FEATURE EXTRACTION
review_features=df.copy()
review_features=df[['text']].reset_index(drop=True)
tfidf_vectorizer = TfidfVectorizer(max_features=5000,ngram_range=(2,2))
X= tfidf_vectorizer.fit_transform(review_features['text'])
y=df['airline_sentiment']
print(f'Original dataset shape : {Counter(y)}')
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
print(f'Resampled dataset shape {Counter(y_res)}')


# In[26]:


#SPLIT DATA
X_train,X_test,y_train,y_test = train_test_split(X_res, y_res, test_size=0.25, random_state=0)


# In[28]:


#MultinomialNB 
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
prediction={}
model1 = MultinomialNB().fit(X_train , y_train)
prediction['Multinomial'] = model1.predict_proba(X_test)[:,1]
print("Multinomial Accuracy : {}".format(model1.score(X_test ,y_test)))


# In[29]:


#Fitiing Bernouli NB
from sklearn.naive_bayes import BernoulliNB
model2 = BernoulliNB().fit(X_train,y_train)
prediction['Bernoulli'] = model2.predict_proba(X_test)[:,1]
print("Bernoulli Accuracy : {}".format(model2.score(X_test , y_test)))


# In[30]:


#Fitiing LogisticRegression
from sklearn import linear_model
logreg = linear_model.LogisticRegression(solver='lbfgs' , C=1000)
logistic = logreg.fit(X_train, y_train)
prediction['LogisticRegression'] = logreg.predict_proba(X_test)[:,1]
print("Logistic Regression Accuracy : {}".format(logreg.score(X_test , y_test)))


# In[31]:


def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:        print('Confusion matrix, without normalization')
    thresh = cm.max() / 2.
    for i in range (cm.shape[0]):
        for j in range (cm.shape[1]):
            plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[32]:


nb=BernoulliNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
cm = metrics.confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, classes=['Negative','Neutral','Positive'])


# In[33]:


print("Classification Report:\n",classification_report(y_test, y_pred))


# In[ ]:




