from time import time 
import pandas as pd 
import sys 
import tweepy 
from tweepy import OAuthHandler 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer 
from sklearn.decomposition import NMF, LatentDirichletAllocation 
from nltk.tokenize.casual import TweetTokenizer 
import string
import nltk 
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
from nltk.corpus import stopwords 
from nltk.util import everygrams 
from collections import Counter 
import itertools as it 
import re 
import urllib 
from nltk.stem import PorterStemmer 
from nltk.stem.wordnet import WordNetLemmatizer 
from sklearn.cluster import KMeans 
from sklearn.metrics import pairwise_distances_argmin
from time import time 
from sklearn.cluster import KMeans 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn import metrics 
from os import path 
from PIL import Image 
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator 
import matplotlib.pyplot as plt 
import collections 

def custom_stopwords(max_length=4,mychar='.'):     
    def stopword_maker(length): 
        return ((''.join(x) for x in it.product(mychar, repeat=length))) 
    words = it.chain.from_iterable((stopword_maker(length) for length in range(max_length+1)))     
    return list(words) 
 
def preprocess_tweet(input_string):     
    out_string=re.sub(contains_URL, '', input_string)     
    out_string=re.sub(contains_html_tag, '', out_string)     
    out_string=re.sub(contains_backslash, '', out_string)     
    out_string=re.sub(contains_forwardslash, '', out_string)     
    return out_string 
 
def tokenize_tweet(tweet): 
    tokens = (TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False).tokenize(tweet))     
    tokens = [tkn for tkn in tokens if not tkn in string.punctuation]     
    return tokens 

#print topic words 
def print_topic_words(model, feature_names, n_top_words):     
    for topic_idx, topic in enumerate(model.components_):         
        message = "Topic #%d: " % topic_idx         
        for i in topic.argsort()[:-n_top_words - 1:-1]:             
            message += " " +feature_names[i]             
            top_messages.append(feature_names[i]) 
        print(message) 
    print() 

#set twitter authentication keys 
consumer_key="Enter Your Consumer Key here" 
consumer_secret="Enter Your Consumer Secret here" 
access_token="Enter Your Access token here" 
access_secret="Enter Your Access secret here" 
auth = OAuthHandler(consumer_key, consumer_secret) 
auth.set_access_token(access_token, access_secret) 
api = tweepy.API(auth) 
encodingTot = sys.stdout.encoding or 'utf8' 
path="res/consolidatedtweets.csv" 
columns = ['Tweet'] 
tweetDF = pd.DataFrame(columns=columns) 
 
t0 = time() 
num_tweets = 1000 
#load tweets using tweepy library and twitter API
for tweet in tweepy.Cursor(api.search, q="pluralsight -filter:retweets", lang="en").items(num_tweets):         
    lenDF=len(tweetDF)     
    tweetDF.loc[lenDF]=[tweet.text] 
tweetDF.to_csv(path, sep='\t', encoding = 'utf8')  
tweetDF = pd.read_csv(open(path,'rU', encoding='utf8'), sep='\t', engine='c',encoding='utf8') 
tweetDF.drop_duplicates(subset=['Tweet'],inplace=True) 
tweetDF["Tweet"].head()

tweets = tweetDF['Tweet'].tolist()  #regular expressions to preprocess tweets 
contains_only_numbers = r'(?:(?:\d+,?)+(?:\.?\d+)?)' 
contains_URL = r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+' 
contains_html_tag = r'<[^>]+>' 
contains_backslash =  r'\s*(?:[\w_]*\\(?:[\w_]*\\)*[\w_]*)' 
contains_forwardslash = r'\s*(?:[\w_]*/(?:[\w_]*/)*[\w_]*)' 
 
stop_words = set(stopwords.words('english')) 
context_words_exclude=set(["pluralsight","#pluralsight","https","best","back","wish","great","inc","co","@ pluralsight","using","really","yes","hi","please","thank"]) 
tweet_stopwords = list(it.chain(stop_words,context_words_exclude, custom_stopwords(max_length=4,mychar='.'))) 

#intialize variables 
top_messages=[] 
n_samples = len(tweetDF) 
n_features = 8000 
n_components = 50 
n_top_words = 10 
 
#Use tf-idf features to vectorize features 
#max_df is the threshold to ignore terms that have a document frequency strictly higher than the given threshold  
#max_features: builds a vocabulary that only considers the top max_features ordered by term frequency across the corpus. 
#TfidfVectorizer uses L2 norm (Eucleadian distances) by default and use_idf=true 
vectorizer = TfidfVectorizer(preprocessor=preprocess_tweet,tokenizer=tokenize_tweet,max_features=n_features, stop_words=tweet_stopwords,max_df=0.95, min_df=2,) 
t0 = time() 
tfidf = vectorizer.fit_transform(tweets) 
tfidf_feature_names = vectorizer.get_feature_names() 
print(len(tfidf_feature_names)) 
t1=time() 
print("done vectorization in %0.3fs." % (t1 - t0)) 

#Fit NMF Model to the Vectorized tweets 
print("Fitting the NMF model with tf-idf features, sample size=%d and features=%d..." % (n_samples, n_features)) 
t0 = time() 
nmf = NMF(n_components=n_components, random_state=1,alpha=.1, l1_ratio=.5).fit(tfidf) 
print("done in %0.3fs." % (time() - t0)) 
print("\nTopics in NMF model (Frobenius norm):") 
tfidf_feature_names = vectorizer.get_feature_names() 
print_topic_words(nmf, tfidf_feature_names, n_top_words) 
 
#calculate metrics print("Calculating NMF Metrics...") 
t0 = time() 
nmftrans = NMF(n_components=n_components, random_state=1,alpha=.1, l1_ratio=.5).fit_transform(tfidf) 
nmflabels = nmftrans.argmax(axis=1) 
print("V-measure: %0.3f" % metrics.v_measure_score(tweets, nmflabels)) 
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(tfidf, nmflabels)) 
print("done in %0.3fs." % (time() - t0)) 
#display results 
#prepare a string of top topic words for word cloud 
print("Starting visualization...") 
t0 = time() 
output=" ".join(top_messages) 
wordcloud = WordCloud(max_font_size=50, max_words=100, 
background_color="white").generate(output) 
plt.figure() 
plt.imshow(wordcloud, interpolation="bilinear") 
plt.axis("off") 
plt.show() 
print("done in %0.3fs." % (time() - t0)) 