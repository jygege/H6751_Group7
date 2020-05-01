# H6751_Group_Project

##Introduction
In this project, our team participant in Kaggle Competetion: REal Or Not? NLP with Disaster Tweets
Started from a trainining set contains tweets, location data, and a label: real or not disaster.
To preidct the disaster laebl on a given test data.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Librarys you need to install for running the script 

```
pip install pandas
pip install numpy
pip install re

pip install seaborn 
pip install matplotlib

pip install nltk
pip install sklearn
pip install scipy
pip install catboost
```

## Running the tests

The jupyter notebook file can be run at once, comments are avalible for coding explaination.
```
# inspect the target class distribution
x=train.target.value_counts()
sns.barplot(x.index,x)
plt.gca().set_ylabel('tweets')
# There are slightly more Real Disaster then Fake One.
```

There are three sections in the script
### Exploratory data analysis 
EDA was done on train and test. 
For example, the most common (non stop words) words/ bigram, word counts in tweest distribution in the two class data.

### Data cleaning 
URL, html, punctuation, stopwords, and uppercase removal. 
```
def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)
```

Contractions
```
tweet = re.sub(r"We're", "We are", tweet)
```

### Feature Engineering 
Apart from the pre-processed tweets text data, 'word_count', 'unique_word_count', 'stop_word_count','mean_word_length', 'char_count','punctuation_count','num_char','num_punctuation', 'keyword'(provided by original Kaggle dataset) are extracted from the tweets text as features.

CounterVectorizer and TFIDFVectorizer were used for vectoring.

### Model training and validation
1. Standalone Model training and validation
  LogisticRegression, SVC, MultinomialNB, GradientBoostingClassifier, CatBoostClassifier are trained and validated with features from     both CounterVectorizer and TFIDFVectorizer.
  Prediction result favours the CounterVectorizer.
2. Ensemble Learning Model
  Using the Ensemble learning: VotingClassifier with AdaBoostClassifer, NaiveBayesClassifer, RandomForestClassifier and     LogisticRegressionClassifer as base learners.


