import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

file = 'SPAM text message 20170820 - Data.csv'
file_name = {'train' : 'train_cleaned_'+file,
             'test'  : 'test_cleaned_'+file  }

train_data = pd.read_csv(file_name['train']).dropna()

train_msg = train_data['Message']
result = train_data['Category']

vectorizer = CountVectorizer()

x = vectorizer.fit_transform(train_msg)


tfidf = TfidfTransformer()
x = tfidf.fit_transform(x)

clf = MultinomialNB().fit(x, result)

#-------------------------------------------------------------

test_data = pd.read_csv(file_name['test'])

e_result = test_data['Category']
test_msg = test_data['Message']

x_new = vectorizer.transform(test_msg)
x_new = tfidf.transform(x_new)

predicted = clf.predict(x_new)

zip_result = zip(predicted, e_result)

percent = sum([ 100 for i in zip_result if len(set(i)) == 1 ])/len(predicted)

print(percent)