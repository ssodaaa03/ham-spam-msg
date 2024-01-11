import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

from model import file_name, x, clf

test_data = pd.read_csv(file_name['test'])

e_result = test_data['Category'].tolist()
test_msg = test_data['Message'].tolist()

x_new = CountVectorizer().transform(test_msg)
x_new = tfidf.transform(x_new)

# Make predictions
predicted = clf.predict(x_new)
i = 1
for e in predicted:
  print(str(i) + '. ' + str(e))
  i += 1