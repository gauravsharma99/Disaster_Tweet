import pandas as pd
import numpy as np
# from sklearn.impute import SimpleImputer
def pre_process(train_file, test_file):
  train_data = pd.read_csv(train_file)
  test_data = pd.read_csv(test_file)
  train_data = train_data.fillna(value = "NA")
  test_data = test_data.fillna(value = "NA")
  import nltk
  # nltk.download('stopwords')
  from nltk.corpus import stopwords
  from nltk.tokenize import word_tokenize  
  stop_words = stopwords.words('english')
  train_data['text'] = train_data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
  train_data['text'] = train_data['text'].str.replace('[^\w\s]','')
  train_data['text'] = train_data['text'].str.replace("https?://[A-Za-z0-9./]*", "")

  test_data['text'] = test_data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
  test_data['text'] = test_data['text'].str.replace('[^\w\s]','')
  test_data['text'] = test_data['text'].str.replace("https?://[A-Za-z0-9./]*", "")


  from sklearn import feature_extraction
  count_vectorizer = feature_extraction.text.CountVectorizer()
  train_vectors = count_vectorizer.fit_transform(train_data["text"])
  test_vectors = count_vectorizer.transform(test_data["text"])
  # train_vectors.append(data['keyword'])
  # print(train_vectors)
  
  return train_vectors, train_data.target, test_vectors, test_data.id


train_x, train_y, test_x, test_id = pre_process("train.csv","test.csv")


print(np.shape(train_x))
print(np.shape(test_x))

from sklearn.model_selection import train_test_split
# x_train, x_val, y_train, y_val = train_test_split(train_x, train_y)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0, max_iter = 150).fit(train_x, train_y)
# clf.score(x_val, y_val)

# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(random_state=0).fit(x_train, y_train)
# clf.score(x_val, y_val)

y_pred = clf.predict(test_x)

output = pd.DataFrame({'id': test_id,
                       'target': y_pred})
output.to_csv('submission.csv', index = False)