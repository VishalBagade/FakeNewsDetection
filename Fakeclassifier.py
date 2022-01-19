import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

dt = pd.read_csv("C:/Users/hp/Downloads/news/news.csv")
print(dt)
x = dt['text']
dt['class'] = dt['label'].map({'FAKE': 0, 'REAL': 1})
y = dt['class']
print(x.head(5))
print(y.head(5))
vectorizer = TfidfVectorizer()
vectorizer.fit(x)
X = vectorizer.transform(x)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# nb = MultinomialNB()
pd = PassiveAggressiveClassifier()
# nb.fit(X_train, Y_train)
pd.fit(X_train, Y_train)
# result = nb.score(X_test, Y_test)
pre = pd.predict(X_test)
result = pd.score(X_test, Y_test)
print(result)
acc = result * 100
print("Accuracy=", acc)

con = confusion_matrix(Y_test, pre, labels=['FAKE', 'REAL'],)
print(con)
