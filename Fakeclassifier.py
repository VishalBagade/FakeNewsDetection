import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tkinter as tk

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
pad = PassiveAggressiveClassifier()
# nb.fit(X_train, Y_train)
pad.fit(X_train, Y_train)
# result = nb.score(X_test, Y_test)
pre = pad.predict(X_test)
result = pad.score(X_test, Y_test)
print(result)
acc = result * 100
print("Accuracy=", acc)


def Fake_News_Detection():
    news=e.get("1.0","end-1c")
    new_df=[news]   
    vfid=vectorizer.transform(new_df)
    pre=pad.predict(vfid)
    if new_df[0] == "":
        label2.config(text="Please Enter the News")
    else:
        if pre == 0:
            label2.config(text="News is Fake")
        
        else:
            label2.config(text="News is Real")


root=tk.Tk()
root.title("Fake News Classifier")
root.geometry('680x680')

label1=tk.Label(root,text="Fake News Detection",font="None 26 bold")
label1.pack(pady="20")
e=tk.Text(root,height='20',width='60',bg="light cyan")
e.pack()
mybutton=tk.Button(root,text="Submit",command=Fake_News_Detection)
mybutton.pack(pady="10")
label2=tk.Label(root,text="",font="None 10 bold")
label2.pack()
root.mainloop()

