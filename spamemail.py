from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

emails = ["Win money now", "Hello friend", "Claim your prize"]
labels = [1, 0, 1]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

model = MultinomialNB()
model.fit(X, labels)

test = vectorizer.transform(["Win prize"])
print(model.predict(test))
