# Spam-Email-Detector
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
data = pd.read_csv('spam.csv', encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'text']
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split the data
X = data['text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Naive Bayes Model
nb_pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB())
])
nb_pipeline.fit(X_train, y_train)
y_pred_nb = nb_pipeline.predict(X_test)

# SVM Model
svm_pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SVC(kernel='linear'))
])
svm_pipeline.fit(X_train, y_train)
y_pred_svm = svm_pipeline.predict(X_test)

# Evaluation
print('Naive Bayes Classification Report:\n', classification_report(y_test, y_pred_nb))
print('Naive Bayes Confusion Matrix:\n', confusion_matrix(y_test, y_pred_nb))
print('Naive Bayes Accuracy:', accuracy_score(y_test, y_pred_nb))

print('SVM Classification Report:\n', classification_report(y_test, y_pred_svm))
print('SVM Confusion Matrix:\n', confusion_matrix(y_test, y_pred_svm))
print('SVM Accuracy:', accuracy_score(y_test, y_pred_svm))
