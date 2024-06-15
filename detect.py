!pip install nltk 
import pandas as pd
import numpy as np
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load dataset
df = pd.read_csv('train.csv')

# Display first few rows and check columns
print(df.head())

# Check for any missing values
print(df.isnull().sum())

# Drop rows with missing values, if any
df = df.dropna()

# Reset index after dropping rows
df = df.reset_index(drop=True)

# Check the distribution of labels
print(df['Label'].value_counts())

# Preprocessing function
def preprocess_text(text):
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize text
    words = nltk.word_tokenize(text)
    # Remove stopwords and lemmatize
    words = [WordNetLemmatizer().lemmatize(word) for word in words if word not in stopwords.words('english')]
    # Join words back into sentence
    return ' '.join(words)

# Apply preprocessing to 'statement' column
df['Statement'] = df['Statement'].apply(preprocess_text)

# Display preprocessed data
print(df.head())

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df['Statement'], df['Label'], test_size=0.3, random_state=42)

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))

# Fit and transform TF-IDF Vectorizer on training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Transform test data using fitted TF-IDF Vectorizer
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Initialize Passive Aggressive Classifier
classifier = PassiveAggressiveClassifier(max_iter=50)

# Train the classifier
classifier.fit(X_train_tfidf, y_train)

# Predict on test data
y_pred = classifier.predict(X_test_tfidf)

# Calculate accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Plot confusion matrix
cm = metrics.confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Save the model
pickle.dump(classifier, open('model.pkl', 'wb'))

# Example function to load and use the model
def predict_fake_news(text):
    # Load the saved model
    loaded_model = pickle.load(open('model.pkl', 'rb'))
    # Preprocess input text
    processed_text = preprocess_text(text)
    # Vectorize using TF-IDF
    vectorized_text = tfidf_vectorizer.transform([processed_text])
    # Predict using loaded model
    prediction = loaded_model.predict(vectorized_text)
    return prediction[0]

# Example usage
test_news = "This news article is fake."
prediction = predict_fake_news(test_news)
if prediction == 1:
    print("Prediction: Real News")
else:
    print("Prediction: Fake News")
from sklearn.model_selection import learning_curve

# Function to plot learning curve
def plot_learning_curve(estimator, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title("Learning Curve")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# Plot learning curve
plot_learning_curve(classifier, X_train_tfidf, y_train, cv=5, n_jobs=-1)
plt.show()
from sklearn.model_selection import validation_curve

# Function to plot validation curve
def plot_validation_curve(estimator, X, y, param_name, param_range, cv=None, scoring="accuracy"):
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring=scoring, n_jobs=-1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title("Validation Curve")
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.plot(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    return plt

# Define parameter range for validation curve (e.g., max_iter for PassiveAggressiveClassifier)
param_range = np.arange(1, 100, 10)

# Plot validation curve
plot_validation_curve(classifier, X_train_tfidf, y_train, param_name="max_iter", param_range=param_range, cv=5)
plt.show()
