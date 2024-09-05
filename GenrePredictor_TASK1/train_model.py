import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# Load the training data
train_data = pd.read_csv('C:/Users/nk452/MachineLearninhML/movieGenrePredictor/test_data_solution.txt', delimiter=' ::: ', header=None, names=['ID', 'Title', 'Genre', 'Description'], engine='python')
train_data.dropna(subset=['Description', 'Genre'], inplace=True)

# Prepare features and target
X_train = train_data['Description']
y_train = train_data['Genre']

# Vectorize the text data
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)

# Split the data
X_train_tfidf, X_val_tfidf, y_train, y_val = train_test_split(X_train_tfidf, y_train, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_tfidf, y_train)

# Save the model and vectorizer
joblib.dump(model, 'movie_genre_predictor.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

print("Model and vectorizer saved successfully.")
