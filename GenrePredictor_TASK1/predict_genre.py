import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load the model and vectorizer
try:
    model = joblib.load('movie_genre_predictor.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    print("Model and vectorizer loaded successfully.")
except Exception as e:
    print("Error loading .pkl files:", e)
    exit()

# Load the test data
test_data = pd.read_csv('C:/Users/nk452/MachineLearninhML/movieGenrePredictor/test_data_solution.txt', delimiter=' ::: ', header=None, names=['ID', 'Title', 'Description'], engine='python')
test_data.dropna(subset=['Description'], inplace=True)

# Prepare features
X_test = test_data['Description']

# Transform the text data
X_test_tfidf = tfidf.transform(X_test)

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Add predictions to the dataframe
test_data['Predicted_Genre'] = y_pred

# Save the predictions to a CSV file
test_data[['ID', 'Title', 'Description', 'Predicted_Genre']].to_csv('predicted_genres.csv', index=False)

print("Predictions saved to 'predicted_genres.csv'.")
