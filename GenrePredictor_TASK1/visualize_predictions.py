import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the predictions data
predicted_data = pd.read_csv(r'C:\Users\nk452\MachineLearninhML\newMovie\predicted_genres.csv')

# Plot the distribution of predicted genres
plt.figure(figsize=(10, 6))
sns.countplot(y='Predicted_Genre', data=predicted_data, order=predicted_data['Predicted_Genre'].value_counts().index)
plt.title('Distribution of Predicted Genres')
plt.xlabel('Count')
plt.ylabel('Genre')
plt.show()
