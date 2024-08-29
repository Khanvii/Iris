import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Set a seed for reproducibility
seed = 42

# Read original dataset
iris_df = pd.read_csv('data/iris.csv')

# Shuffle the dataset
iris_df = iris_df.sample(frac=1, random_state=seed)

# Selecting features and target data
X = iris_df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
y = iris_df["Species"]  # Changed to a Series for compatibility with sklearn

# Split data into train and test sets (70% training and 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=seed, stratify=y
)

# Create an instance of the random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=seed)

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")  # Accuracy: 0.96

# save the model to disk
joblib.dump(clf, 'rf_model.sav')

