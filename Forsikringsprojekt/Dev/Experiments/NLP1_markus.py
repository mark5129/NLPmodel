# Import necessary libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score

# Sample dataset
texts = [
    "I love this movie, it was fantastic!",
    "The food was terrible and service was slow.",
    "Had a great time at the park today.",
    "Absolutely worst this product. Will never buy again.",
    "What a wonderful experience, truly enjoyed it.",
    "It was a boring and dull evening.",
]
labels = ["positive", "negative", "positive", "negative", "positive", "negative"]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)

# Build the pipeline: TF-IDF vectorizer + Naive Bayes classifier
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))

# Test with new samples
new_samples = ["I absolutely loved the show!", "This was the worst meal I've ever had."]
predictions = model.predict(new_samples)
print("\nPredictions on new samples:")
for text, pred in zip(new_samples, predictions):
    print(f"'{text}' -> {pred}")
