import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')

def clean_review(review):
    return ' '.join(word for word in review.split() if word not in set(stopwords.words('english')))

def preprocess_data(file_path):
    data = pd.read_csv(file_path)

    # Clean reviews
    data['text'] = data['text'].astype(str).apply(clean_review)

    # Replace labels
    data['sentiment'] = data['sentiment'].replace(['pos', 'neg'], [1, 0])

    # Vectorization
    cv = TfidfVectorizer(max_features=2500)
    reviews = cv.fit_transform(data['text']).toarray()
    y = data['sentiment']

    return reviews, y, cv, data
