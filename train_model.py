from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def train_model(reviews, sentiments):
    X_train, X_test, y_train, y_test = train_test_split(
        reviews, sentiments, test_size=0.2, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test
