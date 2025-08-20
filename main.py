from preprocessing import preprocess_data
from train_model import train_model
from evaluate import evaluate_model

if __name__ == "__main__":
    print("Step 1: Preprocessing...")
    reviews, sentiments, cv, data = preprocess_data("data/Movie_Review.csv")

    print("Step 2: Training...")
    model, X_train, X_test, y_train, y_test = train_model(reviews, sentiments)  

    print("Step 4: Evaluating...")
    evaluate_model(model, X_test, y_test)
