from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import cv2
import os

def load_dataset():
    """Load the digits dataset."""
    digits = load_digits()
    X , y = digits.data , digits.target
    return X , y

def show_samples(X,y):
    """show 10 samples from dataset."""
    for i in range(10):
        img = X[i].reshape(8,8)
        img = cv2.resize(img,(200,200))
        img = (img*16).astype("uint8")
        img = cv2.putText(img,str(y[i]), (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1,255, 2)
        cv2.imshow(f"number is {y[i]}",img)
        os.makedirs("images", exist_ok=True)
        cv2.imwrite(f"images/sample_digit_{i}.png",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def preprocess_data(X,y):
    """preprocess data for train."""
    X_train , X_test , y_train , y_test = train_test_split(X,y,test_size = 0.2,random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train , X_test , y_train , y_test

def train_model(X_train,y_train):
    """Train a Logistic Regression classifier."""
    model = LogisticRegression()
    model.fit(X_train,y_train)
    return model

def evaluate_model(X_test,y_test,model):
    """predict test data and compute accuracy."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test,y_pred)
    return acc

def save_results(acc):
    """save accuracy result."""
    os.makedirs("results",exist_ok=True)
    with open("results/results.txt","w") as file:
        file.write(f"Accuracy : {acc:.4f}")

def main():
    X, y = load_dataset()
    show_samples(X,y)
    X_train , X_test , y_train , y_test = preprocess_data(X,y)
    model = train_model(X_train,y_train)
    acc = evaluate_model(X_test,y_test,model)
    save_results(acc)

if __name__ == "__main__":
    main()
