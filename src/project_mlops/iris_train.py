import pickle
from typing import Annotated

import typer
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

app = typer.Typer()

# Load the dataset
data = load_breast_cancer()
x = data.data
y = data.target

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


@app.command()
def svc(output_file: Annotated[str, typer.Option("--output", "-o")] = "model.ckpt") -> None:
    """Train the svc model."""
    # Train a Support Vector Machine (SVM) model
    model = SVC(kernel="linear", random_state=42)
    model.fit(x_train, y_train)

    with open(output_file, "wb") as f:
        pickle.dump(model, f)

@app.command()
def knn(output_file: Annotated[str, typer.Option("--output", "-o")] = "model.ckpt") -> None:
    """Train the knn model."""

    # Train a K-Nearest Neighbors (KNN) model
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(x_train, y_train)

    with open(output_file, "wb") as f:
        pickle.dump(model, f)


@app.command()
def evaluate(model_file):
    """Evaluate the model."""
    with open(model_file, "rb") as f:
        model = pickle.load(f)

    # Make predictions on the test set
    y_pred = model.predict(x_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)
    return accuracy, report

def main() -> None:
    app()

if __name__ == "__main__":
    app()