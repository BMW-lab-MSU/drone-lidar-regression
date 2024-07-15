# Importing libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

def main():
    print("Loading the dataset...")

    # Load the dataset
    try:
        print("Loading stacked image data...")
        X = pd.read_csv('stacked_total_image_data.csv')
        print("Loaded stacked image data")
        print("Loading frequency labels...")
        y = pd.read_csv('rpm1_values.csv')
        print("Loaded frequency labels...")
        print("Datasets loaded successfully...")
    except FileNotFoundError as e:
        print(f"Error loading datasets: {e}")
        return

    print("Making regression model...")
    # Fit regression model
    regr_1 = DecisionTreeRegressor(max_depth=2)
    regr_2 = DecisionTreeRegressor(max_depth=5)
    regr_1.fit(X, y)
    regr_2.fit(X, y)
    print("Beautiful regression model...")

    print("PREDICTING THE DATA...")
    # Predict
    X_test = np.linspace(min(X), max(X), 500).reshape(-1,1)
    y_1 = regr_1.predict(X_test)
    y_2 = regr_2.predict(X_test)
    print("Data was predicted...")

    print("Plotting the results...")
    # Plot the results
    plt.figure(figsize=(10,6))
    plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data")
    plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=2", linewidth=2)
    plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
    plt.xlabel("data")
    plt.ylabel("target")
    plt.title("Decision Tree Regression")
    plt.legend()
    plt.show()
    print("Results plotted!")

if __name__ == "__main__":
        main()