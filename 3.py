import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def run_fit(df, train_size, feature):
    def print_performance(test, pred, model):
        print(f"LR Beta  : {model.coef_}")
        print(f"LR Alpha : {model.intercept_}")

        print(f"Coefficient of Determination : {r2_score(test, pred)}")
        print(f"RMSE : {np.sqrt(mean_squared_error(test, pred))}")
        print(f"MSE  : {mean_squared_error(test, pred)}")

    print(f"======= Train Size {train_size} =======")
    X = df.drop(labels=feature, axis=1)
    y = df[feature]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_size, random_state=42)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr.predict(X_test)
    y_pred = lr.predict(X_test)

    print_performance(y_test, y_pred, lr)

    print("\n")
    return lr


def predict(df, sample, model, feature):
    x = df.iloc[50, :].drop(feature).values
    y = df.iloc[50, :][feature]
    return y, model.predict(x.reshape(1,-1))[0]


def print_sample_performance(y, sample1, sample2):
    print("Data   | Value | RMSE")
    print("---------------------")
    print(f"Actual |  {y:3.2f} | 0.0")
    print(f"0.3    |  {sample1:3.2f} | {np.sqrt(mean_squared_error([y], [sample1])):3.2f}")
    print(f"0.7    |  {sample2:3.2f} | {np.sqrt(mean_squared_error([y], [sample2])):3.2f}")


def main():
    ### Load in Iris data
    iris = load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    #convert_species = np.vectorize(lambda x : "setosa" if x==0 else ("versicolor" if x==1 else "virginica"))
    #iris_df["species"] = convert_species(iris.target)

    FEATURE = "petal length (cm)"
    lr30 = run_fit(iris_df, 0.3, FEATURE)
    lr70 = run_fit(iris_df, 0.7, FEATURE)

    y, sample30 = predict(iris_df, 50, lr30, FEATURE)
    _, sample70 = predict(iris_df, 50, lr70, FEATURE)

    print_sample_performance(y, sample30, sample70)

if __name__ == "__main__":
    main()
