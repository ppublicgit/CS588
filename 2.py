import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def lin_reg(x, y):
    SSxy = np.sum(y*x) - x.shape[0]*np.mean(y)*np.mean(x)
    SSxx = np.sum(x*x) - x.shape[0]*np.mean(x)*np.mean(x)
    beta = SSxy / SSxx
    alpha = np.mean(y) - beta* np.mean(x)
    return alpha, beta


def sklearn_lin_reg(x, y):
    reg = LinearRegression().fit(x.reshape(-1, 1), y)
    return reg.intercept_, reg.coef_[0]


def plot_lin_reg(x, y, alpha, beta, **kwargs):
    title = kwargs.get("title", "Linear Regression Model")
    xlabel = kwargs.get("xlabel", "x")
    ylabel = kwargs.get("ylabel", "y")
    scatter = kwargs.get("scatter", True)
    label = kwargs.get("label", "")
    fig = kwargs.get("fig", None)
    ax = kwargs.get("ax", None)
    linestyle = kwargs.get("linestyle", "-")

    if ((fig is not None) + (ax is not None)) == 1:
        raise ValueError("Must specify both or neither a fig and ax kwarg")
    elif ((fig is not None) + (ax is not None)) == 0:
        fig, ax = plt.subplots(1, squeeze=False)
    ax[0,0].set_xlabel(xlabel)
    ax[0,0].set_ylabel(ylabel)
    fig.suptitle(title)

    if scatter:
        ax[0,0].scatter(x, y, color="k")
    y_pred = alpha + beta*x
    ax[0,0].plot(x, y_pred, label=label, linestyle=linestyle)

    return fig, ax


def predict(x, alpha, beta):
    return alpha + x *beta


def calc_profit_ratios(alpha, beta, ratio):
    x = np.linspace(3, 12, num=4)
    y_pred = lambda x : predict(x, alpha, beta)
    y = np.array([y_pred(xi) for xi in x])
    index = len(x)
    rat = np.inf
    while rat > ratio:
        x = np.append(x, index*3+3)
        y = np.append(y, y_pred(x[index]))
        rat = y[index]/y[index-4]
        index += 1
    x = np.append(x, index*3+3)
    y = np.append(y, y_pred(x[index]))
    return x[4:], y[4:]/y[:-4]

def main():
    months = np.array([3, 5, 7, 9, 12,15,18])
    profits = np.array([100, 250, 330, 590, 660, 780, 890])

    alpha_m, beta_m = lin_reg(months, profits)

    fig, ax = plot_lin_reg(months, profits, alpha_m, beta_m,
                           xlabel="Months", ylabel="Profits", label="mine")

    alpha_s, beta_s = sklearn_lin_reg(months, profits)

    fig, ax = plot_lin_reg(months, profits, alpha_s, beta_s,
                           xlabel="Months", ylabel="Profits", fig=fig, ax=ax,
                           label="sklearn", linestyle="--")

    ax[0,0].legend()
    fig.show()

    print("======== Linear Regresssion =========")
    print("Model   |   Alpha  |    Beta   |  Month  |   Prediction")
    print("-------------------------------------------------------")
    print(f"Mine    |  {alpha_m:6.2f}  |   {beta_m:6.2f}  |    12   |       {predict(12, alpha_m, beta_m):6.2f}")
    print(f"Sklearn |  {alpha_s:6.2f}  |   {beta_s:6.2f}  |    12   |       {predict(12, alpha_s, beta_s):6.2f}\n")

    xs, ratios = calc_profit_ratios(alpha_m, beta_m, 1.5)

    plt.figure()
    plt.plot(xs, ratios)
    plt.xlabel("Months")
    plt.ylabel("Ratio (Profit at Month/(Profit at Month-12))")
    plt.title("Plot of Ratio vs Month for 12 Month Separation")
    plt.ylim([1, max(ratios)+1])
    plt.xlim([12, max(xs)+3])
    plt.plot([12, max(xs)+3], [1.5, 1.5], "k--", alpha=0.5)
    plt.grid()
    plt.show(block=False)

if __name__ == "__main__":
    main()
    breakpoint()
