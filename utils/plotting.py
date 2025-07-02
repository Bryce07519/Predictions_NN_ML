import matplotlib.pyplot as plt


def plot_equity_curve(cumulative_returns, title="Equity Curve"):
    plt.figure(figsize=(10, 4))
    plt.plot(cumulative_returns)
    plt.title(title)
    plt.xlabel("Steps")
    plt.ylabel("Equity")
    plt.grid(True)
    plt.show()