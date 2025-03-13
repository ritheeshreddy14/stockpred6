# Plot Results function for Prediction code (Python)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns
from sklearn.metrics import mean_squared_error

def plot_results(t, y, name):
    """
    Plot results of prediction model similar to MATLAB's PlotResults function

    Parameters:
    t (array): Target values
    y (array): Predicted values
    name (str): Title for the plots
    """
    fig = plt.figure(figsize=(14, 10))

    # t and y
    plt.subplot(2, 2, 1)
    plt.plot(y, 'k', label='Outputs')
    plt.plot(t, color=[0, 0.4470, 0.7410], label='Targets')
    plt.legend()
    plt.title(name)
    plt.grid(True, alpha=0.3)

    # Correlation Plot
    plt.subplot(2, 2, 2)
    plt.scatter(t, y, color='k', alpha=0.5)
    xmin = min(min(t), min(y))
    xmax = max(max(t), max(y))
    plt.plot([xmin, xmax], [xmin, xmax], 'b', linewidth=2)

    # Calculate correlation coefficient
    R, _ = pearsonr(t.flatten(), y.flatten())
    plt.title(f'R = {R:.4f}')
    plt.xlabel('Targets')
    plt.ylabel('Outputs')
    plt.grid(True, alpha=0.3)

    # Error plot
    plt.subplot(2, 2, 3)
    e = t - y
    plt.plot(e, 'b')
    plt.legend(['Error'])
    MSE = np.mean(np.square(e))
    RMSE = np.sqrt(MSE)
    plt.title(f'MSE = {MSE:.4f}, RMSE = {RMSE:.4f}')
    plt.grid(True, alpha=0.3)

    # Error distribution
    plt.subplot(2, 2, 4)
    sns.histplot(e, kde=True, bins=50)
    eMean = np.mean(e)
    eStd = np.std(e)
    plt.title(f'μ = {eMean:.4f}, σ = {eStd:.4f}')
    plt.xlabel('Error')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

