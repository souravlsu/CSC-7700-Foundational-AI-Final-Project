import matplotlib.pyplot as plt          # For plotting
import seaborn as sns                   # For enhanced visualization
import pandas as pd                     # For data manipulation
from sklearn import metrics             # For model evaluation metrics
from sklearn.metrics import classification_report  # For classification performance summary
import numpy as np                      # For numerical operations
from sklearn import tree                # For decision tree models
from sklearn.tree import export_text    # For exporting tree rules as text

# Defining a utility class for plotting and evaluation
class myutils():
    def __init__(self, target):
        """
        Initializes the utility class with a target list (likely target variable names).
        This class is designed to make visualization and evaluation easier.
        """
        self.target = target

    def SingleParityPlot(self, plt, no, ypredict, yactual):
        """
        Draws a parity plot (scatter plot comparing predicted and actual values)
        for a specific output variable (indexed by 'no').

        Parameters:
        - plt: The matplotlib plotting interface
        - no: Index of the target variable to plot
        - ypredict: Numpy array of predicted values
        - yactual: Numpy array of actual/true values
        """
        # Determining plot bounds
        a = np.max(ypredict[:, no])
        b = np.max(yactual[:, no])
        d = np.min(ypredict[:, no])
        c = np.min(yactual[:, no])
        max_value = max(a, b)
        min_value = min(c, d)
        x_values = np.linspace(min_value, max_value, 50)

        # Preparing DataFrame for seaborn plotting
        sns.set(style="whitegrid")
        df = pd.DataFrame({
            'Actual': yactual[:, no],
            'Predicted': ypredict[:, no]
        })

        # Creating scatter plot
        sns.scatterplot(data=df, x='Actual', y='Predicted', s=10, color='blue')
        plt.plot(x_values, x_values, 'k--', linewidth=1)  # Diagonal line y = x for reference

        # Computing metrics
        mse = self.mean_squared_error(yactual[:, no], ypredict[:, no])
        rsq = self.r_squared(yactual[:, no], ypredict[:, no])

        # Setting plot labels and title
        plt.xlabel('Actual values')
        plt.ylabel('Predicted values')
        plt.title(f'Target: {self.target[no+4]} MSE: {mse:.7f} R²: {rsq:.3f}')

    def plot_result(self, results, save_path=None):
        """
        Plots a training/testing loss curve (MSE vs epoch).

        Parameters:
        - results: List or array of MSE values (typically per epoch)
        - save_path: If specified, saves the plot to this path instead of displaying
        """
        plt.figure()
        plt.plot(results, '.-', label='Test', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title('MSE vs Epoch during training')
        plt.grid(True)
        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()

    def mean_squared_error(self, ypredict, yactual):
        """
        Computes Mean Squared Error (MSE) between predicted and actual values.
        Works element-wise across vectors or columns of a matrix.
        """
        return np.mean((ypredict - yactual) ** 2, axis=0)

    def r_squared(self, ypredict, yactual):
        """
        Computes R-squared (coefficient of determination) between predicted and actual values.
        """
        corr = np.corrcoef(ypredict, yactual)
        r_squared = corr[0, 1] ** 2
        return r_squared