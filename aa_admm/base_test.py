# Base class for unit tests.
from unittest import TestCase
import matplotlib.pyplot as plt

class BaseTest(TestCase):
    def plot_results(self, results, labels, ts=None, linestyles=None, pltError=False, yLabel=None, maxIt=None):
        n = len(results)  # number of residual plots
        if linestyles is None:
            for i in range(n):
                result = results[i]
                if maxIt is not None:
                    result = result[:maxIt]
                if ts == None:
                    plt.semilogy(result, label=labels[i], linewidth=2.0)
                else:
                    x = ts[i]
                    plt.semilogy(x, result, label=labels[i], linewidth=2.0)
        else:
            for i in range(n):
                result = results[i]
                if maxIt is not None:
                    result = result[:maxIt]
                if ts == None:
                    plt.semilogy(result, label=labels[i], linewidth=2.0, linestyle=linestyles[i])
                else:
                    x = ts[i]
                    plt.semilogy(x, result, label=labels[i], linewidth=2.0, linestyle=linestyles[i])
            
        plt.legend(prop={'size': 10},loc="upper right")
        
        if ts == None:
            plt.xlabel("Iteration", fontsize=10)
        else:
            plt.xlabel("Time (s)", fontsize=10)
            
        if not pltError:
            plt.ylabel("Norm of combined residual", fontsize=10)
        else:
            if yLabel is None:
                plt.ylabel(r"$||z-z^*||_2$", fontsize=10)
            else:
                plt.ylabel(yLabel)
            
        plt.show()
