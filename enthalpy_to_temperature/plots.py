from _commons import DB_SIZES, DEG
from matplotlib import pyplot as plt
import numpy as np


def plot_table_evalutation(results_table: dict[dict[str, float]]):
    """
    
    Plot the evaluation of the table method in terms of accuracy and
    computational time for different dataset sizes.
    
    Parameters
    ----------
    results_table : dict[dict[str, float]]
        Dictionary containing the evaluation results for different dataset 
        sizes. Each key is a dataset size and the value is another dictionary 
        with keys "emax", "eave", "emin", and "time" representing the maximum,
        average, minimum error and computational time respectively.
    """    
    fig, axs = plt.subplots(1, 2, figsize=(11,5))
    x = np.arange(len(DB_SIZES))
    width = 0.25
    axs[0].bar(x - width, [results_table[s]["emax"] for s in DB_SIZES], width, 
               label="Max. error")
    axs[0].bar(x,         [results_table[s]["eave"] for s in DB_SIZES], width, 
               label="Ave. error")
    axs[0].bar(x + width, [results_table[s]["emin"] for s in DB_SIZES], width, 
               label="Min. error")
    axs[0].set_yscale('log')
    axs[0].set_xlabel('Dataset width')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels([str(s) for s in DB_SIZES], rotation=90)
    axs[0].set_ylabel('Error [-]')
    axs[0].set_title('Accuracy')
    axs[0].legend(fontsize='small')
    axs[1].bar([str(s) for s in DB_SIZES], [results_table[s]["time"] 
                                         for s in DB_SIZES])
    axs[1].set_xlabel('Dataset width')
    axs[1].set_xticks(x)
    axs[1].set_xticklabels([str(s) for s in DB_SIZES], rotation=90)
    axs[1].set_ylabel('Computational time [s]')
    axs[1].set_title('Computational Time')
    axs[1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axs[1].yaxis.get_offset_text().set_fontsize(10)
    plt.subplots_adjust(wspace=0.4)
    plt.savefig('accuracy_time_tradeoff.png')


def plot_polynomium_results(eave_poly: list[float], time_poly: list[float]):
    """
    Plot the evaluation of the polynomial method in terms of accuracy and
    computational time for different polynomial degrees, compared to the table 
    method.
    
    Parameters
    ----------
    eave_poly : list[float]
        List of average errors for the polynomial method for different 
        polynomial degrees.
    time_poly : list[float]
        List of computational times for the polynomial method for different 
        polynomial degrees.
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(range(1, DEG), eave_poly, '--o', label='Ave. error polynomium')
    axs[0].set_xticks(range(1, DEG))
    axs[0].set_xlim(1, DEG-1)
    axs[0].set_yscale('log')
    axs[0].set_xlabel('Polynomium Degree')
    axs[0].set_ylabel('Error [-]')
    axs[0].set_title('Accuracy')
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(range(1, DEG), time_poly, '--o', label='Ave. time polynomium')
    axs[1].set_xlabel('Polynomium Degree')
    axs[1].set_xticks(range(1, DEG))
    axs[1].set_xlim(1, DEG-1)
    axs[1].set_ylabel('Computational time [s]')
    axs[1].set_title('Computational Time')
    axs[1].legend()
    axs[1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axs[1].yaxis.get_offset_text().set_fontsize(10)
    axs[1].grid()

    plt.subplots_adjust(wspace=0.4)
    plt.savefig('polynomial_degree.png')
    
    
def plot_accuracy_comparison(reference: np.ndarray, err_newton: np.ndarray,
                             err_table: np.ndarray, err_poly: np.ndarray):
    """
    Plot the accuracy comparison between different methods.
    
    Parameters
    ----------
    reference : np.ndarray
        Reference data to compare against
    err_newton : np.ndarray
        Relative error of the Newton method
    err_table : np.ndarray
        Relative error of the table method
    err_poly : np.ndarray
        Relative error of the polynomial method
    """
    plt.semilogy(reference[:,0][1:], err_newton, label='Newton', color='blue')
    plt.semilogy(reference[:,0][1:], err_table, label='Table', color='orange')
    plt.semilogy(reference[:,0][1:], err_poly, label='Polynomium (10 deg)', 
                 color='green')
    plt.xlabel('Temperature [K]')
    plt.ylabel('Relative Error [-]')
    plt.legend(bbox_to_anchor=(1, 1.02))
    plt.grid()
    plt.savefig('accuracy_comparison.png') 