import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""

    # Number of data points: n
    #n = len(data)

    # x-data for the ECDF: x
    #x = np.sort(data)

    # y-data for the ECDF: y
    #y = np.arange(1, (len(data) + 1) / len(data))

    #return x, y

    return np.sort(data), np.arange(1, len(data)+1) / len(data)


# Draw 100000 samples from Normal distribution with stds of interest: samples_std1, samples_std3, samples_std10
samples_std1 = np.random.normal(20, 1, size=100000)
samples_std3 = np.random.normal(20, 3, size=100000)
samples_std10 = np.random.normal(20, 10, size=100000)

# Make histograms
#plt.hist(samples_std1, normed=True, histtype='step', bins=100)
#plt.hist(samples_std10, normed=True, histtype='step', bins=100)

# Make a legend, set limits and show plot
#_ = plt.legend(('std = 1', 'std = 3', 'std = 10'))
#plt.ylim(-0.01, 0.42)
#

"""
Now that you have a feel for how the Normal PDF looks, let's consider its CDF.
Using the samples you generated in the last exercise (in your namespace as samples_std1, samples_std3, and samples_std10), generate and plot the CDFs.
"""
# Generate CDFs
x_std1, y_std1 = ecdf(samples_std1)
x_std3, y_std3 = ecdf(samples_std3)
x_std10, y_std10 = ecdf(samples_std10)

# Plot CDFs
plt.plot(x_std1, y_std1, marker='.', linestyle='none')
plt.plot(x_std3, y_std3, marker='.', linestyle='none')
plt.plot(x_std10, y_std10, marker='.', linestyle='none')

# Make 2% margin
plt.margins(0.02)

# Make a legend and show the plot
_ = plt.legend(('std = 1', 'std = 3', 'std = 10'), loc='lower right')
plt.show()
