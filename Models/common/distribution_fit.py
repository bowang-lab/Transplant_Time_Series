import scipy
import scipy.stats

import matplotlib
import matplotlib.pyplot as plt


class Distribution(object):
    def __init__(self, dist_names_list=[]):
        self.dist_names = ["norm", "lognorm", "expon", "pareto"]
        self.dist_results = []
        self.params = {}

        self.DistributionName = ""
        self.PValue = 0
        self.Param = None

        self.isFitted = False

    def Fit(self, y):
        self.dist_results = []
        self.params = {}
        for dist_name in self.dist_names:
            dist = getattr(scipy.stats, dist_name)
            param = dist.fit(y)

            self.params[dist_name] = param
            # Applying the Kolmogorov-Smirnov test
            D, p = scipy.stats.kstest(y, dist_name, args=param)
            self.dist_results.append((dist_name, p))

        # select the best fitted distribution
        sel_dist, p = max(self.dist_results, key=lambda item: item[1])
        # store the name of the best fit and its p value
        self.DistributionName = sel_dist
        self.PValue = p

        self.isFitted = True
        return self.DistributionName, self.PValue

    def Random(self, n=1):
        if self.isFitted:
            dist_name = self.DistributionName
            param = self.params[dist_name]
            # initiate the scipy distribution
            dist = getattr(scipy.stats, dist_name)
            return dist.rvs(*param[:-2], loc=param[-2], scale=param[-1], size=n)
        else:
            raise ValueError("Must first run the Fit method.")

    def Plot(self, y):
        x = self.Random(n=len(y))
        plt.hist(x, alpha=0.5, label="Fitted")
        plt.hist(y, alpha=0.5, label="Actual")
        plt.legend(loc="upper right")
