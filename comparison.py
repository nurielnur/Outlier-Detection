import numpy as np
import matplotlib.pyplot as plt

from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn import svm

from datasets import make_datasets
from OutlierDetection import OutlierDetection

import warnings
warnings.filterwarnings('ignore')

n_samples = 300
outliers_fraction = 0.2
n_features = 2 #For generating 2 dimensional Dataset

datasets = make_datasets(n_samples, outliers_fraction, n_features)


anomaly_algorithms = [
    ("Robust covariance", EllipticEnvelope(contamination=outliers_fraction)),
    ("One-Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel="rbf",
                                      gamma=0.1)),
    ("Isolation Forest", IsolationForest(contamination=outliers_fraction,
                                         random_state=42)),
    ("Proposed Algorithm", OutlierDetection(thrs = 0.025, coef = 0.30))
]



plt.figure(figsize=(len(anomaly_algorithms) * 2 + 5, 12.5))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)
plot_num = 1
rng = np.random.RandomState(42)

for i_dataset, X in enumerate(datasets):
    for name, algorithm in anomaly_algorithms:
        plt.subplot(len(datasets), len(anomaly_algorithms), plot_num)
        if i_dataset == 0:
            plt.title(name, size=18)

        y_pred = algorithm.fit_predict(X)


        plt.scatter(X[:, 0], X[:, 1], s=10, c=y_pred)
        plt.xticks(())
        plt.yticks(())
        plot_num += 1

plt.show()