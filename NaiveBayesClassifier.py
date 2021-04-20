import math
import numpy as np

"""
** Naive Bayes Classifier **
Simple implementation of the Naive Bayes Classifier.

"""


class NaiveBayesClassifier:
    def __init__(self, features, labels):
        self.means, self.std = [], []
        self.features, self.labels = features, labels
        self.classes = np.unique(labels)
        cls_idx, cls_features = [0] * \
            len(self.classes), [0] * len(self.classes)
        label, counts = np.unique(labels, return_counts=True)
        self.prior = dict(zip(label, counts))
        for c in self.classes:
            cls_idx[c] = np.argwhere(labels == c)
            cls_features[c] = features[cls_idx[c], :]
            self.prior[c] = self.prior[c] / sum(list(self.prior.values()))
        n_features = features.shape[1]
        self.means = [np.mean(cls_features[c], axis=0).reshape(n_features,)
                      for c in self.classes]
        self.std = [np.std(cls_features[c], axis=0)[0]
                    for c in self.classes]

    def _calc_likelihood(self, x, mean, stdev):
        exponent = math.exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    def _predict(self, sample):
        self.posterior = {label: math.log(
            self.prior[label], math.e) for label in self.classes}
        for c in self.classes:
            # calculate likelihood for each feature
            for i in range(len(self.means)):
                # log(prior) += log(feature_1 | class) + log(feature_2 | class) + ...
                self.posterior[c] += math.log(self._calc_likelihood(
                    sample[i], self.means[c][i], self.std[c][i]), math.e)

        # math.log(sum([math.e ** val for val in self.posterior.values()]), math.e)
        evidence = 0
        # posterior = prior * likelihood / evidence
        self.posterior = {
            c: (math.e ** (self.posterior[c] - evidence)) for c in self.posterior}

        return self.posterior

    def predict(self, samples):
        predictions = []
        for sample in samples:
            predicted_c, max_p = None, 0
            for c, p in self._predict(sample).items():
                if p > max_p:
                    max_p, predicted_c = p, c
            predictions.append((max_p, predicted_c))
        return predictions
