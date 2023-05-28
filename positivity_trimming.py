from __future__ import annotations

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import pandas as pd
from typing import Tuple
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

OPTIMAL_THRESHOLD_ACCURACY = 5e-6


def _check_is_valid_threshold_value(threshold_value):
    if not isinstance(threshold_value, (float, type(None))):
        raise ValueError("invalid threshold_value")
    return threshold_value


def _check_is_valid_threshold_method(threshold_method):
    threshold_method = "crump" if threshold_method == "auto" else threshold_method
    if threshold_method not in cutoff_optimizers:
        raise ValueError("invalid threshold_method")
    return threshold_method


def _check_propensities(prob):
    """ check if the treatment assignment is binary"""
    if prob.shape[1] > 2:
        raise ValueError('This threshold selection method is applicable only '
                         'for binary treatment assignment')
    else:
        propensities = prob.iloc[:, 1]
        return propensities


def crump_cutoff(prob, segments=10000):
    """
    A systematic approach to find the optimal trimming cutoff, based on the
    marginal distribution of the propensity score,
    and according to a variance minimization criterion.
    "Crump, R. K., Hotz, V. J., Imbens, G. W., & Mitnik, O. A. (2009).
    Dealing with limited overlap in estimation of average treatment effects."
    Args:
        prob (pd.Series): probability of be assign to a group
                          (n_samples, n_classes)
        segments (int): number of exclusive segments of the interval (0, 0.5].
                        more segments results with more precise cutoff
    Returns:
        float: the optimal cutoff,
               i.e. the smallest value that satisfies the criterion.
    """
    propensities = _check_propensities(prob)
    alphas = np.linspace(1e-7, 0.5, segments)
    alphas_weights = alphas * (1 - alphas)
    overlap_weights = propensities * (1 - propensities)
    for i in range(segments):
        obs_meets_criterion = overlap_weights >= alphas_weights[i]
        criterion = 2 * (np.sum(obs_meets_criterion / overlap_weights) /
                         np.maximum(np.sum(obs_meets_criterion), 1e-7))
        if (1 / alphas_weights[i]) <= criterion:
            break
    return alphas[i]


cutoff_optimizers = {'crump': crump_cutoff}


def _lookup_method(threshold_method):
    if threshold_method in cutoff_optimizers:
        return cutoff_optimizers[threshold_method]
    else:
        raise Exception("Method %s does not exist" % threshold_method)


class Trimming(ABC, BaseEstimator):
    def __init__(self,
                 learner=LogisticRegression(),
                 threshold="auto"):
        """
        Args:
            learner (sklearn object): Initialized sklearn model
            threshold (str | float) : The threshold method or value.
                - if auto: finding the optimized threshold in a principled way.
                - if float, hard-coded value between 0 to 0.5 is used
                  in order to clip the propensity estimation.
        """
        self.learner = learner
        if not hasattr(self.learner, "predict_proba"):
            raise AttributeError("Propensity Estimator must use a machine "
                                 "learning that can predict probabilities"
                                 "(i.e., have predict_proba method)")

        if isinstance(threshold, str):
            self.threshold = _check_is_valid_threshold_method(threshold)
        else:
            self.threshold_ = _check_is_valid_threshold_value(threshold)

    def _fit_threshold(self, X):
        """Fit threshold in a principled way"""
        prob = self.learner.predict_proba(X)
        prob = pd.DataFrame(prob, index=X.index, columns=self.learner.classes_)
        method = _lookup_method(self.threshold)
        threshold = method(prob)
        return threshold

    def fit(self, X, a):
        """Fit propensity model for positivity.
        Args:
            X (pd.DataFrame): covariate matrix of size
                              (num_subjects, num_features)
            a (pd.Series): treatment assignment of size (num_subjects,)
        """
        self.learner.fit(X, a)
        if hasattr(self, 'threshold'):
            self.threshold_ = self._fit_threshold(X)
        return self

    def predict(self, X, a, threshold=None):
        """Predict whether or not a sample is in the overlap region.
        Find samples that have probabilities to be assigned to one of the
        treatment groups, that is bigger than the cutoff threshold.
        return a boolean indexer which is `True` if their probabilities are
        higher than the cutoff threshold and `False` otherwise.
        Args:
            X (pd.DataFrame): covariate matrix of size
                              (num_subjects, num_features)
            a (pd.Series): treatment assignment of size (num_subjects,)
            threshold (float|None): The cutoff threshold.
                - if float, an optional value between 0 to 0.5 to clip the
                    propensity estimation.
                - if None, use the optimized cutoff in a principled way.
        Returns:
            pd.Series: a Series of length `X.shape[0]` with the same index as
               `X` and only boolean values
        """
        prob = self.learner.predict_proba(X)
        prob = pd.DataFrame(prob, index=X.index, columns=self.learner.classes_)

        threshold_value = _check_is_valid_threshold_value(threshold)
        threshold_to_use = (self.threshold_ if threshold_value is None
                            else threshold_value)

        untrimmed_indices = (prob >= threshold_to_use).all(axis=1)
        return untrimmed_indices

    def transform(self,
                  X: pd.DataFrame, a: pd.Series, *args: pd.Series
                  ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Transform the input data to remove positivity violations.
        Args:
            X (pd.DataFrame): Covariate matrix of size (num_subjects, num_features).
            a (pd.Series): Treatment assignment of size (num_subjects,).
            *args (pd.Series): Zero or more pd.Series objects corresponding to 
               outcomes. Each argument must be indexed the same as the other 
               arguments and have size (num_subjects,).
        Returns:
            Tuple[pd.DataFrame, pd.Series, pd.Series]: Subsets of `X`, `a` and
            the output series objects of `args` corresponding to the samples
            which do not violate the positivity assumption.
        """
        indices_to_keep = self.predict(X, a)
        return_list = [X.loc[indices_to_keep], a.loc[indices_to_keep]]
        for output in args:
            return_list.append(output.loc[indices_to_keep])
        return return_list

    def fit_predict(self, X: pd.DataFrame, a: pd.Series) -> pd.Series:
        """Fit positivity checker and predict overlap membership.
        This is a convenience function that calls `fit` and `predict`.
        Args:
            X (pd.DataFrame): Covariate matrix of size (num_subjects, num_features)
            a (pd.Series): Treatment assignment of size (num_subjects,).
        Returns:
            pd.Series: a Series of length `X.shape[0]` with the same index as
               `X` and only boolean values
        """
        self.fit(X, a)
        return self.predict(X, a)

    def fit_transform(self,
                      X: pd.DataFrame, a: pd.Series, *args: pd.Series
                      ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Fit and transform data by removing positivity violations.
        Args:
            X (pd.DataFrame): Covariate matrix of size (num_subjects, num_features).
            a (pd.Series): Treatment assignment of size (num_subjects,).
            *args (pd.Series): Zero or more pd.Series objects corresponding to 
               outcomes. Each argument must be indexed the same as the other 
               arguments and have size (num_subjects,).
        Returns:
            Tuple[pd.DataFrame, pd.Series, pd.Series]: Subsets of `X`, `a` and
            the output series objects of `args` corresponding to the samples
            which do not violate the positivity assumption.
        """
        self.fit(X, a)
        return self.transform(X, a, *args)

    def score(self,
              X: pd.DataFrame, a: pd.Series,
              **kwargs):
        """Score the positivity violation
        This is a generic function, but right now it receives
        only one kind of scorer - cross_covaraince_score
        Args:
            X (pd.DataFrame): Covariate matrix of size (num_subjects, num_features).
            a (pd.Series): Treatment assignment of size (num_subjects,).
            **kwargs : kwargs that are corresponding to the scoring metric.
        Returns:
            float: a non-negative score that quantifies the violation
            of positivity
        """
        from .scorers.metrics import cross_covariance_score
        X_trans, a_trans = self.transform(X, a)
        return cross_covariance_score(X_trans, a_trans, **kwargs)