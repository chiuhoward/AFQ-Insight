import numpy as np
from scipy.linalg import lstsq
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


def find_subset_indices(X_full, X_subset, method="hash"):
    """
    Find row indices in X_full that correspond to rows in X_subset.
    Supports 'hash' (fast) and 'precise' (element-wise) matching.
    """
    if X_full.shape[1] != X_subset.shape[1]:
        raise ValueError(
            f"Feature dimensions don't match: {X_full.shape[1]} vs {X_subset.shape[1]}"
        )
    indices = []
    if method == "precise":
        for i, subset_row in enumerate(X_subset):
            matches = [
                j
                for j, full_row in enumerate(X_full)
                if np.array_equal(full_row, subset_row, equal_nan=True)
            ]
            if not matches:
                raise ValueError(f"No matching row found for subset row {i}")
            indices.append(matches[0])
    elif method == "hash":
        full_hashes = [hash(row.tobytes()) for row in X_full]
        for i, subset_row in enumerate(X_subset):
            subset_hash = hash(subset_row.tobytes())
            try:
                indices.append(full_hashes.index(subset_hash))
            except ValueError as e:
                raise ValueError(f"No matching row found for subset row {i}") from e
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'hash' or 'precise'.")
    return np.array(indices)


class IdentityTransformer(BaseEstimator, TransformerMixin):
    """A transformer that returns the input unchanged."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


class CovariateRegressor(BaseEstimator, TransformerMixin):
    """
    Fits covariate(s) onto each feature in X and returns their residuals.
    """

    def __init__(
        self,
        covariate,
        X,
        pipeline=None,
        cross_validate=True,
        precise=False,
        unique_id_col_index=None,
        stack_intercept=True,
    ):
        """Regresses out a variable (covariate) from each feature in X.

        Parameters
        ----------
        covariate : numpy array
            Array of length (n_samples, n_covariates) to regress out of each
            feature; May have multiple columns for multiple covariates.
        X : numpy array
            Array of length (n_samples, n_features), from which the covariate
            will be regressed. This is used to determine how the
            covariate-models should be cross-validated (which is necessary
            to use in in scikit-learn Pipelines).
        cross_validate : bool
            Whether to cross-validate the covariate-parameters (y~covariate)
            estimated from the train-set to the test set (cross_validate=True)
            or whether to fit the covariate regressor separately on the test-set
            (cross_validate=False). Setting this parameter to True is equivalent
            to "foldwise covariate regression" (FwCR) as described in our paper
            (https://www.biorxiv.org/content/early/2018/03/28/290684). Setting
            this parameter to False, however, is NOT equivalent to "whole
            dataset covariate regression" (WDCR) as it does not apply covariate
            regression to the *full* dataset, but simply refits the covariate
            model on the test-set. We recommend setting this parameter to True.
        precise: bool
            Transformer-objects in scikit-learn only allow to pass the data
            (X) and optionally the target (y) to the fit and transform methods.
            However, we need to index the covariate accordingly as well. To do so,
            we compare the X during initialization (self.X) with the X passed to
            fit/transform. As such, we can infer which samples are passed to the
            methods and index the covariate accordingly. When setting precise to
            True, the arrays are compared feature-wise, which is accurate, but
            relatively slow. When setting precise to False, it will infer the index
            by looking at the hash of all the features, which is much
            faster. Also, to aid the accuracy, we remove the features which are constant
            (0) across samples.
        stack_intercept : bool
            Whether to stack an intercept to the covariate (default is True)

        Attributes
        ----------
        weights_ : numpy array
            Array with weights for the covariate(s).

        Notes
        -----
        This is a modified version of the ConfoundRegressor from [1]_.

        References
        ----------
        .. [1] Lukas Snoek, Steven Miletić, H. Steven Scholte,
            "How to control for confounds in decoding analyses of neuroimaging data",
            NeuroImage, Volume 184, 2019, Pages 741-760, ISSN 1053-8119,
            https://doi.org/10.1016/j.neuroimage.2018.09.074.
        """
        self.covariate = covariate.astype(np.float64)
        self.cross_validate = cross_validate
        self.X = X
        self.precise = precise
        self.stack_intercept = stack_intercept
        self.weights_ = None
        self.pipeline = pipeline
        self.imputer = SimpleImputer(strategy="median")
        self.X_imputer = SimpleImputer(strategy="median")
        self.unique_id_col_index = unique_id_col_index

    def _prepare_covariate(self, covariate):
        """Prepare covariate matrix (adds intercept if needed)"""
        if self.stack_intercept:
            return np.c_[np.ones((covariate.shape[0], 1)), covariate]
        return covariate

    def fit(self, X, y=None):
        """Fits the covariate-regressor to X.

        Parameters
        ----------
        X : numpy array
            An array of shape (n_samples, n_features), which should correspond
            to your train-set only!
        y : None
            Included for compatibility; does nothing.
        """

        # Prepare covariate matrix (adds intercept if needed)
        covariate = self._prepare_covariate(self.covariate)

        # Find indices of X subset in the original X
        method = "precise" if self.precise else "hash"
        fit_idx = find_subset_indices(self.X, X, method=method)

        # Remove unique ID column if specified
        if self.unique_id_col_index is not None:
            X = np.delete(X, self.unique_id_col_index, axis=1)

        # Extract covariate data for the fitting subset
        covariate_fit = covariate[fit_idx, :]

        # Conditional imputation for covariate data
        if np.isnan(covariate_fit).any():
            covariate_fit = self.imputer.fit_transform(covariate_fit)
        else:
            # Still fit the imputer for consistency in transform
            self.imputer.fit(covariate_fit)

        # Apply pipeline transformation if specified
        if self.pipeline is not None:
            X = self.pipeline.fit_transform(X)

        # Conditional imputation for X
        if np.isnan(X).any():
            X = self.X_imputer.fit_transform(X)
        else:
            # Still fit the imputer for consistency in transform
            self.X_imputer.fit(X)

        # Fit linear regression: X = covariate * weights + residuals
        # Using scipy's lstsq for numerical stability
        self.weights_ = lstsq(covariate_fit, X)[0]

        return self

    def transform(self, X):
        """Regresses out covariate from X.

        Parameters
        ----------
        X : numpy array
            An array of shape (n_samples, n_features), which should correspond
            to your train-set only!

        Returns
        -------
        X_new : ndarray
            ndarray with covariate-regressed features
        """

        if not self.cross_validate:
            self.fit(X)

        # Prepare covariate matrix (adds intercept if needed)
        covariate = self._prepare_covariate(self.covariate)

        # Find indices of X subset in the original X
        method = "precise" if self.precise else "hash"
        transform_idx = find_subset_indices(self.X, X, method=method)

        # Remove unique ID column if specified
        if self.unique_id_col_index is not None:
            X = np.delete(X, self.unique_id_col_index, axis=1)

        # Extract covariate data for the transform subset
        covariate_transform = covariate[transform_idx]

        # Conditional imputation for covariate data (use fitted imputer)
        if np.isnan(covariate_transform).any():
            covariate_transform = self.imputer.transform(covariate_transform)

        # Apply pipeline transformation if specified
        if self.pipeline is not None:
            X = self.pipeline.transform(X)

        # Conditional imputation for X (use fitted imputer)
        if np.isnan(X).any():
            X = self.X_imputer.transform(X)

        # Compute residuals
        X_new = X - covariate_transform.dot(self.weights_)

        # Ensure no NaNs in output
        X_new = np.nan_to_num(X_new)

        return X_new
