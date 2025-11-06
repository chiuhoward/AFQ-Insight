import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from afqinsight.covariate_regressor import (
    CovariateRegressor,
    IdentityTransformer,
    find_subset_indices,
)


class TestFindSubsetIndices:
    def test_find_subset_indices_precise(self):
        """Test find_subset_indices with precise method"""
        np.random.seed(42)
        X_full = np.random.rand(100, 10)
        X_subset = X_full[20:30]  # Known subset

        indices = find_subset_indices(X_full, X_subset, method="precise")
        expected_indices = np.arange(20, 30)

        np.testing.assert_array_equal(indices, expected_indices)

    def test_find_subset_indices_hash(self):
        """Test find_subset_indices with hash method"""
        np.random.seed(42)
        X_full = np.random.rand(50, 10)
        X_subset = X_full[10:20]  # Known subset

        indices = find_subset_indices(X_full, X_subset, method="hash")
        expected_indices = np.arange(10, 20)

        np.testing.assert_array_equal(indices, expected_indices)

    def test_dimension_mismatch_error(self):
        """Test error when dimensions don't match"""
        X_full = np.random.rand(50, 10)
        X_subset = np.random.rand(20, 5)  # Different number of features

        with pytest.raises(ValueError, match="Feature dimensions don't match"):
            find_subset_indices(X_full, X_subset)

    def test_no_matching_row_error(self):
        """Test error when no matching row is found"""
        X_full = np.random.rand(50, 10)
        X_subset = np.random.rand(20, 10)  # Different data, no matches

        with pytest.raises(ValueError, match="No matching row found"):
            find_subset_indices(X_full, X_subset, method="precise")

    def test_invalid_method_error(self):
        """Test error with invalid method"""
        X_full = np.random.rand(50, 10)
        X_subset = X_full[10:20]

        with pytest.raises(ValueError, match="Unknown method"):
            find_subset_indices(X_full, X_subset, method="invalid")


class TestIdentityTransformer:
    def test_identity_transformer(self):
        """Test that IdentityTransformer returns input unchanged"""
        X = np.random.rand(100, 20)
        transformer = IdentityTransformer()

        # Test fit returns self
        assert transformer.fit(X) is transformer

        # Test transform returns unchanged data
        X_transformed = transformer.transform(X)
        np.testing.assert_array_equal(X, X_transformed)


class TestCovariateRegressor:
    def test_basic_functionality(self):
        """Test basic covariate regression"""
        np.random.seed(42)
        n_samples, n_features = 100, 20

        # Create synthetic data
        X = np.random.randn(n_samples, n_features)
        covariate = np.random.randn(n_samples, 2)  # 2 covariates

        # Initialize regressor
        regressor = CovariateRegressor(covariate=covariate, X=X)

        # Fit and transform
        X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)
        regressor.fit(X_train)
        X_residuals = regressor.transform(X_test)

        # Check output shape
        assert X_residuals.shape == X_test.shape

        # Check that weights were fitted
        assert regressor.weights_ is not None
        assert regressor.weights_.shape[0] == 3  # 2 covariates + intercept
        assert regressor.weights_.shape[1] == n_features

    def test_cross_validate_false(self):
        """Test with cross_validate=False"""
        np.random.seed(42)
        n_samples, n_features = 50, 10

        X = np.random.randn(n_samples, n_features)
        covariate = np.random.randn(n_samples, 1)

        regressor = CovariateRegressor(covariate=covariate, X=X, cross_validate=False)

        X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)
        regressor.fit(X_train)
        X_residuals = regressor.transform(X_test)

        assert X_residuals.shape == X_test.shape

    def test_with_missing_values(self):
        """Test handling of missing values"""
        np.random.seed(42)
        n_samples, n_features = 100, 15

        X = np.random.randn(n_samples, n_features)
        covariate = np.random.randn(n_samples, 1)

        # Introduce some NaN values
        X[10:15, 5] = np.nan
        covariate[20:25] = np.nan

        regressor = CovariateRegressor(covariate=covariate, X=X)

        X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)
        regressor.fit(X_train)
        X_residuals = regressor.transform(X_test)

        # Should not contain NaN values
        assert not np.isnan(X_residuals).any()
        assert X_residuals.shape == X_test.shape

    def test_no_intercept(self):
        """Test with stack_intercept=False"""
        np.random.seed(42)
        n_samples, n_features = 50, 8

        X = np.random.randn(n_samples, n_features)
        covariate = np.random.randn(n_samples, 1)

        regressor = CovariateRegressor(covariate=covariate, X=X, stack_intercept=False)

        X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)
        regressor.fit(X_train)
        X_residuals = regressor.transform(X_test)

        # Check that weights shape reflects no intercept
        assert regressor.weights_.shape[0] == 1  # Only covariate, no intercept
        assert X_residuals.shape == X_test.shape

    def test_precise_vs_hash_methods(self):
        """Test that both methods give same results"""

        np.random.seed(42)
        X_full = np.random.rand(50, 10)
        X_subset = X_full[10:20]

        indices_precise = find_subset_indices(X_full, X_subset, method="precise")
        indices_hash = find_subset_indices(X_full, X_subset, method="hash")

        np.testing.assert_array_equal(indices_precise, indices_hash)

    def test_pipeline_integration(self):
        """Test with a custom pipeline"""
        from sklearn.preprocessing import StandardScaler

        np.random.seed(42)
        n_samples, n_features = 100, 10

        X = np.random.randn(n_samples, n_features)
        covariate = np.random.randn(n_samples, 1)
        pipeline = StandardScaler()

        regressor = CovariateRegressor(covariate=covariate, X=X, pipeline=pipeline)

        X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)
        regressor.fit(X_train)
        X_residuals = regressor.transform(X_test)

        assert X_residuals.shape == X_test.shape

    def test_unique_id_column_removal(self):
        """Test removal of unique ID column"""
        np.random.seed(42)
        n_samples, n_features = 50, 10

        # Add a unique ID column at index 0
        X = np.random.randn(n_samples, n_features)
        ids = np.arange(n_samples).reshape(-1, 1)
        X_with_id = np.hstack([ids, X])

        covariate = np.random.randn(n_samples, 1)

        regressor = CovariateRegressor(
            covariate=covariate, X=X_with_id, unique_id_col_index=0
        )

        X_train, X_test = train_test_split(X_with_id, test_size=0.3, random_state=42)
        regressor.fit(X_train)
        X_residuals = regressor.transform(X_test)

        # Output should have one less column (ID removed)
        assert X_residuals.shape[1] == n_features
