import pickle
import pandas as pd
import pandas.api.types as pdtypes
import pytest
import sys

from sklearn.model_selection import train_test_split

try:
    from module.data import process_data
    from module.model import inference, compute_model_metrics
except ModuleNotFoundError:
    sys.path.append('./module')
    from data import process_data
    from model import inference, compute_model_metrics

fake_categorical_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",

]


@pytest.fixture(scope="module")
def data():
    return pd.read_csv("data/census_clean.csv", skipinitialspace=True)


def test_column_presence_and_type(data):
    """Tests that cleaned csv file has expected columns and types.

    Args:
        data (pd.DataFrame): Dataset for testing
    """

    required_columns = {
        "age": pdtypes.is_int64_dtype,
        "workclass": pdtypes.is_object_dtype,
        "fnlgt": pdtypes.is_int64_dtype,
        "education": pdtypes.is_object_dtype,
        "education-num": pdtypes.is_int64_dtype,
        "marital-status": pdtypes.is_object_dtype,
        "occupation": pdtypes.is_object_dtype,
        "relationship": pdtypes.is_object_dtype,
        "race": pdtypes.is_object_dtype,
        "sex": pdtypes.is_object_dtype,
        "capital-gain": pdtypes.is_int64_dtype,
        "capital-loss": pdtypes.is_int64_dtype,
        "hours-per-week": pdtypes.is_int64_dtype,
        "native-country": pdtypes.is_object_dtype,
        "salary": pdtypes.is_object_dtype,
    }

    assert set(data.columns.values).issuperset(set(required_columns.keys()))

    # Check that the columns are of the right dtype
    for col_name, format_verification_funct in required_columns.items():

        assert format_verification_funct(
            data[col_name]
        ), f"Column {col_name} failed test {format_verification_funct}"

def marital_status_values(data):
    """Tests that the marital-status column has the expected values.

    Args:
        data (pd.DataFrame): Dataset for testing
    """
    expected_values = {
        "Married-civ-spouse",
        "Divorced",
        "Never-married",
        "Separated",
        "Widowed",
        "Married-spouse-absent",
        "Married-AF-spouse",
    }

    assert set(data["marital-status"].unique()) == expected_values

def relationship_values(data):
    """Tests that the relationship column has the expected values.

    Args:
        data (pd.DataFrame): Dataset for testing
    """
    expected_values = {
        "Wife",
        "Own-child",
        "Husband",
        "Not-in-family",
        "Other-relative",
        "Unmarried",
    }

    assert set(data["relationship"].unique()) == expected_values


def test_sex_values(data):
    """Tests that the sex column has the expected values.

    Args:
        data (pd.DataFrame): Dataset for testing
    """
    expected_values = {
        "Male",
        "Female"
    }

    assert set(data["sex"]) == expected_values


def test_salary_values(data):
    """Tests that the salary column has the expected values.

    Args:
        data (pd.DataFrame): Dataset for testing
    """
    expected_values = {
        "<=50K",
        ">50K"
    }

    assert set(data["salary"]) == expected_values

def test_column_values(data):
    # Check that the columns are of the right dtype
    for col_name in data.columns.values:
        assert not data[col_name].isnull().any(
        ), f"Column {col_name} has null values"


def test_model_input(data):
    for col_name in data.columns.values:
        assert not data[col_name].isnull().any(
        ), f"Features {col_name} has null values"