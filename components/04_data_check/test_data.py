'''
This .py file runs the necessary tests to check our data
after cleaning it after the "basic_clean" step

Author: Vitor Abdo
Date: March/2023
'''

# import necessary packages
import pandas as pd
import scipy.stats

# DETERMINISTIC TESTS

def test_import_data(data):
    '''Test that the dataset is not empty'''

    assert data.shape[0] > 0
    assert data.shape[1] > 0


def test_column_names(data):
    '''Tests if the column names are the same as the original
    file, including in the same order
    '''
    expected_colums = [
        'corporation', 'lastmonth_activity', 'lastyear_activity',
        'number_of_employees', 'exited']

    these_columns = data.columns.values

    # This also enforces the same order
    assert list(expected_colums) == list(these_columns)


def test_entries_values(data):
    '''Test dataset variable entries'''

    # independent variables
    assert all(data.lastmonth_activity.values) >= 0
    assert all(data.lastyear_activity.values) >= 0
    assert all(data.number_of_employees.values) >= 0

    # label
    known_label_entries = [0, 1]
    label_column = set(data.exited.unique())
    assert set(known_label_entries) == set(label_column)


def test_data_integrity(data):
    '''Test data integrity: when data is missing or invalid, 
    we say that there's a data integrity issue
    '''
    check_null_values = list(data.isna().sum())
    assert all(check_null_values) == 0


# NON DETERMINISTIC TESTS

def test_similar_label_distrib(
        data: pd.DataFrame, ref_data: pd.DataFrame):
    '''Apply a threshold on the KL divergence to detect if the distribution of the new
    data is significantly different than that of the reference dataset
    '''
    dist1 = data['exited'].value_counts().sort_index()
    dist2 = ref_data['exited'].value_counts().sort_index()

    assert scipy.stats.entropy(dist1, dist2, base=2) < 0.20


def test_lastmonth_activity_ttest(
        data: pd.DataFrame, ref_data: pd.DataFrame):
    '''Tests whether the means of two independent samples are significantly different'''

    dist1 = data['lastmonth_activity'].values
    dist2 = ref_data['lastmonth_activity'].values

    ts, pvalues = scipy.stats.ttest_ind(dist1, dist2)

    assert pvalues > 0.05


def test_lastyear_activity_ttest(
        data: pd.DataFrame, ref_data: pd.DataFrame):
    '''Tests whether the means of two independent samples are significantly different'''

    dist1 = data['lastyear_activity'].values
    dist2 = ref_data['lastyear_activity'].values

    ts, pvalues = scipy.stats.ttest_ind(dist1, dist2)

    assert pvalues > 0.05


def test_number_of_employees_ttest(
        data: pd.DataFrame, ref_data: pd.DataFrame):
    '''Tests whether the means of two independent samples are significantly different'''

    dist1 = data['number_of_employees'].values
    dist2 = ref_data['number_of_employees'].values

    ts, pvalues = scipy.stats.ttest_ind(dist1, dist2)

    assert pvalues > 0.05


def test_data_stability(
        data: pd.DataFrame, ref_data: pd.DataFrame):
    '''Test data integrity: when the data contains 
    values different from what we expect. We are 
    checking by the mean and median of the data
    '''
    hist_means = list(ref_data.mean())
    actual_means = list(data.mean())

    hist_medians = list(ref_data.median())
    actual_medians = list(data.median())

    mean_comparison = [abs((actual_means[i] - hist_means[i]) / hist_means[i]) for i in range(len(actual_means))]
    median_comparison = [abs((actual_medians[i] - hist_medians[i]) / hist_medians[i]) for i in range(len(actual_medians))]

    assert all(mean_comparison) <= 0.10
    assert all(median_comparison) <= 0.10
