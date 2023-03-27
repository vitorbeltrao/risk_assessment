'''
This .py file runs the necessary tests to check our data
after cleaning it after the "basic_clean" step

Author: Vitor Abdo
Date: March/2023
'''

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