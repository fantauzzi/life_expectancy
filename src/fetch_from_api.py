import requests
import pandas as pd


def get_who_table(url: str, index: str | None = None) -> pd.DataFrame:
    """
    Fetches data from the GHO OData API and returns them as a dataframe,
    :param url: The URL for a get request to the GHO OData API.
    :param index: The column name to be used as index; optional.
    :return: The requested dataframe.
    """
    resp = requests.get(url)
    resp_json = resp.json()

    resp_value = resp_json['value']  # A sequence of dictionaries
    columns = resp_value[0].keys()  # Use the first item in the sequence to figure out the column names

    items = [list(item[column] for column in columns) for item in resp_value]

    df = pd.DataFrame(data=items,
                      columns=columns)
    if index is not None:
        assert len(df[index].unique()) == len(df)
        df.set_index(index, drop=True, inplace=True)

    return df


def get_who_indicators() -> pd.DataFrame:
    """
    Fetches and returns the indicators available from the GHO OData API.
    :return: a dataframe with the requested indicators.
    """
    df = get_who_table('https://ghoapi.azureedge.net/api/Indicator', index='IndicatorCode')
    return df


def get_who_dimensions() -> pd.DataFrame:
    """
    Fetches and returns the dimensions available from the GHO OData API.
    :return: a dataframe with the requested dimensions.
    """
    df = get_who_table('https://ghoapi.azureedge.net/api/Dimension', index='Code')
    return df


def get_who_life_expectancy() -> pd.DataFrame:
    """
    Fetches and returns the life expectancy information from the GHO OData API.
    :return: a dataframe with the life expectancy information.
    """
    df = get_who_table('https://ghoapi.azureedge.net/api/WHOSIS_000001', index='Id')
    assert (df.IndicatorCode == 'WHOSIS_000001').all()
    assert (df.TimeDimType == 'YEAR').all()
    assert (df.Dim1Type == 'SEX').all()
    must_be_null = ['Dim2Type',
                    'Dim2',
                    'Dim3Type',
                    'Dim3',
                    'DataSourceDimType',
                    'DataSourceDim',
                    'Low',
                    'High',
                    'Comments']

    for col in must_be_null:
        assert df[col].isnull().all(), f'Columns {must_be_null} contains non-null values'

    df.drop(must_be_null + ['TimeDimensionBegin', 'TimeDimensionEnd', 'IndicatorCode', 'TimeDimType', 'Dim1Type'],
            inplace=True, axis=1)
    return df


def main():
    """
    The WHO dataset is composed of indicators. E.g. "Life expectancy at birth" is an indicator. The indicator comes
    with a number of samples. Each sample provides the value for the indicator, and also the values of a number of
    categorical variables. The WHO dataset calls those categorical variables "dimensions". Not all indicators share the
    same set of dimensions: different indicators may, in general, have different dimensions. E.g. the "Life expectancy
    at birth" indicator includes dimensions for country, year and sex.
    """
    who_indicators = get_who_indicators()
    lang_values = who_indicators.Language.unique()
    assert list(lang_values) == ['EN']
    name_values = who_indicators.IndicatorName.unique()
    print(f'Found {len(who_indicators)} unique indicator codes and {len(name_values)} unique indicator names.')

    who_dimensions = get_who_dimensions()

    life_exp = get_who_life_expectancy()
    pass


if __name__ == '__main__':
    main()
