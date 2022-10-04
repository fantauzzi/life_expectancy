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

    assert (df.TimeDim.astype(float) == df.TimeDimensionValue.astype(float)).all()
    df.drop(must_be_null + ['TimeDimensionBegin',
                            'TimeDimensionEnd',
                            'IndicatorCode',
                            'TimeDimType',
                            'Dim1Type',
                            'TimeDimensionValue',
                            'Value',
                            'Date'],
            inplace=True, axis=1)
    return df


def get_who_indicator2(indicator: str, index: str | None = None) -> pd.DataFrame:
    """
    Fetches and returns the life expectancy information from the GHO OData API.
    :return: a dataframe with the life expectancy information.
    """
    df = get_who_table(f'https://ghoapi.azureedge.net/api/{indicator}', index)
    # assert (df.IndicatorCode == indicator).all()
    assert (df.TimeDimType == 'YEAR').all()
    # assert (df.Dim1Type == 'SEX').all()
    must_be_null = ['Dim2Type',
                    'Dim2',
                    'Dim3Type',
                    'Dim3',
                    'Comments']

    for col in must_be_null:
        assert df[col].isnull().all(), f'Columns {must_be_null} contains non-null values'

    assert (df.TimeDim.astype(float) == df.TimeDimensionValue.astype(float)).all()
    df.drop(must_be_null + ['TimeDimensionBegin',
                            'TimeDimensionEnd',
                            'IndicatorCode',
                            'TimeDimType',
                            'Dim1Type',
                            'TimeDimensionValue',
                            'Value',
                            'Date',
                            'DataSourceDimType',
                            'DataSourceDim',
                            'Low',
                            'High'],
            inplace=True, axis=1)
    return df


def get_who_indicator(indicator: str, index: str | None = None) -> pd.DataFrame:
    """
    Fetches and returns the life expectancy information from the GHO OData API.
    :return: a dataframe with the life expectancy information.
    """
    df = get_who_table(f'https://ghoapi.azureedge.net/api/{indicator}', index)
    assert (df.IndicatorCode == indicator).all()
    assert (df.TimeDimType == 'YEAR').all()

    assert (df.TimeDim.astype(float) == df.TimeDimensionValue.astype(float)).all()
    df = df[df.SpatialDimType == 'COUNTRY']
    df.dropna(axis=1, how='all', inplace=True)
    return df


def get_who_dataset() -> pd.DataFrame:
    def drop_btsx_inplace(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        df = df[df[column_name].isin(['MLE', 'FMLE'])]
        return df

    indicator_to_code = {'life expectancy': 'WHOSIS_000001',
                         'adult mortality': 'WHOSIS_000004',
                         'infant deaths': 'CM_02',
                         'alcohol': 'SA_0000001400',
                         'measles': 'WHS3_62',
                         'bmi': 'NCD_BMI_MEAN',
                         'under-five deaths': 'CM_01',
                         'polio': 'WHS3_49',
                         'diphtheria': 'WHS4_100',
                         'HIV/AIDS': 'WHS2_138',
                         'thinness  1-19 years': 'NCD_BMI_MINUS2C'
                         }

    indicators: dict[str, pd.DataFrame] = {indicator_name: get_who_indicator(indicator_code, index='Id') for
                                           indicator_name, indicator_code in
                                           indicator_to_code.items()}

    cols_to_be_dropped = ['IndicatorCode',
                          'SpatialDimType',
                          'TimeDimType',
                          'Value',
                          'Date',
                          'TimeDimensionValue',
                          'TimeDimensionBegin',
                          'TimeDimensionEnd']

    for ind in indicators.values():
        ind.drop(cols_to_be_dropped, inplace=True, axis=1)

    indicators['life expectancy'].drop(['Dim1Type'], inplace=True, axis=1)
    indicators['life expectancy'] = drop_btsx_inplace(indicators['life expectancy'], 'Dim1')

    indicators['adult mortality'].drop(['Dim1Type'], inplace=True, axis=1)
    indicators['adult mortality'] = drop_btsx_inplace(indicators['adult mortality'], 'Dim1')

    indicators['infant deaths'].drop(['Dim1Type', 'Low', 'High'], inplace=True, axis=1)
    indicators['infant deaths'] = drop_btsx_inplace(indicators['infant deaths'], 'Dim1')

    indicators['alcohol'] = indicators['alcohol'][indicators['alcohol'].Dim1 == 'SA_TOTAL']
    indicators['alcohol'].drop(['Dim1Type', 'Low', 'High', 'Dim1', 'DataSourceDim', 'DataSourceDimType'], inplace=True,
                               axis=1)

    indicators['bmi'].drop(['Dim1Type', 'Dim2Type', 'Dim2', 'Low', 'High', 'Comments'], inplace=True, axis=1)
    indicators['bmi'] = drop_btsx_inplace(indicators['bmi'], 'Dim1')

def main():
    """
    The WHO dataset is composed of indicators. E.g. "Life expectancy at birth" is an indicator. The indicator comes
    with a number of samples. Each sample provides the value for the indicator, and also the values of a number of
    categorical variables. The WHO dataset calls those categorical variables "dimensions". Not all indicators share the
    same set of dimensions: different indicators may, in general, have different dimensions. E.g. the "Life expectancy
    at birth" indicator includes dimensions for country, year and sex.
    """
    who_indicators = get_who_indicators()
    who_indicators.to_csv('indicators.csv')
    lang_values = who_indicators.Language.unique()
    assert list(lang_values) == ['EN']
    name_values = who_indicators.IndicatorName.unique()
    print(f'Found {len(who_indicators)} unique indicator codes and {len(name_values)} unique indicator names.')

    who_dimensions = get_who_dimensions()

    life_exp = get_who_life_expectancy()
    dataset = get_who_dataset()


if __name__ == '__main__':
    main()
