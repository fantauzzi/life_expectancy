import pandas as pd
import numpy as np
import requests
import ssl
import urllib3
import wbgapi as wb


class CustomHttpAdapter(requests.adapters.HTTPAdapter):
    # Credit: https://stackoverflow.com/a/73519818/4262324
    # "Transport adapter" that allows us to use custom ssl_context.

    def __init__(self, ssl_context=None, **kwargs):
        self.ssl_context = ssl_context
        super().__init__(**kwargs)

    def init_poolmanager(self, connections, maxsize, block=False):
        self.poolmanager = urllib3.poolmanager.PoolManager(
            num_pools=connections, maxsize=maxsize,
            block=block, ssl_context=self.ssl_context)


def get_legacy_session():
    # Credit: https://stackoverflow.com/a/73519818/4262324
    ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
    ctx.options |= 0x4  # OP_LEGACY_SERVER_CONNECT
    session = requests.session()
    session.mount('https://', CustomHttpAdapter(ctx))
    return session


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


'''
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
'''


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
                         'thinness 1-19 years': 'NCD_BMI_MINUS2C'
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

    indicators['under-five deaths'].drop(['Dim1Type', 'Low', 'High'], inplace=True, axis=1)
    indicators['under-five deaths'] = drop_btsx_inplace(indicators['under-five deaths'], 'Dim1')

    indicators['HIV/AIDS'].drop(['Comments'], inplace=True, axis=1)

    indicators['thinness 1-19 years'].drop(['Dim1Type', 'Dim2Type', 'Dim2', 'Low', 'High', 'Comments'], inplace=True,
                                           axis=1)
    indicators['thinness 1-19 years'] = drop_btsx_inplace(indicators['thinness 1-19 years'], 'Dim1')

    for indicator_name, indicator_code in indicators.items():
        indicators[indicator_name] = indicators[indicator_name].rename({'NumericValue': indicator_name}, inplace=False,
                                                                       axis=1)

    res = indicators['life expectancy']
    for indicator_name in indicators.keys():
        if indicator_name != 'life expectancy':
            res = pd.merge(res, indicators[indicator_name], how='outer')

    # TODO most null are in the HIV/AIDS column, consider dropping it completely before proceeding with the dropna by row
    res.drop(['HIV/AIDS'], inplace=True, axis=1)
    res = res.dropna(axis=0, how='any')
    return res


def main():
    """
    The WHO dataset is composed of indicators. E.g. "Life expectancy at birth" is an indicator. The indicator comes
    with a number of samples. Each sample provides the value for the indicator, and also the values of a number of
    categorical variables. The WHO dataset calls those categorical variables "dimensions". Not all indicators share the
    same set of dimensions: different indicators may, in general, have different dimensions. E.g. the "Life expectancy
    at birth" indicator includes dimensions for country, year and sex.
    """

    pd.options.display.max_rows = 1000

    # Total population by sex and country available. See indicator'26 (id=49)
    # resp = get_legacy_session().get('https://population.un.org/dataportalapi/api/v1/indicators')
    # resp_json = resp.json()
    who_indicators = get_who_indicators()
    who_indicators.to_csv('indicators.csv')
    lang_values = who_indicators.Language.unique()
    assert list(lang_values) == ['EN']
    name_values = who_indicators.IndicatorName.unique()
    print(f'Found {len(who_indicators)} unique indicator codes and {len(name_values)} unique indicator names.')

    who_dimensions = get_who_dimensions()

    # life_exp = get_who_life_expectancy()
    dataset = get_who_dataset()
    value_counts = dataset.SpatialDim.value_counts()
    print(value_counts.iloc[np.lexsort([value_counts.index, value_counts.values])])

    country_codes = list(dataset.SpatialDim.unique())
    un_code_to_indicator = {'NY.GDP.MKTP.CD': 'GDP',
                            'SH.XPD.GHED.GE.ZS': 'health expenditure',
                            'SH.HIV.INCD.TL.P3': 'hiv',
                            'SH.H2O.SMDW.ZS': 'safe drinking water',
                            'SP.POP.TOTL': 'population',
                            'SE.XPD.CTOT.ZS': 'education expenditure'}

    df = wb.data.DataFrame(un_code_to_indicator.keys(), country_codes, time=(2000, 2010, 2015), labels=True)
    df.drop(['Country', 'Series'], axis=1, inplace=True)
    df = df.stack()
    df = df.unstack(1)
    df['country'], df['year'] = zip(*df.index)
    df.rename(un_code_to_indicator, inplace=True, axis=1)
    # TODO there is an issue with the WHO and UN country codes, as they might now use the same values and/or
    # dont' have the same countries.
    pass


if __name__ == '__main__':
    main()


'''
NY.GDP.MKTP.CD 	GDP (current US$) 
SH.XPD.GHED.GE.ZS  Domestic general government health expenditure (% of general government expenditure)
SH.HIV.INCD.TL.P3 Incidence of HIV, all (per 1,000 uninfected population)  
SH.H2O.SMDW.ZS People using safely managed drinking water services (% of population)
SP.POP.TOTL 	Population, total
SE.XPD.CTOT.ZS 	Current education expenditure, total (% of total expenditure in public institutions) 

SH.H2O.BASW.ZS People using at least basic drinking water services (% of population) 
SH.XPD.CHEX.GD.ZS 	Current health expenditure (% of GDP)
GB.XPD.RSDV.GD.ZS Research and development expenditure (% of GDP) 	
'''
