# -*- coding:utf-8 -*-

__version__ = '0.1.0'
__maintainer__ = 'Ying Deng 27.10.2021'
__authors__ = 'Ying Deng'
__credits__ = 'Ying Deng'
__email__ = 'Ying.Deng@dlr.de'
__date__ = '27.10.2021'
__status__ = 'dev'  # options are: dev, test, prod
__copyright__ = 'DLR'

import decimal
import os

import numpy as np
import pandas as pd
from matplotlib.pyplot import rcParams
from matplotlib.pyplot import style
from matplotlib.pyplot import subplots
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
from shapely.geometry import LineString
from shapely.geometry import Point
from sklearn import preprocessing
from sklearn import svm

import app_config_data


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def translate_pt_to_en_df(df, col_name):
    """Translate the column name and the values of each column for a DataFrame
    Args:
        - df: pd.DataFrame, the data to be translated from PT to EN
        - col_name: str, the group of the column name in tools/trans_PT_to_EN.yaml to be used

    Notes:
        - translation of the value in each column is based on the column name
        - different column name but same translation of value is supported by giving the column name in the
            trans_PT_to_EN.yaml
        - the translation is based on the exact string match
    Return:
        df: pd.DataFrame, translated df
    """
    # %% read in translation configuration file
    # get the directory of module which is running + the translation file name
    trans_cfg = read_yaml(file=f'{os.path.dirname(os.path.realpath(__file__))}/{app_config_data.trans_file}')
    # convert the duplicated key-value translation, see trans_PT_to_EN.yaml
    for k, v in trans_cfg.items():
        if not isinstance(v, CommentedMap):
            trans_cfg[k] = trans_cfg[v]

    # %% translate the column name
    # translate pandas.DataFrame value of those columns type are object (string) and exists in the trans_cfg
    # replace or remove the string invisible but lead to failure in rename
    # \n new line; .strip() leading and trailing characters
    df.columns = df.columns.str.replace('\n', ' ').str.strip()
    df = df.rename(trans_cfg[col_name], axis=1)

    # %% translate the value of each column
    translated = df.select_dtypes(include=['object', 'category'])
    for col in translated.columns:
        if col in trans_cfg.keys():
            # to avoid SettingWithCopyException

            translated_cp = translated.copy()
            # first remove the leading and trailing white space
            # then remove the white space including space, tab, form-feed, and so on
            # last replace the PT with EN

            trans_comp = trans_cfg[col]

            # replace the whole string
            df[col] = translated_cp[col].str.strip().replace(r'\s+|\\n', ' ', regex=True
                                                             ).replace(to_replace=trans_comp)
    return df


def title_series_pt_words(series):
    """Convert first character of each work to upper case while keeping the remaining to lowercase. The
    prepositions should always be lower case. Keep the Roman number uppercase (I-V)
    Args:
        - series: pd.Series, the value should be string
    """
    replace_preposition = {' De ': ' de ',
                           ' Do ': ' do ',
                           ' Da ': ' da ',
                           ' Dos ': ' dos ',
                           ' Das ': ' das ',
                           ' E ': ' e '}
    # $ end of the string
    replace_roman = {' Iii$': ' III',
                     ' Iv$': ' IV',
                     ' Vi$': ' VI',
                     ' Vii$': ' VII',
                     ' Viii$': ' VIII',
                     ' Ix$': ' IX'}

    replace_roman = {' Ii$': ' II',
                     ' Iii$': ' III',
                     ' Iv$': ' IV',
                     ' Vi$': ' VI',
                     ' Vii$': ' VII',
                     ' Viii$': ' VIII',
                     ' Ix$': ' IX'}
    replace_roman_revise = {' IIi': ' III',
                            ' VIi': ' VII',
                            ' VIIi': ' VIII'}
    if isinstance(series, pd.Series):
        if isinstance(series.dtypes, object):
            # todo: too ugly
            return series.str.title().replace(replace_preposition, regex=True
                                              ).replace(replace_roman, regex=True, limit=1, method='bfill'
                                                        ).replace(replace_roman_revise, regex=True, limit=1,
                                                                  method='bfill')

        else:
            assert "Values in series are not string"
    else:
        assert "Input not pd.Series"


def delete_accents_pt_words_in_series(series):
    """Delete the accents in values for pd.Series
    see more unicode form: https://docs.python.org/3/library/unicodedata.html#module-unicodedata
    Args:
     - series: pd.Series

    Returns:
     - pd.Series
    """
    return series.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8').str.strip()


def cut_linestring(line, distance):
    """Cut lines at a specified distance.
    source: https://shapely.readthedocs.io/en/stable/manual.html
    Args:
        - line: LineString
        - distance: float
    Return:

    """
    # Cuts a line in two at a distance from its starting point
    if distance <= 0.0 or distance >= line.length:
        return [LineString(line)]
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pd = line.project(Point(p))
        if pd == distance:
            return [
                LineString(coords[:i + 1]),
                LineString(coords[i:])]
        if pd > distance:
            cp = line.interpolate(distance)
            return [LineString(coords[:i] + [(cp.x, cp.y)]), LineString([(cp.x, cp.y)] + coords[i:])]


def float_range(start, stop, step):
    """Generate a float range that keep the precision fixed
    source: https://www.techbeamers.com/python-float-range/
    Args:
        - start: int, start of the range
        - stop: int/float, end of the range
        - step: str, precision

    Return:
    """
    while start < stop:
        yield float(start)
        start += decimal.Decimal(step)


def plot_setting(n_rows=1, n_cols=1, figure_size=(15, 5), **kwargs):
    # inspiration: http://aeturrell.com/2018/01/31/publication-quality-plots-in-python/
    style.use(os.path.join(os.path.realpath(__file__), app_config_data.plot_style))
    if kwargs:
        for k, v in kwargs.copy().items():
            if k == 'font_scale':
                font_scale = v
                font_keys = ["axes.labelsize", "axes.titlesize", "legend.fontsize",
                             "xtick.labelsize", "ytick.labelsize", "font.size", "figure.titlesize",
                             "legend.title_fontsize"]
                font_dict = {k: rcParams[k] * font_scale for k in font_keys}
                rcParams.update(font_dict)
                del kwargs[k]
    # x- or y-axis will be shared among all subplot
    fig, ax = subplots(nrows=n_rows, ncols=n_cols, sharex='all', sharey='all',
                       frameon=True, figsize=figure_size, **kwargs)

    return fig, ax


def set_plot_style(**kwargs):
    style.use(os.path.join(os.path.realpath(__file__), app_config_data.plot_style))

    for k, v in kwargs.items():
        if k == 'font_scale':
            font_scale = v
            font_keys = ["axes.labelsize", "axes.titlesize", "legend.fontsize",
                         "xtick.labelsize", "ytick.labelsize", "font.size", "figure.titlesize", "legend.title_fontsize"]
            font_dict = {k: rcParams[k] * font_scale for k in font_keys}
            rcParams.update(font_dict)


def read_config(scenario_file=None):
    """Update the default yaml file based on the scenario yaml file

    Args:
        - scenario_file: file name of scenario yaml file, should be "XXX.yaml"
    """
    default_file = f'{os.path.dirname(os.path.realpath(__file__))}/{app_config_data.plot_color_name}'
    default = read_yaml(default_file)
    if scenario_file:
        scenario = read_yaml(scenario_file)
        default.update(scenario)
    return default


def read_yaml(file):
    """
    read in yaml file
    """
    # add encoding = 'utf-8' to handle the case of messy code with Portuguese in the file
    with open(file, encoding='utf-8') as file:
        yaml = YAML()
        file = yaml.load(file)
    return file


def sin_state_region_mapping_in_series():
    """Mapping each state to the belonging region in SIN in Portuguese wFith Abbreviation

    Returns:
        - output: pd.Series
            - index: str, abbreviation of states
            - value: str, abbreviation of the belonging geo-electric regions used in SIN
    """
    sin = sin_state_region_mapping()
    output = {}
    for r, s_all in sin.items():
        for s in s_all:
            output.update({s: r})
    output = pd.Series(output)
    output.name = 'region'
    return output


def sin_state_region_mapping():
    """
    mapping each state to the belonging region in SIN in Portuguese wFith Abbreviation

    Returns:
        - output: dict
            - key: str, abbreviation of geo-electric regions used in SIN
            - value: list, list of abbreviation of belonging states

    Notes:
        - SIN defined by ONS: electric region in national grid-SIN;
        - combine the 'Southeast' and 'Midwest' as 'Southeast/Midwest'
        - http://www.ons.org.br/paginas/energia-agora/balanco-de-energia
    """
    # read in the composition of geo-electric regions in states-

    state_region = {'North': ['Pará', 'Tocantins', 'Maranhão', 'Amapá', 'Amazonas', 'Roraima'],
                    'Northeast': ['Piauí', 'Ceará', 'Rio Grande do Norte', 'Paraíba', 'Pernambuco', 'Alagoas',
                                  'Sergipe',
                                  'Bahia'],
                    'Southeast/Midwest': ['Espírito Santo', 'Rio de Janeiro', 'Minas Gerais', 'São Paulo', 'Goiás',
                                          'Distrito Federal', 'Mato Grosso', 'Acre', 'Rondônia', 'Mato Grosso do Sul'],
                    'South': ['Rio Grande do Sul', 'Santa Catarina', 'Paraná']}

    # read in translation file: full name to abbreviations
    # get the directory of module which is running + the translation file name
    trans_cfg = read_yaml(file=f'{os.path.dirname(os.path.realpath(__file__))}/{app_config_data.trans_file}')

    # get the translation from full region name in English to short name
    trans_region = trans_cfg['region_en']
    # get the translation from full state name in English to short name
    trans_state = trans_cfg['state']

    output = {trans_region[k]: [trans_state[element] for element in v] for k, v in state_region.items()}
    return output


def hourly_timestamp_for_given_year(year):
    """
    Prepares the hourly timestamp index for the given year. It is determined by whether the year is a leap year or not
    Args:
        - year: str, int
    Returns:
        - idx: Timestamp
    """
    # noinspection PyTypeChecker
    if pd.Timestamp(str(year)).is_leap_year:
        idx = pd.date_range(f'{str(year)}-01-01', periods=366 * 24, freq='h')
    else:
        idx = pd.date_range(f'{str(year)}-01-01', periods=365 * 24, freq='h')
    return idx


def daily_timestamp_for_given_year(year):
    """Prepares the daily timestamp index for the given year. It is determined by whether the year is a leap year or not
    Args:
        - year: str, int
    Returns:
        - idx: Timestamp
    """
    # noinspection PyTypeChecker
    if pd.Timestamp(str(year)).is_leap_year:
        idx = pd.date_range(f'{str(year)}-01-01', periods=366, freq='d')
    else:
        idx = pd.date_range(f'{str(year)}-01-01', periods=365, freq='d')
    return idx


def fill_nan_7_days_in_df(df, filepath=None):
    """Forward fill missing data in DataFrame of times series (hourly, for a year):
        - drop columns when all NaNs
        - forward fill the NaNs with value one week before (NaNs at first week of the year will not be filled)
        - if NaN still exists, filled with ZERO
    Args:
        - df: pd.DataFrame / pd.Series
                - index: DateTimeIndex
        - filepath: optional, the input file from which df originates

    Returns:
        df: DataFrame or Series
        report_dict: str, the log of the missing value

    Notes:
    only forward fill, reason see: https://towardsdatascience.com/basic-time-series-manipulation-with-pandas-4432afee64ea

    TODO: write the output to a log file
    """
    # for DataFrame with only value, set the missing value to ZERO
    if df.shape == (1, 1):
        df = df.fillna(0)
    # index of df should be DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
        print('Note: The index of input data should be pd.DatetimeIndex')

    # convert Series to DataFrame and process the data as frame, in the end, it will converted to Series
    if isinstance(df, pd.Series):
        # select all rows with NaN under the entire Series
        df = df.to_frame()

    nan_per_year = df.isna().groupby(df.index.year).sum()
    # initialize the report to be printed (update along the code and print out)
    report_dict = {'Input DataFrame size': int(df.size),
                   'Number of NaN per year': nan_per_year.to_dict(),
                   'NaNs total before': int(df.isna().sum().sum())}

    # drop columns that all rows are NaNs
    df = df.dropna(axis=1, how='all')
    dropped_col = df.columns[df.isna().all(axis=0)]
    if not dropped_col.empty:
        report_dict.update({'Columns with all NaNs are dropped': dropped_col.to_list()})

    # select rows that all columns are NaNs
    all_nan_rows = df[df.isna().all(axis=1)]
    if not all_nan_rows.empty:  # if the rows with all NaNs exist, go into th loop
        report_dict.update({'Rows with all NaNs are': all_nan_rows.index.astype(str).to_list()})

        # use the smallest value (day of year) of the nan_values as the maximal week when the data can fill in
        max_week_for_ffill = all_nan_rows.index.dayofyear.min() // 7

        if max_week_for_ffill != 0:
            # when the datetime of the missing date is not within the first week, the missing value is filled with the
            # value one week before
            index_before_week = all_nan_rows.index + pd.DateOffset(-7)
            fill_value = df.loc[index_before_week]
            fill_value.index = all_nan_rows.index
            df = df.fillna(fill_value)

    # select rows that has any NaNs
    any_nan_rows = df[df.isna().any(axis=1)]
    if not any_nan_rows.empty:
        # report_dict.update({'Rows with any NaNs (exclude all NaNs) are': any_nan_rows.index}) # can be to much

        # if first week data contains NaN, not fill any value here
        to_fill = any_nan_rows[any_nan_rows.index.dayofyear > 7]
        index_to_fill = to_fill.index + pd.DateOffset(-7)
        to_fill_value = df.loc[index_to_fill]
        to_fill_value.index = to_fill.index
        df = df.fillna(to_fill_value)

    # print to give insight of the function
    # note: all the type assignment in report_dict helps to dump the dict to json file
    report_dict.update(
        {'NaNs forward filled with value 7 days before': int(report_dict['NaNs total before'] - df.isna().sum().sum()),
         'NaNs filled with normal forward-fill': int(df.isna().sum().sum())})

    print("--------------------------------------------------")
    if filepath:
        report_dict.update({'NaNs exist in file': filepath})
        print(f'Report about the input in {filepath}: \n {report_dict}')
    else:
        print(f'Report about the input: \n {report_dict}')
    print("--------------------------------------------------")
    # if NaNs still exists, forward-fill the previous value.
    df = df.ffill()
    # convert single column dataframe to Series
    df = df.squeeze()
    return df, report_dict


def remove_outliers(ts, outliers_idx, figsize=(8, 4)):
    """Interpolate outliers in a ts
    source: https://towardsdatascience.com/time-series-analysis-for-machine-learning-with-python-626bee0d0205
    """
    ts_clean = ts.copy()
    if ts_clean.name:
        title = f'Removed outliers for {ts_clean.name}'
    else:
        title = f'Removed outliers'
    ts_clean.loc[outliers_idx] = np.nan
    ts_clean = ts_clean.interpolate(method='linear')
    fig, ax = plot_setting(figure_size=figsize)
    ts.plot(ax=ax, color='red', alpha=0.5,
            title=title,
            label='original',
            legend=True,
            linewidth=0.1
            )
    ts_clean.plot(ax=ax, color='black',
                  label='interpolated',
                  legend=True,
                  linewidth=0.1)
    ax.set_xlim(left=ts.index[0], right=ts.index[-1])
    return ts_clean


def find_outliers(ts, perc=0.01, figsize=(8, 4)):
    """Find outliers using sklearn unsupervised support vetcor machine.
    source: https://towardsdatascience.com/time-series-analysis-for-machine-learning-with-python-626bee0d0205
    Args:
        - ts: pandas.Series
        - perc: float, percentage of outliers to look for
        - figsize: tuple, the size of the figure
    Returns:
        - dtf_outliers: dtf with raw ts, outlier 1/0 (yes/no), numeric index
    """
    # fit svm
    scaler = preprocessing.StandardScaler()
    ts_scaled = scaler.fit_transform(ts.values.reshape(-1, 1))
    model = svm.OneClassSVM(nu=perc, kernel="rbf", gamma=0.01)
    model.fit(ts_scaled)  # dtf output
    dtf_outliers = ts.to_frame(name="ts")
    dtf_outliers["index"] = range(len(ts))
    dtf_outliers["outlier"] = model.predict(ts_scaled)
    dtf_outliers["outlier"] = dtf_outliers["outlier"].apply(lambda
                                                                x: 1 if x == -1 else 0)
    # plot
    fig, ax = plot_setting(figure_size=figsize)
    if ts.name:
        ax.set(title=f"Outliers detection for {ts.name}: found "
                     + str(sum(dtf_outliers["outlier"] == 1)))
    else:
        ax.set(title="Outliers detection: found "
                     + str(sum(dtf_outliers["outlier"] == 1)))

    ax.plot(dtf_outliers["index"], dtf_outliers["ts"],
            color="black", linewidth=0.5)
    ax.scatter(x=dtf_outliers[dtf_outliers["outlier"] == 1]["index"],
               y=dtf_outliers[dtf_outliers["outlier"] == 1]['ts'],
               edgecolors='red')
    ax.set_xlim(left=ts.index[0], right=ts.index[-1])
    ax.set_xlabel('')
    return dtf_outliers


def plot_clustered_stacked_bar(df_list, labels=None, title="Clustered stacked bar plot", H="/", legend_y_offset=0,
                               **kwargs):
    """Inspiration: https://stackoverflow.com/questions/22787209/how-to-have-clusters-of-stacked-bars-with-python-pandas

    Args:
        title: title of the plot
        df_list: a list of dataframes, with identical columns and index, create a clustered stacked bar plot.
        labels: a list of the names of the dataframe, used for the legend title is a string for the title of the plot
        H: is the hatch used for identification of the different dataframe
    """

    n_df = len(df_list)
    n_col = len(df_list[0].columns)
    n_ind = len(df_list[0].index)

    fig, ax = plot_setting(figure_size=(15, 6))
    # for each data frame
    for df in df_list:
        ax = df.plot(kind="bar",
                     linewidth=0.8,
                     stacked=True,
                     ax=ax,
                     legend=False,
                     grid=False,
                     edgecolor='0.5',
                     **kwargs)  # make bar plots

    # get the handles and labels to modify
    h, l = ax.get_legend_handles_labels()
    for i in range(0, n_df * n_col, n_col):  # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i + n_col]):
            for rect in pa.patches:  # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(i / n_col))  # edited part
                rect.set_width(1 / float(n_df + 1))
    # make sure the plot start from y axis
    ax.set_xlim(left=min([item[0].xy[0] for item in h]))
    ax.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    ax.set_xticklabels(df.index, rotation=0)
    ax.set_title(title)

    # Add invisible data to add another legend
    n = []
    for i in range(n_df):
        n.append(ax.bar(0, 0, color="white", hatch=H * i))

    l1 = ax.legend(h[:n_col], l[:n_col], bbox_to_anchor=(1.01, 0.8), loc='center left',
                   title='Technology',
                   ncol=1,
                   bbox_transform=ax.transAxes,
                   frameon=False)
    # align the legend title to the left
    l1._legend_box.align = 'left'
    ax.add_artist(l1)
    if labels is not None:
        l2 = ax.legend(n, labels, bbox_to_anchor=(1.01, 0.3), loc='center left',
                       title='Scenario',
                       ncol=1,
                       bbox_transform=ax.transAxes,
                       frameon=False)
        # align the legend title to the left
        l2._legend_box.align = 'left'
        ax.add_artist(l2)
    ax.set_xlabel("")
    return fig, ax

def multiply_if_decimal(value):
    if value % 1 != 0:  # Check if there is a decimal part
        return value * 1000
    else:
        return value