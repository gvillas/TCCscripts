# -*- coding:utf-8 -*-

__version__ = '0.1.0'
__maintainer__ = 'Ying Deng 21.07.2022'
__authors__ = 'Ying Deng'
__credits__ = 'Ying Deng'
__email__ = 'Ying.Deng@dlr.de'
__date__ = '05.03.2021'
__status__ = 'dev'  # options are: dev, test, prod
__copyright__ = 'DLR'

"""Module-level docstring
Disaggregate the regional, hourly electricity demand to state-level, hourly:  
- The electricity demand from ONS is given: per electric region (defined by national interconnect grid), hourly
- Heuristic method to disaggregate (up-sample) it in state (regarded as node in the model) in Brazil
- Apply the heuristic based on the annual electricity demand per state from EPE
    - The hourly, state-level electricity demand is available for 2012-2019 onwards.

Note:
- the electricity demand (load) from ONS = Consumption + Losses served by Type I, Type II-A, Type II-B plants, 
                                           sets of plants and part of Type III plants in the daily programming
    - source: 01-Balanço de Energia/Boletim Diário da Operação [http://sdro.ons.org.br/SDRO/DIARIO/index.htm]
    - this is net load
- there are NaNs in the row dataset
"""
import glob
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

sys.path.append('../')
from utility_for_data import create_folder
from utility_for_data import fill_nan_7_days_in_df
from utility_for_data import find_outliers
from utility_for_data import hourly_timestamp_for_given_year
from utility_for_data import plot_setting
from utility_for_data import remove_outliers
from utility_for_data import sin_state_region_mapping_in_series
from utility_for_data import translate_pt_to_en_df

logger = logging.getLogger(__name__)


def read_ons_load():
    """ Extract the hourly load curve for four SIN regions from ONS

    Notes:
        - source: `ONS Hourly load curve for four SIN regions<http://www.ons.org.br/Paginas/resultados-da-operacao/historico-da-operacao/curva_carga_horaria.aspx>
                - CURVA DE CARGA HORARIA -> Simples
                - access date: 06. July 2021
        - electricity load at hourly resolution, for 4 regions
                Region in raw (PT)	Region in EN
                            Sul 	South
                            Norte	North
                        Nordeste	Northeast
            Sudeste/Centro-Oeste	Southeast/Midwest
        - 1999.01.01 00:00 - 2020.12.31 23:00
        - the unit of original dataset is MWh
        - trim value lower than ZERO:
            for region N:
                          raw data                      after processing
                1999-11-14 05:00:00   -0.119                 0
                1999-11-14 06:00:00   -0.158                 0
                1999-11-14 07:00:00   -0.172                 0
                1999-11-14 08:00:00   -0.073                 0
                1999-11-14 09:00:00   -0.073                 0
                1999-11-14 10:00:00   -0.095                 0
    """
    try:
        ons = pd.read_csv(
            f"{os.path.dirname(os.path.realpath(__file__))}/resource/ons_load_hourly_region_1999_2023.csv",
            index_col=0)
        ons['time'] = pd.to_datetime(ons['time'])
    except FileNotFoundError:
        create_folder('resource')
        # read in dataset
        raw_file =  f"{os.path.dirname(os.path.realpath(__file__))}/raw/CURVA_CARGA_2023.csv"

            # read in raw data
        ons = pd.read_csv(raw_file, sep=';')
        ons.index.name = 'index'
        ons.drop("nom_subsistema", axis=1, inplace=True)
        ons['din_instante'] = pd.to_datetime(ons['din_instante'], format='%Y-%m-%d %H:%M:%S')
        ons['din_instante'] = ons['din_instante'].dt.strftime('%d.%m.%Y %H:%M:%S')
        ons.rename(columns={ons.columns[0]: 'Subsistema', ons.columns[1]: 'Din Instante', ons.columns[2]: 'Val Cargaenergiaconmwmed'}, inplace=True)

        ons = translate_pt_to_en_df(ons, 'column_ons_hourly_load')
        
        # performance improvement: instead of parse date in read_csv, use pd.to_datetime(df, format ='XXX')
        ons['time'] = pd.to_datetime(ons['time'], format='%d.%m.%Y %H:%M:%S')

        ons = ons.pivot_table(index='time', values='load', columns='region')

        # check nan value for the daily index, if exists, fill the NaNs with the value 7 days ago or after
        idt_hourly = [hourly_timestamp_for_given_year(str(y)) for y in
                      range(min(ons.index.year), max(ons.index.year) + 1)]
        idt_hourly = pd.DatetimeIndex(np.hstack(idt_hourly))
        idt_hourly.name = ons.index.name
        # check the length of the DataFrame df whether it equals to the idt_weekly
        if not ons.shape[0] == idt_hourly.shape[0]:  # when nan exist in df, go into the loop
            ons = ons.reindex(idt_hourly)
            # fill the NaNs with the value 7 days ago or after
            ons, report_dict = fill_nan_7_days_in_df(ons)

            # export the data report of missing data
            # chr(92) being the ASCII code for backslash '\\'
            with open(f"resource/DATA_REPORT_{raw_file.split(chr(92))[-1].split('/')[-1]}.txt", 'a') as file:
                # indent keyword argument is set it automatically adds newlines
                json.dump(report_dict, file, indent=2)

        # trim values lower than 0
        ons = ons.clip(lower=0)  # Ying's note: 6 value in North are negative -> unreasonable value in the raw dataset
        # convert the pivot dataframe to unpivot form
        ons = ons.melt(ignore_index=False).reset_index()

        # store the results
        ons.to_csv('resource/ons_load_hourly_region_1999_2023.csv')

    return ons  # MWh


def read_epe_load(t='consumption'):
    """Extract total and sectoral values of annual electricity consumption/consumer for each state from EPE

    Notes:
        - source: Table 4.2-4.28, `EPE<https://www.epe.gov.br/pt/publicacoes-dados-abertos/publicacoes/anuario-estatistico-de-energia-eletrica>`
        - annual electricity demand
        - 2012 - 2020
        - 27 states: 26 federal states + Brasília
        - The information is in English, so only the federal state information is converted to abbreviations.
        - the unit of original dataset is GWh for consumption
        - the unit of the output is MWh for consumption
        - end-use sector defined in EPE are: 'Residential', 'Industrial', 'Commercial', 'Rural', 'Public Sector', 'Public Lighting' 'Public Service', 'Total' (manually aggregated)
        - note: only the "Total" is to be used as the disaggregation factor for load profiles

    Args:
        t: str, define the info to read from EPE
            - 'consumption': default, total and sectoral consumption per state, unit MWh
            - 'consumer': total and sectoral number of consumer, no unit

    Returns:
        epe: pandas.DataFrame.
            - if tag = 'consumption', total and sectoral values of annual consumption per year and per federal state
            - if tag = 'consumer', total and sectoral values of annual consumer per year and per federal state
    """
    try:
        epe = pd.read_csv(f'resource/epe_sectoral annual electricity {t}_2012-2023.csv', index_col=0)

    except FileNotFoundError:
        # dict of sheet name of electricity consumption/consumers and the state, added manually
        sheet_name_state = {'Tabela 4.2': 'Rondônia',
                            'Tabela 4.3': 'Acre',
                            'Tabela 4.4': 'Amazonas',
                            'Tabela 4.5': 'Roraima',
                            'Tabela 4.6': 'Pará',
                            'Tabela 4.7': 'Amapá',
                            'Tabela 4.8': 'Tocantins',
                            'Tabela 4.9': 'Maranhão',
                            'Tabela 4.10': 'Piauí',
                            'Tabela 4.11': 'Ceará',
                            'Tabela 4.12': 'Rio Grande do Norte',
                            'Tabela 4.13': 'Paraíba',
                            'Tabela 4.14': 'Pernambuco',
                            'Tabela 4.15': 'Alagoas',
                            'Tabela 4.16': 'Sergipe',
                            'Tabela 4.17': 'Bahia',
                            'Tabela 4.18': 'São Paulo',
                            'Tabela 4.19': 'Minas Gerais',
                            'Tabela 4.20': 'Espírito Santo',
                            'Tabela 4.21': 'Rio de Janeiro',
                            'Tabela 4.22': 'Paraná',
                            'Tabela 4.23': 'Santa Catarina',
                            'Tabela 4.24': 'Rio Grande do Sul',
                            'Tabela 4.25': 'Mato Grosso do Sul',
                            'Tabela 4.26': 'Mato Grosso',
                            'Tabela 4.27': 'Goiás',
                            'Tabela 4.28': 'Distrito Federal'}

        # read-in the dataset of consumption per state and the number of consumers
        df = pd.read_excel(io='raw/raw_Anuário Estatístico de Energia Elétrica 2021 - Workbook.xlsx',
                           sheet_name=list(sheet_name_state.keys()),
                           usecols=[i for i in range(2, 15) if i not in [12, 13]],  # adaption needed for each workbook
                           skiprows=[i for i in range(0, 8)],
                           index_col=-1
                           )  # GWh for consumption
        # replace the sheet name with the state
        for old, new in sheet_name_state.items():
            df[new] = df.pop(old)

        def extract_total(raw_df, i_row):
            temp_res = pd.DataFrame()
            for state, val in raw_df.items():
                # iloc[i_col] is equivalent to loc['Consumption (GWh)'] or loc['Consumidores (unidades)']
                temp = val.iloc[i_row].rename(state)
                temp_res = pd.concat([temp_res, temp], axis=1)
            # convert the full name of state to abbreviations
            temp_res = translate_pt_to_en_df(temp_res.T.melt(ignore_index=False
                                                             ).reset_index().rename(columns={'index': 'state',
                                                                                             'variable': 'time'}),
                                             'state')
            temp_res['sector'] = 'Total'
            return temp_res

        def extract_per_sector(raw_df, start, end):
            temp_res = pd.DataFrame()
            for state, val in raw_df.items():
                # iloc[start:end] is equivalent to loc['XXX'] and XX is filled with sector name
                temp = val.iloc[start:end].melt(ignore_index=False).reset_index().rename(columns={'  ': 'sector',
                                                                                                  'variable': 'time'})
                temp['state'] = state
                temp_res = pd.concat([temp_res, temp])
            temp_res = translate_pt_to_en_df(temp_res, 'state')
            return temp_res

        if t == 'consumption':
            epe_total = extract_total(raw_df=df, i_row=0)
            epe_sector = extract_per_sector(raw_df=df, start=1, end=9)
            epe = pd.concat([epe_total, epe_sector])
            # convert unit from GWh to MWh
            epe['value'] *= 1e3
        elif t == 'consumer':
            epe_total = extract_total(raw_df=df, i_row=9)
            epe_sector = extract_per_sector(raw_df=df, start=10, end=17)
            epe = pd.concat([epe_total, epe_sector])
            # make the number of consumer 'int' type
            epe['value'] = epe['value'].astype('int64')
        else:
            assert 'Tag is not supported'

        epe['time'] = epe['time'].astype('int64')
        epe = epe.sort_values(['state', 'time', 'sector'])
        epe.to_csv(f'resource/epe_sectoral annual electricity {t}_2012-2023.csv')

    return epe


def compare_ons_epe_total_per_year():
    """ Compare the annual total from ONS with EPE
    Notes:
        - ONS yearly total value is bigger than epe yearly total value, this is due to the reason in P.9
        - https://www.epe.gov.br/sites-pt/publicacoes-dados-abertos/publicacoes/PublicacoesArquivos/publicacao-251/topico-315/NT_Carga_ONS-EPE-CCEE%20_07-12-2016%5B1%5D.pdf
    """
    # read ONS annual national electricity demand - hourly, classified in four SIN regions
    ons_total = read_ons_load().pivot_table(index='time',
                                            columns='region',
                                            values='value').sum(axis=1)
    ons_total = ons_total.groupby(ons_total.index.year).sum()
    ons_total.name = 'ons'

    # read EPE annual national electricity demand - hourly, classified in four 26 federal states + Brasília
    epe_total = read_epe_load(t='consumption').query("sector=='Total'").groupby(['time']).sum().squeeze()
    epe_total.name = 'epe'

    comparison = pd.concat([ons_total, epe_total], join='inner', axis=1)
    comparison['diff(ons-epe)'] = comparison['ons'] - comparison['epe']
    comparison.to_csv('resource/DATA_ANALYSIS_compare_ons_epe_total_per_year.csv')
    return comparison


def compare_ons_epe_per_region_per_year():
    epe = read_epe_load().set_index('state')
    epe['region'] = sin_state_region_mapping_in_series()
    epe = epe.groupby(['region', 'time']).sum().pivot_table(index='time', columns='region', values='value') / 1000

    ons = read_ons_load().pivot_table(index='time', columns='region', values='value')
    ons = ons.groupby(ons.index.year).sum().loc[epe.index] / 1000

    delta_ons_epe = (ons - epe) / ons * 100

    with pd.ExcelWriter('resource/DATA_ANALYSIS_compare_ons_epe_per_region_per_year_GWh.xlsx',
                        engine='openpyxl') as writer:
        epe.to_excel(writer,
                     sheet_name='epe')
        ons.to_excel(writer,
                     sheet_name='ons')
        delta_ons_epe.to_excel(writer,
                               float_format="%.0f",
                               sheet_name='(ons-epe) div ons %')


def get_state_region_factor(t='consumption'):
    """ Get the state region factor as weighting factor to distribute the regional value to federal states

    Args:
        - t: str, the parameter to be used as distribution, support:
            - "consumption": total consumption at federal state to distribute at federal state level
            - "consumer": total consumer at federal state to distribute at federal state level
            - "consumption in sectors": sectoral consumption of electricity to distribute at federal state level and sector
            - "consumer in sectors": sectoral consumer of electricity to distribute at federal state level and sector

    Notes:
        - factor = annual demand of state / annual demand of the sum of the belonging regions
        e.g.,
            state                region              demand at 2012
            Paraná               South                 1.0
            Santa Catarina       South                 1.0
            Rio Grande do Sul    South                 1.0

            factor of Parana at 2012 = 1.0/(1.0 + 1.0 + 1.0) = 0.33
        - dataset from EPE
    Returns: factor: pd.DataFrame
        - index: abbreviation of state name
        - columns: 'time', 'region', 'factor'
    """
    # get the yearly sum for each states from EPE dataset
    # map the states and region according to the classification in ONS data
    epe_annual = read_epe_load(t=t).set_index('state')
    sin = sin_state_region_mapping_in_series()
    epe_annual['region'] = sin

    yearly_region_total = epe_annual.query("sector=='Total'").groupby(['time', 'region']).sum().reset_index().rename(
        columns={'value': 'sum'})

    # plot the annual total value
    plt.close('all')
    fig, ax = plot_setting(figure_size=(8, 4))
    yearly_region_total.pivot(index='time', columns='region', values='sum'
                              ).plot(ax=ax, kind='bar', stacked=True, ylabel='MWh', xlabel='')
    ax.legend(bbox_to_anchor=(1.0, 1.0), title='electric region')
    ax.set_ylim(ymin=0)
    plt.tight_layout()
    fig.savefig(fname=f'resource/epe_annual_electricity_consumption_2012-2022.png')
    factor = epe_annual.reset_index().merge(yearly_region_total, on=['time', 'region'], how='left').set_index('state')
    factor['factor'] = (factor['value'] / factor['sum'])
    factor = factor[['time', 'sector_x', 'region', 'factor']].reset_index()
    factor = factor.rename(columns={'sector_x': 'sector'})
    
    return factor


def get_state_hourly(t='consumption', base_year='2023'):
    """Get the state hourly data: map the annual state data from EPE onto the region hourly data from ONS
    Args:
        - t: str, parameter to distribute the regional demand to federal-state level
        - base_year: int, year of interest
    Notes: unit: MWh
    """
    try:
        res = pd.read_csv(f"results/distribute_by_{t}/Hourly_electricity_demand_per_state_{base_year}.csv",
                          index_col=0)
        res['time'] = pd.to_datetime(res['time'], format="%Y-%m-%d %H:%M:%S")  # format='%d.%m.%Y %H:%M:%S'
    except FileNotFoundError:
        create_folder('results')
        create_folder(f'results/distribute_by_{t}')
        # select the hourly data from ONS dataset based on the time range of EPE dataset
        if not isinstance(base_year, int):
            base_year = int(base_year)
        # the year is limited by EPE dataset since it has fewer data than ONS dataset
        if base_year not in range(2012, 2024):
            assert "Only support year 2012-2023"
        hourly_ons = read_ons_load().pivot_table(index='time', values='value', columns='region').loc[str(base_year)]
        # get the factor of disaggregation (regional -> federal state)
        #state_region_factor = get_state_region_factor(t=t).set_index('state'
        #                                                             ).query("time==@base_year and sector=='Total'")
        # Workaround to apply the factors from 2022 in ONS' demand data of 2023
        state_region_factor = get_state_region_factor(t=t).set_index('state')
        factor_2023 = state_region_factor[state_region_factor['time'] == 2022]
        factor_2023['time'] = 2023
        state_region_factor = pd.concat([state_region_factor, factor_2023]).query(
            "time==@base_year and sector=='Total'"
        )
        

        res = pd.DataFrame(index=hourly_ons.index, columns=state_region_factor.index.unique())
        for i, v in state_region_factor.iterrows():
            state, region, factor = i, v['region'], v['factor']
            res[state] = hourly_ons[region] * factor
        res = res.melt(ignore_index=False).reset_index()  # the type of value has been changed from float to object
        res['value'] = pd.to_numeric(res['value']).round(2)

        # store the results
        res.to_csv(f"results/distribute_by_{t}/Hourly_electricity_demand_per_state_{base_year}.csv")

    return res


def plot_ons_ts_region():
    """ plot the time series before processing from ONS- hourly, per region"""
    # prepare data, unit of GWh
    ons_region_ts = pd.read_csv('resource/ons_load_hourly_region_1999_2023.csv').pivot_table(index='time',
                                                                                             columns='region',
                                                                                             values='value') / 1e3
    ons_region_ts.index = pd.to_datetime(ons_region_ts.index, format='%Y-%m-%d %H:%M:%S')

    # stacked area plot of the time series - daily, hourly, weekly
    plt.close('all')
    for temporal in ['hourly', 'daily', 'weekly']:
        fig, ax = plot_setting(figure_size=(8, 4))
        # inspiration: https://stackoverflow.com/questions/14029245/putting-an-if-elif-else-statement-on-one-line
        df = ons_region_ts if temporal == 'hourly' else (
            ons_region_ts.resample('d').sum() if temporal == 'daily' else ons_region_ts.resample('w').sum())
        df.plot.area(ax=ax, stacked=True, linewidth=0.5, colormap='Blues')
        ax.set_ylabel(f'{temporal.title()} electricity load [GWh]')
        ax.set_xlabel('')
        ax.legend(ncol=1, bbox_to_anchor=(1.0, 1.0))
        ax.get_legend().set_title('region')
        plt.tight_layout()
        fig.savefig(fname=f'resource/{temporal}_per region_electricity load_1999-2023_ONS raw data.png',
                    dpi=500, bbox_inches='tight')

    # one-month moving average
    plt.close('all')
    for level in ['N', 'SE', 'NE', 'S', 'National']:
        # prepare data
        df = ons_region_ts.sum(axis=1) if level == 'National' else ons_region_ts[level]

        # plot
        # https://towardsdatascience.com/time-series-analysis-for-machine-learning-with-python-626bee0d0205
        fig, ax = plot_setting(figure_size=(8, 4))
        window = 24 * 30  # every 30 days
        rolling_mean = df.rolling(window=window).mean()
        rolling_std = df.rolling(window=window).std()

        ax.plot(df[window:], label='original', color="black", linewidth=0.1)
        # plot moving average
        ax.plot(rolling_mean, label='one-month rolling mean',
                color="r", linewidth=0.5)
        # plot upper and lower bounds
        lower_bound = rolling_mean - (1.96 * rolling_std)
        upper_bound = rolling_mean + (1.96 * rolling_std)
        ax.fill_between(x=df.index, y1=lower_bound, y2=upper_bound,
                        color='skyblue', alpha=0.4, label='bounds')

        ax.set_ylabel(f'{level} hourly total power load [GWh]')
        ax.set_xlabel('')
        ax.legend(ncol=1, bbox_to_anchor=(1.0, 1.0))
        ax.set_xlim(left=df.index[0], right=df.index[-1])
        ax.set_ylim(bottom=0)
        plt.tight_layout()

        fig.savefig(fname=f'resource/{level}_electricity load_1999-2023_one month rolling_ONS raw data.png',
                    dpi=500, bbox_inches='tight')


def plot_hourly_load_state(t='consumption'):
    """ plot the time series after processing - hourly, per state"""
    # %% Time Series Analysis
    # TODO: streamchart, https://www.python-graph-gallery.com/web-streamchart-with-matplotlib
    # prepare data, convert MWh to GWh
    state_ts = pd.concat([pd.read_csv(f, index_col=0) for f in glob.glob(f'results/distribute_by_{t}/*.csv')])
    state_ts = state_ts.pivot_table(index='time', columns='state', values='value') / 1e6  # TWh
    state_ts.index = pd.to_datetime(state_ts.index, format='%Y-%m-%d %H:%M:%S')

    plt.close('all')
    fig, ax = plot_setting(figure_size=(8, 4))
    # three day could show to inter day x-labels 12:00
    three_day = state_ts['2023-03-16':'2023-03-18']
    three_day.plot(ax=ax)
    ax.set_ylabel(f'TWh')
    ax.set_xlabel('')
    ax.legend(ncol=2, bbox_to_anchor=(1.0, 1.0))
    ax.get_legend().set_title('state')
    ax.set_ylim(ymin=0)
    plt.tight_layout()
    fig.savefig(fname=f'results/distribute_by_{t}/2023-03-16~18_per_state_electricity load_distributed_by_{t}.png')

    # stacked area plot of the time series - daily, hourly, weekly
    plt.close('all')
    for temporal in ['hourly', 'daily', 'weekly']:
        fig, ax = plot_setting(figure_size=(8, 4))
        # inspiration: https://stackoverflow.com/questions/14029245/putting-an-if-elif-else-statement-on-one-line
        df = state_ts if temporal == 'hourly' else (
            state_ts.resample('d').sum() if temporal == 'daily' else state_ts.resample('w').sum())
        df.plot.area(ax=ax, stacked=True, linewidth=0.1,
                     color=['#2a9d8f', '#264653', '#e9c46a', '#f4a261', '#283618', '#6d6875', '#f1faee',
                            '#a8dadc', '#457b9d', '#ff006e', '#cdb4db', '#ffc8dd', '#ffafcc', '#bde0fe',
                            '#a2d2ff', '#606c38', '#e76f51', '#06d6a0', '#dda15e', '#bc6c25', '#003049',
                            '#d62828', '#80ffdb', '#fcbf49', '#eae2b7', '#3a86ff', '#8338ec'])
        ax.set_ylabel(f'TWh')
        ax.set_xlabel('')
        ax.legend(ncol=2, bbox_to_anchor=(1.01, 1.0), frameon=False)
        ax.get_legend().set_title('state')
        plt.tight_layout()
        fig.savefig(
            fname=f'results/distribute_by_{t}/{temporal}_per_state_electricity load_2012-2023_distributed_by_{t}.png',
            dpi=500, bbox_inches='tight')


def plot_remove_outlier(year=None):
    # https://machinelearningmastery.com/time-series-data-visualization-with-python/
    # prepare data
    ons_region_ts = pd.read_csv('resource/ons_load_hourly_region_1999_2023.csv').pivot_table(index='time',
                                                                                             columns='region',
                                                                                             values='value')
    ons_region_ts.index = pd.to_datetime(ons_region_ts.index, format='%Y-%m-%d %H:%M:%S')

    # plot - find outliers, after removing outliers by interpolations
    create_folder('resource')
    if year:
        ons_region_ts = ons_region_ts.loc[str(year)]

    for col_name in ons_region_ts.columns:
        plt.close('all')
        ts = ons_region_ts.loc[:, col_name]
        dtf_outliers = find_outliers(ts, perc=0.01)
        outliers_index_pos = dtf_outliers[dtf_outliers["outlier"] == 1].index  # exclude outliers
        ts_clean = remove_outliers(ts, outliers_idx=outliers_index_pos)

        # store the plot
        figs = [plt.figure(n) for n in plt.get_fignums()]
        i = 0
        for fig in figs:
            fig.savefig(f'resource/outlier_{col_name}_{i}.png')
            i += 1


def summary_load():
    # regional summary
    plot_ons_ts_region()
    plot_remove_outlier(year=None)
    compare_ons_epe_per_region_per_year()
    compare_ons_epe_total_per_year()
    # state level summary
    plot_hourly_load_state()


if __name__ == '__main__':
    for year in range(2023, 2024):
        for tag in ['consumption', 'consumer']:
            get_state_hourly(t=tag, base_year=str(year))

        plot_hourly_load_state(t='consumption')
        plot_hourly_load_state(t='consumer')
        summary_load()
