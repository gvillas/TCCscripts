# -*- coding: utf-8 -*-
__version__ = '0.1.0'
__maintainer__ = 'Ying Deng 21.07.2022'
__authors__ = 'Ying Deng'
__credits__ = 'Ying Deng'
__email__ = 'ying.deng@dlr.de'
__date__ = '18.06.2021'
__status__ = 'dev'  # options are: dev, test, prod
__copyright__ = 'DLR'

"""Module-level docstring.
Read in the historical daily inflow data. This is called in ``solve_network.py``

Notes:
    The average of all historical years is not used, as this would lead to infeasibility issues in the modeling and 
    smooth out the peaks and valleys of the curve. 
"""

import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

current_path = os.getcwd()
print("Current Path:", current_path)
sys.path.append('../')
from power_plants.get_installed_capacity import get_installed_cap_per_type_state_aneel
from utility_for_data import create_folder
from utility_for_data import daily_timestamp_for_given_year
from utility_for_data import fill_nan_7_days_in_df
from utility_for_data import hourly_timestamp_for_given_year
from utility_for_data import plot_setting
from utility_for_data import sin_state_region_mapping_in_series
from utility_for_data import translate_pt_to_en_df
from utility_for_data import multiply_if_decimal

sns.set_style('white')  # default the white background color of seaborn plot


def get_daily_inflow_per_region():
    """Read daily inflow energy - natural affluent energy (ENA) of reservoirs with daily periodicity by Subsystems

    Important:
        - source: `ONS <http://www.ons.org.br/Paginas/resultados-da-operacao/historico-da-operacao/energia_afluente_subsistema.aspx>`_
        - resolution: daily, four SIN regions (Subsystems)
        - access date: January, 2024
        - note: the unit in the raw dataset is MWmed, daily average of MWh, `data description <https://ons-dl-prod-opendata.s3.amazonaws.com/dataset/ena_subsistema_di/DicionarioDados_EnaPorSubsistema.pdf>`_
        - no negative value
    Return:
        pandas.DataFrame: time series of daily inflow at four ONS electric regions, with columns
            - subsystem: name of the electric subsystem defined in ONS
            - inflow: unit is [MWh], inflow energy at each subsystem
    """
    try:
        inflow = pd.read_csv(
            f"{os.path.dirname(os.path.realpath(__file__))}/resource/ONS_hydro_inflow_region_2000_2023_daily.csv")
        inflow['time'] = pd.DatetimeIndex(inflow['time'])
    except FileNotFoundError:
        create_folder('resource')
        # read in raw data
        files = glob.glob(
            f"{os.path.dirname(os.path.realpath(__file__))}/raw/ENA_DIARIO_SUBSISTEMA*.csv")
        datasets = []
        for fn in files:
            temp = pd.read_csv(fn, sep=";")
            temp = temp[['nom_subsistema', 'ena_data','ena_bruta_regiao_mwmed']]
            temp.index.name = 'index'
            temp.rename(columns={temp.columns[0]: 'Subsistema', temp.columns[1]: 'Din Instante', temp.columns[2]: 'Val Enaarmazenavelmwmes'}, inplace=True)
            rule = {'SUDESTE':'Sudeste', 'NORTE':'Norte', 'NORDESTE':'Nordeste', 'SUL':'Sul'}
            temp['Subsistema'] = temp['Subsistema'].replace(rule)
            temp['Din Instante'] = pd.to_datetime(temp['Din Instante'], format='%Y-%m-%d')
            temp['Din Instante'] = temp['Din Instante'].dt.strftime('%d.%m.%Y %H:%M:%S')
            temp = translate_pt_to_en_df(temp, 'column_ons_hydro')
            # performance improvement: instead of parse date in read_csv, use pd.to_datetime(df, format ='XXX')
            temp['time'] = pd.to_datetime(temp['time'], format='%d.%m.%Y %H:%M:%S')
            # get the daily value since the raw data is hourly (not-continuous)
            temp = temp.pivot_table(index='time', columns='region', values='inflow').resample('d').sum()
            # select the dataset within the desired time range: 2000-2023
            temp = temp.loc['2000':'2023']

            # check nan value for the daily index, if exists, fill the NaNs with the value 7 days ago or after
            idt_daily = [daily_timestamp_for_given_year(str(year)) for year in
                        range(min(temp.index.year), max(temp.index.year) + 1)]
            idt_daily = pd.DatetimeIndex(np.hstack(idt_daily))
            # check the length of the DataFrame df whether it equals to the idt_daily
            if not temp.shape[0] == idt_daily.shape[0]:  # when nan exist in df, go into the loop
                temp = temp.reindex(idt_daily)
                # fill the NaNs with the value 7 days ago or after
                temp, report_dict = fill_nan_7_days_in_df(temp, filepath=fn)

                # export the data report of missing data
                # chr(92) being the ASCII code for backslash '\\'
                with open(f"resource/DATA_REPORT_{fn.split(chr(92))[-1]}.txt", 'a') as file:
                    # indent keyword argument is set it automatically adds newlines
                    json.dump(report_dict, file, indent=2)

            datasets.append(temp)
        # combine the dataset that has all daily historical data, drop duplicated records
        inflow = pd.concat(datasets, axis=1)
        # convert the MWmês to MWh for daily data
        # source: http://www.ons.org.br/sites/multimidia/Documentos%20Compartilhados/dados/DADOS2014_ONS/9_1.html
        inflow = inflow * 30 * 24 / 30
        # convert the pivot dataframe to unpivot form
        inflow = inflow.melt(ignore_index=False).reset_index()
        inflow.to_csv('resource/ONS_hydro_inflow_region_2000_2023_daily.csv', index=False)
    return inflow


def get_state_hourly_inflow(base_year, phase='operation'):
    """Based on the dataset of historical daily inflow of region, create the hourly inflow for state, assuming:
        - inflow of state is proportion to the installed capacity of that state in the belonging region
        - equally distribute the daily inflow to hourly inflow
    """
    try:
        state_inflow = pd.read_csv(
            f'results/state_inflow_reference_year_{base_year}_distribute_by_hydropower_plants_{phase}.csv',
            index_col=0, parse_dates=True)
    except FileNotFoundError:
        # inflow is the same as the base year (not the yearly average)
        region_inflow = get_daily_inflow_per_region().pivot_table(index='time',
                                                                  values='value',
                                                                  columns='region').loc[str(base_year)]
        # get the scale factor between regions and states
        state_cap = get_installed_cap_per_type_state_aneel(base_year)
        hydro_cap = state_cap[state_cap['type'].isin(['hydro', 'small_hydro', 'mini_hydro'])]
        if phase == 'operation':
            hydro_cap = hydro_cap.query('phase==@phase').drop(['reference_year', 'phase'], axis=1).groupby(
                'state').sum()  # .drop(['reference_year', 'phase'], axis=1).
        else:
            hydro_cap = hydro_cap.drop(['reference_year', 'phase'], axis=1).groupby(
                ['state']).sum().reset_index().set_index('state')
            phase = 'operation+planning'
        # # assume the inflow of state is proportion to the installed capacity of that state in the belonging region
        sin = sin_state_region_mapping_in_series()
        factor = pd.concat([sin, hydro_cap], axis=1)

        # get the region sum for each state, note index is the state, value is the sum of p_nom the belonging region
        region_sum = factor.groupby('region').sum().loc[factor.region].set_index(factor.index).squeeze()
        factor['factor'] = factor['value'] / region_sum['value']

        # build up new dataset for daily inflow for state
        state_inflow = pd.DataFrame(index=region_inflow.index, columns=factor.index)  # index=region_inflow_mean.index
        for col in state_inflow.columns:
            r = factor.loc[col, 'region']
            state_inflow[col] = region_inflow[r] * factor.loc[col, 'factor']

        # convert the daily inflow to hourly inflow by equally division
        state_inflow = state_inflow.reindex(hourly_timestamp_for_given_year(base_year)).ffill() / 24

        # save the intermediate results for the per state inflow based on the installed capacity of hydropower
        create_folder('results')
        state_inflow.to_csv(
            f'results/ONS_state_inflow_distribute_by_hydropower_plants_{phase}_reference_year_{base_year}.csv')

    return state_inflow


def plot_historical_daily_inflow_each_year():
    """Plot the historical daily inflow in each year for each region,
    `inspiration <https://seaborn.pydata.org/examples/timeseries_facets.html>`_.

    The figures are saved: ``resource/daily_inflow_region_ONS_XX.png``.
    """
    inflow = get_daily_inflow_per_region()
    create_folder('resource')
    sns.set(font='serif')
    sns.set_style('white')  # default the white background color of seaborn plot
    for region_i in inflow['region'].unique():
        plt.close('all')
        # prepare data
        inflow_r = inflow.query('region==@region_i').copy()
        inflow_r['year'] = inflow_r['time'].dt.year
        inflow_r['dayofyear'] = inflow_r['time'].dt.dayofyear

        # plotting each year's time series in its own facet
        g = sns.relplot(
            data=inflow_r, x="dayofyear", y="value", col="year", hue="year",
            kind="line", linewidth=1, zorder=5, palette="GnBu", col_wrap=3, height=3, aspect=1, legend=False)
        # iterate over each subplot to customize further
        for year, ax in g.axes_dict.items():
            # add the title as an annotation within the plot
            ax.text(.8, .85, year, transform=ax.transAxes, fontweight="bold")

            # plot every year's time series in the background
            sns.lineplot(data=inflow_r, x="dayofyear", y="value", units="year",
                         estimator=None, color=".7", linewidth=0.2, ax=ax)

        # tweak the supporting aspects of the plot
        g.set_titles("")
        g.set(xlim=(0, 366))

        g.set_ylabels('MWh')
        g.tight_layout()

        g.savefig(fname=f'resource/daily_inflow_region_ONS_{region_i}.png',
                  dpi=500, bbox_inches='tight')


def plot_mean_and_historical_inflow():
    """Plot the average of the daily averages for all historical years"""
    inflow = get_daily_inflow_per_region()
    create_folder('resource')
    for region_i in inflow['region'].unique():
        plt.close('all')
        inflow_r = inflow.query('region==@region_i').copy()
        inflow_r['year'] = inflow_r['time'].dt.year
        inflow_r['dayofyear'] = inflow_r['time'].dt.dayofyear

        fig, ax = plot_setting()
        sns.set_style('white')  # default the white background color of seaborn plot
        sns.lineplot(ax=ax, data=inflow_r, x="dayofyear", y="value", units="year",
                     estimator=None, color=".7", linewidth=1)

        inflow_r_mean = inflow_r.pivot_table(index='dayofyear', values='value', columns='year').mean(axis=1)
        g = sns.lineplot(ax=ax, data=inflow_r_mean, label='mean of 2000-2023',
                         estimator=None, color="#0077b6", sizes=0.75, linewidth=5)
        g.set(xlim=(0, 366))

        g.set(xlabel="Day of year", ylabel="MWh")
        fig.savefig(fname=f'resource/daily_mean_historical_2000-2023_inflow_region_ONS_{region_i}.png',
                    dpi=500, bbox_inches='tight')


def plot_month_avg_inflow_per_region():
    """Plot the monthly average of inflow for the historical year per region.

    The figures are saved: ``resource/avg_inflow_per_month_region_2000-2023.png``.

    Hint:
        Possible conclusions from the figure:
            - hydrological complementarity existing between the SE-southeast and central west and S-south region
                - the distribution of the average monthly inflow(affluent natural energy) show that the respective dry and humid periods are not coincident
                - in region SE, the humid period occurs from Dec. to Apr. where the inflow are above the annual average
                - in region S, there is no obvious wet and dry period. However, the S region ex import/export energy surplus from/to SE region when needed, for instance, SE in the dry periods
            - there is no hydrological complementarity between N and NE region
                - their respective dry and wet periods coincide
                - during the wet season, the inflows to the Tucuruí Hydro-Power Complex in the Northern Region are extremely high, resulting in the occurrence of spillways that can be turbined and exported
    """
    inflow = get_daily_inflow_per_region()
    inflow = inflow.pivot_table(index='time', values='value', columns='region').resample('M').sum()
    inflow = inflow.groupby(inflow.index.month).mean().melt(ignore_index=False)
    inflow.index = pd.to_datetime(inflow.index, format='%m').month_name()

    # plotting
    plt.close('all')
    create_folder('resource')
    fig, ax = plot_setting(figure_size=(15, 5))
    sns.set_style('white')  # default the white background color of seaborn plot

    sns.barplot(ax=ax, data=inflow, x=inflow.index, y='value', hue='region', palette='GnBu')
    ax.legend(loc='upper center', ncol=4)
    ax.set_ylabel('MWh')
    ax.set_xlabel('')
    plt.tight_layout()
    fig.savefig(fname=f'resource/avg_inflow_per_month_region_2000-2023',
                dpi=500, bbox_inches='tight')


def plot_per_state_inflow(base_year, phase='operation'):
    plt.close('all')

    state_inflow = get_state_hourly_inflow(base_year, phase=phase)
    if phase != 'operation':
        phase = 'operation+planning'
    fig, ax = plot_setting(figure_size=(8, 4))
    # GWh
    (state_inflow/1e3).plot(ax=ax, linewidth=1.5,
                            color=['#2a9d8f', '#264653', '#e9c46a', '#f4a261', '#283618', '#6d6875', '#f1faee',
                                   '#a8dadc', '#457b9d', '#ff006e', '#cdb4db', '#ffc8dd', '#ffafcc', '#bde0fe',
                                   '#a2d2ff', '#606c38', '#e76f51', '#06d6a0', '#dda15e', '#bc6c25', '#003049',
                                   '#d62828', '#80ffdb', '#fcbf49', '#eae2b7', '#3a86ff', '#8338ec'])
    ax.set_ylabel(f'GWh')
    ax.set_xlabel('')
    ax.legend(ncol=2, bbox_to_anchor=(1.01, 1), frameon=False)
    ax.get_legend().set_title('state')
    ax.set_ylim(ymin=0)
    plt.tight_layout()

    fig.savefig(fname=f'results/ONS_state_inflow_distributed_by_{phase}_reference_year_{base_year}.png')


if __name__ == '__main__':
    for y in list(range(2018, 2024)):
        for p in ['operation', 'planning']:
            get_state_hourly_inflow(base_year=y, phase=p)
            plot_per_state_inflow(base_year=y, phase=p)

            # # % analysis
            region_inflow = get_daily_inflow_per_region()
            plot_month_avg_inflow_per_region()
            plot_historical_daily_inflow_each_year()
            plot_mean_and_historical_inflow()

