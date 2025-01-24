__version__ = '0.1.0'
__maintainer__ = 'Ying Deng 12.07.2022'
__authors__ = 'Ying Deng'
__credits__ = 'Ying Deng'
__email__ = 'ying.deng@dlr.de'
__date__ = '20.09.2021'
__status__ = 'dev'  # options are: dev, test, prod
__copyright__ = 'DLR'

"""Module-level docstring
Time series of the international transmission btw UR-Brazil, Argentina-Brazil
"""
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append('../')
from utility_for_data import (translate_pt_to_en_df, hourly_timestamp_for_given_year, fill_nan_7_days_in_df,
                              create_folder, plot_setting, set_plot_style)


def get_international_transmission_hourly(year=None):
    """
    Args:
        year: int, reference year
    Notes:
        - ONS: http://www.ons.org.br/Paginas/resultados-da-operacao/historico-da-operacao/intercambios_energia.aspx
        - access time: Jul. 2021
        - Tipo de Medida: Intercâmbio físico entre subsistemas (Physical Exchange between Subsystems)
        - hourly, four regions in SIN
        - unit, MWh
    Returns:
        - data: pd.DataFrame, hourly international electricity exchange between Brazilian state and neighborhood country
    """
    try:
        data = pd.read_csv(f"resource/Cross-border_transmission_1999_2020_hourly_4_SIN_regions.csv",
                           index_col=0)
        data.index = pd.to_datetime(data.index, format="%Y-%m-%d %H:%M:%S")  # format='%d.%m.%Y %H:%M:%S'
    except FileNotFoundError:
        create_folder('resource')
        files = glob.glob(f'{os.path.dirname(os.path.realpath(__file__))}'
                          f'/raw/*_Simples_Intercâmbio_de_Energia_Barra_Hora_Full_Data_data.csv')
        datasets = []
        for fn in files:
            temp = pd.read_csv(fn,
                               sep=';', thousands='.', decimal=',',
                               usecols=[9, 13, 14, 23],
                               engine='c',
                               low_memory=True,
                               ).drop_duplicates()
            temp = translate_pt_to_en_df(temp, 'column_ons_transmission')  # unit is MWmed -> here hourly -> = MWh
            temp['time'] = pd.to_datetime(temp['time'], format='%d.%m.%Y %H:%M:%S')
            temp = temp.set_index('time')

            # select the dataset within 1999-2020
            temp = temp.loc['1999':'2020']

            # check nan value for the daily index, if exists, fill the NaNs with the value 7 days ago or after
            idt_hourly = [hourly_timestamp_for_given_year(str(year)) for year in
                          range(min(temp.index.year), max(temp.index.year) + 1)]
            idt_hourly = pd.DatetimeIndex(np.hstack(idt_hourly))
            idt_hourly.name = temp.index.name
            # check the length of the DataFrame df whether it equals to the idt_weekly
            if not temp.shape[0] == idt_hourly.shape[0]:  # when nan exist in df, go into the loop
                temp = temp.reindex(idt_hourly)
                # fill the NaNs with the value 7 days ago or after
                temp, report_dict = fill_nan_7_days_in_df(temp, filepath=fn)

                # export the data report of missing data
                # chr(92) being the ASCII code for backslash '\\'
                with open(f"resource/DATA_REPORT_{fn.split(chr(92))[-1]}.txt", 'a') as file:
                    # indent keyword argument is set it automatically adds newlines
                    json.dump(report_dict, file, indent=2)

            # this is hack coded
            # http://www.ons.org.br/_layouts/download.aspx?SourceUrl=http://www.ons.org.br/Mapas/Mapa%20Sistema%20de%20Transmissao%20-%20Horizonte%202024.pdf
            # replace the "SIN" in column "start_node" with the corresponding state name of Brazil
            temp['start_node'] = 'RS'
            temp.reset_index(inplace=True)
            datasets.append(temp)
        # convert the pivot dataframe to unpivot form
        data = pd.concat(datasets, axis=0).set_index('time')
        data.to_csv(f'resource/Cross-border_transmission_1999_2020_hourly_4_SIN_regions.csv', index=True)
    # select the data for the given year
    if year:
        if year in list(range(1999, 2021)):
            data = data.loc[str(year)]
        else:
            assert "Only year 1999-2020 is supported"
    return data


def get_data_paper_output():
    """international transmission from 2012-2020
    """
    create_folder('results')
    data = get_international_transmission_hourly().loc['2012':'2020']
    data.to_csv('results/Cross-border_transmission_RS-URU_RS-ARG_2012-2020_hourly.csv')
    return data


def analysis_international_transmission(tag='full'):
    # %% Time Series Analysis
    # read in dataset for the year; hourly
    if tag == 'full':
        # since the SIN-Uruguay don't have data for 1999-2001 and a lot of missing data in 2002
        df = get_international_transmission_hourly().loc['2003':]
        fig_name = '_2003_2020'
    else:
        df = get_data_paper_output()
        fig_name = '_2012_2020_data_paper'

    df['name'] = df['start_node'] + '-' + df['end_node']
    df = df.pivot_table(index='time', values='power', columns='name')

    # plot the time series data; daily
    # hourly is too dense -> df.plot(ax=ax, style=['-', '-'], linewidth=0.5, alpha=0.5)
    fig, ax = plot_setting(figure_size=(8, 4))
    daily = df.resample('w').sum() / 1e3  # GWh
    daily.plot(ax=ax, title='',
               style=['-', '-'], linewidth=1.5, color=['#ba181b', '#6c757d']),
    ax.set_ylabel('GWh')
    ax.set_xlabel('')
    ax.legend(bbox_to_anchor=(1.01, 0.8), title='', frameon=False)
    fig.autofmt_xdate(ha='center')
    plt.tight_layout()
    fig.savefig(fname=f'resource/weekly_transmission{fig_name}.png',
                dpi=300, bbox_inches='tight')

    # dig into the data
    # by_time
    fig, ax = plot_setting(figure_size=(8, 4))
    by_time = df.groupby(df.index.time).mean()
    hourly_ticks = 4 * 60 * 60 * np.arange(6)
    by_time.plot(ax=ax, xticks=hourly_ticks, style=['-', '-'], linewidth=0.5, color=['#008083', '#ff703d'])
    ax.set_xlabel('')
    ax.legend(loc='upper left')
    ax.set_xlim(left=by_time.index[0], right=by_time.index[-1])
    ax.get_legend().set_title('')
    fig.autofmt_xdate(ha='center')
    plt.tight_layout()
    fig.savefig(fname=f'resource/transmission_by_hour_of_day{fig_name}.png',
                dpi=300, bbox_inches='tight')

    # plot the weekly seasonality using box plot
    fig, ax = plot_setting(figure_size=(8, 4), n_cols=1, n_rows=2)
    for name, axes in zip(df.columns, ax):
        sns.boxplot(ax=axes, data=df, x=df.index.weekday, y=name,
                    linewidth=0.5, fliersize=1,
                    boxprops={'facecolor': 'none', 'edgecolor': 'black'},
                    medianprops={'color': '#ff703d'}
                    )
        axes.set_xlabel('')
        axes.set_ylabel('MWh')
        axes.set_title(name, loc='center')
        axes.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    fig.autofmt_xdate(ha='center')
    plt.tight_layout()
    fig.savefig(fname=f'resource/transmission_weekly_seasonality_box_plot{fig_name}.png',
                dpi=300, bbox_inches='tight')

    # plot the monthly seasonality
    fig, ax = plot_setting(figure_size=(8, 4), n_cols=1, n_rows=2)
    for name, axes in zip(df.columns, ax):
        sns.boxplot(ax=axes, data=df, x=df.index.month, y=name,
                    linewidth=0.5, fliersize=1,
                    boxprops={'facecolor': 'none', 'edgecolor': 'black'},
                    medianprops={'color': '#ff703d'}
                    )
        axes.set_xlabel('')
        axes.set_ylabel('MWh')
        axes.set_title(name, loc='center')
        axes.set_xticklabels(
            ['Jan.', 'Feb.', 'Mar.', 'Apr.', 'May', 'June', 'July', 'Aug.', 'Sept.', 'Oct.', 'Nov.', 'Dec.'])
    fig.autofmt_xdate(ha='center')
    plt.tight_layout()
    fig.savefig(fname=f'resource/transmission_monthly_seasonality_box_plot{fig_name}.png',
                dpi=300, bbox_inches='tight')

    # plot the seasonal for many years
    # prepare data
    df = df.copy().melt(ignore_index=False)
    df['year'] = [d.year for d in df.index]
    df['month'] = [d.strftime('%b') for d in df.index]

    # draw plot
    plt.close('all')
    set_plot_style()
    # note: to increase space between rows on FacetGrid plot, use gridspec_kws
    g = sns.FacetGrid(df, row='name', hue='year', height=4, aspect=2, sharey=True, gridspec_kws={"hspace": 0.3})
    g = (g.map(sns.lineplot, 'month', 'value', ci=None, linewidth=1).add_legend(ncol=2, bbox_to_anchor=(1.0, 0.5)))

    # decoration
    # make the name of subplot readable
    [plt.setp(ax.texts, text="") for ax in g.axes.flat]  # remove the original texts
    g.set_titles(row_template='{row_name}', col_template='{col_name}')
    # adjust the y range for better plotting
    g.set(ylim=(-2000, 1000))
    g.set_xlabels('')
    g.set_ylabels('MWh')

    plt.savefig(fname=f'resource/transmission_line_plot{fig_name}.png',
                dpi=300, bbox_inches='tight')

    # boxplot of Month-wise (Seasonal) and Year-wise (trend) Distribution
    g = sns.FacetGrid(df, row='name', height=4, aspect=2, sharey=True, gridspec_kws={"hspace": 0.3})
    g = (g.map(sns.boxplot, 'year', 'value', linewidth=0.5, fliersize=1,
               boxprops={'facecolor': 'none', 'edgecolor': 'black'},
               medianprops={'color': '#ff703d'}
               ).add_legend(ncol=2, bbox_to_anchor=(1.0, 0.5)))
    [plt.setp(ax.texts, text="") for ax in g.axes.flat]  # remove the original texts
    g.set_titles(row_template='{row_name}', col_template='{col_name}')
    # adjust the y range for better plotting
    g.set(ylim=(-2000, 1000))
    g.set_xlabels('')
    g.set_ylabels('MWh')
    g.fig.autofmt_xdate(ha='center')

    plt.savefig(fname=f'resource/transmission_box_plot{fig_name}.png',
                dpi=300, bbox_inches='tight')


# if '__name__' == '__main__':
get_data_paper_output()
for t in ['full', 'paper']:
    get_international_transmission_hourly()
    analysis_international_transmission(tag=t)
