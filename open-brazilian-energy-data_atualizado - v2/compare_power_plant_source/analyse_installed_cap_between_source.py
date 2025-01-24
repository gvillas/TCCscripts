# -*- coding:utf-8 -*-

__version__ = '0.1.0'
__maintainer__ = 'Ying Deng 12.12.2022'
__authors__ = 'Ying Deng'
__credits__ = ''
__email__ = 'Ying.Deng@dlr.de'
__date__ = '02.05.2021'
__status__ = 'dev'  # options are: dev, test, prod
__copyright__ = 'DLR'

""" A module-level docstring
This is to analyze the installed capacity of the dataset from different Brazilian institutions aiming to know the 
numerical gap between them. 
In the end, ANEEL SIGA dataset is used as the source for the power plants infrastructure in the model, while the ONS_Cap 
datasets used for the validation
"""
import geopandas as gpd
import numpy as np
import pandas as pd

from grid.prepare_grid import read_shp_power_plants_from_epe
from node.create_node import get_node_by_state
from power_plants.get_installed_capacity import read_aneel_raw
from utility_for_data import create_folder
from utility_for_data import delete_accents_pt_words_in_series
from utility_for_data import translate_pt_to_en_df


def get_state_from_plant_id(plant):
    """Derive the state information from the plant_id for EPE Webmap dataset
    Args:
        - plant: gpd.GeoDataFrame or pd.DataFrame with column plant_id (and geometry)
    """
    # derive the info of state from plant_id
    plant['state'] = plant['plant_id'].str[7:9]
    if isinstance(plant, gpd.GeoDataFrame):
        # fill the missing nan in state (missing 'plant_id')
        empty_plant_id = plant[plant['state'].isna()]
        # get the belonging state through location projection
        state_polygon = get_node_by_state()
        plant['state'] = plant['state'].fillna(
            gpd.sjoin(empty_plant_id, state_polygon, how="left", predicate='intersects')['name'])

    return plant


def harmonise_plant_type_for_group_cap(df, map_dict_name):
    """Aggregate installed capacity in the EPE Webmap/ANEEL-SIGA/ONS dataset based on our classification ("harmonised")
    Args
        - df: DataFrame, pivot format
            - columns: plant type
            - index: state
            - values: capacity [MW]
        - map_dict_name: str, name of the map dict used. Only name in mapping is supported
    Returns
        - summed: pd.DataFrame
    Notes:
        the mapping between EPE/ANEEL and ONS follows:
              ----------------------------------------------------------------
                plant type of each data source (default -> refine = False)
              ----------------------------------------------------------------
              harmonised    ONS               EPE                   ANEEL
              ----------------------------------------------------------------
              solar_pv      solar_pv          solar_pv              solar_pv
              on_wind       on_wind           on_wind               on_wind
              nuclear       nuclear           nuclear               nuclear
              thermal       thermal           biomass_thermal       thermal
                                              fossil_thermal
              hydro         hydro             small_hydro           small_hydro
                            hydro_pump        mini_hydro            mini_hydro
                                              hydro                 hydro
                                                                    wave
              ----------------------------------------------------------------
    epe_map: dict, plant type map between EPE and harmonised
                    - key: the group of classification in harmonised
                    - value: the corresponding plant type in EPE
    aneel_map_default: dict, plant type map between ANEEL and harmonised.
                                (df = get_installed_cap_per_type_state_aneel(refine=False))
                    - key: the group of classification in harmonised
                    - value: the corresponding plant type in ANEEL
    aneel_map: dict, plant type map between ANEEL and harmonised.
                                (df = get_installed_cap_per_type_state_aneel(refine=True))
                    - key: the group of classification in harmonised
                    - value: the corresponding plant type in ANEEL
    ons_map: dict, plant map between the ONS and the harmonised
                    - key: the group of classification in harmonised
                    - value: the corresponding plant type in ONS
    """

    epe_map = {'hydro': ['hydro', 'small_hydro', 'mini_hydro'],
               'thermal': ['biomass_thermal', 'fossil_thermal'],
               'on_wind': 'on_wind',
               'solar_pv': 'solar_pv',
               'nuclear': 'nuclear'}

    aneel_map_default = {'hydro': ['hydro', 'small_hydro', 'mini_hydro'],
                         'thermal': 'thermal',
                         'on_wind': 'on_wind',
                         'solar_pv': 'solar_pv',
                         'nuclear': 'nuclear'}

    aneel_map = {'hydro': ['hydro', 'small_hydro', 'mini_hydro'],
                 'thermal': ['oil', 'gas', 'coal', 'biomass'],
                 'on_wind': 'on_wind',
                 'solar_pv': 'solar_pv',
                 'nuclear': 'nuclear'}

    ons_map = {'hydro': ['hydro', 'hydro_pump'],
               'thermal': 'thermal',
               'on_wind': 'on_wind',
               'solar_pv': 'solar_pv',
               'nuclear': 'nuclear'}

    mapping = {'epe_map': epe_map,
               'aneel_map_default': aneel_map_default,
               'aneel_map': aneel_map,
               'ons_map': ons_map}

    if map_dict_name not in ['epe_map', 'aneel_map_default', 'aneel_map', 'ons_map']:
        assert 'name of mapping dict given, parameter map_dict_name, is wrong'
    else:
        summed = pd.DataFrame()
        for k, v in mapping[map_dict_name].items():
            if isinstance(v, list):
                try:
                    summed[k] = df[v].sum(axis=1)
                # to avoid the keyError when some plant type is filtered due to the "regime" or "start_time"
                except KeyError:
                    pass

            if isinstance(v, str):
                try:
                    summed[k] = df[v]
                except KeyError:
                    pass
        return summed


def read_ons_installed_capacity():
    """read in ONS Historical Database
    data and metadata: https://dados.ons.org.br/dataset/capacidade-geracao
    access date: 2022 12 08
    Notes:
        - the installed capacity for the power plants in operation
    """

    ons = translate_pt_to_en_df(pd.read_csv('raw/raw_CAPACIDADE_GERACAO.csv', sep=';'), 'column_ons')
    # drop the columns as they are duplicated info after translation
    ons = ons.drop(['region_id', 'state_name'], axis=1)
    # rename the state information for Itaipu. Was 'I', change to "PR". The total capacity is 7000MW (Brazil's share)
    to_correct_state = ons.query("state=='I'")
    ons.loc[to_correct_state.index, 'state'] = 'PR'
    return ons


def harmonise_plant_id(df):
    """Revise the format of CEG in dataset for harmonisation
    Notes:
        - reason: EPE Webmap and ONS installed capacity dataset has different format of CEG regarding the last three
                  element; ANEEL don't have the version number
        - explanation: The last three element is the version number
        - revise the CEG will not lead to the change number of unique CEG

                                 ANEEL-SIGA          EPE Webmap       ONS Historical Database (installed capacity)
    data entities                  10541               3178                4191
    unique CEG                     10541               3160                1389
    unique CEG (after revision)    10541               3160                1389

    """
    df['plant_id'] = df['plant_id'].apply(lambda x: x[:-3] if pd.notna(x) else np.nan)
    return df


def get_data():
    """ Get the data the three dataset to compare
    Returns:
        - aneel: ANEEL-SIGA
        - epe: EPE Webmap
        - ons: ONS Historical Database

    """
    # ONS Historical Database (installed capacity) -> operation
    ons = harmonise_plant_id(read_ons_installed_capacity())
    ons.to_csv('resource/ONS_installed_capacity_operation_full.csv')

    # EPE Webmap (installed capacity) -> operation and planned
    epe = harmonise_plant_id(read_shp_power_plants_from_epe())
    epe.to_csv('resource/EPE_Webmap_installed_capacity_operation_and_planned_full.csv')

    # ANEEL-SIGA (installed capacity) -> operation and planed
    aneel = read_aneel_raw()
    aneel.to_csv('resource/ANEEL_SIGA_installed_capacity_operation_and_planned_full.csv')

    return ons, epe, aneel


def compare_nr_data_entities_unique_plant_id():
    """Compare the three dataset
    Notes:
        ANEEL-SIGA and EPE Webmap contains the data entities for power plants both in operation and planned,
        while ONS only contains power plants in operation

    """
    # get the data
    ons, epe, aneel = get_data()

    # get the statistics of numbers for each dataset
    ons_nr = pd.Series([ons.shape[0], ons.shape[1], ons.plant_id.unique().shape[0], ons.plant_name.unique().shape[0]],
                       name='ONS Historical Database')
    epe_nr = pd.Series([epe.shape[0], epe.shape[1], epe.plant_id.unique().shape[0], epe.plant_name.unique().shape[0]],
                       name='EPE Webmap')
    aneel_nr = pd.Series([aneel.shape[0], aneel.shape[1], aneel.plant_id.unique().shape[0],
                         aneel.plant_name.unique().shape[0]], name='ANEEL-SIGA')

    # prepare the summary
    summary = pd.concat([ons_nr, epe_nr, aneel_nr], axis=1)
    summary.index = ['Number of data entities', 'Number of attributes', 'Number of unique plant id',
                     'Number of unique plant name']

    return summary


def compare_installed_cap_per_type():
    """Compare the installed capacity between the dataset per federal state per harmonised plant type

    Returns:

    Notes:
        - ANEEL-SIGA and EPE Webmap contains the data entities for power plants both in operation and planned,
          while ONS only contains power plants in operation
        - the plant type is different across the three dataset, therefore, need to aggregate

    """
    # ONS Historical Database (installed capacity) -> operation
    ons = harmonise_plant_id(read_ons_installed_capacity())
    ons.to_csv('resource/ONS_installed_capacity_operation_full.csv')
    # EPE Webmap (installed capacity) -> operation and planned
    epe = harmonise_plant_id(read_shp_power_plants_from_epe())
    epe.to_csv('resource/EPE_Webmap_installed_capacity_operation_and_planned_full.csv')
    # ANEEL-SIGA (installed capacity) -> operation and planed
    aneel = read_aneel_raw()
    epe.to_csv('resource/ANEEL_SIGA_installed_capacity_operation_and_planned_full.csv')

    ons_agg = ons[['state', 'type', 'capacity']].groupby(['state', 'type']
                                                         ).sum().pivot_table(index='state', columns='type',
                                                                             values='capacity')
    ons_agg = harmonise_plant_type_for_group_cap(ons_agg, 'ons_map').fillna(0)

    epe = get_state_from_plant_id(epe)
    epe_agg = epe[['state', 'type', 'capacity']].groupby(['state', 'type']
                                                         ).sum().pivot_table(index='state', columns='type',
                                                                             values='capacity')
    epe_agg = harmonise_plant_type_for_group_cap(epe_agg, 'epe_map').fillna(0)

    aneel = get_state_from_plant_id(aneel)
    aneel_agg = aneel[['state', 'type', 'capacity']].groupby(['state', 'type']
                                                             ).sum().pivot_table(index='state', columns='type',
                                                                                 values='capacity')
    aneel_agg = harmonise_plant_type_for_group_cap(aneel_agg, 'aneel_map').fillna(0)

    comp_cap_per_state = pd.concat([aneel_agg.stack().rename('ANEEL-SIGA'), epe_agg.stack().rename('EPE Webmap'),
                                    ons_agg.stack().rename('ONS')], axis=1).reset_index().rename(
        columns={'level_1': 'type'})

    return comp_cap_per_state


def calculate_installed_cap_diff(df):
    """Compare the difference of installed capacity for each plant type from ONS, ANEEL-SIGA, EPE
    Notes:
        - the ONS is used as reference to calculate the relative deviation
        - four new columns are added
            - 'ANEEL-SIGA - ONS[MW]': absolute deviation of installed capacity between ANEEL-SIGA and ONS, in MW
            - 'EPE Webmap - ONS[MW]': absolute deviation of installed capacity between EPE Webmap and ONS, in MW
            - 'ANEEL-SIGA - ONS)/ONS[%]': relative deviation of installed capacity between ANEEL-SIGA and ONS, in %
            - 'EPE Webmap - ONS)/ONS[%]': relative deviation of installed capacity between EPE Webmap and ONS, in %
    """

    df['ANEEL-SIGA - ONS[MW]'] = df['ANEEL-SIGA'] - df['ONS']
    df['(ANEEL-SIGA - ONS)/ONS[%]'] = (df['ANEEL-SIGA'] - df['ONS']) / df['ONS'] * 100

    df['EPE Webmap - ONS[MW]'] = df['EPE Webmap'] - df['ONS']
    df['(EPE Webmap - ONS)/ONS[%]'] = (df['EPE Webmap'] - df['ONS']) / df['ONS'] * 100
    return df


def match_aneel_epe():
    """Match each power plants
        - set the plant type based on EPE
        - 'wave' reminds
        - group EPE by plant_id for matching (note: to be improved, as it may introduce errors)

    Notes:
        - unit of capacity is MW
        - both ANEEL-SIGA and EPE contains capacity in operation and planned
        - the plant_id in ANEEL-SIGA is unique
        - the plant_id is not unique and has missing value (2), therefore
            - we first group the EPE by its plant_id and the plant_name (no missing values)
        - by matching, we have to steps:
            plant_id matched?
                if yes, check:
                    if value of capacity matched? (even small difference is regarded as unmatched)
                    if value of plant name matched? (the plant name is harmonised to upper wo accents)
                    if value of state matched
                if not, provide info:
                    the plant_id, only ANEEL-SIGA has: column "_merge" := "ANEEL_only"
                    the plant_id, only EPE has: column "_merge" := "EPE_only"
    """
    # read the dataset
    _, epe, aneel = get_data()

    # process the data for merging
    epe = get_state_from_plant_id(epe)

    epe_unique = epe.groupby(['plant_id', 'plant_name']).agg({'capacity': sum, 'phase': list,
                                                              'type': list, 'state': list}).reset_index()
    epe_unique['type'] = epe_unique['type'].apply(lambda x: list(set(x))).apply(lambda x: x[0] if len(x) == 1 else x)
    epe_unique['state'] = epe_unique['state'].apply(lambda x: list(set(x))).apply(lambda x: x[0] if len(x) == 1 else x)
    epe_unique['phase'] = epe_unique['phase'].apply(lambda x: list(set(x))).apply(lambda x: x[0] if len(x) == 1 else x)

    # change the value of 'type' (plant type) based on the classification of 'type' defined in EPE
    # i.e., refine thermal in column 'type' to fossil_thermal and biomass_thermal based on the column 'fuel_type'
    aneel.loc[aneel['fuel_type'] == 'Fossil', 'type'] = 'fossil_thermal'
    aneel.loc[aneel['fuel_type'] == 'Biomass', 'type'] = 'biomass_thermal'

    # merge ANEEL-SIGA and EPE Webmap data based on attribute "plant_id"
    merge_res = pd.merge(aneel, epe_unique, on='plant_id', how='outer', indicator=True
                         ).sort_values(by=['type_x', 'type_y'])
    # case of matching
    matched = merge_res.query("_merge=='both'").copy()
    # check whether the capacities are matched for EPE and ANEEL (observation: only several doesn't match)
    matched.loc[:, 'matched_cap'] = matched.apply(lambda x: x.capacity_x == x.capacity_y, axis=1)
    # check whether the federal state are matched for EPE and ANEEL (observation: only several doesn't match)
    matched.loc[:, 'matched_state'] = matched.apply(lambda x: str(x.state_x) in str(x.state_y), axis=1)
    # check whether the plant_name are matched for EPE and ANEEL (observation: only several doesn't match)
    matched.loc[:, 'matched_plant_name'] = matched.apply(lambda x: str(x.plant_name_x) in str(x.plant_name_y), axis=1)

    # case of not matching
    # to avoid SettingWithCopyWarning: set .copy()
    unmatched = merge_res.query("_merge!='both'").copy()
    unmatched['_merge'] = unmatched['_merge'].apply(lambda x: 'ANEEL_only' if x == 'left_only' else 'EPE_only')
    return matched, unmatched


def match_aneel_ons():
    """Match ANEEL-SIGA and ONS dataset by the plant_id
    Notes:
        - unit of capacity is MW
        - ANEEL-SIGA contains capacity in operation and planned, while ONS only in operation
        - the plant_id in ANEEL-SIGA is unique
        - the plant_id in ONS is not unique and has missing value (69), therefore:
            - we first group the ONS by its plant_id and plant_name (no missing values)
        - by matching, we have to steps:
            plant_id matched?
                if yes, check:
                    if value of capacity matched? (even small difference is regarded as unmatched)
                    if value of plant name matched? (the plant name is harmonised to upper wo accents)
                    if value of state matched
                if not, provide info:
                    the plant_id, only ANEEL-SIGA has: column "_merge" := "ANEEL_only"
                    the plant_id, only EPE has: column "_merge" := "ONS_only"
    """
    # read the dataset
    ons, _, aneel = get_data()

    # group the ons based on the unique pair of plant_id and plant_name
    ons_unique = ons.groupby(['plant_id', 'plant_name']).agg(
        {'ons_name': list, 'capacity': sum, 'type': list, 'grid_mode': list, 'state': list,
         'primary': list}).reset_index()
    ons_unique['ons_name'] = ons_unique['ons_name'].apply(lambda x: list(set(x))).apply(
        lambda x: x[0] if len(x) == 1 else x)
    ons_unique['type'] = ons_unique['type'].apply(lambda x: list(set(x))).apply(lambda x: x[0] if len(x) == 1 else x)
    ons_unique['grid_mode'] = ons_unique['grid_mode'].apply(lambda x: list(set(x))).apply(
        lambda x: x[0] if len(x) == 1 else x)
    ons_unique['state'] = ons_unique['state'].apply(lambda x: list(set(x))).apply(lambda x: x[0] if len(x) == 1 else x)
    ons_unique['primary'] = ons_unique['primary'].apply(lambda x: list(set(x))).apply(
        lambda x: x[0] if len(x) == 1 else x)

    # merge ANEEL-SIGA and ONS data based on attribute "plant_id"
    merge_res = pd.merge(aneel, ons_unique, on='plant_id', how='outer', indicator=True)

    # case of matching
    matched = merge_res.query("_merge=='both'").copy()
    # check whether the capacities are matched for EPE and ANEEL (observation: only several doesn't match)
    matched.loc[:, 'matched_cap'] = matched.apply(lambda x: x.capacity_x == x.capacity_y, axis=1)
    # check whether the federal state are matched for EPE and ANEEL (observation: only several doesn't match)
    matched.loc[:, 'matched_state'] = matched.apply(lambda x: str(x.state_x) in str(x.state_y), axis=1)
    # check whether the plant name are matched (observation: only several doesn't match)
    matched['plant_name_x'] = delete_accents_pt_words_in_series(matched['plant_name_x']).apply(lambda x: x.upper())
    matched['plant_name_y'] = delete_accents_pt_words_in_series(matched['plant_name_y']).apply(lambda x: x.upper())
    matched.loc[:, 'matched_plant_name'] = matched.apply(lambda x: str(x.plant_name_x) in str(x.plant_name_y), axis=1)

    # case of not matching
    unmatched = merge_res.query("_merge!='both'").copy()
    unmatched['_merge'] = unmatched['_merge'].apply(lambda x: 'ANEEL_only' if x == 'left_only' else 'ONS_only')
    return matched, unmatched


# %% report

def main_generate_report():
    create_folder('results')
    with pd.ExcelWriter(f'results/installed_capacity_comparison.xlsx', engine='openpyxl') as writer:
        # compare number of data entities between ANEEL-SIGA, EPE Webmap and ONS
        compare_nr_data_entities_unique_plant_id().to_excel(writer,
                                                            sheet_name='data_info')

        # difference of installed capacity between ANEEL-SIGA and ONS
        # national per type
        calculate_installed_cap_diff(compare_installed_cap_per_type().groupby('type').sum()
                                     ).to_excel(writer, sheet_name='per_type_cap')
        # federal state level per type
        calculate_installed_cap_diff(compare_installed_cap_per_type()).to_excel(writer,
                                                                                sheet_name='per_type_per_state_cap')

        # difference of installed capacity between ANEEL-SIGA and EPE Webmap
        # match each power plants and sum the results by states
        plant_id_matched_aneel_epe, plant_id_unmatched_aneel_epe = match_aneel_epe()
        plant_id_matched_aneel_epe.to_excel(writer, sheet_name='id_matched_aneel_epe')
        plant_id_unmatched_aneel_epe.to_excel(writer, sheet_name='id_unmatched_aneel_epe')

        # difference of installed capacity between ANEEL-SIGA and ONS
        # match each power plants and sum the results by states
        plant_id_matched_aneel_ons, plant_id_unmatched_aneel_ons = match_aneel_ons()
        plant_id_matched_aneel_ons.to_excel(writer, sheet_name='id_matched_aneel_ons')
        plant_id_unmatched_aneel_ons.to_excel(writer, sheet_name='id_unmatched_aneel_ons')


# if '__name__' == '__main__':
main_generate_report()
