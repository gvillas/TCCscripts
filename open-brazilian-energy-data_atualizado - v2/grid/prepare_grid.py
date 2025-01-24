# -*- coding:utf-8 -*-

__version__ = '0.1.0'
__maintainer__ = 'Ying Deng 21.07.2022'
__authors__ = 'Ying Deng'
__credits__ = 'Ying Deng'
__email__ = 'Ying.Deng@dlr.de'
__date__ = '27.10.2021'
__status__ = 'dev'  # options are: dev, test, prod
__copyright__ = 'DLR'

"""Module-level docstring
Prepare input data for transmission lines (AC and HVDC lines) and create network topologies from EPE Webmap (download 
March 2021, September 2020 version).

Use spatial and topological analysis to transform the map objects in the EPE Webmap into a network model of the electric 
power system (being used in script solve_network). The calculation of the transfer capacity of AC lines is based on the 
transmission model (also known as the net transfer capacity model) calculation, where each line has different starting 
and ending nodes (in this model, each federal state in Brazil is considered as a node, so that there are 27 nodes in 
total). 

Due to the missing information in the original dataset, there are several assumptions by adding electrical parameters 
for each line. The basis for all assumptions is provided inside the functions.

Many improvements are needed in the future (see TODO).

Abbreviations:
    - sub: substation
    - trans_cap: transfer_capacity
    - _start: _0
    - _end: _1
    - distance_: dis_
    - l: line
    - s: substations
    - pp: power plants

TODO: compare or compensate the info with the ONS data
    - short comparison: EPE (operation) -> 1586 pieces; ONS (only operation) -> 2007
    - ONS dataset has the information of "Line type (AC, DC, or line extension)"
"""
import glob
import logging
import os
import sys

import fiona
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
import shapely
from shapely.geometry import Point

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.append(parent_dir)

from utility_for_data import (read_config, create_folder, translate_pt_to_en_df, cut_linestring,
                              float_range, plot_setting)
from node.create_node import get_node_by_state


# %  pre-processing of the shapefile
def read_shp_line_subs_from_epe(shp_kind_en):
    """Read the substation/line shapefile
    combine the shapefile of different phase, add the phase information of "operation" or "construction",
    set the CRS, and drop unwanted columns

    Args:
        - shp_kind_en: str, 'substation' OR 'line', the kind of shapefile in English, to get the data in gpd.DataFrame

    Important:
        - source: 'EPE Webmap <https://gisepeprd2.epe.gov.br/WebMapEPE/>'
        - access date: 2021 03 10
        - the CRS in raw shapefile is 'EPSG: 4674 (SIRGAS 2000)<(https://epsg.io/4674)>, unit- degree, see https://arquivos.epe.gov.br/DEA/SMA/Metadados%20WebMapEPE/Sistema%20El%C3%A9trico%20Existente/Metadados%20-%20Linhas%20de%20Transmiss%C3%A3o%20-%20Base%20Existente.pdf
            - in the dataset this information might not be provided (value is "None") or EPSG: 4674
            - to use the calculation, e.g, "centroid", "sjoin", in geopandas, which calculate the distance, should use the projected CRS, unit in metre
                - see Geopandas, https://geopandas.org/en/stable/docs/reference/api/geopandas.sjoin_nearest.html#geopandas.sjoin_nearest
            - for this reason, the map has been re-projected, .to_crs('epsg:4087')

    Return:
        - gdf: gpd.GeoDataFrame, a shape file with added information
    """
    try:
        gdf = gpd.read_file(f'resource/EPE_{shp_kind_en}_operation_and_construction.shp')  # 'epsg:4087'
    except (FileNotFoundError, fiona.errors.DriverError) as e:
        create_folder('resource')
        # the "phase" and "shp" is extracted from the filename of shapefile
        phase = {'Base_Existente': 'operation',
                 'Expansão_Planejada': 'construction'}
        shp_name_pt = {'substation': 'Subestações',
                       'line': 'Linhas_de_Transmissão'}[shp_kind_en]

        var = []
        for phase_name_pt, phase_name_en in phase.items():
            gdf = gpd.read_file(f'raw/{shp_name_pt}_-_{phase_name_pt}.shp'
                                ).to_crs('epsg:4087')
            gdf = translate_pt_to_en_df(gdf, 'column_epe_shp_line_subs')
            # manual add the information of phase since this is not included in the raw shapefile
            gdf['phase'] = phase_name_en
            gdf['voltage'] = gdf['voltage'].astype(str).str.replace(',', '.').astype(int, errors='ignore')
            gdf['start_time'] = gdf['start_time'].replace(['-', 0.0], np.NaN)
            var.append(gdf)
        gdf = pd.concat(var, axis=0).reset_index(drop=True)
        # drop the unwanted information, set errors='ignore' because substation shapefile don't have all those columns
        gdf = gdf.drop(['Shape_STLe', 'created_da', 'created_us', 'last_edite', 'last_edi_1'], axis=1, errors='ignore')
        # store the processed shapefile - translation, drop unwanted columns, add phase information
        # known Warning (FutureWarning: pandas.Int64Index) of Geopandas: https://github.com/geopandas/geopandas/issues/2347
        gdf.to_file(f'resource/EPE_{shp_kind_en}_operation_and_construction_epsg4087.shp')
    return gdf


def read_shp_power_plants_from_epe():
    """Read in the power plants of installed and planned from EPE Webmap dataset; installed capacity is converted
    from kW to MW.
    Notes:
        - plant_id has duplicates
        - Gindaí and Espraiado don't have plant_id

    Returns
        - gdf: gpd.GeoDataFrame, the cleaned dataset of power plants information from EPE Webmap
    """
    try:
        gdf = gpd.read_file(
            f'{os.path.dirname(os.path.realpath(__file__))}/resource/EPE_power_plants_operation_and_construction_epsg4087.shp')
        # Avoiding problematic data set as 'YYYY'
        gdf['start_time'] = gdf['start_time'].fillna(pd.to_datetime('1/1/1956')).apply(lambda x: pd.to_datetime('1/1/' + str(x), format='%m/%d/%Y', errors='coerce') if isinstance(x, (int, str)) and x.isdigit() else x)
        gdf['start_time'] = pd.to_datetime(gdf['start_time'])

    except (FileNotFoundError, fiona.errors.DriverError) as e:
        create_folder(f'{os.path.dirname(os.path.realpath(__file__))}/resource')
        # the "phase" and "shp" is extracted from the filename of shapefile
        phase = {'Base_Existente': 'operation',
                 'Expansão_Planejada': 'construction'}
        shp_name = {'small_hydro': 'PCH',
                    'mini_hydro': 'CGH',
                    'on_wind': 'EOL',
                    'solar_pv': 'UFV',
                    'hydro': 'UHE',
                    'biomass_thermal': 'UTE_Biomassa',
                    'fossil_thermal': 'UTE_Fóssil',
                    'nuclear': 'UTE_Nuclear',
                    }

        var = []
        for phase_name_pt, phase_name_en in phase.items():
            for shp_name_en, shp_name_pt in shp_name.items():
                gdf = gpd.read_file(f'{os.path.dirname(os.path.realpath(__file__))}/raw/{shp_name_pt}_-_{phase_name_pt}.shp'
                                    ).to_crs('epsg:4087')
                gdf = translate_pt_to_en_df(gdf, 'column_epe_shp_plant')
                # manual add the information of phase since this is not included in the raw shapefile
                gdf['phase'] = phase_name_en
                gdf['type'] = shp_name_en
                gdf['start_time'] = gdf['start_time'].replace(['-', 0.0], np.NaN)
                # drop the unwanted cols, set errors='ignore' because substation shapefile don't have all those columns
                gdf = gdf.drop(['Shape_STLe', 'created_da', 'created_us', 'last_edite', 'last_edi_1'],
                               axis=1, errors='ignore')
                var.append(gdf)
        gdf = pd.concat(var, axis=0).reset_index(drop=True)
        # convert the capacity from kW to MW
        gdf['capacity'] = gdf['capacity']/1e3
        # store the processed shapefile - translation, drop unwanted columns, add phase information
        gdf.to_file(
            f'{os.path.dirname(os.path.realpath(__file__))}/resource/EPE_power_plants_operation_and_construction_epsg4087.shp')

    return gdf


def add_foreign_substation():
    """Revise manually the line substation mapping with international transmission lines, update the 'ac_dc',
    and update 'trans_cap' if "hvdc" line

    Args:
        - l_s_pp: GeoDataFrame, dataset of line substation mapping, get from def join_start_end_in_line_with_substations_and_power_plants()
    Return:
        - l_s_pp: GeoDataFrame, revised dataset of line substation mapping with the international transmission lines

    Notes:
        - hard coded
        - the 'trans_cap' is not fully revised here
        - the geometry is from Google Maps (``WGS84<https://epsg.io/4326>``), either the exact location of the substation or the location of the city
        - the init_distance is the same as in the function join_start_end_in_line_with_substations_and_power_plants(init_distance)
    """
    # % collect information

    #  Brazil -Venezuela
    # source: p.11, http://www.uhebemquerer.com.br/wp-content/uploads/2020/10/EPE-Bruno-Silveira-Planejamento-do-sistema-de-transmisao-de-energia.pdf
    # source: epe, p. 130/p.127, https://www.epe.gov.br/sites-pt/publicacoes-dados-abertos/publicacoes/PublicacoesArquivos/publicacao-40/topico-70/Cap4_Texto.pdf
    # Note: from epe report, p. 130: "Although the capacity of this system is 200 MW, due to the reactive power deficit on the Venezuelan side, even after the commissioning of the static compensator in the 230 kV sector of SE Boa Vista, it is not possible for Brazil to import more than 150 MW."
    # Note: AC line, https://www.power-technology.com/marketdata/engenheiro-lechuga-equador-boa-vista-line-brazil/
    # Note: the string "Venezuela" is by testing, check the substation can be 'Boa Vista'
    ven_var = {'name': 'SE Macagua',
               'agent': 'None',
               'start_time': 'None',
               'voltage': 230,
               'geometry': Point(4.5306490, -61.1380975),
               # the actual location is Point(8.304186251846579, -62.6680881882042), but this line from EPE layer is not complete. so that the end point of the line is used
               'phase': 'operation',
               'state': 'VEN'  # iso_a3
               }

    # Brazil - Argentina
    # SE Paso de Los Libres
    # source: epe, p. 129, https://www.epe.gov.br/sites-pt/publicacoes-dados-abertos/publicacoes/PublicacoesArquivos/publicacao-40/topico-70/Cap4_Texto.pdf
    arg_var_1 = {'name': 'SE Paso de Los Libres',
                 'agent': 'None',
                 'start_time': 'None',
                 'voltage': 132,
                 'geometry': Point(-29.728507, -57.114561),  # the actual Point(-29.716667, -57.083333)
                 'phase': 'operation',
                 'state': 'ARG'  # iso_a3
                 }

    # SE Rincón de Santa Maria
    # total transfer capacity is 2200 MW, epe, p.129, https://www.epe.gov.br/sites-pt/publicacoes-dados-abertos/publicacoes/PublicacoesArquivos/publicacao-40/topico-70/Cap4_Texto.pdf
    # but for each is 1100 MW, see # https://library.e.abb.com/public/0d50a8fce76db2c9c1256fda003b4d43/THE%20GARABI%202000%20MW%20INTERCONNECTION.pdf
    # Note: HVDC line, back-to-back

    arg_var_2 = {'name': 'SE Rincón de Santa Maria',
                 'agent': 'ABB',
                 'start_time': 'None',
                 'voltage': 500,
                 'geometry': Point(-28.230610, -55.719075),
                 # the actual is Point(-27.48493901325796, -56.69641712729585),
                 'phase': 'operation',
                 'state': 'ARG'  # iso_a3
                 }

    # Brazil -Uruguay
    # SE Rivera
    # source: epe, p.129, https://www.epe.gov.br/sites-pt/publicacoes-dados-abertos/publicacoes/PublicacoesArquivos/publicacao-40/topico-70/Cap4_Texto.pdf
    # agent source: https://en.wikipedia.org/wiki/Rivera_HVDC_Back-to-back_station
    # back-to-back
    ury_var_1 = {'name': 'SE Rivera',
                 'agent': 'Areva and inaugurated',
                 'start_time': 'None',
                 'voltage': 230,
                 'geometry': Point(-30.940934, -55.560149),  # the actual is Point(-32.941389, -55.559444)
                 'phase': 'operation',
                 'state': 'URY'  # iso_a3
                 }

    ury_var_2 = {'name': 'SE Melo',
                 'agent': 'Areva',
                 'start_time': 'None',
                 'voltage': 500,
                 'geometry': Point(-32.366667, -54.183333),
                 'phase': 'operation',
                 'state': 'URY'  # iso_a3
                 }
    # % create geoDataFrame
    foreign_subs = gpd.GeoDataFrame(pd.DataFrame([ven_var, arg_var_1, arg_var_2, ury_var_1, ury_var_2]),
                                    geometry='geometry', crs='epsg:4326').to_crs(epsg=4087)
    # interchange (x, y) in the coordinates from Google Maps to (y, x) used in the map layer in EPE
    foreign_subs.geometry = foreign_subs.geometry.map(lambda point: shapely.ops.transform(lambda x, y: (y, x), point))

    return foreign_subs


def join_start_end_in_line_with_substations_and_power_plants(init_distance=1000):
    """Join the substation layer (sub) & power plants layer (pp) to the line layer according to the starting and ending
    points of the LineString.

    Notes:
        - the initial distance is assumed to be 1000m (1km), which assumes the circle of <=1km of the start and end point
           of the Line layer is recognised as the sub/pp
        - The distance of the start and end point with the sub/pp is indicated in column "distance"
        - 0.1° = 11.1 metre
        - The number of not found matched substations:  -> reason, add the map layer of pp
            init_distance     sub_0    sub_1
            1000(1km)         101      129
            10000(10km)        38      34
            50000(50km)        3       4 (hydropower plants)

           The number of not found matched sub/pp:
           init_distance     sub_0    sub_1
            1000(1km)         79      115
            50000(50km)        0      2
            85000(85km)        0      0 *
            * LT 230 kV Itapaci - Mineradora Maracá C1 (82km)
            * SECC LT 500 kV Sobradinho - Juazeiro III, C1, na SE UFV Futura (50km)

    Return:
        l_s_pp : gpd.GeoDataFrame, the lines with added information of substations for its start and end points.
            - the column "distance" is the mapping distance of the start and end point of the line with the substations

    """
    # %% read in the processed shapefile
    # the transmission lines
    line = read_shp_line_subs_from_epe('line')
    # only as .astype(int) doesn't work because of '230.0'
    # see https://stackoverflow.com/questions/1841565/valueerror-invalid-literal-for-int-with-base-10
    line['voltage'] = line['voltage'].astype(float).astype(int)
    # the substation
    s = read_shp_line_subs_from_epe('substation')
    # the power plants
    pp = read_shp_power_plants_from_epe()

    # %% add the federal state information for substations
    state = get_node_by_state()
    s = gpd.sjoin(s, state, how='left').rename(columns={'name_right': 'state',
                                                        'name_left': 'name'}).drop(labels=['index_right', 'x', 'y'],
                                                                                   axis=1)
    foreign_s = add_foreign_substation()
    # %% add the foreign substations
    s = pd.concat([s, foreign_s])
    # add the missing state information of "SE Margem Direita" (hard code)
    # source: https://www.itaipu.gov.py/en/energy/itaipu-transmission-systems
    s.loc[s['name'].str.contains('SE Margem Direita'), 'state'] = 'PRY'  # iso_a3

    # %% add the federal state information for power plants
    pp = gpd.sjoin_nearest(pp, state, how='left').rename(columns={'name': 'state'}).drop(labels=['index_right', 'x', 'y'],
                                                                                 axis=1)
    
    # create a new column "name", which is the same as in "s"
    pp['name'] = pp['type'] + '_' + pp['plant_name']

    # %% revise the shapefile in the raw data
    # convert MultiLineString to LineString  in geometry -> to get the start and end point of the LineString wo problem
    multi_linestring = line[~line.apply(lambda row: gpd.GeoSeries(row.geometry).geom_type,
                                        axis=1).squeeze().str.match('LineString')]
    multi_linestring.to_file('resource/DATA_ANALYSIS_unwanted MultiLineString_before revision_step1.shp')
    revise_multi_linestring = multi_linestring.copy()
    revise_multi_linestring.geometry = revise_multi_linestring.convex_hull.boundary
    revise_multi_linestring.to_file('resource/DATA_ANALYSIS_MultiLineString to LineString_after revision_step1.shp')
    line.loc[multi_linestring.index] = revise_multi_linestring

    # find out those LineString which is closed and open them but cut the string to half
    closed_string = line[line.geometry.boundary.is_empty]
    closed_string.to_file('resource/DATA_ANALYSIS_open the closed LineString_before revision_step2.shp')
    open_string = closed_string.copy()
    # note: Point(row.geometry.bounds[0:2]) is the starting point of the LineString
    open_string.geometry = closed_string.apply(lambda row: shapely.ops.split(row.geometry,
                                                                             Point(row.geometry.bounds[0:2])).geoms[0],
                                               axis=1).apply(lambda row: cut_linestring(row, row.length / 2)[0])
    open_string.to_file('resource/DATA_ANALYSIS_open the closed LineString_after revision_step2.shp')
    line.loc[open_string.index] = open_string

    # %% get the start and end point of the LineString in the shapefile of line
    l_s_pp = line.copy()  # to store the results in l_s_pp -> joint line and substation
    l_s_pp['sub_0'] = l_s_pp.apply(lambda row: Point(row.geometry.coords[0]), axis=1)
    l_s_pp['sub_1'] = l_s_pp.apply(lambda row: Point(row.geometry.coords[-1]), axis=1)

    # %% join the start and end point from line shapefile with the substation shapefile via geometry
    # TODO: the replacement of the geometry in start and end substation (column "sub_0", "sub_1");
    #  Now - no replacement; expected - replace the geometry of substations if matches
    to_join = pd.concat([s[['name', 'geometry', 'state']], pp[['name', 'geometry', 'state']]])
    for sub in ['sub_0', 'sub_1']:
        # Note: this value must be greater than 0, in the same units as the map layer, for epsg:4087, in meters.
        joint_gdf = gpd.GeoDataFrame(l_s_pp[sub].rename('geometry'))
        joint_gdf = gpd.sjoin_nearest(left_df=joint_gdf, right_df=to_join, how='left', max_distance=init_distance)
        joint_gdf['distance'] = init_distance
        not_found_sub = joint_gdf[joint_gdf['index_right'].isna()]
        # 85km, the upper limit is determined experimentally - until the not_found_sub is empty
        for distance in list(float_range(0, 85000, '1000')):
            if (not not_found_sub.empty) and (distance not in [0, init_distance]):  # Todo: this is too ugly
                second_joint_gdf = gpd.GeoDataFrame(not_found_sub.geometry).sjoin_nearest(right=to_join, how='left',
                                                                                          max_distance=distance)
                second_joint_gdf['distance'] = distance
                second_found_gdf = second_joint_gdf[second_joint_gdf['index_right'].notna()]
                not_found_sub = second_joint_gdf[second_joint_gdf['index_right'].isna()]

                joint_gdf.loc[second_found_gdf.index] = second_found_gdf
        # add the information of substations to the transmission line
        joint_gdf = joint_gdf.drop(labels=['geometry', 'index_right'], axis=1)
        joint_gdf.columns = joint_gdf.columns + '_' + sub
        l_s_pp = l_s_pp.merge(joint_gdf, left_index=True, right_index=True)

        # TODO: need improvement -> the phase of the matched substation is not the same as the line
        # pd.DataFrame(l.filter(like='phase'))[~pd.DataFrame(l.filter(like='phase')).nunique(axis = 1).eq(1)]
        # TODO: add the hist plot of the distance
        # l_s_pp.filter(like='distance').plot.hist(alpha=0.5)
    # TODO: save the processed shape file, so far geopandas does not allows to store multiple geometry columns
    # follow: https://github.com/geopandas/geopandas/issues/1490
    # try:
    #     l_s_pp = gpd.read_file('resource/PROCESSED_line_substation_operation_and construction.shp')
    # except (FileNotFoundError, fiona.errors.DriverError) as e:
    # l_s_pp.to_file('resource/PROCESSED_line_substation_operation_and construction.shp')
    pd.DataFrame(l_s_pp).to_csv('resource/EPE_processed_line_sub_pp_operation_and construction.csv')
    return l_s_pp


# % revise the line substation mapping after pre-processing

def update_sub(mask_string, variable_dict, l_s_pp):
    """Update the values in the line substation dataset

    Args:
        - mask_string: str, use to match the column of "name" (name of transmission line) in the l_s_pp dataset
        - variable_dict: dict, the column and the value pair used to revise
        - l_s_pp: gpd.GeoDataFrame, dataset of line substation mapping, get from def join_start_end_in_line_with_substations_and_power_plants()
    """
    mask = l_s_pp[l_s_pp.name.str.contains(mask_string)]
    to_replace = pd.DataFrame.from_dict(variable_dict, orient='index').T
    l_s_pp.loc[mask.index, to_replace.columns] = to_replace.values


def revise_trans_cap_for_hvdc_lines(l_s_pp):
    """Revise manually the transfer capacity (column 'trans_cap') for HVDC lines (column 'ac_dc' with value 'hvdc')

    Notes:
        - this is hard coded
        - the mask_string in the update_sub() is hard coded by experimental filtering
        - the 'trans_cap' is not fully revised here, further steps needed
    Args:
        - l_s_pp: gpd.GeoDataFrame, line substation mapping obtained from def join_start_end_in_line_with_substations_and_power_plants()

    Return:
        - l_s_pp: gpd.GeoDataFrame, revised dataset of line substation mapping with the HVDC lines
    """

    # Itaipu 1 and Itaipu 2
    # see https://en.wikipedia.org/wiki/HVDC_Itaipu
    # also see https://www.itaipu.gov.py/en/energy/itaipu-transmission-systems
    # also see https://www.researchgate.net/publication/331042253_Analysis_of_the_Incidence_of_Direct_Lightning_over_a_HVDC_Transmission_Line_through_EFD_Model
    # mask_string = 'Foz do Iguaçu - Ibiúna', because start = 'Foz do Iguaçu', end = 'Ibiúna'
    # trans_cap = 3150[MW]
    # source: divided by two since each bipolar transmission consists two line but each bipolar deliver 3150 MW
    # https: // web.archive.org / web / 20051115122539 / http: // www.transmission.bpa.gov / cigresc14 / Compendium / ITAIPU.htm
    var = {'ac_dc': 'hvdc',
           'trans_cap': 3150}
    update_sub(mask_string='Foz do Iguaçu - Ibiúna', variable_dict=var, l_s_pp=l_s_pp)

    # Rio Madeira HVDC system
    # see https://search.abb.com/library/Download.aspx?DocumentID=9AKK105713A1117&LanguageCode=en&DocumentPartId=&Action=Launch
    # also see https://en.wikipedia.org/wiki/Rio_Madeira_HVDC_system
    # drop the duplicated lines
    # Note: there are duplicated lines: should be TWO, but Four in raw shapefile
    duplicated = l_s_pp[l_s_pp.name.str.contains('Porto Velho - Araraquara')].name.drop_duplicates()
    l_s_pp = l_s_pp.drop(duplicated.index)
    # see https://en.wikipedia.org/wiki/Rio_Madeira_HVDC_system
    # [1] two bipolar ±600 kV DC  transmission lines with a capacity of 3150 MW
    # mask_string = 'Porto Velho - Araraquara', because start = 'Porto Velho', end = 'Araraquara'
    # trans_cap = 3150 [MW]
    var = {'ac_dc': 'hvdc',
           'trans_cap': 3150}
    update_sub(mask_string='Porto Velho - Araraquara', variable_dict=var, l_s_pp=l_s_pp)
    # [2] two bipoles 400 MW back-to-back converters to supply power to the local 230 kV AC system
    var = {'ac_dc': 'hvdc',
           'trans_cap': 400}
    update_sub(mask_string='Porto Velho - Porto Velho', variable_dict=var, l_s_pp=l_s_pp)

    # Xingu-Estreito HVDC transmission line
    # drop the duplicated lines
    # Note: there are duplicated lines: should be TWO, but Four in raw shapefile
    duplicated = l_s_pp[l_s_pp.name.str.contains('Xingu - Estreito')].name.drop_duplicates()
    l_s_pp = l_s_pp.drop(duplicated.index)
    # see https://en.wikipedia.org/wiki/Xingu-Estreito_HVDC_transmission_line
    # also see https://energia.gob.cl/sites/default/files/mini-sitio/07_stategrid_paulo_esmeraldo.pdf
    # mask_string = 'Xingu - Estreito', because start = 'Xingu', end = 'Estreito'
    # trans_cap = 4000 [MW]
    var = {'ac_dc': 'hvdc',
           'trans_cap': 4000}
    update_sub(mask_string='Xingu - Estreito', variable_dict=var, l_s_pp=l_s_pp)

    # Xingu-Terminal Rio HVDC transmission line
    # see https://en.wikipedia.org/wiki/Xingu-Estreito_HVDC_transmission_line
    # also see nom_tipolinha = LINHA DC, https://dados.ons.org.br/dataset/linha-transmissao/resource/79cb787b-4c34-447d-8c65-2f1ce5b5d031
    # also see p.3, https://energia.gob.cl/sites/default/files/mini-sitio/07_stategrid_paulo_esmeraldo.pdf
    var = {'ac_dc': 'hvdc',
           'trans_cap': 4000}
    update_sub(mask_string='Xingu - Terminal Rio', variable_dict=var, l_s_pp=l_s_pp)

    # Graça Aranha - Silvânia HVDC transmission line
    # see p.3, https://energia.gob.cl/sites/default/files/mini-sitio/07_stategrid_paulo_esmeraldo.pdf
    var = {'ac_dc': 'hvdc',
           'trans_cap': 4000}
    update_sub(mask_string='Graça Aranha - Silvânia', variable_dict=var, l_s_pp=l_s_pp)

    # SE Rincón de Santa Maria
    # total transfer capacity is 2200 MW, epe, p.129, https://www.epe.gov.br/sites-pt/publicacoes-dados-abertos/publicacoes/PublicacoesArquivos/publicacao-40/topico-70/Cap4_Texto.pdf
    # but for each is 1100 MW, see # https://library.e.abb.com/public/0d50a8fce76db2c9c1256fda003b4d43/THE%20GARABI%202000%20MW%20INTERCONNECTION.pdf
    # Note: HVDC line, back-to-back

    var = {'ac_dc': 'hvdc',
           'trans_cap': 1100}
    update_sub(mask_string='Rincón de Santa Maria', variable_dict=var, l_s_pp=l_s_pp)

    # Brazil -Uruguay
    # SE Rivera
    # source: epe, p.129, https://www.epe.gov.br/sites-pt/publicacoes-dados-abertos/publicacoes/PublicacoesArquivos/publicacao-40/topico-70/Cap4_Texto.pdf
    # agent source: https://en.wikipedia.org/wiki/Rivera_HVDC_Back-to-back_station
    # back-to-back
    var = {'ac_dc': 'hvdc',
           'trans_cap': 70}
    update_sub(mask_string='Rivera', variable_dict=var, l_s_pp=l_s_pp)

    var = {'ac_dc': 'hvdc',
           'trans_cap': 500}
    update_sub(mask_string='Melo', variable_dict=var, l_s_pp=l_s_pp)

    return l_s_pp


def add_addition_line_info(l_s_pp):
    """Add/revise additional information of the transmission line

    Note:
        - add additional information of circuit and line type(AC or HVDC)
        - revise the transfer capacity (trans_cap) for the HVDC lines
        - the column "trans_cap" of the output: HVDC is reliable, AC is NOT reliable

    Returns:
        l_s_pp: gpd.GeoDataFrame, revised dataset of line substation mapping with a set of revision operations
    """

    # add information of 'ac_dc' status, default line type is 'AC'
    l_s_pp['ac_dc'] = 'ac'
    # add information of 'trans_cap', default 'None'
    l_s_pp['trans_cap'] = None
    # add the information of circuit
    # default circuit = 1 -> the data provides the info for each parallel lines, e.g., C1 - C4 in the line name
    # "CD" is double circuit
    # p. 114, https://www.epe.gov.br/sites-pt/publicacoes-dados-abertos/publicacoes/PublicacoesArquivos/publicacao-276/topico-525/EPE-DEE-RE-025-2020-rev0+SMA%20-%20Estudo%20para%20Controle%20de%20Tens%C3%A3o%20e%20Suprimento%20ao%20Extremo%20Sul%20da%20Bahia.pdf
    l_s_pp['circuit'] = 1
    l_s_pp.loc[l_s_pp.name.str.contains('CD'), 'circuit'] = 2
    # revision
    l_s_pp = revise_trans_cap_for_hvdc_lines(l_s_pp)

    return l_s_pp


def plot_line_substation_for_analysis(l_s_pp, name_tag_in_filename=None):
    """Plot the transmission lines and substations

    Args:
        - l_s_pp: gpd.GeoDataFrame, the data to plot
        - name_tag_in_filename: str, default None, to distinguish the saved plots
    """
    plt.close('all')
    # update the string in the file name for the stored plot
    string_in_filename = name_tag_in_filename if name_tag_in_filename else ''

    # by voltage
    # base map of the state
    fig, ax = plot_setting(figure_size=(4, 4))
    get_node_by_state().plot(ax=ax, color='white', edgecolor='grey', linewidth=0.2)
    # plot the substation
    l_s_pp['sub_0'].plot(ax=ax, color='black', markersize=1, linewidth=0)
    l_s_pp['sub_1'].plot(ax=ax, color='black', markersize=1, linewidth=0)
    # plot the line at different voltage
    # note: to put the legend outside the plot, set bbox_to_anchor, BUT after 'title'
    # note: set categorical=True, because voltage is numeric, so legend_kwds cannot be set.
    l_s_pp.plot(ax=ax, column='voltage', cmap='Paired', linewidth=0.5,
                legend=True, categorical=True, legend_kwds={'title': 'Voltage', 'bbox_to_anchor': (0.8, 1),
                                                            'frameon': False})
    # decoration
    ax.set_aspect('equal')
    ax.axis('off')
    # store the plot
    fig.savefig(f'resource/DATA_ANALYSIS_{string_in_filename}_map_of_line_substation_by_voltage.png',
                dpi=300, bbox_inches='tight')

    # by phase
    # base map of the state
    fig, ax = plot_setting(figure_size=(4, 4))
    get_node_by_state().plot(ax=ax, color='white', edgecolor='grey', linewidth=0.2)
    # plot the substation
    gpd.GeoDataFrame(l_s_pp[['sub_0', 'phase']], geometry='sub_0'
                     ).plot(ax=ax, column='phase', cmap='Paired', linewidth=0, markersize=1)
    gpd.GeoDataFrame(l_s_pp[['sub_1', 'phase']], geometry='sub_1'
                     ).plot(ax=ax, column='phase', cmap='Paired', linewidth=0, markersize=1)
    # plot the line at different phase
    # note: to put the legend outside of the plot, set bbox_to_anchor, BUT after 'title'
    l_s_pp.plot(ax=ax, column='phase', cmap='Paired', linewidth=0.5,
                legend=True, legend_kwds={'title': 'Phase', 'bbox_to_anchor': (0.8, 1), 'frameon': False})
    # decoration
    ax.set_aspect('equal')
    ax.axis('off')
    # store the plot
    fig.savefig(f'resource/DATA_ANALYSIS_{string_in_filename}_map_of_line_substation_by_phase.png',
                dpi=300, bbox_inches='tight')

    # by ac dc line type
    # base map of the state
    fig, ax = plot_setting(figure_size=(4, 4))
    get_node_by_state().plot(ax=ax, color='white', edgecolor='grey', linewidth=0.2)
    # plot the substation
    l_s_pp['sub_0'].plot(ax=ax, color='black', markersize=1, linewidth=0)
    l_s_pp['sub_1'].plot(ax=ax, color='black', markersize=1, linewidth=0)
    # plot the line at different line type
    # note: to put the legend outside the plot, set bbox_to_anchor, BUT after 'title'
    l_s_pp.plot(ax=ax, column='ac_dc', cmap='Paired', linewidth=0.5,
                legend=True, legend_kwds={'title': 'Links type', 'bbox_to_anchor': (0.8, 1), 'frameon': False})
    # decoration
    ax.set_aspect('equal')
    ax.axis('off')
    # store the plot
    fig.savefig(f'resource/DATA_ANALYSIS_{string_in_filename}_map_of_line_substation_by_ac-dc.png',
                dpi=300, bbox_inches='tight')


# % main function
def get_ac_hvdc_network(only_operation=False):
    """Calculate the transfer capacity for the line that across the federal states (defined node in the PyPSA-Brazil)

    Note:
        - Use transport model method
        - calculate the transfer capacity based on the existing + construction grid topology
        - the obtained results is the parameter for ``Link`` Component in pypsa
    """
    # TODO: build a power grid network and aggregation, similar to Gridkit, https://github.com/bdw/GridKit/tree/v1.0
    # clustering methodology:
    # - source 1: eGo, https://github.com/openego/eTraGo/blob/630a2189a2003eab4c2e02fb17d4331df6a76ebd/etrago/cluster/disaggregation.py#L14
    # - source 2: pypsa, https://github.com/PyPSA/PyPSA/blob/dcf94567b5221e2928cd6e2986ed342f227f0bd6/pypsa/networkclustering.py#L148

    # %% add the belonging federal state information of start and end of the transmission line based on the EPE map of substations (sub) and power plants (pp)
    line_substation_powerplant = join_start_end_in_line_with_substations_and_power_plants()

    # %% read in the revised layer of transmission line with mapped start and end substations
    line_substation_powerplant = add_addition_line_info(l_s_pp=line_substation_powerplant)
    # TODO: more manual revision for the lines with the same starting and ending substation information
    # test = line_substation[['name_sub_0', 'name_sub_1']]
    # test[test.apply(lambda x: min(x) == max(x), 1)]
    # line_substation.loc[test[test.apply(lambda x: min(x) == max(x), 1)].index]

    # select the lines across federal states
    across_state = line_substation_powerplant[
        line_substation_powerplant['state_sub_0'] != line_substation_powerplant['state_sub_1']]

    # %% save and plot the lines before and after selection
    plot_line_substation_for_analysis(line_substation_powerplant, name_tag_in_filename='full_network')
    plot_line_substation_for_analysis(across_state, name_tag_in_filename='selected_network')
    # TODO: save the processed shape file, so far geopandas does not allows to store multiple geometry columns
    # save the file of revised full network
    line_substation_powerplant[['sub_0', 'sub_1']] = line_substation_powerplant[['sub_0', 'sub_1']].astype(str)
    line_substation_powerplant.to_file('resource/PROCESSED_full_network.shp')
    # save the file of revised and selected network across the state
    across_state[['sub_0', 'sub_1']] = across_state[['sub_0', 'sub_1']].astype(str)
    across_state.to_file('resource/PROCESSED_across_federal_state_network.shp')

    # %% prepare the model input
    create_folder('results')
    model_lines = across_state[
        ['name', 'voltage', 'length', 'ac_dc', 'circuit', 'trans_cap', 'phase', 'geometry',
         'sub_0', 'state_sub_0', 'sub_1', 'state_sub_1',
         ]].rename(columns={'name': 'name',
                            'voltage': 'voltage',
                            'length': 'length',
                            'state_sub_0': 'node0',
                            'state_sub_1': 'node1',
                            'ac_dc': 'carrier',
                            'circuit': 'num_parallel',
                            'trans_cap': 'transfer_capacity'})

    # %% ac lines
    if only_operation:
        model_lines = model_lines.query("phase=='operation'")
        file_tag = 'only_operation'
    else:
        file_tag = 'operation_and_planned'
    ac = model_lines.loc[model_lines['carrier'] == 'ac'].copy()

    # simplify network to 380 kV -> convert the voltage to 380 kV
    ac['voltage_s'] = 380
    ac.loc[ac['voltage'] != 380, 'num_parallel'] *= (ac.loc[ac['voltage'] != 380, 'voltage'] / 380.) ** 2

    # get the electrical parameters for the give line type from pypsa component, no information about the line type for each line
    # assumption took from epe by similarity with the available line type defined in pypsa Line Type component
    # Table 16-2 - Electrical Parameters of Transmission Lines - Winning Alternative, https://www.epe.gov.br/sites-pt/publicacoes-dados-abertos/publicacoes/PublicacoesArquivos/publicacao-276/topico-525/EPE-DEE-RE-025-2020-rev0+SMA%20-%20Estudo%20para%20Controle%20de%20Tens%C3%A3o%20e%20Suprimento%20ao%20Extremo%20Sul%20da%20Bahia.pdf
    ac['type'] = '490-AL1/64-ST1A 380.0'
    network = pypsa.Network()
    line_type = network.line_types.loc['490-AL1/64-ST1A 380.0']
    # formulation: # p653, D.Oeding · B. R. Oswald: Elektrische Kraftwerke und Netze, 7. Auflage, Springer 2011
    ac['transfer_capacity'] = np.sqrt(3) * line_type.i_nom * ac['voltage_s'] * ac['num_parallel']
    ac['efficiency'] = 1 - (3 * line_type.r_per_length * line_type.i_nom ** 2 * 4) / ac['transfer_capacity']

    # aggregation - aggregate the ac line transmitted between states by the same start and end point, inspirited by
    # https://github.com/PyPSA/PyPSA/blob/dcf94567b5221e2928cd6e2986ed342f227f0bd6/pypsa/networkclustering.py#L167
    ac_links = ac[['name', 'node0', 'node1', 'carrier', 'efficiency', 'transfer_capacity', 'length']].copy()

    # sort the pair of "node0" and "node1" to avoid the duplication due to reverse order in the aggregation
    ac_links[['node0', 'node1']] = [sorted([a, b]) for a, b in zip(ac_links.node0, ac_links.node1)]
    # aggregation
    ac_links = ac_links.groupby(['node0', 'node1']
                                ).agg(
        {'transfer_capacity': 'sum', 'efficiency': 'mean', 'name': '_'.join, 'length': 'mean'}
    ).reset_index()
    ac_links['carrier'] = 'ac'

    # %% hvdc lines
    hvdc_links = model_lines.loc[model_lines['carrier'] == 'hvdc'][
        ['name', 'node0', 'node1', 'carrier', 'transfer_capacity', 'length']].copy()
    hvdc_links['efficiency'] = 1
    hvdc_links[['node0', 'node1']] = [sorted([a, b]) for a, b in zip(hvdc_links.node0, hvdc_links.node1)]
    hvdc_links = hvdc_links.groupby(['node0', 'node1']
                                    ).agg(
        {'transfer_capacity': 'sum', 'efficiency': 'mean', 'name': '_'.join, 'length': 'mean'}
    ).reset_index()
    hvdc_links['carrier'] = 'hvdc'

    # %% export the results
    transmission = pd.concat([ac_links, hvdc_links]).reset_index(drop=True)
    transmission.to_csv(f'results/EPEWebmap_equivalent_grid_aggregate_by_state_{file_tag}.csv')


def plot_model_input_ac_hvdc(line_volume=False):
    # % read in maps/ data
    # map of South America - country level
    sm_map = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')
                           ).query("continent == 'South America' & name !='Brazil'"
                                   ).to_crs('epsg:4087').rename(columns={'iso_a3': 'name', 'name': 'name_full'})
    # map of Brazil - federal state level
    state_map = get_node_by_state()
    # data of lines
    line = pd.concat([pd.read_csv(f, index_col=0) for f in glob.glob('results/*.csv')])
    line['line_volume'] = line['transfer_capacity'] * line['length']

    # % plot setting
    fig, ax = plot_setting(figure_size=(4, 4))
    cfg = read_config()
    tech_color = cfg["tech_color"]
    text_color = '#000000'  # '#343a40'
    edge_color = '#343a40'
    nice_name = cfg["nice_name"]

    # % plotting country boundary in South America
    sm_map.apply(lambda x: ax.annotate(text=x['name'], xy=x.geometry.centroid.coords[0], fontsize=3,
                                       ha='center', color=text_color, alpha=0.5), axis=1)
    # sm_map.apply(lambda x: ax.annotate(text=x['name_full'],
    #                                    xy=(x.geometry.centroid.coords[0][0],
    #                                        x.geometry.centroid.coords[0][1] - 55000),
    #                                    ha='center', color=text_color,  # grey
    #                                    fontsize=2, alpha=0.5), axis=1)
    sm_map.plot(ax=ax, color='white', edgecolor=edge_color, linewidth=0.1, alpha=0.5)
    # plot federal state boundary in Brazil
    state_map.apply(lambda x: ax.annotate(text=x['name'], xy=x.geometry.centroid.coords[0], fontsize=3,
                                          ha='center', color=text_color), axis=1)
    # state_map.apply(lambda x: ax.annotate(text=x['state_full'],
    #                                       xy=(x.geometry.centroid.coords[0][0],
    #                                           x.geometry.centroid.coords[0][1] - 55000),
    #                                       ha='center', color=text_color,  # grey
    #                                       fontsize=2, zorder=5), axis=1)
    state_map.plot(ax=ax, color='#f5f5f5', edgecolor=edge_color, linewidth=0.1)

    # plot the lines
    # create gpd.GeoDataframe for the lines
    geometry = line[['node0', 'node1']].apply(
        lambda row: pd.concat([state_map, sm_map]).set_index('name').centroid.loc[row].unary_union.convex_hull, axis=1)
    line = gpd.GeoDataFrame(line, geometry=geometry)
    if line_volume:
        line_size_factor = 2e6  # manually determined
        line_width = line['line_volume'] / line_size_factor
    else:
        line_size_factor = 5e3
        line_width = line['transfer_capacity'] / line_size_factor

    line.plot(ax=ax, color=line['carrier'].replace(tech_color), alpha=0.8,
              linewidth=line_width)

    # line.plot(ax=ax, column='carrier', color=line['carrier'].replace(tech_color), alpha=line_width)

    # % styling
    # delete the axis
    ax.set_aspect('equal')
    ax.axis('off')
    # display the shown information on the left bar
    if line_volume:
        plot_info, label_unit, s_list = 'Line volume', 'GWkm', (10000, 1000)
    else:
        plot_info, label_unit, s_list = 'Line transfer capacity', 'GW', (10, 1)

    ax.text(-0.05, 0.5, plot_info, transform=ax.transAxes,
            fontsize=6, color=text_color, alpha=0.5,
            ha='center', va='center', rotation=90)

    # legend of line width
    handles = []
    labels = []
    for s in s_list:
        handles.append(plt.Line2D([0], [0.5],
                                  color=text_color,
                                  linewidth=s * 1e3 / line_size_factor))
        # the whitespace here is to put legend title at left
        labels.append(f"{s}                    ")

    l1 = ax.legend(handles, labels,
                   fontsize=4, title_fontsize=4.5,
                   loc="lower left",
                   frameon=False,
                   bbox_to_anchor=(0, 0.4),
                   labelspacing=0.8,
                   title=f'{label_unit}                     ',
                   columnspacing=0.5)

    ax.add_artist(l1)

    # legend of technology
    techs = line['carrier'].unique()
    handles = []
    labels = []
    for t in techs:
        handles.append(plt.Line2D([0], [0], color=tech_color[t],
                                  marker='o', markersize=3, markeredgewidth=0, linewidth=0))
        labels.append(nice_name.get(t))
    l2 = ax.legend(handles, labels,
                   fontsize=4, title_fontsize=4.5,
                   loc="lower left",
                   frameon=False,
                   bbox_to_anchor=(0.7, 0.8),
                   handletextpad=0., columnspacing=0.5)
    ax.add_artist(l2)
    plt.legend(prop={'family': 'monospace'})

    # saving
    fig.savefig(f'resource/model_input_grid_{plot_info}.png', dpi=300, bbox_inches='tight')
    plt.close('all')


def processed_line_statistics():
    # insights of the line before aggregation
    fig, ax = plot_setting(figure_size=(8, 4))
    df = pd.read_csv('resource/EPE_processed_line_sub_pp_operation_and construction.csv',
                     index_col=0).filter(regex='distance', axis=1)
    df.plot(ax=ax, kind='hist', alpha=0.5, edgecolor='black', rwidth=0.9, bins=10, color=['blue', 'brown'])
    fig.savefig(f'resource/EPE_statistics_processed_line_sub_pp_operation_and construction.png', dpi=300,
                bbox_inches='tight')
    plt.close('all')
    df.describe(percentiles=[.5, .90, .95]).to_csv(
        'resource/EPE_statistics_processed_line_sub_pp_operation_and construction.csv')


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.info("Prepare AC and HVDC transmission lines (aggregated to inter-state transmission) with NTC assumptions.")
    get_ac_hvdc_network(only_operation=False)
    get_ac_hvdc_network(only_operation=True)
    processed_line_statistics()
    plot_model_input_ac_hvdc(line_volume=False)
    plot_model_input_ac_hvdc(line_volume=True)
