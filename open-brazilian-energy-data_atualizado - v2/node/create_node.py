# -*- coding: utf-8 -*-
__version__ = '0.1.0'
__maintainer__ = 'Ying Deng 21.07.2022'
__authors__ = 'Ying Deng'
__credits__ = 'Ying Deng'
__email__ = 'ying.deng@dlr.de'
__date__ = '18.06.2021'
__status__ = 'dev'  # options are: dev, test, prod
__copyright__ = 'DLR'

import os
import sys

import fiona
import geopandas as gpd
import matplotlib.pyplot as plt

sys.path.append('../')
from utility_for_data import create_folder
from utility_for_data import plot_setting
from utility_for_data import translate_pt_to_en_df


def get_node_by_state():
    """Consider each federal state of Brazil as a node in PyPSA-Brazil.
    The centroid of the polygon for states is the position of the node.

    Important:
    The projection of the maps is ``EPSG:4674 (SIRGAS 2000) <https://epsg.io/4674>``, unit- degree, geographic CRS.
    To use calculations, e.g, "centroid", "sjoin",  the map needs to be reprojected, ``.to_crs('epsg:4087')``

    EPSG:4326 is also a geographic CRS, unit - degree,
    # issue: UserWarning: Geometry is in a geographic CRS. Results from 'centroid' are likely incorrect. Use 'GeoSeries.to_crs()' to re-project geometries to a projected CRS before this operation.
    # previous: to_crs('epsg:4326'), now: set .to_crs('epsg:4087') solved: see https://gis.stackexchange.com/questions/372564/userwarning-when-trying-to-get-centroid-from-a-polygon-geopandas

    """
    try:
        state_node = gpd.read_file(f'{os.path.dirname(os.path.realpath(__file__))}/results/node_epsg4087.shp')
    except (FileNotFoundError, fiona.errors.DriverError) as e:
        create_folder('results')
        state_node = gpd.read_file(f'{os.path.dirname(os.path.realpath(__file__))}/raw/BR_UF_2020.shp'
                                   ).set_crs('epsg:4674').to_crs(crs=4087, epsg='epsg')
        state_node = translate_pt_to_en_df(state_node, 'column_state_shp')[['state', 'state_name', 'geometry']]
        # use the centroid of the polygon to represent the position of the node (attributes: "x", "y" in REFuelsESM)
        state_node["x"] = state_node.centroid.x
        state_node["y"] = state_node.centroid.y

        # rename the state with "name" since "name"
        state_node = state_node.rename(columns={'state': 'name'})

        state_node.to_file(f'{os.path.dirname(os.path.realpath(__file__))}/results/node_epsg4087.shp')

    return state_node


def plot_node():
    # % plot setting
    create_folder('resource')
    fig, ax = plot_setting(figure_size=(5, 5))

    # % plotting
    state = get_node_by_state()

    # abbreviation name
    state.apply(lambda x: ax.annotate(text=x['name'], xy=x.geometry.centroid.coords[0], fontsize=5, ha='center'),
                axis=1)
    # # full name
    # state.apply(lambda x: ax.annotate(text=x['state_full'],
    #                                   xy=(x.geometry.centroid.coords[0][0], x.geometry.centroid.coords[0][1] - 55000),
    #                                   ha='center',
    #                                   color='#343a40',  # grey
    #                                   fontsize=2), axis=1)
    # plot the polygon
    state.plot(ax=ax, cmap='Blues', alpha=0.5, edgecolor='grey', linewidth=0.2)
    # styling
    ax.set_aspect('equal')
    ax.axis('off')
    # saving
    fig.savefig(f'resource/node_definition.png',
                dpi=300, bbox_inches='tight')
    plt.close('all')


# if '__name__' == '__main__':
get_node_by_state()
plot_node()
