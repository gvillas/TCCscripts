# -*- coding:utf-8 -*-

__version__ = '0.1.0'
__maintainer__ = 'Ying Deng 22.03.2022'
__authors__ = 'Ying Deng'
__credits__ = 'Ying Deng'
__email__ = 'Ying.Deng@dlr.de'
__date__ = '25.09.2020'
__status__ = 'dev'  # options are: dev, test, prod
__copyright__ = 'DLR'

"""A module-level docstring
Get the installed capacity for each type of power plants

Assumptions:
- the installed capacity will remind the same as base year (e.g., 2018)
- the power plants with the phase == 'operation' (in PT: 'Fase' == 'Operação') is considered as lower bound
"""

import os
import sys

import geopandas as gpd  # To create GeoDataFrame
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim
from shapely import wkt  # Well Known Text (WKT)
from shapely.geometry import Point  # Shapely for converting latitude/longitude to geometry

sys.path.append('../')
from utility_for_data import (read_config, translate_pt_to_en_df, plot_setting, create_folder,
                              sin_state_region_mapping_in_series)
from node.create_node import get_node_by_state


# % processing helper functions
def get_longitude_latitude(city):
    """By giving the name of the city, get the corresponding coordinates.

    Args:
        - city: str, the name of the city

    Returns:
        tuple, the coordinates

    Notes:
        Error exists: requests.exceptions.SSLError: HTTPSConnectionPool(host='nominatim.openstreetmap.org', port=443)
    https://github.com/geopy/geopy/issues/321

    """
    geo_locator = Nominatim(timeout=10, user_agent="pypsa-br")
    # passing country_codes = 'BR' to limit the search in Brazil
    # country code, see: https://en.wikipedia.org/wiki/List_of_ISO_3166_country_codes

    # Create a geopy rate limiter:
    geocode_with_delay = RateLimiter(geo_locator.geocode)
    location = geocode_with_delay(city, country_codes='BR')
    return location.longitude, location.latitude

def extract_coordinates_decimal(df):
    """Extract the coordinates from ANEEL dataset, considering that the latitude and longitude data
    were provided in decimals"""
    df = pd.to_numeric(df, errors='coerce')
    return df

def extract_coordinates(df):
    """Extract the coordinates for the ANEEL dataset"""
    # regex = r"(\d{1,2})°(\d{1,2})'(\d{1,2})\.(\d{1,2,3})\"(N|S|E|W)"
    df = df.str.replace(' ', '').str.replace(',', '.'
                                             ).str.replace("“", "\""
                                                           ).str.replace("’", "'"
                                                                         ).str.replace("”", "\""
                                                                                       ).str.replace("''", "\""
                                                                                                     ).str.replace('º',
                                                                                                                   '°')
    regex = r"(-?\d+)°(\d+)'(\d+)\.(\d+)['?|\"?](N|S|E|W)?"
    e = df.str.extract(regex, expand=True)
    coord = (e[0].astype(float)
             + (e[1].astype(float)
                + (e[2].astype(float) + e[3].astype(float) / 100) / 60.
                ) / 60.
             ) * e[4].map({'N': +1., 'S': -1., 'E': +1., 'W': -1.})
    regex_2 = r"(-?\d+)°(\d+)'(\d+)['?|\"?](N|S|E|W)?"
    e_1 = df.loc[coord[coord.isna()].index].str.extract(regex_2, expand=True)
    coord_1 = (e_1[0].astype(float)
               + (e_1[1].astype(float) + e_1[2].astype(float) / 60.) / 60.
               ) * e_1[3].map({'N': +1., 'S': -1., 'E': +1., 'W': -1.})
    coord[coord_1[coord_1.notna()].index] = coord_1[coord_1.notna()]
    # for those the geolocation is missing, set as nan
    coord[coord.isna()] = np.nan
    return coord


def refine_plant_type(raw):
    """Refine the plant type
    Args:
        - raw: pd.DataFrame, read-in data
    Return:
        - output: dict

    Notes:
        identifier: default identifier for filter each power plants classification of plants type in the data.
                    See README and tools/trans_PT_to_EN.yaml and README.docx
                    key is the value of column 'type', 'thermal' will be further classified with 'oil', 'gas',
                    'coal', 'biomass'.
                     - keys: first-layer key is the name of the 'type', which is the generation power plants type
                     - values: empty, '', if the power plants can be identified by the 'type'
                               non-empty, dict type, contains further identifier for the power plants
                                        - key: further identifier which is column name in data, e.g, fuel_type
                                        - value: value of the identifier
    """
    identifier = {'mini_hydro': ['mini_hydro', 'wave'],
                  'small_hydro': '',
                  'hydro': '',
                  'on_wind': '',
                  'solar_pv': '',
                  'nuclear': '',
                  'thermal': {'oil': {'fuel_type': 'Fossil',
                                      'fuel_source': ['Oil', 'Other fossil energy']},
                              'gas': {'fuel_type': 'Fossil',
                                      'fuel_source': 'Natural gas'},
                              'coal': {'fuel_type': 'Fossil',
                                       'fuel_source': 'Mineral coal'},
                              'biomass': {'fuel_type': 'Biomass'}}}
    for name, value in identifier.items():
        if not value:  # empty identifier; power plants can be determined by column 'type'
            pass
        elif isinstance(value, list):
            select = raw[raw["type"].isin(value)]
            raw.loc[select.index, "type"] = name
        elif isinstance(value, dict):
            for second_name, second_val in value.items():
                select = raw[raw["type"] == name]
                for k, v in second_val.items():
                    if isinstance(v, str):
                        select = select[select[k].isin([v])]
                    if isinstance(v, list):
                        select = select[select[k].isin(v)]
                raw.loc[select.index, 'type'] = second_name
                # output[second_name] = select
        else:
            assert f"identifier of {name} is not valid"
    return raw


# % data processing
def read_aneel_raw():
    """Read raw file from ANEEL, rename the column name, fulfill the missing coordinates for each plant.
    The types of thermal power plants are subdivided into gas-fired thermal, oil-fired thermal, coal-fired thermal,
    and biomass-fired thermal.

    Important:
        - source: 'ANEEL SIGA <https://app.powerbi.com/view?r=eyJrIjoiNjc4OGYyYjQtYWM2ZC00YjllLWJlYmEtYzdkNTQ1MTc1NjM2IiwidCI6IjQwZDZmOWI4LWVjYTctNDZhMi05MmQ0LWVhNGU5YzAxNzBlMSIsImMiOjR9>'
        - access date: 2021 06 09
        - local file: raw/raw_2021_06_ANEEL_BD SIGA_09062021.xlsx
        - details of the raw data, see raw/README.docx file
    select the data:
        - installed capacity is the granted power (in PT: Potência Outorgada) in kW, unit is converted to MW
    translation from PT to EN:
        - see REFuelsESM/tools/trans_PT_to_EN.yaml

    Notes:
        - the file read is the translated file, column name and several content are renamed with EN name, see README
        - get the coordinates of the power plants
            - parse the longitude and latitude from the raw data (however, there are missing data)
            - for those plants (rows) with missing geolocation:
                - use pkg geopy to get the longitude and latitude based on 'city' in raw dataset (no missing data)
                - when more than one city provided, only the first city will be used
        - the plant_id (CEG) is not in the form of GGG.FF.UF.999999-D; some version has no UF
            - GGG: Generation Type
            - FF: fuel source
            - UF: state
            - 999999-D: core of the id, sequential numeric with verifying digit (D)
        - the column 'Type' will be further classified based on the attributes in "fuel_source", classified as gas-fired
          , oil-fired, coal-fired, and biomass-fired. Value of 'type' in ANEEL raw data:
            - 'mini_hydro'
            - 'small_hydro'
            - 'hydro'
            - 'on_wind'
            - 'solar_pv'
            - 'wave'
            - 'nuclear'
            - 'thermal'

    Returns:
        - output: geopandas
    """
    try:
        df = pd.read_csv(f'{os.path.dirname(os.path.realpath(__file__))}/resource'
                         f'/convert_ANEEL_geolocation_added_state_updated_2021_06.csv')
        df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
        df['end_time'] = pd.to_datetime(df['end_time'], errors='coerce')
        # use wkt.loads to deserialize a string and get a new geometric object of the appropriate type
        df['geometry'] = df['geometry'].apply(wkt.loads)
        output = gpd.GeoDataFrame(df, crs='epsg:4087')

    except FileNotFoundError:
        create_folder('resource')
        # read-in raw data and translate the columns name
        input_path = f'{os.path.dirname(os.path.realpath(__file__))}/raw/raw_2021_06_ANEEL_BD SIGA_09062021.xlsx'
        data = pd.read_excel(input_path, sheet_name=0, skiprows=1, na_values='-', decimal=',')

        # translate the PT words
        data = translate_pt_to_en_df(data, 'column_aneel')

        # Note: fuel_type, Type of fuel, which is a secondary classification ("type") of fuel ("fuel") for
        # thermal power plants, such as oil, natural gas and coal.

        # check if the code is unique value, which used to present the unique name of each power plants
        if not data["plant_id"].nunique() == len(data):
            print(f'column of "plant_id" is NOT unique')

        # hack coded: get the regime to determine whether they are connected to the grid
        # see trans_PT_to_EN.yaml/regime
        data['regime'] = data['regime'].str[-4:-1].replace('(SP', 'REG')

        # get the geometry
        data['latitude'] = extract_coordinates_decimal(data['latitude'])
        data['longitude'] = extract_coordinates_decimal(data['longitude'])
        # creating a geometry column
        geometry = [Point(xy) if not any(np.isnan(xy)) else None for xy in zip(data['longitude'], data['latitude'])]
        # default Coordinate Reference System of ANEEL dataset is EPSG:4674 (SIRGAS 2000)
        crs = 'epsg:4674'
        # Creating a Geographic data frame
        data_gdf = gpd.GeoDataFrame(data, crs=crs, geometry=geometry).to_crs('epsg:4087')

        # Assign the geometry based on the city for those power plants whose geometry is outside Brazil or missing
        # Assumption: when more than one city information is provided, just the first city will be considered
        # get the rows of power plants whose geometry is outside Brazil or missing
        state = get_node_by_state()
        within = gpd.sjoin(data_gdf, state, how='left', predicate='within')
        outside_or_missing = within[within['name'].isna()]  # has 847 records; here "name" is the "state" information

        # get the first city when more than one city exists in column 'city', drop the duplicates to save time
        used_city = outside_or_missing['city'].str.split(',').apply(lambda x: x[0]
                                                                    ).str.split(' - ').apply(lambda x: x[0].strip())
        orig_city_city = pd.concat([outside_or_missing['city'], used_city], axis=1)
        orig_city_city.columns = ['raw_city', 'convert_city']
        simple_city = pd.DataFrame({'convert_city': used_city.unique()})
        simple_city['geometry'] = simple_city['convert_city'].map(get_longitude_latitude).apply(
            lambda row: Point(row))  # the CRS is EPSG 4326

        city_log = gpd.GeoDataFrame(orig_city_city.join(simple_city.set_index('convert_city'),
                                                        on='convert_city')['geometry'].to_frame(),
                                    crs='epsg:4326').to_crs('epsg:4087')
        # instead of fillna use 'update'
        data_gdf.update(city_log, overwrite=True)

        # convert the time to datetime
        data_gdf['start_time'] = pd.to_datetime(data_gdf['start_time'], errors='coerce')
        data_gdf['end_time'] = pd.to_datetime(data_gdf['end_time'], errors='coerce')
        # convert kW in raw data to MW
        data_gdf["capacity"] = data_gdf["capacity"].astype(np.float64) / 1e3

        # reassign the "state" based on the "geometry"
        output = gpd.sjoin(data_gdf, get_node_by_state())
        output['state'] = output['name']
        output = refine_plant_type(output)

        # select the output columns
        output = output[['plant_name', 'plant_id', 'state', 'type', 'phase', 'fuel_type', 'fuel_source', 'primary',
                         'capacity', 'start_time', 'end_time', 'basin', 'regime', 'city', 'geometry']]
        
        # drop rows with solar plants data which construction have not started yet
        output = output.drop(output[(output['type'] == 'solar_pv') & (output['phase'] == 'construction not started')].index)

        output.to_csv('resource/convert_ANEEL_geolocation_added_state_updated_2021_06.csv', index=False)
        output.to_excel('resource/convert_excel_ANEEL_geolocation_added_state_updated_2021_06.xlsx', index=False)
        output_to_shp = output.copy()
        output_to_shp['start_time'] = output_to_shp['start_time'].dt.strftime('%Y-%m-%d')
        output_to_shp['end_time'] = output_to_shp['end_time'].dt.strftime('%Y-%m-%d')
        output_to_shp.to_file('resource/convert_excel_ANEEL_geolocation_added_state_updated_2021_06.shp', index=False)
        # Ying's note: Tubarão oil power plants is on an island of PE, where the coordinates are correct
    return output


def get_installed_cap_per_plant_aneel(base_year, current):
    """For a given base year, the total installed capacity of all types of power plants in each state is obtained.

    Args:
        - current: Bool, This is used to determine the lower bound (existing) and upper bound (existing + planning)
            - True: the plant with the phase of "operation"
            - False: plants with all phase
        - base_year: str: to select the installed capacity before base_year
    Returns
        - output: pd.Dataframe,
            - columns are information of each power plant:
              plant_name, plant_id, state, type, phase, fuel_type, fuel_source, start_time(have NaNs), capacity,
              basin (have NaNs), regime, latitude, longitude, city, geometry
            - index has no meaning
            - the geometry could be different from the raw data, see read_aneel_raw()
            - the values in the column 'type' could be further classified
    Notes:
        - Assumptions: the nan value in the column 'start_time' is assumed as the plants installed before the base_year
        - the DataFrame contains NaNs
        - 'capacity' in unit of [MW]
        - phase == operation (9455)
                                start_time   end_time  records
                                  -            nan        3 (2 x nuclear, 1 oil in RR)
                                  nan           -         0
                                  -             -         2199
                                  nan          nan        7253

          phase == construction / construction not started (1086)

                                  nan          nan        0
                                  -            nan        1 (nuclear)
                                  -            -          1085

    """
    # read in data
    dataset = read_aneel_raw()
    # revise the dataset where to decommission (end_time) is the same as the commission (start:time)
    dataset.loc[dataset.start_time == dataset.end_time, 'end_time'] = 'NaT'

    if current:
        # select the existing power plants are those with "phase" of "operation" if "current" is True
        dataset = dataset.loc[dataset["phase"].isin(['operation'])]
        # select the power plants in operation after the given time + which the start_time is empty
        dataset = dataset[(dataset['start_time'].dt.year <= int(base_year)) | (dataset['start_time'].isna())]

    # else:
    #     # TODO: check the time for each type of plants
    #     # assumptions: the scenario year is 2050
    #     dataset = dataset[(dataset['end_time'].dt.year > scenario_year) | (dataset['start_time'].isna())]
    # if the plant type will be further classified
    else:
        # planed capacity with the phase not 'operation' -> 'construction' or 'construction not started
        dataset = dataset.loc[~dataset["phase"].isin(['operation'])]

    return dataset


def get_installed_cap_per_type_state_aneel(base_year):
    """Get installed capacity sum [MW] per states and power plant type
    Args
        - base_year: str, to select the installed capacity before base_year
    Returns
        - output: pd.DataFrame:
            - columns: different type of power plants (the values in the column 'Type' were further classified)
            - index: states
            - values: installed capacity [MW]
    """
    try:
        output = pd.read_csv(
            f'{os.path.dirname(os.path.realpath(__file__))}/results/ANEEL_powerplants_per_state_per_type_reference_year_{base_year}.csv')
    except FileNotFoundError:
        create_folder(f'{os.path.dirname(os.path.realpath(__file__))}/results')
        # the phase of the plant with "operation"
        existed = get_installed_cap_per_plant_aneel(base_year, current=True)
        # the phase of the plant with all phase based on the phase
        planning = get_installed_cap_per_plant_aneel(base_year=None, current=False)
        planning['phase'] = "planning"
        # planning.loc[planning.start_time.dt.year < base_year, 'phase']='operation'
        # planning.loc[planning.start_time.dt.year >= base_year, 'phase'] = "planning" -> incorrect capacity for nuclear
        output = pd.concat([existed, planning])
        # by_plant = by_plant[by_plant['regime'].isin(["PIE", "REG"])] Ying note: not affect the validation results
        output = output[['state', 'type', 'capacity', 'phase']].groupby(['state', 'type', 'phase']).sum().reset_index()
        # build a table which allows comparison between different states and plants type
        # fill NaNs with ZERO: those states does not have the specific power plants, set the installed capacity to ZERO
        output = output.pivot_table(index='state',
                                    columns=['type', 'phase'],
                                    values='capacity').fillna(0).melt(ignore_index=False).reset_index()
        output['reference_year'] = base_year

        output.to_csv(
            f'{os.path.dirname(os.path.realpath(__file__))}/results/ANEEL_powerplants_per_state_per_type_reference_year_{base_year}.csv',
            index=False)

    return output


# % plotting

def plot_pie_per_state_aneel():
    create_folder('resource')
    cfg = read_config()
    tech_color = cfg["tech_color"]
    nice_name = cfg["nice_name"]

    installed = read_aneel_raw().groupby(['type', 'state'])['capacity'].sum()
    for state in installed.index.levels[1]:
        plt.close('all')
        fig, ax = plot_setting(figure_size=(4, 4))
        val = installed.iloc[installed.index.get_level_values('state') == state]
        p, t = ax.pie(val,
                      startangle=90,
                      shadow=False,
                      colors=[tech_color[tech] for tech in val.index.get_level_values('type')]
                      )

        label = val.rename(nice_name).reset_index().type
        plt.legend(p, label, bbox_to_anchor=(1.01, 1), frameon=False)
        plt.title(
            f'{val.rename(nice_name).reset_index().state.unique()[0]}, '
            f'{val.rename(nice_name).reset_index().capacity.sum().round(2)}MW')
        fig.savefig(f'resource/installed_capacity_pieplot_aneel_{state}_all_phase.png',
                    dpi=300, bbox_inches='tight')


def plot_map_per_plant_aneel():
    # Notes: there is one oil plant outside Brazil, it is correct
    plt.close('all')
    # prepare data
    data_gdf = read_aneel_raw()

    # plot setting
    fig, ax = plot_setting(figure_size=(4, 4))
    cfg = read_config()
    tech_color = cfg["tech_color"]
    nice_name = cfg["nice_name"]

    # plot the base map of the federal states
    get_node_by_state().plot(ax=ax, color='white', edgecolor='#343a40', linewidth=0.2, alpha=0.5)

    # plot the installed capacity of each type of plant
    scale = 200
    data_gdf.plot(ax=ax, markersize=data_gdf['capacity'] / scale,
                  alpha=0.5,
                  linewidth=0.2,
                  color=[tech_color[tech] for tech in data_gdf['type']],
                  )

    # add three phantom data points
    handles = [plt.Line2D([0], [0], color='w',
                          marker='o', markersize=size / scale, markeredgewidth=0.2, linewidth=0) for size in
               [100, 500, 1000]]
    labels = [str(size) for size in [100, 500, 1000]]
    l1 = ax.legend(handles, labels,
                   fontsize=4, title_fontsize=4.5,
                   loc="lower left",
                   frameon=False,
                   bbox_to_anchor=(0.25, 0.22),
                   title='MW   ',  # the whitespace here is to put legend title at left
                   handletextpad=0., columnspacing=0.5)

    ax.add_artist(l1)

    # add a rectangular patch for each technology
    label_color = pd.DataFrame([tech_color[tech] for tech in data_gdf['type']],
                               [nice_name[tech] for tech in data_gdf['type']]).drop_duplicates().squeeze()
    patch_list = []
    for label, color in label_color.iteritems():
        patch_list.append(patches.Patch(facecolor=color,
                                        label=label,
                                        alpha=0.8,
                                        linewidth=0.4,
                                        edgecolor='None'))

    # create a legend with the list of patches above.
    ax.legend(handles=patch_list, fontsize=4, loc='lower left', ncol=1,
              bbox_to_anchor=(0.8, 0.5), frameon=False)

    ax.set_aspect('equal')
    ax.axis('off')

    fig.savefig(f'resource/ANEEL_SIGA_processed_plants_all_phase.jpg',
                dpi=500, bbox_inches='tight')


def summary_installed_cap_per_type_state_aneel(base_year):
    create_folder(f'resource/{base_year}')
    with pd.ExcelWriter(f'resource/{base_year}/aneel_installed_cap_per_state_operation_GW_{base_year}.xlsx',
                        engine='openpyxl') as writer:
        for phase in ['operation', 'planning']:
            df = pd.concat([sin_state_region_mapping_in_series(),
                            get_installed_cap_per_type_state_aneel(base_year).query('phase==@phase').pivot(
                                index='state',
                                columns='type',
                                values='value'
                            ) / 1e3],
                           axis=1).reset_index().rename(columns={'index': 'state'}
                                                        ).sort_values('region')

            df.loc['Column_Total'] = df.sum(numeric_only=True, axis=0)
            df.loc[:, 'Row_Total'] = df.sum(numeric_only=True, axis=1)

            df.to_excel(excel_writer=writer,
                        index=False,
                        sheet_name=phase)  # GW


# % analysis
def get_number_of_plant_per_state_type_aneel(base_year):
    """Get the number of power plants when summed up by state and plant type
    Args:
        - base_year: string

    Returns
        - plant_number: pd.Series
            - index: MultiIndex: type/state
            - value: number of power plants for the plant type and state considered
    """
    by_plant = get_installed_cap_per_plant_aneel(base_year, current=True)
    plant_number = by_plant.value_counts(subset=['type', 'state'], sort=False)
    return plant_number

def get_decomissioning():
    """Get the decomissioned capacity over the years according to ANEEL file"""
    df = pd.read_csv(f'{os.path.dirname(os.path.realpath(__file__))}/resource'
                         f'/convert_ANEEL_geolocation_added_state_updated_2021_06.csv')
    df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
    df['end_time'] = pd.to_datetime(df['end_time'], errors='coerce')
    df = df.dropna(subset=['end_time'])
    df['year'] = df['end_time'].dt.year.astype(int)
    df = df.groupby(['state', 'type', 'phase', 'year'])['capacity'].sum()
    df.to_csv(
            f'{os.path.dirname(os.path.realpath(__file__))}/results/decomissioning_schedule.csv')
    


if __name__ == '__main__':
    for year in list(range(2012, 2025)):
        summary_installed_cap_per_type_state_aneel(year)
        get_installed_cap_per_type_state_aneel(year)
        
    plot_pie_per_state_aneel()
    plot_map_per_plant_aneel()
