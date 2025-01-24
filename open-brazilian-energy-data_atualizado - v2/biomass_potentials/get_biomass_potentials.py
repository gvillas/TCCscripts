# -*- coding:utf-8 -*-

__version__ = '0.1.0'
__maintainer__ = 'Ying Deng 12.07.2022'
__authors__ = 'Ying Deng'
__credits__ = 'Ying Deng'
__email__ = 'Ying.Deng@dlr.de'
__date__ = '23.11.2021'
__status__ = 'dev'  # options are: dev, test, prod
__copyright__ = 'DLR'

"""Module-level docstring
Obtaining estimated economic potential (geographically installable potentials, MW) for biomass power plant expansion

Note:
- operation: installed capacity of biomass thermal power plants, based on ANEEL dataset
- planning: capacity of biomass thermal power plants with status "under construction" or "construction not started", based on ANEEL dataset
- potential: additional installable capacity based on the estimated economic potential of biomass power plant expansion, based on Portugal-Pereira, Joana, Soria, Rafael, et al, 2015

"""
import sys
import pandas as pd

sys.path.append('../')
from power_plants.get_installed_capacity import get_installed_cap_per_type_state_aneel
from utility_for_data import translate_pt_to_en_df, create_folder, plot_setting


def get_biomass_geographically_installable_potential(year):
    """Get the potential of geographically installable capacity for the biomass thermal power plants

    Args:
        year: int, the reference year

    Notes:
    - The data source is the estimated economic potential (energy, unit in MWh).
        - source: Portugal-Pereira, Joana, Soria, Rafael, et al, 2015: Agricultural and agro-industrial residues-to-energy: Techno-economic and environmental assessment in Brazil. DOI: 10.1016/j.biombioe.2015.08.010.
            - 'Potencial económico_dentro del círculo2.xlsx'
        - license: get email approve from the author Soria, Rafael
        - resolution: municipality level, annual
        - what is the economic potential in paper: the potential for electricity generation from biomass residues distributed within a 50 km radius of the biomass thermal plant. The biomass thermal power plants includes the installed and the planning one from ANEEL. However, the raw dataset of power plants is not accessible. We instead use the current version of ANEEL SIGA datasset.
        - notes:
            - in the paper the total theoretic potential is 143 TWh/y, however, the data is not provided by the author. In the "Copia de Dados GIS potencial biomassa3_potencial técnico y sustentable_v1.xlsx", the total 127 TWh/y, which is an out-of-date version.
            - in the data, no state "DF"
            - unit [MWh/y], note in cell K2 of the "raw_Potencial económico_dentro del círculo2_unit correct.xlsx", the unit [KWh/y], which is corrected by K1 (marked in red)

    - Data processing: convert the estimated economic potential in the paper to the geographic installable capacity (output of this script)

        - geographic installable capacity of biomass power plant expansion = installed capacity (ANEEL, phase->operation) + planning capacity (ANEEL, phase->planning) + the potential value in the paper / (0.6*8760h) (0.6 is the "availability value" for "Biomass -steam turbine" from Table A3, Soria, R., Lucena, A. F., Tomaschek, J., Fichter, T., Haasz, T., Szklo, A., ... & Kern, J. (2016). Modelling concentrated solar power (CSP) in the Brazilian energy system: A soft-linked model coupling approach. Energy, 116, 265-280.)
            - availability value is the "availability of the time through the year?", confirmed by Soria via email.
            - The value of 0.6 was chosen as a conservative assessment because the economic potential includes several types of biomass, with sugarcane having a relatively high availability factor (>0.8), while other types are lower.

    """
    # economic potentials from Portugal-Pereira, Joana, Soria, Rafael, et al, 2015 paper, converted with capacity factor
    skiprows = list(range(4850, 5019))
    skiprows.append(0)
    potential_gen = pd.read_excel('raw/raw_Potencial económico_dentro del círculo2_unit correct.xlsx',
                                  sheet_name=0, usecols=[4, 10], skiprows=skiprows, na_values='')  # MWh
    potential_gen = translate_pt_to_en_df(potential_gen, 'column_bio_potential')
    potential_gen = potential_gen.groupby('state').sum().squeeze()
    potential_gen = potential_gen[potential_gen != 0].rename('value').round(decimals=2)  # drop zeros
    gen_cap = potential_gen / (0.6 * 8760)  # convert to capacity MW
    # get the sum of the ANEEL capacity, installed + planning
    aneel_cap = get_installed_cap_per_type_state_aneel(year).query("type=='biomass'")
    installed_plan = get_installed_cap_per_type_state_aneel(year).query("type=='biomass'").groupby(
        'state').sum().reset_index()
    installed_plan['type'] = 'biomass'
    installed_plan['reference_year'] = year
    installed_plan['phase'] = 'operation+planning'

    # get the geographic installable potentials
    cap_potential = installed_plan.copy().set_index('state')
    cap_potential['value'] = cap_potential['value'] + gen_cap
    cap_potential = cap_potential.fillna(0).reset_index()
    cap_potential['phase'] = 'potential'

    # export the results
    create_folder('results')
    cap_potential.to_csv(f'results/biomass_geographic_potential_reference_year_{year}.csv', index=False)

    # compare the new upper bound with the upper bound (exist+planning) and lower bound(exist) from ANEEL dataset
    # plotting
    create_folder('resource')

    to_plot = pd.concat([aneel_cap.query("phase=='operation'"), installed_plan, cap_potential])

    fig, ax = plot_setting(figure_size=(8, 4))
    (to_plot.pivot_table(index='state', values='value', columns='phase')[
         ['operation', 'operation+planning', 'potential']] / 1e3).plot(ax=ax, kind='bar')  # unit GW

    ax.set(ylabel="GW",  # unit GW
           xlabel="")
    ax.legend(ncol=1, bbox_to_anchor=(1.01, 0.8), title='phase', frameon=False)
    fig.autofmt_xdate(rotation=90)

    fig.savefig(fname=f'resource/comparison_Soria_ANEEL_installed_planning_potential_reference_year_{year}.png',
                dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    for base_year in list(range(2012, 2021)):
        get_biomass_geographically_installable_potential(year=base_year)
