# Open Brazilian Energy Data

## Publication
This repository is the official implementation of the paper [Harmonized and Open Energy Dataset for Modeling a Highly Renewable Brazilian Power System](https://rdcu.be/c6c7D).


Citation:
```
@article{deng_harmonized_2023,
  title={Harmonized and Open Energy Dataset for Modeling a Highly Renewable Brazilian Power System},
  author={Deng, Ying and Cao, Karl-Kien and Hu, Wenxuan and Stegen, Roland and von Krbek, Kai and Soria, Rafael and Rochedo, Pedro Rua Rodriguez and Jochem, Patrick},
  journal={Scientific Data},
  year={2023},
  volume={10},
  pages={103},
  doi={https://doi.org/10.1038/s41597-023-01992-9}
}
```

## Description
We aim to provide a first publicly available, spatially explicit, harmonized and English version of Brazil’s energy data, we enable researchers to replicate the Brazilian energy system and/or to improve the integration into global energy models starting from a common basis.

This dataset can be used in popular open energy system models such as PyPSA and other modeling frameworks. In the near future, we plan to release an application of this dataset, the PyPSA-Brazil model. PyPSA-Brazil is a novel open source Brazilian energy system based on publicly available datasets and an open modeling framework, PyPSA, designed to study the impact of incorporating renewable synthetic kerosene production into a highly renewable Brazilian energy system.

There is an overview of the categories in the dataset.

| No       | Category                                       | Raw Data                                                                                                                                                                                                                                                                                                                                                                                                                                   | License of Raw Data |        Folder name in code |
|----------|------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------|---------------------------:|
| i        | Geospatial data for Brazil                     | [IBGE - Municipal Mesh Data](https://www.ibge.gov.br/geociencias/organizacao-do-territorio/malhas-territoriais/15774-malhas.html?=&t=sobre)                                                                                                                                                                                                                                                                                                | ODbL                |                       node |
| ii       | Aggregated grid network topology               | [EPE Webmap](https://gisepeprd2.epe.gov.br/WebMapEPE/)                                                                                                                                                                                                                                                                                                                                                                                     | ODbL                |                       grid |
| iii      | Variable renewable potentials (wind and solar) | [EnDAT](http://dx.doi.org/10.1841)                                                                                                                                                                                                                                                                                                                                                                                                         | citation            |              re_potentials | 
| iv       | Installable capacity for biomass thermal plant | [ANEEL - SIGA](https://www.aneel.gov.br/)                                                                                                                                                                                                                                                                                                                                                                                                  | ODbL                |         biomass_potentials |       
| v        | Inflow for the hydropower plants               | [ONS - Historical Natural Energy Inflow for Each Region](http://www.ons.org.br/Paginas/resultados-da-operacao/historico-da-operacao/energia_afluente_subsistema.aspx)                                                                                                                                                                                                                                                                      | ODbL                |               hydro_inflow |       
| vi       | Power plants                                   | [ANEEL - SIGA](https://www.aneel.gov.br/)                                                                                                                                                                                                                                                                                                                                                                                                  | ODbL                |               power_plants |
| vii      | Electricity load profiles                      | [ONS - Historical Regional Load Curve](http://www.ons.org.br/Paginas/resultados-da-operacao/historico-da-operacao/curva_carga_horaria.aspx), [EPE-Statistical Yearbook of Electricity](https://www.epe.gov.br/sites-pt/publicacoes-dados-abertos/publicacoes/PublicacoesArquivos/publicacao-160/topico-168/Workbook_2021.xlsx)                                                                                                             | ODbL                |           load_time_series |
| viii     | Scenarios of electricity demand                | [IEA - World Energy Outlook](https://www.iea.org/reports/world-energy-outlook-2021), [EPE - National energy plan 2050](https://www.epe.gov.br/pt/publicacoes-dados-abertos/publicacoes/Plano-Nacional-de-Energia-2050), [Riahi, K., et al.](https://doi.org/10.1038/s41558-021-01215-2), [Van Soest, H.L., et al.](https://doi.org/10.1038/s41467-021-26595-z), [Baptista, L. B., et al.](https://doi.org/10.1016/j.gloenvcha.2022.102472) | citations           |    energy_demand_scenarios |
| ix       | Cross-border electricity exchanges             | [ONS - Historical Energy Exchanges](http://www.ons.org.br/Paginas/resultados-da-operacao/historico-da-operacao/intercambios_energia.aspx)                                                                                                                                                                                                                                                                                                  | ODbL                |                elec_import |
| appendix | Capacity comparison of power plants            | [ANEEL - SIGA](https://www.aneel.gov.br/), [EPE Webmap](https://gisepeprd2.epe.gov.br/WebMapEPE/), [ONS Historical Database](https://dados.ons.org.br/dataset/capacidade-geracao)                                                                                                                                                                                                                                                          | ODbL                | compare_power_plant_source |  


## How to use
To use the harmonized English version of the dataset directly, you can download it from the [Zenodo](https://doi.org/10.5281/zenodo.6951435). The metadata can be found in the paper [Harmonized and Open Energy Dataset for Modeling a Highly Renewable Brazilian Power System](https://rdcu.be/c6c7D).  

This repository contains scripts for data processing, plotting and analysis covered in the paper.  

Each folder is a subcategory presented in the paper. To get the results presented in the paper, you should run the script inside each subfolder. In the repository, we provide the codes for most of the categories, except iii) and viii). 

    open-brazilian-energy-data
        ├──dataset_name
        │   ├── raw
        │   │   ├── raw_XXX.csv
        │   │   ├── dict__XX.docx
        │   │   └── README.txt
        │   ├── resource
        │   │   ├── DATA_ANALYSIS_XXX.csv (.xlsx)
        │   │   ├── figure_XXX.png
        │   │   └── data_filename_without_prefix_XXX.csv
        │   ├── results
        │   │   └── model_inputs_filename.csv
        │   └── processing.py
        ├──plotting.yaml
        ├──pypsa-brazil_data_env.yml
        ├──pypsa-brazil_plot_style.txt
        ├──trans_PT_to_EN.yaml
        ├──utility_for_data.py
        └──README.md

- ``dataset_name``: Store the dataset with categories (cf. Data Records in the paper).

- ``raw``: Store the raw datasets. The meaning of prefix of the file are:

    - **raw_**: raw data file. Note: for the folder without another file with prefix, all files are raw data file.
    - **dict_**: This file contains the information relevant to the raw data. Usually, the translation of Portuguese and English are provided.
    - **README.txt**: Detail information of the data sources.

- ``resource``: Stores intermediate results, which can be picked up again by the other functions. Such a strategies is taken to store the large dataset. Store the results of data analysis. The meaning of prefix of the file are:

    - **DATA_ANALYSIS_**: usually .csv or .xlsx file, which aims for data analysis purpose between several datasets of same kind but from different source.
    - **DATA_REPORT_**: usually .txt or .html file, which aims for providing insight on the data quality.
    - **figures**: usually .png file, which aims to provide insight of the raw datasets, the intermediate results by data processing, or the output file under the folder of results.
    - **csv file without prefix**: the intermediate processed data.

- ``results``: the output of the ``script.py``, which is also the model input. In the ``script.py``, the functions to get the output is usually called ``def get_XX(XX)``. Users can adapt it to their purposes.

- ``script.py``: Python script for data processing includes:

    - pre-processing of the raw data
    - store the intermediate results for processing for large datasets
    - data analysis to provide insight of the raw data, intermediate processed data, and final results.
    - note: some scripts might run slowly, e.g., get_electricity_demand.py/def summary_load(), get_inflow.py

- ``plotting.yaml``: the color, nice name used in plotting for each technology 
- ``pypsa-brazil_data_env.yml``: the Python packages necessary to run the model. 
  - packages can be over-installed, as these are also the packages needed to run the pypsa-brazil model (coming soon) 
- ``pypsa-brazil_plot_style.txt``: the plotting style
- ``trans_PT_to_EN.yaml``: translation from Protheses to English, including the metadata (not necessary complete)
- ``utility_for_data.py``: utility function used in all scripts
- ``README.md``: readme file

## Installation

Have ``conda`` (alternative: minconda, mamba) installed on your operating system. If not, follow the [installation guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/>).

Download ``Open Brazilian Energy Data`` from our Gitlab repository  

```bash 
    /path/without/spaces % git clone https://gitlab.com/dlr-ve/esy/open-brazil-energy-data/open-brazilian-energy-data.git
```

Install python dependency
```bash
    /open_brazilian_energy_data % conda env create -f pypsa_brazil_data_env.yaml
    /open_brazilian_energy_data % conda activate pypsa-br
```

## License
Copyright <2022> [German Aerospace Center Institute for Networked Energy Systems](https://www.dlr.de/ve/desktopdefault.aspx/tabid-12472/21440_read-49440/)

Under the open source [BSD-3-Clause](LICENSE.txt)


## Contact
Deng, Ying [dengying8421@gmail.com](mailto:dengying8421@gmail.com)

