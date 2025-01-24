import pandas as pd
import numpy as np

# Path do arquivo gerado pelo script pypsa-brazil que contém dados da rede por estado 
path = "C:\\TCC\\ScriptsPypsaBrazil\\open-brazilian-energy-data_atualizado\\grid\\results\\EPEWebmap_equivalent_grid_aggregate_by_state_operation_and_planed.csv"
grid = pd.read_csv(path)

# Mapeamento dos estados nas respectivas regiões (seguindo padrão do ONS)
regiao_estado ={'PA': 'NORTE',
                'TO': 'NORTE',
                'MA': 'NORTE',
                'AP': 'NORTE',
                'AM': 'NORTE',
                'RR': 'NORTE',
                'PI': 'NORDESTE',
                'CE': 'NORDESTE',
                'RN': 'NORDESTE',
                'PB': 'NORDESTE',
                'PE': 'NORDESTE',
                'AL': 'NORDESTE',
                'SE': 'NORDESTE',
                'BA': 'NORDESTE',
                'ES': 'SUDESTE',
                'RJ': 'SUDESTE',
                'MG': 'SUDESTE',
                'SP': 'SUDESTE',
                'GO': 'SUDESTE',
                'DF': 'SUDESTE',
                'MT': 'SUDESTE',
                'AC': 'SUDESTE',
                'RO': 'SUDESTE',
                'MS': 'SUDESTE',
                'RS': 'SUL',
                'SC': 'SUL',
                'PR': 'SUL',
                'ARG': 'ARGENTINA',
                'PRY': 'PARAGUAI',
                'VEN': 'VENEZUELA',
                'URY': 'URUGUAI'}


# grid = grid.drop(columns=["name", "length", "carrier"])

# Gera colunas no arquivo lido, associando regiões aos estados
grid["regiao0"] = grid["node0"].map(regiao_estado)
grid["regiao1"] = grid["node1"].map(regiao_estado)

# Agregação dos dados 'por estado' em 'por regiao'
# Gera o dataframe no qual serão inseridos os dados por regiao
grid_subsistema = pd.DataFrame(columns = ['regiao0', 'regiao1', 'transfer_capacity', 'efficiency'])

# Define as regioes por estado
regioes_brasil = ['SUDESTE', 'SUL', 'NORDESTE', 'NORTE']

# Para cada regiao do brasil, busca por conexões com estados que pertencem a outra regiao
for regiao0 in regioes_brasil:
    regioes_total = ['SUDESTE', 'SUL', 'NORDESTE', 'NORTE', 'ARGENTINA','PARAGUAI','VENEZUELA', 'URUGUAI']
    regioes_total.remove(regiao0)
    for regiao1 in regioes_total:
        grid_regiao = grid[((grid['regiao0'] == regiao0) & (grid['regiao1'] == regiao1)) |
                           ((grid['regiao0'] == regiao1) & (grid['regiao1'] == regiao0))]
        if not grid_regiao.empty:
            transfer_capacity = grid_regiao['transfer_capacity'].sum()
            efficiency = np.average(grid_regiao['efficiency'], weights=grid_regiao['transfer_capacity'])
            grid_subsistema = pd.concat([grid_subsistema,pd.DataFrame(
                                                    {'regiao0': regiao0, 'regiao1': regiao1,
                                                    'transfer_capacity': transfer_capacity,'efficiency': efficiency},
                                                     index=[0])])
            
grid_subsistema.to_csv('grid_subsistema.csv', index=False)
        