"""
Cada função descrita abaixo é responsável pela leitura dos dados tratados, filtragem/agregações de acordo
com a configuração fornecida pelo usuário e conversão para o arquivo de dados correspondente do OSeMOSYS.
"""

# Bibliotecas
import yaml
import pandas as pd
import os
import sys

# Define o diretório de trabalho para o diretório atual do script
diretorio_corrente = os.path.dirname(os.path.realpath(__file__))
os.chdir(diretorio_corrente)

# SETs
def SetDAILYTIMEBRACKET():
    pass

def SetDAYTYPE():
    pass

def SetEMISSION():
    pass

def SetFUEL(f):
    pass

def SetMODE_OF_OPERATION():
    pass

def SetRegion():
    pass

def SetSEASON():
    pass

def SetSTORAGE():
    pass

def SetTECHNOLOGY():
    pass

def SetTIMESLICE():
    pass

def SetYEAR():
    pass


# params
def AccumulatedAnnualDemand(r,f,y,file_name):
    df = CarregaArquivo(file_name)
    if len(df.index) == 0:
        EscreveArquivo(df, "AccumulatedAnnualDemand")
    else:
        pass

def AnnualEmissionLimit(r,e,y,file_name):
    """Limites de emissão definidos por ano. A função está preparada para replicar os limites de emissão definidos
    para o SIN para as demais regiões do caso, já que a granularidade mínima no momento era o SIN.

    Args:
        r (dict): set REGION
        e (list): set EMISSION
        y (list): set YEAR
        file_name (str): Nome do arquivo de dados tratados de limites de emissão
    """
    df = CarregaArquivo(file_name)
    if len(df.index) == 0:
        EscreveArquivo(df, "AnnualEmissionLimit")
    else:
        df_filtered = df[(df['YEAR'].isin(y)) & (df['EMISSION'].isin(e))]
        regions = set(r.values())
        regions -= set(['ARGENTINA', 'PARAGUAI', 'VENEZUELA', 'URUGUAI'])
        df_final = pd.DataFrame()
        for region in regions:
            df_filtered['REGION'] = region
            df_final = pd.concat([df_final, df_filtered], ignore_index=True)
        EscreveArquivo(df_final, "AnnualEmissionLimit") 
                

def AnnualExogenousEmission(r,f,y,file_name):
    df = CarregaArquivo(file_name)
    if len(df.index) == 0:
        EscreveArquivo(df, "AnnualExogenousEmission")
    else:
        pass

def AvailabilityFactor(r,t,y,file_name):
    df = CarregaArquivo(file_name)
    if len(df.index) == 0:
        EscreveArquivo(df, "AvailabilityFactor")
    else:
        pass

def CapacityFactor(r,t,l,y,on_wind,off_wind,solar_pv,other_tec):
    """Cálculo do fator de capacidade para cada tecnologia. O tratamento é diferenciado para tecnologias
    convencionais e fontes intermitentes. Para a primeira, é utilizado um fator de capacidade constante,
    enquanto as fontes renováveis possuem dados que variam com granularidade horária.

    
    Args:
        r (dict): set REGION
        t (list): set TECHNOLOGY
        l (list): set TIMESLICE
        y (list): set YEAR
        on_wind (str): Nome do arquivo de dados com fator de capacidade de eólicas onshore
        off_wind (str): Nome do arquivo de dados com fator de capacidade de eólicas offshore
        solar_pv (str): Nome do arquivo de dados com fator de capacidade de solar fotovoltaica
        other_tec (str): Nome do arquivo de dados com fator de capacidade para outras tecnologias
    """
    regions = r
    for i in ['ARG', 'PRY', 'VEN', 'URY']:
        regions.pop(i)
    
    column_order = ["REGION", "TECHNOLOGY", "hours", "YEAR", "VALUE"]
    df_onwind = CarregaArquivo(on_wind)
    df_onwind_melted = df_onwind.melt(id_vars="hours", var_name="variable", value_name="VALUE")
    df_onwind_melted["REGION"] = df_onwind_melted["variable"].map(regions)
    df_onwind_final = df_onwind_melted.groupby(["REGION","hours"])['VALUE'].mean().reset_index()
    df_onwind_final["TECHNOLOGY"] = 'on_wind'
    df_onwind_final.rename(columns={'hours': 'TIMESLICE'}, inplace=True)

    df_offwind = CarregaArquivo(off_wind)
    df_offwind_melted = df_offwind.melt(id_vars="hours", var_name="variable", value_name="VALUE")
    df_offwind_melted["REGION"] = df_offwind_melted["variable"].map(regions)
    df_offwind_final = df_offwind_melted.groupby(["REGION","hours"])['VALUE'].mean().reset_index()
    df_offwind_final["TECHNOLOGY"] = 'off_wind'
    df_offwind_final.rename(columns={'hours': 'TIMESLICE'}, inplace=True)

    df_solar_pv = CarregaArquivo(solar_pv)
    df_solar_pv_melted = df_solar_pv.melt(id_vars="hours", var_name="variable", value_name="VALUE")
    df_solar_pv_melted["REGION"] = df_solar_pv_melted["variable"].map(regions)
    df_solar_pv_final = df_solar_pv_melted.groupby(["REGION","hours"])['VALUE'].mean().reset_index()
    df_solar_pv_final["TECHNOLOGY"] = 'solar_pv'
    df_solar_pv_final.rename(columns={'hours': 'TIMESLICE'}, inplace=True)

    df_othertec = CarregaArquivo(other_tec)
    df_othertec_filtered = df_othertec[df_othertec['TECHNOLOGY'].isin(t)]
    regions = set(r.values())
    regions -= set(['ARGENTINA', 'PARAGUAI', 'VENEZUELA', 'URUGUAI'])
    df_final = pd.DataFrame()
    for region in regions:
        df_othertec_filtered['REGION'] = region
        df_final = pd.concat([df_final, df_othertec_filtered], ignore_index=True)

    df = pd.concat([df_final, df_onwind_final, df_offwind_final, df_solar_pv_final], ignore_index=True)
    
    EscreveArquivo(df, "CapacityFactor")

def CapacityOfOneTechnologyUnit(r,t,y,file_name):
    df = CarregaArquivo(file_name)
    if len(df.index) == 0:
        EscreveArquivo(df, "CapacityOfOneTechnologyUnit")
    else:
        pass

def CapacityToActivityUnit(r,t,file_name):
    """Fator de conversão relacionado à energia que seria produzida quando uma unidade de capacidade é completamente
    utilizada no ano. A função está preparada para replicar os dados definidos para o SIN para as demais regiões do
    caso, já que a granularidade mínima no momento era o SIN.

    Args:
        r (dict): set REGION
        t (list): set TECHNOLOGY
        file_name (str): Nome do arquivo de dados tratados de 'CapacityToActivityUnit'
    """
    df = CarregaArquivo(file_name)
    if len(df.index) == 0:
        EscreveArquivo(df, "CapacityToActivityUnit")
    else:
        df_filtered = df[df['TECHNOLOGY'].isin(t)]
        regions = set(r.values())
        regions -= set(['ARGENTINA', 'PARAGUAI', 'VENEZUELA', 'URUGUAI'])
        df_final = pd.DataFrame()
        for region in regions:
            df_filtered['REGION'] = region
            df_final = pd.concat([df_final, df_filtered], ignore_index=True)
        EscreveArquivo(df_final, "CapacityToActivityUnit")

def CapitalCost(r,t,y,file_name):
    """Custo de investimento de cada tecnologia, por unidade de capacidade. A função está preparada para replicar os
    dados definidos para o SIN para as demais regiões do caso, já que a granularidade mínima no momento era o SIN.

    Args:
        r (dict): set REGION
        t (list): set TECHNOLOGY
        y (list): set YEAR
        file_name (str): Nome do arquivo de dados tratados contendo dados de custos de investimento
    """
    df = CarregaArquivo(file_name)
    if len(df.index) == 0:
        EscreveArquivo(df, "CapitalCost")
    else:
        df_filtered = df[(df['TECHNOLOGY'].isin(t)) & df['YEAR'].isin(y)]
        regions = set(r.values())
        regions -= set(['ARGENTINA', 'PARAGUAI', 'VENEZUELA', 'URUGUAI'])
        df_final = pd.DataFrame()
        for region in regions:
            df_filtered['REGION'] = region
            df_final = pd.concat([df_final, df_filtered], ignore_index=True)
        EscreveArquivo(df_final, "CapitalCost")

def CapitalCostStorage(r,s,y,file_name):
    df = CarregaArquivo(file_name)
    if len(df.index) == 0:
        EscreveArquivo(df, "CapitalCostStorage")
    else:
        pass

def Conversionld():
    pass

def Conversionls():
    pass

def Conversionlh():
    pass

def DaySplit():
    pass

def DaysInDayType():
    pass

def DepreciationMethod(r,file_name):
    df = CarregaArquivo(file_name)
    if len(df.index) == 0:
        EscreveArquivo(df, "DepreciationMethod")
    else:
        pass

def DiscountRate(r,file_name):
    df = CarregaArquivo(file_name)
    if len(df.index) == 0:
        EscreveArquivo(df, "DiscountRate")
    else:
        pass

def DiscountRateStorage(r,s,file_name):
    df = CarregaArquivo(file_name)
    if len(df.index) == 0:
        EscreveArquivo(df, "DiscountRateStorage")
    else:
        pass

def EmissionActivityRatio(r,t,e,m,y,file_name):
    """Fator de emissão para cada tecnologia por unidade, por modo de operação. A função está preparada para
    replicar os dados definidos para o SIN para as demais regiões do caso, já que a granularidade mínima no momento
    era o SIN.

    Args:
        r (dict): set REGION
        t (list): set TECHNOLOGY
        e (list): set EMISSION
        m (list): set MODE_OF_OPERATION
        y (list): set YEAR
        file_name (str): Nome do arquivo contendo dados de fator de emissão
    """
    df = CarregaArquivo(file_name)
    if len(df.index) == 0:
        EscreveArquivo(df, "EmissionActivityRatio")
    else:
        df_filtered = df[(df['TECHNOLOGY'].isin(t)) & df['EMISSION'].isin(e) & df['MODE_OF_OPERATION'].isin(m) & df['YEAR'].isin(y)]
        regions = set(r.values())
        regions -= set(['ARGENTINA', 'PARAGUAI', 'VENEZUELA', 'URUGUAI'])
        df_final = pd.DataFrame()
        for region in regions:
            df_filtered['REGION'] = region
            df_final = pd.concat([df_final, df_filtered], ignore_index=True)
        EscreveArquivo(df_final, "EmissionActivityRatio")

def EmissionsPenalty(r,e,y,file_name):
    df = CarregaArquivo(file_name)
    if len(df.index) == 0:
        EscreveArquivo(df, "EmissionsPenalty")
    else:
        pass

def FixedCost(r,t,y,file_name):
    """Custos fixos de O&M de cada technologia, por unidade de capacidade. A função está preparada para replicar os
    dados definidos para o SIN para as demais regiões do caso, já que a granularidade mínima no momento era o SIN.

    Args:
        r (dict): set REGION
        t (list): set TECHNOLOGY
        y (list): set YEAR
        file_name (str): Nome do arquivo de dados tratados de custos fixos
    """
    df = CarregaArquivo(file_name)
    if len(df.index) == 0:
        EscreveArquivo(df, "FixedCost")
    else:
        df_filtered = df[(df['TECHNOLOGY'].isin(t)) & df['YEAR'].isin(y)]
        regions = set(r.values())
        regions -= set(['ARGENTINA', 'PARAGUAI', 'VENEZUELA', 'URUGUAI'])
        df_final = pd.DataFrame()
        for region in regions:
            df_filtered['REGION'] = region
            df_final = pd.concat([df_final, df_filtered], ignore_index=True)
        EscreveArquivo(df_final, "FixedCost")

def InputActivityRatio(r,t,f,m,y,file_name):
    """Taxa de uso do combustível/commodity de uma tecnologia. A função está preparada para replicar os
    dados definidos para o SIN para as demais regiões do caso, já que a granularidade mínima no momento era o SIN.

    Args:
        r (dict): set REGION
        t (list): set TECHNOLOGY
        f (list): set FUEL
        m (list): set MODE_OF_OPERATION
        y (list): set YEAR
        file_name (str): Nome do arquivo de dados tratados contendo a 'ActivityRatio'
    """
    df = CarregaArquivo(file_name)
    if len(df.index) == 0:
        EscreveArquivo(df, "InputActivityRatio")
    else:
        df_filtered = df[(df['TECHNOLOGY'].isin(t)) & df['FUEL'].isin(f) & df['MODE_OF_OPERATION'].isin(m) & df['YEAR'].isin(y)]
        regions = set(r.values())
        regions -= set(['ARGENTINA', 'PARAGUAI', 'VENEZUELA', 'URUGUAI'])
        df_final = pd.DataFrame()
        for region in regions:
            df_filtered['REGION'] = region
            df_final = pd.concat([df_final, df_filtered], ignore_index=True)
        EscreveArquivo(df_final, "InputActivityRatio")

def ModelPeriodEmissionLimit(r,e,file_name):
    """Limite anual de emissões geradas por cada região para todo horizonte de estudo. A função está preparada para
    replicar os dados definidos para o SIN para as demais regiões do caso, já que a granularidade mínima no momento era o SIN.

    Args:
        r (dict): REGION
        e (list): EMISSION
        file_name (str): Nome do arquivo de dados tratados com limites de emissão
    """
    df = CarregaArquivo(file_name)
    if len(df.index) == 0:
        EscreveArquivo(df, "ModelPeriodEmissionLimit")
    else:
        df_filtered = df[df['EMISSION'].isin(e)]
        regions = set(r.values())
        regions -= set(['ARGENTINA', 'PARAGUAI', 'VENEZUELA', 'URUGUAI'])
        df_final = pd.DataFrame()
        for region in regions:
            df_filtered['REGION'] = region
            df_final = pd.concat([df_final, df_filtered], ignore_index=True)
        EscreveArquivo(df_final, "ModelPeriodEmissionLimit")

def ModelPeriodExogenousEmission(r,e,file_name):
    df = CarregaArquivo(file_name)
    if len(df.index) == 0:
        EscreveArquivo(df, "ModelPeriodExogenousEmission")
    else:
        pass

def OperationalLife(r,t,file_name):
    """Vida útil das tecnologias. A função está preparada para replicar os dados definidos para o SIN para as
    demais regiões do caso, já que a granularidade mínima no momento era o SIN.

    Args:
        r (dict): set REGION
        t (list): set TECHNOLOGY
        file_name (str): Nome do arquivo de dados tratados com vida útil das tecnologias
    """
    df = CarregaArquivo(file_name)
    if len(df.index) == 0:
        EscreveArquivo(df, "OperationalLife")
    else:
        df_filtered = df[df['TECHNOLOGY'].isin(t)]
        regions = set(r.values())
        regions -= set(['ARGENTINA', 'PARAGUAI', 'VENEZUELA', 'URUGUAI'])
        df_final = pd.DataFrame()
        for region in regions:
            df_filtered['REGION'] = region
            df_final = pd.concat([df_final, df_filtered], ignore_index=True)
        EscreveArquivo(df_final, "OperationalLife")

def OperationalLifeStorage(r,s,file_name):
    df = CarregaArquivo(file_name)
    if len(df.index) == 0:
        EscreveArquivo(df, "OperationalLifeStorage")
    else:
        pass

def OutputActivityRatio(r,t,f,m,y,file_name):
    """_summary_

    Args:
        r (dict): set REGION
        t (list): set TECHNOLOGY
        f (list): set FUEL
        m (list): set MODE_OF_OPERATION
        y (list): set YEAR
        file_name (str): Nome do arquivo de dados tratados com 'OutputActivityRatio'
    """
    df = CarregaArquivo(file_name)
    if len(df.index) == 0:
        EscreveArquivo(df, "OutputActivityRatio")
    else:
        df_filtered = df[(df['TECHNOLOGY'].isin(t)) & (df['FUEL'].isin(f)) & (df['MODE_OF_OPERATION'].isin(m)) & (df['YEAR'].isin(y))]
        regions = set(r.values())
        regions -= set(['ARGENTINA', 'PARAGUAI', 'VENEZUELA', 'URUGUAI'])
        df_final = pd.DataFrame()
        for region in regions:
            df_filtered['REGION'] = region
            df_final = pd.concat([df_final, df_filtered], ignore_index=True)
        EscreveArquivo(df_final, "OutputActivityRatio")

def REMinProductionTarget(r,y,file_name):
    df = CarregaArquivo(file_name)
    if len(df.index) == 0:
        EscreveArquivo(df, "REMinProductionTarget")
    else:
        pass

def RETagFuel(r,f,y,file_name):
    df = CarregaArquivo(file_name)
    if len(df.index) == 0:
        EscreveArquivo(df, "RETagFuel")
    else:
        pass

def RETagTechnology(r,t,y,file_name):
    df = CarregaArquivo(file_name)
    if len(df.index) == 0:
        EscreveArquivo(df, "RETagTechnology")
    else:
        pass

def ReserveMargin(r,y,file_name):
    df = CarregaArquivo(file_name)
    if len(df.index) == 0:
        EscreveArquivo(df, "ReserveMargin")
    else:
        df_filtered = df[df['YEAR'].isin(y)]
        regions = set(r.values())
        regions -= set(['ARGENTINA', 'PARAGUAI', 'VENEZUELA', 'URUGUAI'])
        df_final = pd.DataFrame()
        for region in regions:
            df_filtered['REGION'] = region
            df_final = pd.concat([df_final, df_filtered], ignore_index=True)
        EscreveArquivo(df_final, "ReserveMargin")

def ReserveMarginTagFuel(r,f,y,file_name):
    df = CarregaArquivo(file_name)
    if len(df.index) == 0:
        EscreveArquivo(df, "ReserveMarginTagFuel")
    else:
        df_filtered = df[(df['FUEL'].isin(f)) & (df['YEAR'].isin(y))]
        regions = set(r.values())
        regions -= set(['ARGENTINA', 'PARAGUAI', 'VENEZUELA', 'URUGUAI'])
        df_final = pd.DataFrame()
        for region in regions:
            df_filtered['REGION'] = region
            df_final = pd.concat([df_final, df_filtered], ignore_index=True)
        EscreveArquivo(df_final, "ReserveMarginTagFuel")

def ReserveMarginTagTechnology(r,t,y,file_name):
    df = CarregaArquivo(file_name)
    if len(df.index) == 0:
        EscreveArquivo(df, "ReserveMarginTagTechnology")
    else:
        df_filtered = df[(df['TECHNOLOGY'].isin(t)) & (df['YEAR'].isin(y))]
        regions = set(r.values())
        regions -= set(['ARGENTINA', 'PARAGUAI', 'VENEZUELA', 'URUGUAI'])
        df_final = pd.DataFrame()
        for region in regions:
            df_filtered['REGION'] = region
            df_final = pd.concat([df_final, df_filtered], ignore_index=True)
        EscreveArquivo(df_final, "ReserveMarginTagTechnology")
    

def ResidualCapacity(r,t,y,file_name):
    """Cálculo da capacidade residual do sistema para cada ano do estudo. Os dados são apresentados por tecnologia e para cada agregação espacial determinada
    pelo usuário.

    Args:
        r (dict): REGION
        t (list): TECHNOLOGY
        y (list): YEAR
        file_name (str): Nome do arquivo de dados tratados com capacidade residual
    """
    df_inst_cap = CarregaArquivo(file_name)
    df_inst_cap = df_inst_cap.rename(columns={'state': 'REGION', 'type': 'TECHNOLOGY', 'phase': 'PHASE', 'value':'VALUE', 'reference_year': 'YEAR'})
    df_inst_cap = df_inst_cap.groupby(["REGION","TECHNOLOGY","YEAR"])["VALUE"].sum().reset_index()

    df = pd.DataFrame()

    #TODO: Incluir descomissionamento de usinas no cálculo da capacidade instalada para anos seguintes do ano base
    for year in y:

        # Replica valores do ano inicial para todos os anos do estudo
        df_inst_cap_ano = df_inst_cap
        df_inst_cap_ano['YEAR'] = year 

        df = pd.concat([df, df_inst_cap_ano], ignore_index=True)

    df_filtered = df[df['TECHNOLOGY'].isin(t)]

    # Mapeia os dados de agregação de região no dicionário já filtrado por tecnologia
    df_filtered['REGION_AUX'] = df['REGION'].map(r)
    df_filtered.drop(columns=['REGION'])

    # Agrega os dados de capacidade de acordo com a 'REGION' fornecida pelo usuário
    df_final = df_filtered.groupby(["REGION_AUX", "TECHNOLOGY", "YEAR"])["VALUE"].sum().reset_index()
    df_final = df_final.rename(columns={'REGION_AUX': 'REGION'})
    
    # Escreve arquivo .csv    
    EscreveArquivo(df_final, "ResidualCapacity")

def ResidualStorageCapacity(r,s,y,file_name):
    df = CarregaArquivo(file_name)
    if len(df.index) == 0:
        EscreveArquivo(df, "ResidualStorageCapacity")
    else:
        pass

def SpecifiedAnnualDemand(r,f,y,file_name):
    """Cálculo da demanda total anual do sistema. A demanda de cada estado é agregada nas regiões de acordo com a especificação do usuário.
    
    **Os dados tratados de demanda são gerados a partir do script pypsa-brazil a partir da fonte:**
    https://dados.ons.org.br/dataset/carga-energia-verificada
    
    Args:
        r (dict): set REGION
        f (list): set FUEL
        y (list): set YEAR
        file_name (str): Nome do arquivo 'csv' que contém os dados tratados de demanda
    """

    df = CarregaArquivo(file_name)
    
    # Agrega demanda do ano por UF
    df = df.groupby(['state'])['value'].sum().reset_index()
    
    # Mapeia regioes de acordo com 'REGION' fornecida pelo usuário
    df['REGION'] = df['state'].map(r)
    df.drop(columns=['state'])

    # Agrega os dados de demanda de acordo com a 'REGION' fornecida pelo usuário
    df = df.groupby(["REGION"])["value"].sum().reset_index()

    # Ajustes na formatação do arquivo
    df = df.rename(columns={'value': 'VALUE'})
    df['FUEL'] = 'DEM'

    df_final = pd.DataFrame()
    for year in y:
        df_aux = df
        df_aux['YEAR'] = year
        df_final = pd.concat([df_final, df_aux])
        
    ordem_colunas = ['REGION', 'FUEL', 'YEAR', 'VALUE']
    df_final = df_final[ordem_colunas]

    # Escreve arquivo .csv    
    EscreveArquivo(df_final, "SpecifiedAnnualDemand")

def SpecifiedDemandProfile(r,f,t,y,file_name):
    """Cálculo do perfil da demanda do sistema. O perfil é obtido a partir dos dados de demanda horária fornecidos.
    
    **Os dados tratados de demanda são gerados a partir do script pypsa-brazil a partir da fonte:**
    https://dados.ons.org.br/dataset/carga-energia-verificada
    
    Args:
        r (dict): set REGION
        f (list): set FUEL
        t (list): set TIMESLICE
        y (list): set YEAR
        file_name (str): Nome do arquivo 'csv' que contém os dados tratados de demanda
    """
    df = CarregaArquivo(file_name)
    
    df_perfil = df
    # Agrega demanda do ano por UF
    df_demand = df.groupby(['state'])['value'].sum().reset_index()

    # Mapeia regioes de acordo com 'REGION' fornecida pelo usuário
    df_demand['REGION'] = df_demand['state'].map(r)
    df_demand.drop(columns=['state'])
    
    df_perfil['REGION'] = df_perfil['state'].map(r)
    df_perfil.drop(columns=['state'])

    # Agrega os dados de demanda de acordo com a 'REGION' fornecida pelo usuário
    df_demand = df_demand.groupby(["REGION"])["value"].sum().reset_index()
    
    df_perfil = df_perfil.groupby(["REGION", "time"])["value"].sum().reset_index()

    # Ajustes na formatação do arquivo
    df_demand = df_demand.rename(columns={'value': 'TOTAL'})

    df_perfil = df_perfil.rename(columns={'value': 'VALUE', 'time': 'TIMESLICE'})
    df_perfil['FUEL'] = 'DEM'
    df_perfil['YEAR'] = y[0]
    # Merge dos DataFrames
    df_final = pd.merge(df_perfil, df_demand, on='REGION')

    # Cálculo do perfil da demanda para cada REGION
    df_final['VALUE'] = df_final['VALUE']/df_final['TOTAL']

    # Converter para datetime
    df_final['TIMESLICE'] = pd.to_datetime(df_final['TIMESLICE'])
    # Calcular o dia do ano
    df_final['dia_do_ano'] = df_final['TIMESLICE'].dt.dayofyear

    # Calcular a hora do dia
    df_final['hora_do_dia'] = df_final['TIMESLICE'].dt.hour

    # Calcular a hora do ano
    df_final['TIMESLICE'] = (df_final['dia_do_ano'] - 1) * 24 + df_final['hora_do_dia'] + 1
    # Ajuste estrutura 
    ordem_colunas = ['REGION', 'FUEL', 'TIMESLICE', 'YEAR', 'VALUE']
    df_final = df_final[ordem_colunas]
    
    

    EscreveArquivo(df_final, "SpecifiedDemandProfile")


def MinStorageCharge(r,t,s,y,file_name):
    df = CarregaArquivo(file_name)
    if len(df.index) == 0:
        EscreveArquivo(df, "MinStorageCharge")
    else:
        pass


def StorageLevelStart(r,s,file_name):
    df = CarregaArquivo(file_name)
    if len(df.index) == 0:
        EscreveArquivo(df, "StorageLevelStart")
    else:
        pass

def StorageMaxChargeRate(r,s,file_name):
    df = CarregaArquivo(file_name)
    if len(df.index) == 0:
        EscreveArquivo(df, "StorageMaxChargeRate")
    else:
        pass

def StorageMaxDischargeRate(r,s,file_name):
    df = CarregaArquivo(file_name)
    if len(df.index) == 0:
        EscreveArquivo(df, "StorageMaxDischargeRate")
    else:
        pass

def TechnologyFromStorage(r,t,s,m,file_name):
    df = CarregaArquivo(file_name)
    if len(df.index) == 0:
        EscreveArquivo(df, "TechnologyFromStorage")
    else:
        pass

def TechnologyToStorage(r,t,s,m,file_name):
    df = CarregaArquivo(file_name)
    if len(df.index) == 0:
        EscreveArquivo(df, "TechnologyToStorage")
    else:
        pass

def TotalAnnualMaxCapacity(r,t,y,file_name):
    df = CarregaArquivo(file_name)
    if len(df.index) == 0:
        EscreveArquivo(df, "TotalAnnualMaxCapacity")
    else:
        pass

def TotalAnnualMaxCapacityInvestment(r,t,y,file_name):
    df = CarregaArquivo(file_name)
    if len(df.index) == 0:
        EscreveArquivo(df, "TotalAnnualMaxCapacityInvestment")
    else:
        pass

def TotalAnnualMinCapacity(r,t,y,file_name):
    df = CarregaArquivo(file_name)
    if len(df.index) == 0:
        EscreveArquivo(df, "TotalAnnualMinCapacity")
    else:
        pass

def TotalAnnualMinCapacityInvestment(r,t,y,file_name):
    df = CarregaArquivo(file_name)
    if len(df.index) == 0:
        EscreveArquivo(df, "TotalAnnualMinCapacityInvestment")
    else:
        pass

def TotalTechnologyAnnualActivityLowerLimit(r,t,y,file_name):
    df = CarregaArquivo(file_name)
    if len(df.index) == 0:
        EscreveArquivo(df, "TotalTechnologyAnnualActivityLowerLimit")
    else:
        pass

def TotalTechnologyAnnualActivityUpperLimit(r,t,y,file_name):
    df = CarregaArquivo(file_name)
    if len(df.index) == 0:
        EscreveArquivo(df, "TotalTechnologyAnnualActivityUpperLimit")
    else:
        df_filtered = df[(df['TECHNOLOGY'].isin(t)) & (df['YEAR'].isin(y))]
        regions = set(r.values())
        regions -= set(['ARGENTINA', 'PARAGUAI', 'VENEZUELA', 'URUGUAI'])
        df_final = pd.DataFrame()
        for region in regions:
            df_filtered['REGION'] = region
            df_final = pd.concat([df_final, df_filtered], ignore_index=True)
        EscreveArquivo(df_final, "TotalTechnologyAnnualActivityUpperLimit")

def TotalTechnologyModelPeriodActivityLowerLimit():
    pass

def TotalTechnologyModelPeriodActivityUpperLimit():
    pass

def TradeRoute(r,f,y,file_name):
    df = CarregaArquivo(file_name)
    if len(df.index) == 0:
        EscreveArquivo(df, "TradeRoute")
    else:
        pass

def VariableCost(r,t,m,y,file_name):
    df = CarregaArquivo(file_name)
    if len(df.index) == 0:
        EscreveArquivo(df, "VariableCost")
    else:
        df_filtered = df[(df['TECHNOLOGY'].isin(t)) & (df['MODE_OF_OPERATION'].isin(m)) & (df['YEAR'].isin(y))]
        regions = set(r.values())
        regions -= set(['ARGENTINA', 'PARAGUAI', 'VENEZUELA', 'URUGUAI'])
        df_final = pd.DataFrame()
        for region in regions:
            df_filtered['REGION'] = region
            df_final = pd.concat([df_final, df_filtered], ignore_index=True)
        EscreveArquivo(df_final, "VariableCost")

def YearSplit(t,y,file_name):
    pass



# Funções utilitárias
def CarregaArquivo(file_name):
    """Função auxiliar para leitura de arquivo de dados tratados.

    Args:
        file_name (str): Nome do arquivo de dados tratados que será lido

    Returns:
        Pandas DataFrame: Dataframe lido
    """
    path = os.path.join("DadosTratados", file_name)
    df = pd.read_csv(path)
    return df

def EscreveArquivo(df, file_name):
    """Função Auxiliar para escrita de arquivo de dados.

    Args:
        df (Pandas DataFrame): Pandas DataFrame com os dados que serão escritos em arquivo 'csv'
        file_name (str): Nome do arquivo de dados
    """
    if not os.path.exists("data"):
        os.makedirs("data")
    else:
        path = os.path.join("data", file_name + ".csv")
        df.to_csv(path, index=False)

def CarregaArquivoYAML(file_name="input.yaml"):
    """Função auxiliar para realizar a leitura de arquivo YAML

    Args:
        file_name (str, optional): Nome do arquivo YAML com configurações definidas pelo usuário.
        Padrão "input.yaml".

    Returns:
        dict: Dicionário contendo dados do arquivo YAML
    """
    path = os.path.join("DadosTratados", file_name)
    with open(path, 'r') as file:
        yaml_dict = yaml.safe_load(file)
        
        return yaml_dict

# Rotinas
if __name__ == "__main__":

    configs = CarregaArquivoYAML()
    regions = configs['regions']
    emissions = configs['emissions']
    years = configs['years']
    fuels = configs['fuels']
    techs = configs['techs']
    timeslice = False #TODO
    storages = False #TODO
    modes_of_operation = configs['modes_of_operation']

    #Escrita dos parâmetros
    AccumulatedAnnualDemand(r=regions, f=fuels, y=years, file_name="in_AccumulatedAnnualDemand.csv")
    AnnualEmissionLimit(r=regions, e=emissions, y=years, file_name="in_AnnualEmissionLimit.csv")
    AnnualExogenousEmission(r=regions,f=fuels,y=years,file_name="in_AnnualExogenousEmission.csv")
    AvailabilityFactor(r=regions,t=techs,y=years,file_name="in_AvailabilityFactor.csv")
    CapacityFactor(r=regions, t=techs, l=timeslice, y=years, on_wind="in_CapacityFactorOnWind.csv", off_wind="in_CapacityFactorOffWind.csv", solar_pv="in_CapacityFactorSolarPV.csv", other_tec="in_CapacityFactor.csv")
    CapacityOfOneTechnologyUnit(r=regions,t=techs,y=years,file_name="in_CapacityOfOneTechnologyUnit.csv")
    CapacityToActivityUnit(r=regions,t=techs,file_name="in_CapacityToActivityUnit.csv")
    CapitalCost(r=regions,t=techs,y=years,file_name="in_CapitalCost.csv")
    CapitalCostStorage(r=regions,s=storages,y=years,file_name="in_CapitalCostStorage.csv")
    DepreciationMethod(r=regions,file_name="in_DepreciationMethod.csv")
    DiscountRate(r=regions,file_name="in_DiscountRate.csv")
    DiscountRateStorage(r=regions,s=storages,file_name="in_DiscountRateStorage.csv")
    EmissionActivityRatio(r=regions,t=techs,e=emissions,m=modes_of_operation,y=years,file_name="in_EmissionActivityRatio.csv")
    EmissionsPenalty(r=regions,e=emissions,y=years,file_name="in_EmissionsPenalty.csv")
    FixedCost(r=regions,t=techs,y=years,file_name="in_FixedCost.csv")
    InputActivityRatio(r=regions,t=techs,f=fuels,m=modes_of_operation,y=years,file_name="in_InputActivityRatio.csv")
    ModelPeriodEmissionLimit(r=regions,e=emissions,file_name="in_ModelPeriodEmissionLimit.csv")
    ModelPeriodExogenousEmission(r=regions,e=emissions,file_name="in_ModelPeriodExogenousEmission.csv")
    OperationalLife(r=regions,t=techs,file_name="in_OperationalLife.csv")
    OperationalLifeStorage(r=regions,s=storages,file_name="in_OperationalLifeStorage.csv")
    OutputActivityRatio(r=regions,t=techs,f=fuels,m=modes_of_operation,y=years,file_name="in_OutputActivityRatio.csv")
    REMinProductionTarget(r=regions,y=years,file_name="in_REMinProductionTarget.csv")
    RETagFuel(r=regions,f=fuels,y=years,file_name="in_RETagFuel.csv")
    RETagTechnology(r=regions,t=techs,y=years,file_name="in_RETagTechnology.csv")
    ReserveMargin(r=regions,y=years,file_name="in_ReserveMargin.csv")
    ReserveMarginTagFuel(r=regions,f=fuels,y=years,file_name="in_ReserveMarginTagFuel.csv")
    ReserveMarginTagTechnology(r=regions,t=techs,y=years,file_name="in_ReserveMarginTagTechnology.csv")
    ResidualCapacity(r=regions,t=techs,y=years,file_name="in_ResidualCapacity.csv")
    ResidualStorageCapacity(r=regions,s=storages,y=years,file_name="in_ResidualStorageCapacity.csv")
    SpecifiedAnnualDemand(r=regions,f=fuels,y=years,file_name="in_Demand.csv")
    SpecifiedDemandProfile(r=regions,f=fuels,t=timeslice,y=years,file_name="in_Demand.csv")
    MinStorageCharge(r=regions,t=timeslice,s=storages,y=years,file_name="in_MinStorageCharge.csv")
    StorageLevelStart(r=regions,s=storages,file_name="in_StorageLevelStart.csv")
    StorageMaxChargeRate(r=regions,s=storages,file_name="in_StorageMaxChargeRate.csv")
    StorageMaxDischargeRate(r=regions,s=storages,file_name="in_StorageMaxDischargeRate.csv")
    TechnologyFromStorage(r=regions,t=techs,s=storages,m=modes_of_operation,file_name="in_TechnologyFromStorage.csv")
    TechnologyToStorage(r=regions,t=techs,s=storages,m=modes_of_operation,file_name="in_TechnologyToStorage.csv")
    TotalAnnualMaxCapacity(r=regions,t=techs,y=years,file_name="in_TotalAnnualMaxCapacity.csv")
    TotalAnnualMaxCapacityInvestment(r=regions,t=techs,y=years,file_name="in_TotalAnnualMaxCapacityInvestment.csv")
    TotalAnnualMinCapacity(r=regions,t=techs,y=years,file_name="in_TotalAnnualMinCapacity.csv")
    TotalAnnualMinCapacityInvestment(r=regions,t=techs,y=years,file_name="in_TotalAnnualMinCapacityInvestment.csv")
    TotalTechnologyAnnualActivityLowerLimit(r=regions,t=techs,y=years,file_name="in_TotalTechnologyAnnualActivityLowerLimit.csv")
    TotalTechnologyAnnualActivityUpperLimit(r=regions,t=techs,y=years,file_name="in_TotalTechnologyAnnualActivityUpperLimit.csv")
    TotalTechnologyModelPeriodActivityLowerLimit()
    TotalTechnologyModelPeriodActivityUpperLimit()
    TradeRoute(r=regions,f=fuels,y=years,file_name="in_TradeRoute.csv")
    VariableCost(r=regions,t=techs,m=modes_of_operation,y=years,file_name="in_VariableCost.csv")
    YearSplit(t=techs,y=years,file_name="in_YearSplit.csv")
    