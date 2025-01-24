.. PreparaBase documentation master file, created by
   sphinx-quickstart on Tue Apr  9 23:32:28 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Bem-vindo à documentação do Gerador de Base do SIN
======================================================

Módulo com métodos responsáveis por montar base de dados CSV, que posteriormente será convertida para arquivo datafile (input do OSeMOSYS)

Inputs: 
        - arquivos de dados tratados, que devem estar contidos no folder DadosTratados
        - arquivo de configuração YAML, que deve estar contido no folder DadosTratados

Output: 
        - folder 'data' com arquivos CSVs estruturados para o Otoole
        - arquivo datafile utilizado pelo Otoole
        
.. toctree::
   :maxdepth: 2
   :caption: Funções:

   modules

