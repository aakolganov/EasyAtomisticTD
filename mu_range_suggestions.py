from EasyAtomisticTD_functions.TD_data import Chem_pot

min_p = 10

max_p = 100

T = 370

Compound = 'H2O'

chpt_gas_min = Chem_pot(1000+273,min_p, Compound)

chpt_gas_max = Chem_pot(1000+273,max, Compound)

