# import_wrappers.py など
from utils.import_helpers import read_sympy_csv

def read_KZZb(i: int):
    return read_sympy_csv(f'Potential_data/KZZb_{i}.csv', kind='matrix')

def read_DZW(i: int):
    return read_sympy_csv(f'Potential_data/DZW_{i}.csv', kind='vector')

def read_DZbW(i: int):
    return read_sympy_csv(f'Potential_data/DZbW_{i}.csv', kind='vector')

def read_EK(i: int):
    return read_sympy_csv(f'Potential_data/EK_{i}.csv', kind='scalar')

def read_DtauW(i: int):
    return read_sympy_csv(f'Potential_data/DtauW_{i}.csv', kind='scalar')

def read_DtaubW(i: int):
    return read_sympy_csv(f'Potential_data/DtaubW_{i}.csv', kind='scalar')

def read_h21(i: int):
    return read_sympy_csv(f'Potential_data/h21_{i}.csv', kind='scalar')

def read_Retausol(i: int):
    return read_sympy_csv(f'SUSY_data/Retausol_{i}.csv', kind='scalar')

def read_Imtausol(i: int):
    return read_sympy_csv(f'SUSY_data/Imtausol_{i}.csv', kind='scalar')

def read_ReDtauW(i: int):
    return read_sympy_csv(f'SUSY_data/ReDtauW_{i}.csv', kind='scalar')

def read_ImDtauW(i: int):
    return read_sympy_csv(f'SUSY_data/ImDtauW_{i}.csv', kind='scalar')

# def read_ReDZW(i: int):
#     return read_sympy_csv(f'SUSY_data/ReDZW_{i}.csv', kind='vector')

# def read_ImDZW(i: int):
#     return read_sympy_csv(f'SUSY_data/ImDZW_{i}.csv', kind='vector')

# def read_DW(i: int):
#     return read_sympy_csv(f'SUSY_data/DW_{i}.csv', kind='matrix')