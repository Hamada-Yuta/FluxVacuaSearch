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
