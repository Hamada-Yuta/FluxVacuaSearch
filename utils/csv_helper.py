import numpy as np
import sympy as sp
import csv


def initialize_csv(filename, header):
    """CSVの初期化"""
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)