"""Prepotential correction through total q-degree seven, including multicovers.

Integer instanton numbers: Candelas, Font, Katz and Morrison, hep-th/9403187,
Table 7.1 (PDF page 46). Row a, column b corresponds to q1**a q2**b.
The sign convention is F_inst = -sum(n_ab Li_3(q1**a q2**b))/(2*pi*i)**3.
The cutoff is applied AFTER expanding the multicovers, not to a sum of full
polylogarithms. No cached solution diagnostics or previous results are read.
"""
from fractions import Fraction
from math import gcd

import numpy as np
from scipy.special import zeta

GV_ROWS = {
    0: [0, 3, -6, 27, -192, 1695, -17064, 188454],
    1: [540, -1080, 2700, -17280, 154440, -1640520, 19369800],
    2: [540, 143370, -574560, 5051970, -57879900, 751684050],
    3: [540, 204071184, 74810520, -913383000, 13593850920],
    4: [540, 21772947555, -49933059660, 224108858700],
    5: [540, 1076518252152, 7772494870800],
    6: [540, 33381348217290],
    7: [540],
}


def taylor_coefficients():
    """Return exact rational c_ab in F_inst=sum(c_ab q1^a q2^b)/(2*pi*i)^3."""
    coefficients = {}
    for a in range(8):
        for b in range(8-a):
            if a+b:
                coefficients[a, b] = sum(
                    (-Fraction(GV_ROWS[a//k][b//k], k**3)
                     for k in range(1, gcd(a, b)+1)
                     if a % k == 0 and b % k == 0), Fraction(0))
    assert len(coefficients) == 35
    # These are the five coefficients printed explicitly in the manuscript.
    assert [coefficients[k] for k in [(1,0), (0,1), (2,0), (1,1), (0,2)]] == [
        -540, -3, Fraction(-1215, 2), 1080, Fraction(45, 8)]
    return coefficients


def polynomial_prepotential(z1, z2):
    """Manuscript polynomial, including the perturbative zeta(3) constant."""
    return (-(9*z1**3+9*z1*z1*z2+3*z1*z2*z2)/6
            +(9*z1*z1+6*z1*z2)/4+(17*z1+6*z2)/4
            -270j*zeta(3)/(2*np.pi)**3)


def correction_ratio(frame, batch_size=65536):
    """Compute |F_inst^(<=7)|/|F_poly| at every unchanged saved coordinate.

    Exponentially small terms may underflow to zero in float64 at very large
    imaginary coordinates. This preserves their location in the first histogram
    bin; it is not a claim that the exact correction vanishes.
    """
    coefficients = taylor_coefficients()
    values = frame[['Z1r', 'Z1i', 'Z2r', 'Z2i']].to_numpy(dtype=float)
    result = np.empty(len(values), dtype=float)
    for lo in range(0, len(values), batch_size):
        q = values[lo:lo+batch_size]
        z1, z2 = q[:, 0]+1j*q[:, 1], q[:, 2]+1j*q[:, 3]
        poly = polynomial_prepotential(z1, z2)
        q1, q2 = np.exp(2j*np.pi*z1), np.exp(2j*np.pi*z2)
        correction = np.zeros(len(q), dtype=complex)
        for (a, b), coefficient in coefficients.items():
            correction += float(coefficient)*q1**a*q2**b/(2j*np.pi)**3
        if not np.isfinite(poly).all() or not np.all(np.abs(poly) > 0):
            raise ValueError('Nonfinite or vanishing polynomial prepotential')
        result[lo:lo+len(q)] = np.abs(correction)/np.abs(poly)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite instanton correction ratio')
    return result


def coefficient_records():
    return [{'a': a, 'b': b, 'integer_instanton_number': GV_ROWS[a][b],
             'taylor_numerator': c.numerator, 'taylor_denominator': c.denominator}
            for (a, b), c in taylor_coefficients().items()]
