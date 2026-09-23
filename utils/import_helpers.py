# utils/import_helpers.py
from pathlib import Path
import csv, re, sympy as sp
from sympy.parsing.mathematica import parse_mathematica
from pathlib import Path
from typing import Union, List

#Expr = sp.Expr

# ① 置換は Regex オブジェクトをキャッシュ
_RE_PATTERNS = [
    (re.compile(r'tau1'), r'taur'),
    (re.compile(r'tau2'), r'taui'),
    (re.compile(r'Z\[(\d+)\]'), r'Z\1'),
    (re.compile(r'Zb\[(\d+)\]'), r'Zb\1'),
    (re.compile(r'Zre\[(\d+)\]'), r'Z\1r'),
    (re.compile(r'Zim\[(\d+)\]'), r'Z\1i'),
    (re.compile(r'm\[(\d+)\]'), r'm\1'),
    (re.compile(r'n\[(\d+)\]'), r'n\1'),
]
def _replace(expr: str) -> str:
    for pat, repl in _RE_PATTERNS:
        expr = pat.sub(repl, expr)
    return expr


# ② 汎用読み込み関数
def read_sympy_csv(path: Union[str, Path], kind: str = 'matrix'):
    """
    Parameters
    ----------
    path : str | Path
        CSV ファイルパス
    kind : {'scalar', 'vector', 'matrix'}
        返す構造を指定
    Returns
    -------
    sympy.Expr, list[Expr], list[list[Expr]]
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(newline='') as f:
        reader = csv.reader(f)
        if kind == 'scalar':
            row = next(reader)
            return parse_mathematica(_replace(row[0]))

        if kind == 'vector':
            row = next(reader)
            return [parse_mathematica(_replace(x)) for x in row]

        if kind == 'matrix':
            return [
                [parse_mathematica(_replace(x)) for x in row]
                for row in reader
            ]
        raise ValueError(f"unknown kind {kind}")

# def _parse_cell(cell: str) -> List[Expr]:
#     """1セルを SymPy Expr に変換し、tuple なら展開して list で返す"""
#     obj = parse_mathematica(_replace(cell))
#     if isinstance(obj, (tuple, list)):
#         return list(obj)              # 展開
#     return [obj]

# def read_sympy_csv(
#     path: Union[str, Path],
#     kind: str = 'matrix'
# ) -> Union[Expr, List[Expr], List[List[Expr]]]:

#     path = Path(path)
#     if not path.exists():
#         raise FileNotFoundError(path)

#     with path.open(newline='') as f:
#         reader = csv.reader(f)

#         if kind == 'scalar':
#             first = next(reader)
#             return _parse_cell(first[0])[0]

#         if kind == 'vector':
#             first = next(reader)
#             vec: List[Expr] = []
#             for cell in first:
#                 vec.extend(_parse_cell(cell))
#             return vec

#         if kind == 'matrix':
#             mat: List[List[Expr]] = []
#             for row in reader:
#                 parsed_row: List[Expr] = []
#                 for cell in row:
#                     parsed_row.extend(_parse_cell(cell))
#                 mat.append(parsed_row)
#             return mat

#         raise ValueError(f"unknown kind {kind!r}")

