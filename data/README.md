# Adopted sample

`selected_solutions/part-00000.parquet` contains the first 648,317 rows and
`part-00001.parquet` the final 648,316 rows of the adopted Chapter 4/5 sample.
Run `python -m analysis.prepare_data` from the repository root to create the
single analysis input `data/selected_solutions.parquet`.

| Columns | Type | Meaning |
|---|---|---|
| `FunctionValue` | float64 | Recorded nonnegative potential value used in the analyses. |
| `taur`, `taui` | float64 | Real and imaginary parts of the axio-dilaton. |
| `Z1r`, `Z1i`, `Z2r`, `Z2i` | float64 | Real and imaginary parts of the two complex-structure moduli. |
| `m1`, `m2`, `m3`, `n1`, `n2`, `n3` | int16 | Six integer flux parameters. |

Total: 1,296,633 reliable solutions and 13 columns. There are 127,241 SUSY
solutions with $V < 10^{-4.5}$. All values, dtypes and row order are preserved from the adopted
sample; private workflow metadata is omitted. No new selection or numerical
rounding was applied when distributing the data. The coordinates are the saved
representatives before fundamental-domain mapping. A sample identifier in the
analysis metadata refers to the zero-based row position after concatenation.
