# Reference results

These numerical summaries, software versions, model settings and sample split
indices correspond to Chapters 4 and 5 of the current manuscript. They use
the distributed sample of 1,296,633 reliable solutions and Python 3.11.3.
Both potential and moduli regression use the full sample. No individual
predictions or fitted model weights are required here.

NPZ arrays are read with `numpy.load`. Sample indices refer to the zero-based
positions in the concatenated Parquet input. See the main README and the
array-writing code in each analysis entry point for the individual arrays.
Preserve dataset order when reproducing these splits.

The five classification input comparisons include Brier scores. Ranking
summaries give selection counts, recall and precision. Regression summaries
report R-squared and MAE for the transformed targets in Tables 10 and 11.
Kendall correlation CSVs support the qualitative statement in Section 4.3;
there is no corresponding figure.

Reproduction concerns the analyses of the supplied sample, not a new flux
scan or reconstruction of the sample from the original minimization endpoints.
