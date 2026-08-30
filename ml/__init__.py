"""ML pipeline: feature engineering, triple-barrier labeling, training, and inference.

This package is independent of the live rule-based screener (strategy.py /
ta_engine.py) - it operates on a historical, long-format (ticker, date) panel
rather than single-ticker "latest value" snapshots, since model training needs
a full time series of features and forward-looking labels per row.
"""
