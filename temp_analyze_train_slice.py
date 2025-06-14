import polars as pl

# Load the dataframe
df = pl.read_parquet("/root/crypto-models/features/PEPE.parquet")

# Define feature columns (as identified from metadata.json)
feature_cols = [
    "open", "high", "low", "close", "volume", "volume_spike", "volume_momentum",
    "buy_pressure", "high_low_range", "ret_1h", "ret_4h", "ret_24h",
    "volatility_3h", "volatility_12h", "price_vs_sma12", "price_vs_sma48",
    "vwap_20", "price_vs_vwap", "ret_1", "rsi14", "bb_width", "tr", "atr14",
    "funding_rate", "open_interest", "imbalance", "btc_ret", "funding_rate_z",
    "funding_8h_avg", "oi_change_1h", "ob_imbalance", "btc_return", "btc_corr_proxy"
]

# Sort by dt if not already
df = df.sort("dt")

# Perform 80/20 chronological split for training data
n = len(df)
split_idx = int(n * 0.8)
train_df_slice = df[:split_idx]

print(f"Full Parquet Shape: {df.shape}")
print(f"Training slice shape: {train_df_slice.shape}")
print(f"Analyzing {len(feature_cols)} features in the training slice ({train_df_slice.height} rows):\n")

results = []
for col_name in feature_cols:
    if col_name in train_df_slice.columns:
        col_series = train_df_slice[col_name]
        null_count = col_series.null_count()
        unique_count = len(col_series.unique()) # Counts unique values, null is counted as one if present
        results.append({
            "feature": col_name,
            "null_count": null_count,
            "unique_count": unique_count,
            "dtype": str(col_series.dtype)
        })
    else:
        results.append({
            "feature": col_name,
            "null_count": "N/A - col not found",
            "unique_count": "N/A - col not found",
            "dtype": "N/A - col not found"
        })

print(f"{'Feature':<20} | {'Dtype':<10} | {'Null Count':<10} | {'Unique Count':<12}")
print("-" * 65)
for res in results:
    print(f"{res['feature']:<20} | {res['dtype']:<10} | {str(res['null_count']):<10} | {str(res['unique_count']):<12}")

dropped_by_lgbm_count = 0
print("\nFeatures likely dropped by LightGBM (all nulls or 1 unique value in training slice):")
for res in results:
    # A feature is problematic if all its values are null, or if it has only one unique value (constant)
    # If unique_count is 1 and null_count is also train_df_slice.height, it's an all-null column.
    # If unique_count is 1 and null_count is 0, it's a constant value column.
    # If unique_count is 1 and null_count > 0 but < height, it's a constant value + some nulls (still effectively constant for non-nulls)
    # If unique_count is 2 and null_count > 0 and (train_df_slice.height - null_count) results in one actual value, it's also problematic.
    is_all_null = (res['null_count'] == train_df_slice.height)
    is_constant = (res['unique_count'] == 1 and not is_all_null) # Constant non-null value
    is_constant_with_nulls = (res['unique_count'] == 2 and res['null_count'] > 0 and res['null_count'] < train_df_slice.height)
    
    if is_all_null or is_constant:
        print(f"- {res['feature']} (Nulls: {res['null_count']}, Uniques: {res['unique_count']})")
        dropped_by_lgbm_count +=1
    elif is_constant_with_nulls:
        # Check if the non-null values are all the same
        non_null_uniques = train_df_slice[res['feature']].drop_nulls().n_unique()
        if non_null_uniques == 1:
            print(f"- {res['feature']} (Nulls: {res['null_count']}, Uniques: {res['unique_count']}, Non-Null Uniques: {non_null_uniques}) <- Effectively constant")
            dropped_by_lgbm_count +=1
            
print(f"\nFound {dropped_by_lgbm_count} features that would likely be dropped by LightGBM in the training slice.")
