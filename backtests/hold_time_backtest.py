#!/usr/bin/env python3
"""Sweep hold-time horizons (30–180 min) for specialist models.
Reads only local data – production services untouched.
Usage example:
  python hold_time_backtest.py --tokens FET,BAT --hold-mins 30,60,120,180
Results CSV → /root/crypto-models/backtests/hold_time_results.csv
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import polars as pl, numpy as np  # type: ignore
import lightgbm as lgb
import joblib            # type: ignore
from tqdm import tqdm
import sys # Added for sys.path.append
import os # Added for os.getenv

# Ensure feature_engineering can be imported
ROOT_ML_ENGINE = os.getenv("ML_ENGINE_ROOT", "/root/ml-engine")
if ROOT_ML_ENGINE not in sys.path:
    sys.path.append(ROOT_ML_ENGINE)
from feature_engineering import build_features


MODELS_DIR = Path('/root/crypto-models/models')
FEATURES_DIR = Path('/root/crypto-models/features')
BUCKET_MAP_PATH = Path('/root/analytics-tool-v2/bucket_mapping.csv')
BAR_SEC = 900  # 15-min bars
FEE_PCT = 0.06 # Combined trading fee and slippage (0.03% maker/taker + 0.03% slippage = 0.06%)
BUCKET_TP_SL = {'stable':(1.0,0.3),'low':(2.0,0.6),'mid':(3.5,1.1),'high':(6.0,1.5),'ultra':(10.0,2.0)}

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _load_bucket_map() -> dict[str,str]:
    if not BUCKET_MAP_PATH.exists():
        print(f"[WARN] Bucket map not found: {BUCKET_MAP_PATH}")
        return {}
    try:
        df = pl.read_csv(BUCKET_MAP_PATH)
        return dict(zip(df['symbol'], df['bucket']))
    except Exception as e:
        print(f"[WARN] Failed to load bucket map {BUCKET_MAP_PATH}: {e}")
        return {}

def _simulate(df: pl.DataFrame, mask: np.ndarray, tp: float, sl: float, stop_bars: int):
    closes, highs, lows = (df[col].to_numpy() for col in ('close','high','low'))
    n = len(df); i = trades = wins = 0; total_pnl = 0.0
    
    if not n or not mask.any(): # No data or no entry signals
        return 0, 0.0, 0.0

    current_idx = 0
    while current_idx < n:
        if mask[current_idx]:
            entry_price = closes[current_idx]
            tp_price = entry_price * (1 + tp/100)
            sl_price = entry_price * (1 - sl/100)
            
            exit_price = closes[current_idx] # Default if no exit condition met
            exit_bar_offset = stop_bars # Assume time stop unless TP/SL hit earlier
            exited_early = False

            for j_offset in range(1, stop_bars + 1):
                trade_exit_idx = current_idx + j_offset
                if trade_exit_idx >= n: # End of data
                    exit_price = closes[n-1] # Exit at last known close
                    exit_bar_offset = j_offset -1 # exited on the last available bar before stop_bars
                    exited_early = True
                    break 
                
                if highs[trade_exit_idx] >= tp_price:
                    exit_price = tp_price
                    exit_bar_offset = j_offset
                    exited_early = True
                    break
                if lows[trade_exit_idx]  <= sl_price:
                    exit_price = sl_price
                    exit_bar_offset = j_offset
                    exited_early = True
                    break
            
            if not exited_early: # Means time stop was hit
                exit_price = closes[current_idx + stop_bars] if (current_idx + stop_bars) < n else closes[n-1]
            
            pnl_pct = (exit_price / entry_price - 1) * 100 - FEE_PCT
            if pnl_pct > 0: wins += 1
            total_pnl += pnl_pct
            trades += 1
            current_idx += exit_bar_offset # Advance main loop index
        else:
            current_idx += 1 # No entry signal, advance to next bar

    avg_pnl = total_pnl / trades if trades else 0.0
    win_rate = wins / trades if trades else 0.0
    return trades, avg_pnl, win_rate

# ------------------------------------------------------------
# Core routine
# ------------------------------------------------------------

def run_backtest(model_paths: list[Path], hold_time_bars_list: list[int], min_signal_proba: float = 0.60, results_csv_path: Path | None = None):
    bucket_map_dict = _load_bucket_map()
    all_results_rows: list[dict] = []

    print(f"Found {len(model_paths)} models to backtest.")
    print(f"Using hold times (bars): {hold_time_bars_list}")

    for model_dir_path in tqdm(model_paths, desc='Processing Models', ncols=100):
        metadata_file_path, model_lgb_file_path = model_dir_path / 'metadata.json', model_dir_path / 'model.pkl'
        
        if not (metadata_file_path.exists() and model_lgb_file_path.exists()):
            print(f"[SKIP] Missing metadata or model file in {model_dir_path}")
            continue
        
        try:
            model_metadata = json.loads(metadata_file_path.read_text())
        except json.JSONDecodeError:
            print(f"[SKIP] Invalid JSON in metadata {metadata_file_path}")
            continue

        raw_token_name = model_metadata.get('token', model_dir_path.name.split('_')[0])
        feature_file_token_name = raw_token_name.replace('USDT','')
        
        features_parquet_path = FEATURES_DIR / f'{feature_file_token_name}.parquet'
        if not features_parquet_path.exists():
            print(f"[SKIP] Features parquet not found for {feature_file_token_name} at {features_parquet_path}")
            continue

        token_bucket_from_map = bucket_map_dict.get(feature_file_token_name) # Check if token is in map
        if token_bucket_from_map and token_bucket_from_map in BUCKET_TP_SL:
            token_bucket = token_bucket_from_map
        else:
            if token_bucket_from_map: # It was in map, but not a valid bucket key in BUCKET_TP_SL
                 print(f"[WARN] Invalid bucket '{token_bucket_from_map}' for token {feature_file_token_name} found in mapping. Defaulting to 'ultra'.")
            else: # Not in map at all
                 print(f"[WARN] Token {feature_file_token_name} not found in bucket mapping. Defaulting to 'ultra'.")
            token_bucket = 'ultra'
        
        tp_target_pct, sl_target_pct = BUCKET_TP_SL[token_bucket] # This will now use a valid bucket (either from map or default 'ultra')
        
        try:
            # 1. Load raw features from parquet
            raw_feature_df = pl.read_parquet(str(features_parquet_path))

            # Ensure 'dt' (original datetime column) is present, then rename to 'datetime'
            if 'dt' not in raw_feature_df.columns:
                print(f"[SKIP] Raw feature file {features_parquet_path.name} is missing 'dt' column. Available: {raw_feature_df.columns[:10]}")
                continue
            raw_feature_df = raw_feature_df.rename({'dt': 'datetime'})

            # Add 'symbol' if missing (should ideally be in raw_feature_df)
            if 'symbol' not in raw_feature_df.columns:
                raw_feature_df = raw_feature_df.with_columns(pl.lit(feature_file_token_name).alias('symbol'))

            # 2. Build features. Assumes build_features takes df with 'datetime' and other base columns,
            # and returns a df containing all necessary columns (original + engineered).
            features_df = build_features(raw_feature_df, bar_sec=BAR_SEC)

            # 3. Validate essential columns for simulation are present in the output of build_features
            sim_cols_check = ['datetime', 'open', 'high', 'low', 'close', 'symbol']
            missing_sim_cols = [col for col in sim_cols_check if col not in features_df.columns]
            if missing_sim_cols:
                print(f"[SKIP] Post-build_features, DataFrame for {model_dir_path.name} is missing simulation columns: {missing_sim_cols}. DF Columns: {features_df.columns[:20]}")
                continue

            features_df = features_df.drop_nulls() # Drop rows with NaNs after all feature engineering
            if features_df.is_empty():
                print(f"[SKIP] DataFrame empty after build_features and drop_nulls for {model_dir_path.name}.")
                continue

            # 4. Load the model
            lgbm_model = joblib.load(model_lgb_file_path)

            # 5. Ensure features required by the model are present for prediction
            model_req_features = model_metadata['feature_cols']
            missing_model_features = [col for col in model_req_features if col not in features_df.columns]
            if missing_model_features:
                available_cols_preview = features_df.columns[:20]
                print(f"[SKIP] Model {model_dir_path.name} requires features not in final features_df: {missing_model_features}. Available: {available_cols_preview}")
                continue
            
            ordered_features_df_for_predict = features_df.select(model_req_features)
            all_probas = lgbm_model.predict_proba(ordered_features_df_for_predict)[:, 1]

        except Exception as e:
            print(f"[SKIP] Error in main data/model processing for {model_dir_path.name}: {e}")
            continue

        if features_df.is_empty():
            print(f"[SKIP] No data left after feature engineering/null drop for {model_dir_path.name}")
            continue
            
        # X_test_data is not needed here as all_probas is already calculated
        # probabilities = lgbm_model.predict(X_test_data, num_iteration=lgbm_model.best_iteration) # This was incorrect for sklearn wrapper
        
        # all_probas was calculated earlier using lgbm_model.predict_proba(ordered_features_df_for_predict)[:, 1]
        # Ensure eval_df is constructed correctly using these probabilities and the full features_df for alignment
        # We need to align all_probas (which was based on ordered_features_df_for_predict) with the full features_df
        # This assumes ordered_features_df_for_predict is a subset of features_df and maintains row order.

        # Create a temporary DataFrame from all_probas with an index to join back to features_df
        # This ensures that probabilities are correctly aligned with their corresponding rows in features_df,
        # especially after any drop_nulls operations on features_df.
        # First, get the indices from features_df that correspond to the rows used for prediction.
        # This requires that ordered_features_df_for_predict was created from features_df *after* its final null drop.
        # The current logic is: features_df.drop_nulls() -> ordered_features_df_for_predict = features_df.select(model_req_features) -> all_probas
        # This means all_probas should align with features_df if no rows were dropped *after* all_probas was computed.

        # Let's ensure features_df used for eval_df is the one that all_probas corresponds to.
        # The `ordered_features_df_for_predict` was derived from `features_df` before prediction.
        # So, `features_df` at this stage should be the correct one to select 'datetime', 'high', 'low', 'close' from.
        eval_df = features_df.select(['datetime','high','low','close','symbol']).with_columns(pl.Series('proba', all_probas))
        
        selected_signals_df = (
            eval_df.with_columns([
                pl.col('datetime').dt.date().alias('date_col'),
            ]).with_columns([
                pl.col('proba').rank('ordinal', descending=True).over('date_col').alias('daily_rank'),
                pl.count().over('date_col').alias('daily_signals_count'),
            ])
            .with_columns(((pl.col('daily_signals_count') * 0.03) + 0.999).floor().cast(pl.Int64).alias('num_to_keep_daily'))
            .filter((pl.col('daily_rank') <= pl.col('num_to_keep_daily')) & (pl.col('proba') >= min_signal_proba))
            .select(['datetime', 'high', 'low', 'close', 'proba'])
        )
        
        # Create a mask that aligns with eval_df for entries
        # This involves finding which rows of eval_df correspond to selected_signals_df
        # A common way is to use 'is_in' if datetimes are unique and sorted, or join
        # For simplicity, we'll pass selected_signals_df to _simulate and it will use all its rows.
        # The mask passed to _simulate will be for selected_signals_df itself.
        entry_signals_mask = np.ones(len(selected_signals_df), dtype=bool) 

        for hold_bars_val in hold_time_bars_list:
            if selected_signals_df.is_empty():
                num_trades, avg_pnl_pct, win_rate_val = 0, 0.0, 0.0
            else:
                num_trades, avg_pnl_pct, win_rate_val = _simulate(selected_signals_df, entry_signals_mask, tp_target_pct, sl_target_pct, hold_bars_val)
            
            all_results_rows.append({
                'model_name': model_dir_path.name, 
                'token': feature_file_token_name, 
                'hold_minutes': hold_bars_val * (BAR_SEC // 60),
                'num_trades': num_trades, 
                'avg_pnl_pct': round(avg_pnl_pct, 4), 
                'win_rate': round(win_rate_val, 4),
                'bucket': token_bucket,
                'tp_pct': tp_target_pct,
                'sl_pct': sl_target_pct
            })

    if not all_results_rows:
        print('[BACKTEST] No results generated. Check model paths, feature data, and signal criteria.')
        return

    results_df = pl.DataFrame(all_results_rows)
    
    summary_df = results_df.groupby('hold_minutes').agg([
        pl.mean('avg_pnl_pct').alias('mean_avg_pnl_pct'), 
        pl.sum('num_trades').alias('total_trades'),
        pl.median('avg_pnl_pct').alias('median_avg_pnl_pct'),
        (pl.filter(pl.col('avg_pnl_pct') > 0).count() / pl.count() * 100).alias('pct_profitable_model_hold_configs')
    ]).sort('hold_minutes')
    print("\n--- Backtest Summary (Aggregated by Hold Time) ---")
    print(summary_df)
    
    if results_csv_path:
        try:
            results_csv_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.write_csv(results_csv_path)
            print(f'\n[BACKTEST] Detailed results saved to: {results_csv_path}')
        except Exception as e:
            print(f"[ERROR] Failed to write results CSV: {e}")

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Backtest specialist models with varying hold times.")
    parser.add_argument('--tokens', type=str, help='Comma-separated list of base token symbols (e.g., FET,BAT). If omitted, runs all specialist models found.')
    parser.add_argument('--hold-mins', type=str, default='30,60,120,180', help='Comma-separated list of hold times in minutes (e.g., 30,60,120,180).')
    parser.add_argument('--min-proba', type=float, default=0.60, help='Minimum probability for a signal to be considered.')
    parser.add_argument('--output-csv', type=str, default='/root/crypto-models/backtests/hold_time_results.csv', help='Path to save detailed CSV results.')
    
    args = parser.parse_args()

    if args.tokens:
        target_tokens = {t.strip().upper() for t in args.tokens.split(',') if t.strip()}
        model_paths_to_run = [
            p for p in MODELS_DIR.iterdir() 
            if p.is_dir() and (p.name.endswith('_long') or p.name.endswith('_short')) 
               and any(token_part.replace('USDT','') in target_tokens for token_part in p.name.split('_')[:-1] if token_part)
        ]
    else: 
        model_paths_to_run = [
            p for p in MODELS_DIR.iterdir() 
            if p.is_dir() and (p.name.endswith('_long') or p.name.endswith('_short'))
        ]
    
    if not model_paths_to_run:
        print("No models found matching criteria. Exiting.")
        return

    try:
        bar_duration_mins = BAR_SEC // 60
        hold_times_in_bars = []
        for m_str in args.hold_mins.split(','):
            m = int(m_str.strip())
            if m >= bar_duration_mins and m % bar_duration_mins == 0:
                hold_times_in_bars.append(m // bar_duration_mins)
            else:
                print(f"[WARN] Hold time {m} mins is not a multiple of bar duration ({bar_duration_mins} mins) or is too short. Skipping.")
        if not hold_times_in_bars:
            raise ValueError("No valid hold times (in bars) derived from input.")
    except ValueError as e:
        print(f"Invalid --hold-mins argument: {args.hold_mins}. Error: {e}")
        return
        
    output_file_path = Path(args.output_csv)
    run_backtest(model_paths_to_run, hold_times_in_bars, args.min_proba, output_file_path)

if __name__ == '__main__':
    main()
