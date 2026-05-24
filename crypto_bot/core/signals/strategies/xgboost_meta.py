"""
XGBoost Meta-Classifier (Strategy 7).

Acts as a go/no-go gate on top of Strategies 1–6. Does NOT generate signals
independently and does NOT allocate separate capital.

Usage pattern (enforced by BacktestOrchestrator):
  1. Run Strategies 1–6, collect all (timestamp, symbol, strategy, direction) signals.
  2. Train XGBoostMetaStrategy on in-sample data.
  3. For each signal from 1–6 in out-of-sample: call filter_signal() → keep or suppress.

The filter_signal() method returns True if the signal should be acted on, False to suppress.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any
from crypto_bot.core.signals.base import BaseStrategy
from crypto_bot.core.signals.models import Signal
from crypto_bot.core.signals.indicators import rsi, macd, bb_percent_b, bb_width, atr, adx


class XGBoostMetaStrategy(BaseStrategy):
    """
    XGBoost meta-classifier. Does not produce independent signals.
    Trained via train() on historical feature/label pairs.
    Used via filter_signal() at inference time.
    """

    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__(params)
        self._model = None  # set after train()

    @property
    def param_space(self) -> dict:
        return {
            "n_estimators":   (50,  300, "int"),
            "max_depth":      (3,   8,   "int"),
            "learning_rate":  (0.01, 0.3),
            "subsample":      (0.6,  1.0),
            "forward_window": (3,   12,  "int"),
        }

    def generate_signals(self, candles: pd.DataFrame, aux_data=None) -> list[Signal]:
        # XGBoost Meta does not generate independent signals.
        return []

    def build_features(self, candles: pd.DataFrame, aux_data=None) -> pd.DataFrame:
        """
        Build feature matrix from candles.
        Returns DataFrame with one row per candle, NaNs where insufficient history.
        """
        close = candles["close"]
        high  = candles["high"]
        low   = candles["low"]
        vol   = candles["volume"]

        rsi14 = rsi(close, 14)
        _, _, macd_hist = macd(close, 12, 26, 9)
        pct_b = bb_percent_b(close, 20, 2.0)
        bbw20 = bb_width(close, 20, 2.0)
        atr14 = atr(high, low, close, 14)
        adx14 = adx(high, low, close, 14)

        vol_mean = vol.rolling(20).mean().replace(0, np.nan)
        vol_zscore = (vol - vol_mean) / vol.rolling(20).std(ddof=0).replace(0, np.nan)

        funding = pd.Series(0.0, index=candles.index)
        if aux_data and "funding_rates" in aux_data:
            fr_df = aux_data["funding_rates"]
            sym = str(candles["symbol"].iloc[0])
            fr_sym = fr_df[fr_df["symbol"] == sym].sort_values("timestamp")
            if not fr_sym.empty:
                merged = pd.merge_asof(
                    candles[["timestamp"]].copy(),
                    fr_sym[["timestamp", "rate"]].rename(columns={"timestamp": "fr_ts"}),
                    left_on="timestamp", right_on="fr_ts", direction="backward",
                )
                funding = merged["rate"].fillna(0.0).values

        features = pd.DataFrame({
            "rsi14":      rsi14.values,
            "macd_hist":  macd_hist.values,
            "pct_b":      pct_b.values,
            "bbw20":      bbw20.values,
            "atr14_pct":  (atr14 / close.replace(0, np.nan)).values,
            "adx14":      adx14.values,
            "vol_zscore": vol_zscore.values,
            "funding":    funding if isinstance(funding, np.ndarray) else funding.values,
            "hour":       pd.to_datetime(candles["timestamp"]).dt.hour.values,
            "dow":        pd.to_datetime(candles["timestamp"]).dt.dayofweek.values,
        }, index=candles.index)

        return features

    def train(
        self,
        candles: pd.DataFrame,
        aux_data: dict | None = None,
    ) -> None:
        """
        Train the XGBoost model on candles.
        Label: 1 if forward_window candle return > fees threshold (0.2%), else 0.
        Uses only non-NaN rows.
        """
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost is required: pip install xgboost")

        p = self.params
        fw = int(p.get("forward_window", 6))
        fee_threshold = 0.002  # 0.2% net return hurdle

        features = self.build_features(candles, aux_data)
        close = candles["close"].values
        fwd_return = np.full(len(close), np.nan)
        for i in range(len(close) - fw):
            fwd_return[i] = (close[i + fw] - close[i]) / close[i]

        labels = (fwd_return > fee_threshold).astype(int)

        # Mask valid rows (no NaN features, no NaN labels)
        feat_matrix = features.values.astype(float)
        valid = ~np.isnan(feat_matrix).any(axis=1) & ~np.isnan(fwd_return)

        X = feat_matrix[valid]
        y = labels[valid]

        if len(X) < 50:
            return  # Not enough data to train

        xgb_kwargs = dict(
            n_estimators=int(p.get("n_estimators", 100)),
            max_depth=int(p.get("max_depth", 5)),
            learning_rate=float(p.get("learning_rate", 0.1)),
            subsample=float(p.get("subsample", 0.8)),
            eval_metric="logloss",
            verbosity=0,
            random_state=42,
        )

        # use_label_encoder was removed in newer xgboost versions
        import xgboost as xgb
        import inspect
        if "use_label_encoder" in inspect.signature(xgb.XGBClassifier.__init__).parameters:
            xgb_kwargs["use_label_encoder"] = False

        self._model = xgb.XGBClassifier(**xgb_kwargs)
        self._model.fit(X, y)

    def filter_signal(self, candle_features: pd.Series) -> bool:
        """
        Returns True if the signal should be acted on.
        Falls back to True (pass-through) if model is not trained.
        """
        if self._model is None:
            return True
        feat = candle_features.values.astype(float).reshape(1, -1)
        if np.isnan(feat).any():
            return True
        pred = self._model.predict(feat)[0]
        return bool(pred == 1)
