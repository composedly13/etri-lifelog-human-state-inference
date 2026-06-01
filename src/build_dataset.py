# -*- coding: utf-8 -*-
"""
데이터셋 빌더 (src/build_dataset.py)

대회 원본 구조:
  - data/raw/ch2025_data_items/ch2025_*.parquet : 라이프로그 12종 (subject_id, timestamp, ...)
  - data/raw/ch2026_metrics_train.csv           : 라벨 (subject_id, sleep_date, lifelog_date, Q1~S4)
  - data/raw/ch2026_submission_sample.csv        : 제출 대상 (동일 키, 타깃은 placeholder)

이 모듈은 12종 센서를 (subject_id, date) 단위 **일별 피처**로 집계한 뒤,
(subject_id == subject_id) & (date == lifelog_date) 로 라벨/제출 행에 join하여
모델이 바로 쓰는 train / test 테이블을 만든다.

결과는 data/processed/daily_features.parquet 에 캐시되어 재실행이 빠르다.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple

ID = "subject_id"
DATE = "date"            # 일별 집계 키 (lifelog_date와 매칭)

# 라벨/제출 CSV의 키 컬럼과 타깃
KEY_COLS    = ["subject_id", "sleep_date", "lifelog_date"]
TARGET_COLS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]


# ──────────────────────────────────────────────────────────────────────────────
# 공통 유틸
# ──────────────────────────────────────────────────────────────────────────────

def _add_date(df: pd.DataFrame) -> pd.DataFrame:
    """timestamp → 'YYYY-MM-DD' 문자열 date 컬럼 추가 (라벨의 lifelog_date와 동일 포맷)."""
    df = df.copy()
    df[DATE] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d")
    return df


def _agg_scalar(df: pd.DataFrame, value_col: str, prefix: str,
                aggs: List[str]) -> pd.DataFrame:
    """스칼라 수치 컬럼을 (subject_id, date) 단위로 집계."""
    df = _add_date(df)
    g = df.groupby([ID, DATE])[value_col].agg(aggs)
    g.columns = [f"{prefix}_{a}" for a in aggs]
    return g.reset_index()


# ──────────────────────────────────────────────────────────────────────────────
# 센서별 일별 집계기
# ──────────────────────────────────────────────────────────────────────────────

def agg_hr(path: Path) -> pd.DataFrame:
    """wHr: heart_rate가 레코드마다 HR 값 배열 → 레코드 reduce 후 일별 집계."""
    df = _add_date(pd.read_parquet(path))
    arr = df["heart_rate"].to_numpy()
    rec_mean = np.array([np.mean(a) if len(a) else np.nan for a in arr])
    rec_min  = np.array([np.min(a)  if len(a) else np.nan for a in arr])
    rec_max  = np.array([np.max(a)  if len(a) else np.nan for a in arr])
    tmp = pd.DataFrame({ID: df[ID], DATE: df[DATE],
                        "m": rec_mean, "mn": rec_min, "mx": rec_max})
    g = tmp.groupby([ID, DATE]).agg(
        hr_mean=("m", "mean"), hr_std=("m", "std"),
        hr_min=("mn", "min"), hr_max=("mx", "max"),
        hr_records=("m", "size"),
    )
    return g.reset_index()


def agg_pedo(path: Path) -> pd.DataFrame:
    """wPedo: 보행 수치 다수 → 합계/평균/최대 일별 집계."""
    df = _add_date(pd.read_parquet(path))
    g = df.groupby([ID, DATE]).agg(
        pedo_step_sum=("step", "sum"),
        pedo_dist_sum=("distance", "sum"),
        pedo_cal_sum=("burned_calories", "sum"),
        pedo_run_sum=("running_step", "sum"),
        pedo_walk_sum=("walking_step", "sum"),
        pedo_speed_mean=("speed", "mean"),
        pedo_speed_max=("speed", "max"),
        pedo_freq_mean=("step_frequency", "mean"),
        pedo_records=("step", "size"),
    )
    return g.reset_index()


def agg_binary(path: Path, col: str, prefix: str) -> pd.DataFrame:
    """0/1 상태 컬럼(m_charging, m_screen_use): 비율/합계/개수."""
    return _agg_scalar(pd.read_parquet(path), col, prefix, ["mean", "sum", "size"])


def agg_float(path: Path, col: str, prefix: str) -> pd.DataFrame:
    """연속 수치(m_light, w_light): mean/std/min/max/개수."""
    return _agg_scalar(pd.read_parquet(path), col, prefix,
                       ["mean", "std", "min", "max", "size"])


def agg_activity(path: Path) -> pd.DataFrame:
    """m_activity: 활동 코드 → 개수/고유수/평균/표준편차 (코드 분포의 거친 요약)."""
    df = _add_date(pd.read_parquet(path))
    g = df.groupby([ID, DATE])["m_activity"].agg(
        ["size", "nunique", "mean", "std"])
    g.columns = ["act_records", "act_nunique", "act_mean", "act_std"]
    return g.reset_index()


def agg_gps(path: Path) -> pd.DataFrame:
    """m_gps: 좌표 dict 배열 → 속도/이동범위/포인트 수."""
    df = _add_date(pd.read_parquet(path))
    def per(v):
        if v is None or len(v) == 0:
            return (np.nan, np.nan, np.nan, np.nan, 0)
        sp  = np.array([d.get("speed", np.nan) for d in v], dtype=float)
        lat = np.array([d.get("latitude", np.nan) for d in v], dtype=float)
        lon = np.array([d.get("longitude", np.nan) for d in v], dtype=float)
        return (np.nanmean(sp), np.nanmax(sp), np.nanmean(lat), np.nanmean(lon), len(v))
    arr = [per(v) for v in df["m_gps"].to_numpy()]
    rec = pd.DataFrame(arr, columns=["sp_m", "sp_x", "lat", "lon", "n"])
    rec[ID] = df[ID].to_numpy(); rec[DATE] = df[DATE].to_numpy()
    g = rec.groupby([ID, DATE]).agg(
        gps_speed_mean=("sp_m", "mean"), gps_speed_max=("sp_x", "max"),
        gps_lat_std=("lat", "std"), gps_lon_std=("lon", "std"),
        gps_points=("n", "sum"), gps_records=("n", "size"),
    )
    return g.reset_index()


def agg_ambience(path: Path) -> pd.DataFrame:
    """m_ambience: [label, prob] 배열 → 태그 수/최고확률."""
    df = _add_date(pd.read_parquet(path))
    def per(v):
        if v is None or len(v) == 0:
            return (0, np.nan)
        probs = np.array([float(t[1]) for t in v], dtype=float)
        return (len(v), float(np.max(probs)))
    arr = [per(v) for v in df["m_ambience"].to_numpy()]
    rec = pd.DataFrame(arr, columns=["ntag", "top"])
    rec[ID] = df[ID].to_numpy(); rec[DATE] = df[DATE].to_numpy()
    g = rec.groupby([ID, DATE]).agg(
        amb_tags_mean=("ntag", "mean"), amb_topprob_mean=("top", "mean"),
        amb_records=("ntag", "size"),
    )
    return g.reset_index()


def agg_devices(path: Path, col: str, prefix: str) -> pd.DataFrame:
    """m_ble / m_wifi: {..., rssi} dict 배열 → 기기수/평균 rssi."""
    df = _add_date(pd.read_parquet(path))
    def per(v):
        if v is None or len(v) == 0:
            return (0, np.nan)
        rssi = np.array([d.get("rssi", np.nan) for d in v], dtype=float)
        return (len(v), np.nanmean(rssi))
    arr = [per(v) for v in df[col].to_numpy()]
    rec = pd.DataFrame(arr, columns=["n", "rssi"])
    rec[ID] = df[ID].to_numpy(); rec[DATE] = df[DATE].to_numpy()
    g = rec.groupby([ID, DATE]).agg(
        **{f"{prefix}_count_mean": ("n", "mean"),
           f"{prefix}_rssi_mean":  ("rssi", "mean"),
           f"{prefix}_records":    ("n", "size")}
    )
    return g.reset_index()


def agg_usage(path: Path) -> pd.DataFrame:
    """m_usage_stats: {app_name, total_time} 배열 → 총 사용시간/앱 수."""
    df = _add_date(pd.read_parquet(path))
    def per(v):
        if v is None or len(v) == 0:
            return (0, 0.0)
        tt = np.array([d.get("total_time", 0) for d in v], dtype=float)
        return (len(v), float(np.sum(tt)))
    arr = [per(v) for v in df["m_usage_stats"].to_numpy()]
    rec = pd.DataFrame(arr, columns=["napp", "ttime"])
    rec[ID] = df[ID].to_numpy(); rec[DATE] = df[DATE].to_numpy()
    g = rec.groupby([ID, DATE]).agg(
        usage_app_mean=("napp", "mean"), usage_time_sum=("ttime", "sum"),
        usage_records=("napp", "size"),
    )
    return g.reset_index()


# ──────────────────────────────────────────────────────────────────────────────
# 일별 피처 테이블 빌드 (캐시)
# ──────────────────────────────────────────────────────────────────────────────

def build_daily_features(raw_dir: Path, cache_path: Path,
                         use_cache: bool = True) -> pd.DataFrame:
    """12종 센서를 (subject_id, date) 단위로 집계·병합한 일별 피처 테이블."""
    if use_cache and cache_path.exists():
        print(f"[build_dataset] 캐시 로드: {cache_path}")
        return pd.read_parquet(cache_path)

    items = raw_dir / "ch2025_data_items"
    print("[build_dataset] 일별 피처 집계 시작 (12종)...")

    tables = [
        ("wHr",          agg_hr(items / "ch2025_wHr.parquet")),
        ("wPedo",        agg_pedo(items / "ch2025_wPedo.parquet")),
        ("mACStatus",    agg_binary(items / "ch2025_mACStatus.parquet", "m_charging", "charge")),
        ("mScreenStatus",agg_binary(items / "ch2025_mScreenStatus.parquet", "m_screen_use", "screen")),
        ("mLight",       agg_float(items / "ch2025_mLight.parquet", "m_light", "mlight")),
        ("wLight",       agg_float(items / "ch2025_wLight.parquet", "w_light", "wlight")),
        ("mActivity",    agg_activity(items / "ch2025_mActivity.parquet")),
        ("mGps",         agg_gps(items / "ch2025_mGps.parquet")),
        ("mAmbience",    agg_ambience(items / "ch2025_mAmbience.parquet")),
        ("mBle",         agg_devices(items / "ch2025_mBle.parquet", "m_ble", "ble")),
        ("mWifi",        agg_devices(items / "ch2025_mWifi.parquet", "m_wifi", "wifi")),
        ("mUsageStats",  agg_usage(items / "ch2025_mUsageStats.parquet")),
    ]

    daily = None
    for name, t in tables:
        print(f"  - {name:13s}: {t.shape[0]:>6d} (subject,date) rows, {t.shape[1]-2} feats")
        daily = t if daily is None else daily.merge(t, on=[ID, DATE], how="outer")

    daily = daily.sort_values([ID, DATE]).reset_index(drop=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(cache_path, index=False)
    print(f"[build_dataset] 일별 피처 완료: {daily.shape} → 캐시 저장 {cache_path}")
    return daily


# ──────────────────────────────────────────────────────────────────────────────
# train / test 테이블 구성
# ──────────────────────────────────────────────────────────────────────────────

def build_dataset(raw_dir: str = "data/raw",
                  cache_dir: str = "data/processed",
                  use_cache: bool = True
                  ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """
    Returns:
        train       : 라벨 + 일별 피처 (KEY_COLS + TARGET_COLS + feature_cols)
        test        : 제출 대상 + 일별 피처 (KEY_COLS + feature_cols)
        submission  : 원본 제출 샘플 (컬럼 순서 기준)
        target_cols : TARGET_COLS
        feature_cols: 일별 피처 컬럼명
    """
    raw   = Path(raw_dir)
    cache = Path(cache_dir) / "daily_features.parquet"

    daily = build_daily_features(raw, cache, use_cache=use_cache)
    feature_cols = [c for c in daily.columns if c not in (ID, DATE)]

    labels = pd.read_csv(raw / "ch2026_metrics_train.csv")
    sub    = pd.read_csv(raw / "ch2026_submission_sample.csv")

    # (subject_id, lifelog_date) ↔ (subject_id, date) 로 일별 피처 join
    def attach(df: pd.DataFrame) -> pd.DataFrame:
        out = df.merge(
            daily.rename(columns={DATE: "lifelog_date"}),
            on=["subject_id", "lifelog_date"], how="left",
        )
        return out

    train = attach(labels)
    test  = attach(sub)

    # 피처가 하나도 매칭 안 된 행 비율 점검
    miss_tr = train[feature_cols].isna().all(axis=1).mean()
    miss_te = test[feature_cols].isna().all(axis=1).mean()
    print(f"[build_dataset] train {train.shape} (피처 전부결측 {miss_tr*100:.1f}%) | "
          f"test {test.shape} (피처 전부결측 {miss_te*100:.1f}%)")
    print(f"[build_dataset] feature_cols: {len(feature_cols)}개")

    return train, test, sub, TARGET_COLS, feature_cols


# ── 독립 실행 테스트 ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    train, test, sub, target_cols, feature_cols = build_dataset()
    print("\n[train head]")
    print(train[KEY_COLS + target_cols].head().to_string(index=False))
    print(f"\nfeature_cols ({len(feature_cols)}): {feature_cols}")
