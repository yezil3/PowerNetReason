import numpy as np
import pandas as pd
import json
import math
from pathlib import Path
import re
import os, json, re, datetime as dt
from pathlib import Path
from typing import Any, Union


def save_json(a: Union[str, Path, Any], b: Union[str, Path, Any]):
    """
    Save a Python object as pretty JSON. Creates parent dirs if needed.
    Accepts arguments in either order: (obj, path) or (path, obj).
    """
    # detect which arg is path
    if isinstance(a, (str, Path)) and not isinstance(b, (str, Path)):
        path, obj = Path(a), b
    elif isinstance(b, (str, Path)) and not isinstance(a, (str, Path)):
        path, obj = Path(b), a
    else:
        # fallback: 尽量把 a 当成 path
        path, obj = Path(a), b

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    """
    Load JSON to Python object.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_text(text, path):
    """
    Save plain text. Creates parent dirs if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def robust_z(x: np.ndarray):
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) + 1e-9
    return (x - med) / (1.4826 * mad)


def _to_1d_float(arr):
    """确保是一维 float 数组"""
    a = np.asarray(arr, dtype="float64")
    return a.ravel()


def pick_earliest_above_thresh(zone, z_thresh):
    pool = [
        c
        for c in by_zone.get(zone, [])
        if c["zscore"] >= z_thresh and not c["context_only"]
    ]
    if not pool:
        return None
    pool.sort(key=lambda x: (x["time"]))  # 字符串可直接比较或转 datetime
    return pool[0]


def _iso_time(t):
    """把 pandas/np 时间安全转成 ISO 字符串"""
    try:
        return pd.Timestamp(t).isoformat(sep=" ")
    except Exception:
        return str(t)


def robust_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Return robust z-score per column as a DataFrame with same index/columns."""
    med = df.median(axis=0, skipna=True)
    mad = (df - med).abs().median(axis=0, skipna=True)
    eps = 1e-9
    z = (df - med) / (1.4826 * (mad + eps))
    return z  # DataFrame, same index/columns as df


def build_zone_lines_from_z(
    df,
    delta_df,
    zones,
    z_thresh: float = 6.0,
    topk: int = 3,
    prefer_earliest: bool = True,
    use: str = "delta",  # "delta" 或 "raw"
):
    """
    生成:
      1) zone_lines: 供 LLM 读的字符串行
      2) picks: [(zone_id, time_idx, z_at, delta_at)]
      3) zscore_df: 各列的 robust zscore DataFrame
      4) candidates: 结构化候选列表(带 id、规范化字段)，便于后续调试/可视化/重新打分
    """
    base = delta_df if use == "delta" else df
    zscore_df = robust_zscore(base)
    times = base.index

    zone_lines = []
    picks = []
    candidates = []  # ← 新增

    def _fmt_time(t):
        return t.strftime("%Y-%m-%d %H:%M:%S") if hasattr(t, "strftime") else str(t)

    for zi, cols in enumerate(zones, start=1):
        cols_in = [c for c in cols if c in zscore_df.columns]
        if not cols_in:
            zone_lines.append(f"Zone {zi}: none")
            continue

        z_block = zscore_df[cols_in].to_numpy(dtype=float)
        z_zone = np.nanmax(np.abs(z_block), axis=1)

        if delta_df is not None:
            d_block = delta_df[cols_in].to_numpy(dtype=float)
            delta_zone = np.nanmax(np.abs(d_block), axis=1)
        else:
            delta_zone = np.zeros_like(z_zone)

        pass_idx = np.where(z_zone >= float(z_thresh))[0]
        if pass_idx.size == 0:
            zone_lines.append(f"Zone {zi}: none")
            continue

        if prefer_earliest:
            j0 = int(pass_idx.min())
            chosen = [j0]
            remain = [j for j in pass_idx if j != j0]
            remain.sort(key=lambda j: (-abs(z_zone[j]), j))
            chosen.extend(remain[: max(0, topk - 1)])
        else:
            chosen = list(pass_idx)
            chosen.sort(key=lambda j: (-abs(z_zone[j]), j))
            chosen = chosen[:topk]

        chosen.sort()  # 展示时按时间升序
        for j in chosen:
            t = times[j]
            t_str = _fmt_time(t)
            z_at = float(z_zone[j])
            d_at = float(delta_zone[j])

            # ① 文本行（给 LLM）
            line = (
                f"Zone {zi}: {list(cols_in)}:  "
                f"t = {t_str}  ->  zscore = {z_at:.2f}  Δ = {d_at:.2f}%"
            )
            zone_lines.append(line)

            # ② 调试简表
            picks.append((zi, j, z_at, d_at))
            z_at_t = zscore_df.loc[t_str, cols_in].abs()  # 只看这个 zone 的列
            bus_name = z_at_t.idxmax()  # 列名，例如 "Appliance3"
            bus_idx = int(df.columns.get_loc(bus_name))  # 在 df.columns 中的整数位置
            # ③ 结构化候选（给你做更精细的选择或可视化）
            candidates.append(
                {
                    "id": f"Z{zi}-{t_str.replace(' ', 'T')}",
                    "zone_id": int(zi),
                    # 新增：具体触发该候选的通道
                    "bus_name": bus_name,  # 例如 "Appliance3"
                    "bus_idx": bus_idx,  # 例如 7（0-based）
                    # 保留你原有的字段
                    "buses": list(cols_in),
                    "time_iso": t_str,
                    "zscore": float(
                        z_at
                    ),  # 你原来计算的候选 z（可用 z_at_t.max() 也行）
                    "delta_pct": float(d_at),  # 已是数值，展示时再加 %
                    "source": use,
                    "passed": True,
                    "threshold": float(z_thresh),
                }
            )

    return zone_lines, picks, zscore_df, candidates


def compare_llm_vs_gt(llm_json: dict, gt: dict, time_tolerance_steps: int = 1):
    """返回三个布尔：zone/bus/time 是否匹配"""

    def norm_zone(x):
        try:
            return int(x) if x is not None else None
        except:
            return None

    def norm_bus(x):
        try:
            return int(x) if x is not None else None
        except:
            return None

    z_ok = norm_zone(llm_json.get("initial_zone")) == norm_zone(gt.get("initial_zone"))
    b_ok = norm_bus(llm_json.get("initial_bus")) == norm_bus(gt.get("initial_bus"))

    # 时间：允许 ±k 步的容差（既支持整数步，也支持时间戳字符串）
    lt, gt_t = llm_json.get("initial_time"), gt.get("initial_time")
    try:
        lt_ts = pd.to_datetime(str(lt))
        gt_ts = pd.to_datetime(str(gt_t))
        # 推断一个时间步长
        step = None
        # 这里使用 delta.index 的频率并不总能拿到；退化到 1 步
        step_ok = abs((lt_ts - gt_ts)) <= pd.Timedelta(minutes=1e9)  # 占位，下面替换
        # 实际容差：直接比较“是否相等”或字符串是否一致；再退化为近似比较
        t_ok = (str(lt) == str(gt_t)) or (
            abs((lt_ts - gt_ts)) <= pd.Timedelta(seconds=60 * time_tolerance_steps)
        )
    except Exception:
        t_ok = str(lt) == str(gt_t)

    return z_ok, b_ok, t_ok


def _ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


# ===== Debug helpers =====
DEBUG = True  # 一键开关


def debug_dump(save_dir, name, content, is_json=False):
    if not DEBUG:
        return
    _ensure_dir(save_dir)
    p = Path(save_dir) / name
    if is_json:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(content, f, indent=2, ensure_ascii=False)
    else:
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)


def coerce_llm_json(raw_text):
    """尽量从 LLM 文本中抽取第一个 {...} JSON；并做 key 规范化、缺省填充。"""
    # 抓第一个大括号 JSON
    m = re.search(r"\{.*\}", raw_text, re.S)
    js = {}
    if m:
        try:
            js = json.loads(m.group(0))
        except Exception:
            js = {}
    # 宽松地兼容字段名
    if "initial_anomaly_source" in js and isinstance(
        js["initial_anomaly_source"], dict
    ):
        src = js["initial_anomaly_source"]
        js = {
            "initial_zone": src.get("zone"),
            "initial_bus": src.get("bus_id") or src.get("bus"),
            "initial_time": src.get("time"),
            "propagation": js.get("propagation"),
            "root_cause": js.get("root_cause"),
            "recommendation": js.get("recommendation", []),
        }
    # 保证三个关键键存在
    for k in ["initial_zone", "initial_bus", "initial_time"]:
        js.setdefault(k, None)
    return js


def compress_ranges(numbers):
    numbers = sorted(set(numbers))
    ranges = []
    start = end = None
    for n in numbers:
        if start is None:
            start = end = n
        elif n == end + 1:
            end = n
        else:
            ranges.append((start, end))
            start = end = n
    if start is not None:
        ranges.append((start, end))
    return ", ".join(f"{s}" if s == e else f"{s}–{e}" for s, e in ranges)


def robust_z_s(x):
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    mad = mad if mad > 1e-12 else 1e-12
    return 0.6745 * (x - med) / mad


# utils.py
import json
from datetime import datetime


def pick_initial_from_candidates(path, z_thresh=6.0):
    import json, datetime as dt

    with open(path, "r") as f:
        C = json.load(f)

    # 过滤：通过阈值
    C = [
        c
        for c in C
        if c.get("passed")
        and float(c.get("threshold", z_thresh)) <= float(c.get("zscore", 0))
    ]

    # 规则：选 “**全体候选中** 时间最早 的那条”；若同一时间多条，则选 zone_id 最小
    C.sort(key=lambda c: (dt.datetime.fromisoformat(c["time_iso"]), int(c["zone_id"])))

    if not C:
        return {
            "initial_zone": None,
            "initial_time": None,
            "zscore": None,
            "delta_pct": None,
        }

    c = C[0]
    return {
        "initial_zone": int(c["zone_id"]),
        "initial_time": c["time_iso"],  # ← 只放纯时间
        "zscore": float(c.get("zscore", 0.0)),
        "delta_pct": float(c.get("delta_pct", 0.0)),
    }


def detect_time_column(df: pd.DataFrame) -> str:
    """Pick a datetime-like column (case-insensitive heuristic)."""
    candidates = [c for c in df.columns if re.search(r"time|date", str(c), re.I)]
    for c in candidates + list(df.columns):
        try:
            pd.to_datetime(df[c], errors="raise")
            return c
        except Exception:
            continue
    # fallback: create synthetic time index
    df["_idx_time_"] = np.arange(len(df))
    return "_idx_time_"


def load_clean_csv(path: str, resample: str | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    tcol = detect_time_column(df)
    # Parse time
    if tcol != "_idx_time_":
        df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
        df = df.dropna(subset=[tcol]).sort_values(tcol)
        df = df.set_index(tcol)
        if resample:
            # numeric-only resample with mean
            num = df.select_dtypes(include=[np.number])
            df = num.resample(resample).mean().dropna(how="all")
    else:
        # simple integer index
        df = df.set_index(tcol)

    # Keep only numeric columns as "buses"
    buses = df.select_dtypes(include=[np.number]).copy()
    # Remove constant columns to avoid NaN percent changes
    const_cols = [c for c in buses.columns if buses[c].nunique(dropna=True) <= 1]
    buses = buses.drop(columns=const_cols, errors="ignore")
    return buses


def _times_as_str(index_like):
    """把 DatetimeIndex/一般索引统一成字符串数组，便于安全索引与展示。"""
    if isinstance(index_like, pd.DatetimeIndex):
        return index_like.strftime("%Y-%m-%d %H:%M:%S").to_numpy()
    # 其他类型直接 str 化
    return np.array([str(x) for x in index_like])


def compute_delta(buses: pd.DataFrame, mode: str = "pct") -> pd.DataFrame:
    """
    Return absolute delta per step:
      - pct: abs(pct_change)*100  (percent)
      - abs: abs(diff)
      - z:   abs(zscore of diff)
    """
    if mode == "pct":
        d = buses.pct_change().abs() * 100.0
    elif mode == "abs":
        d = buses.diff().abs()
    elif mode == "z":
        diff = buses.diff()
        mu = diff.mean()
        sd = diff.std().replace(0, np.nan)
        d = ((diff - mu) / sd).abs()
    else:
        raise ValueError("mode must be one of {'pct','abs','z'}")
    d = d.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return d


def split_zones(bus_names: list[str], num_zones: int) -> list[list[str]]:
    """Evenly split bus list into zones."""
    n = len(bus_names)
    num_zones = max(1, min(num_zones, n))
    k = math.ceil(n / num_zones)
    zones = [bus_names[i : i + k] for i in range(0, n, k)]
    return zones


def top_event_per_zone(delta: pd.DataFrame, zone_buses: list[str]):
    """Find (bus, time_index, value) with the largest Δ within this zone."""
    sub = delta[zone_buses]
    # locate max absolute value
    idx = np.unravel_index(np.nanargmax(sub.values), sub.shape)
    t_idx = sub.index[idx[0]]
    bus = sub.columns[idx[1]]
    val = sub.values[idx]
    return bus, t_idx, float(val)


def zone_prompt_text(bus, t, val, unit_label: str) -> str:
    # Show time as index if not datetime; if datetime, show integer step and iso time
    if isinstance(t, (np.datetime64, pd.Timestamp)):
        t_disp = f"{t}"
    else:
        t_disp = f"{t}"
    # Example line style used by your previous pipeline
    return f"Bus {bus}:  t = {t_disp} → Δ = {val:.2f}{unit_label}"


def build_structure_prompt(net: pp.pandapowerNet) -> str:
    plant_buses = set(net.gen["bus"])
    load_buses = set(net.load["bus"])
    substation_buses = set(net.bus.index) - plant_buses - load_buses

    lines = net.line[["from_bus", "to_bus"]]
    adjacency = {b: set() for b in net.bus.index}
    for _, row in lines.iterrows():
        adjacency[row["from_bus"]].add(row["to_bus"])
        adjacency[row["to_bus"]].add(row["from_bus"])

    out = []
    for sb in sorted(substation_buses):
        neighbors = adjacency[sb]
        loads = sorted(b for b in neighbors if b in load_buses)
        plants = sorted(b for b in neighbors if b in plant_buses)

        entry = f"- Substation {sb}:"
        if loads:
            entry += f"\n    - Connected Loads: {compress_ranges(loads)}"
        if plants:
            entry += f"\n    - Connected Plants: {', '.join(map(str, plants))}"
        if not loads and not plants:
            entry += "\n    - No direct connections"
        out.append(entry)

    return "\n".join(out)


def summarize_voltage_data(sim, llm, tokenizer, delta_th=0.3):
    vm_df = sim.data_vm.rename(columns=lambda i: f"bus_{i}")
    va_df = sim.data_va.rename(columns=lambda i: f"bus_{i}")

    print("🔍 Summarizing Voltage Magnitude (vm_pu)...")
    vm_summary = summarize_columns_with_llm(
        vm_df, vm_df.columns.tolist(), llm, tokenizer, delta_th
    )

    print("\n🔍 Summarizing Voltage Angle (va_degree)...")
    va_summary = summarize_columns_with_llm(
        va_df, va_df.columns.tolist(), llm, tokenizer, delta_th
    )

    return vm_summary, va_summary


def summarize_columns_with_llm(df, columns, llm, tokenizer, delta_th: float = 0.3):
    results = {}
    device = llm.model.device if hasattr(llm, "model") else llm.device

    for col in columns:
        stats = summarize_series_with_delta(df[col], delta_th)
        prompt = (
            "You are a power‑system time‑series analyst.\n"
            "Return a single concise sentence (max 25 words) describing trend and anomalies, using the JSON stats below.\n"
            f"Column: {col}\n"
            f"Stats: {json.dumps(stats)}"
        )
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        out = llm.generate(
            **inputs, max_new_tokens=64, pad_token_id=tokenizer.eos_token_id
        )
        resp = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        results[col] = resp
        print(f"[{col}] {resp}")
    return results


def summarize_series_with_delta(series: pd.Series, delta_th: float = 0.3) -> dict:
    v = series.values.astype(float)
    diff = np.diff(v)
    sharp_idx = np.where(np.abs(diff) > delta_th)[0]

    slope = np.polyfit(np.arange(len(v)), v, 1)[0] if len(v) > 1 else 0.0
    if slope > 0.1:
        trend = "increasing"
    elif slope < -0.1:
        trend = "decreasing"
    else:
        trend = "stable"

    return {
        "mean": round(float(v.mean()), 3),
        "min": round(float(v.min()), 3),
        "max": round(float(v.max()), 3),
        "trend": trend,
        f"n_sharp_changes(>|{delta_th:.2f}|)": int(len(sharp_idx)),
        "sharp_idx_sample": sharp_idx[:5].tolist(),
    }


def summarize_system_level(llm, tokenizer, structure_prompt, vm_summary, va_summary):
    def clean(text):
        return text.split("ASSISTANT:")[-1].strip()

    vm_lines = "\n".join(f"{k} (vm_pu): {clean(v)}" for k, v in vm_summary.items())
    va_lines = "\n".join(f"{k} (va_deg): {clean(v)}" for k, v in va_summary.items())

    prompt = (
        "You are a power‑grid reliability expert.\n"
        "Based on the voltage behavior of each bus and the structure of the grid, "
        "identify major failures, likely root causes, and possible propagation paths.\n\n"
        f"Structure:\n{structure_prompt.strip()}\n\n"
        f"Voltage Magnitude Summary:\n{vm_lines}\n\n"
        f"Voltage Angle Summary:\n{va_lines}"
    )

    device = llm.model.device if hasattr(llm, "model") else llm.device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    output = llm.generate(
        **inputs, max_new_tokens=300, pad_token_id=tokenizer.eos_token_id
    )
    result = tokenizer.decode(output[0], skip_special_tokens=True).strip()

    print("\n================ SYSTEM‑LEVEL SUMMARY ================\n")
    print(result)
    return result
