# main.py
# Run REDD/REFIT-style cleaned CSV (e.g., CLEAN_House1.csv) through the LLM anomaly detector.
# It auto-detects the time column and treats each numeric column as a "bus".

import argparse, os, re, json, random
from typing import List, Dict, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import networkx as nx

from utils import *  # 依赖：load_clean_csv / compute_delta / split_zones / build_zone_lines_from_z

#      coerce_llm_json / debug_dump / save_json / compare_llm_vs_gt / Z_THRESH

DEBUG = True  # 一键开关

# ---- Optional LLM (Ollama) ----
try:
    from langchain_community.llms import Ollama

    OLLAMA_OK = True
except Exception:
    OLLAMA_OK = False


# -------------------- misc helpers --------------------
def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def try_parse_int(s):
    try:
        return int(str(s))
    except Exception:
        return None


def _one_liner_method(gt_json: dict, z_thresh: float) -> str:
    """
    生成一句话的方法说明。
    """
    sel = gt_json.get("selector")
    line = (
        f"We compute robust MAD-z per bus on the Δ series, "
        f"then select the earliest timestamp with z ≥ {z_thresh:.1f}; "
        f"if none pass the threshold, we fall back to the global max-Δ point."
    )
    if sel:
        line += f' (selector = "{sel}")'
    return line


def _clean_time_str(s: Optional[str]) -> Optional[str]:
    if not isinstance(s, str):
        return s
    # 把 “... -> zscore = ... Δ = ...%” 等尾巴去掉，只保留左侧时间
    if "->" in s:
        s = s.split("->", 1)[0].strip()
    return s.strip()


# -------------------- prompt --------------------
def build_master_prompt(zone_lines: List[str]) -> str:
    """
    关闭式指令：全局阈值 z>=6.0，挑“所有通过阈值里的最早时间”；若并列取 zone id 最小；
    禁止发明不存在的项；只输出 JSON（不准加解释）。
    """

    EXAMPLE = """
[Example Zones]
[id=101] Zone 1: t=2013-10-09 13:06:00 -> zscore=7.10 Δ=0.10%
[id=102] Zone 1: t=2013-10-09 13:12:00 -> zscore=9.90 Δ=52.46%
[id=201] Zone 2: none
[id=301] Zone 3: t=2014-05-01 08:00:00 -> zscore=6.05 Δ=0.00%

[What you must do]
- Threshold: zscore ≥ 6.0.
- Pick the EARLIEST timestamp among ALL items that pass the threshold.
- If a zone has multiple passing items, still use that zone’s earliest timestamp.
- If no item passes the threshold, set all fields to null.
- Never invent a time that is not listed.
- If there is a tie on time, choose the smallest zone id.
- Output ONLY the following JSON (no extra text):

{
  "initial_zone": <int or null>,
  "initial_bus": null,
  "initial_time": "<YYYY-MM-DD HH:MM:SS or null>",
  "propagation": null,
  "root_cause": "unknown",
  "recommendation": []
}
    """.strip()

    rules = """
You are an assistant analyzing power-grid zone summaries.

Each line below is either:
  - "Zone k: none"
  - or "Zone k: t=<timestamp> -> zscore=<value> Δ=<pct>%"
  (Some lines may include an "[id=...]" tag for reference — the id is optional.)

Rules you MUST follow:
- Threshold: zscore ≥ 6.0.
- Among ALL items that pass the threshold, choose the EARLIEST timestamp globally.
- If a zone has multiple passing items, still use that zone's earliest timestamp.
- If there are ties on time, choose the smallest zone id.
- Never invent zones/buses/times not present in [Zones].
- Output ONLY the JSON object with EXACTLY these keys:
  initial_zone (int or null), initial_bus (null), initial_time (string or null),
  propagation (null), root_cause ("unknown"), recommendation ([]).
- Do NOT add explanations or extra text.
    """.strip()

    return f"""{rules}

{EXAMPLE}

[Zones]
{chr(10).join(zone_lines)}
""".strip()


# -------------------- GT & report --------------------
EPS = 1e-9


def compute_mad_z(x: pd.Series) -> pd.Series:
    med = np.nanmedian(x.values)
    mad = np.nanmedian(np.abs(x.values - med))
    denom = (1.4826 * mad) if mad > 0 else (np.nanstd(x.values) + EPS)
    z = np.abs((x - med) / (denom + EPS))
    return z.fillna(0.0)


# ---------- NEW: helper for visualization ----------
def build_graph_and_anomaly_scores(
    csv_path: str,
    resample: str,
    delta_mode: str,
    num_zones: int,
    k_neighbors: int = 5,
):
    """
    从 CSV 读数据 + 构建“相关性网络” + 基于 MAD-z 的 anomaly 分数。
    """

    # ---- 读数据 ----
    # 这一行是之前漏掉的 👇
    resample_arg = None if resample == "" else resample

    buses = load_clean_csv(csv_path, resample=resample_arg)
    delta = compute_delta(buses, mode=delta_mode)

    bus_names = list(buses.columns)

    # 小工具：根据列名前缀识别属于哪一户（H1, H2, ...）
    def house_of(name: str):
        if "_" in name:
            return name.split("_", 1)[0]
        return None  # 老的单户版本没有前缀 → 当作同一户

    # --------- 1) 建图：相关性网络 ----------
    G = nx.Graph()
    for b in bus_names:
        G.add_node(str(b), label=str(b))

    # 用原始值的相关性（Pearson），取绝对值
    corr = buses.corr().abs()
    np.fill_diagonal(corr.values, 0.0)

    for b in bus_names:
        scores = corr[b].sort_values(ascending=False)
        neighbors = scores.head(k_neighbors).index.tolist()
        for nb in neighbors:
            if b == nb:
                continue

            # 🚫 不同 house（前缀不同）之间不连边
            hb, hn = house_of(b), house_of(nb)
            if hb is not None and hn is not None and hb != hn:
                continue

            w = float(corr.loc[b, nb])
            if not G.has_edge(str(b), str(nb)):
                G.add_edge(str(b), str(nb), weight=w)

    # --------- 2) anomaly scores：对 Δ 做 MAD-z ----------
    zdf = delta.apply(compute_mad_z)
    anomaly_scores = {str(col): float(zdf[col].max()) for col in zdf.columns}

    return G, anomaly_scores


def auto_ground_truth(
    delta: pd.DataFrame, zones: List[List[str]], z_thresh: float = 6.0
):
    zdf = delta.apply(compute_mad_z)
    hits = []
    for b in zdf.columns:
        idx = np.where(zdf[b].values >= z_thresh)[0]
        if len(idx) > 0:
            t0 = zdf.index[idx[0]]
            hits.append(("hit", b, t0, float(zdf.loc[t0, b]), float(delta.loc[t0, b])))
    if not hits:
        i, j = np.unravel_index(np.nanargmax(delta.values), delta.shape)
        t0 = delta.index[i]
        b = delta.columns[j]
        z0 = float(compute_mad_z(delta[b]).loc[t0])
        hits = [("max", b, t0, z0, float(delta.loc[t0, b]))]

    hits.sort(key=lambda x: str(x[2]))
    _, bus_id, t_first, zval, dval = hits[0]

    zone_id = None
    for zi, zb in enumerate(zones, start=1):
        if bus_id in zb:
            zone_id = zi
            break

    return {
        "initial_zone": zone_id,
        "initial_bus": int(bus_id) if str(bus_id).isdigit() else None,
        "initial_time": str(t_first),
        "selector": f"z>={z_thresh:.1f}",
        "z_at_event": zval,
        "delta_at_event": dval,
    }


def make_html_rich(
    llm_json: dict,
    gt_json: dict,
    ok_zone: bool,
    ok_bus: bool,
    ok_time: bool,
    run_meta: Optional[Dict] = None,
    rolling_df: Optional[pd.DataFrame] = None,
) -> str:
    import html as _html
    import json, re
    import pandas as pd

    def badge(ok: bool) -> str:
        return "✅" if ok else "❌"

    def gv(d, k, default="N/A"):
        return d.get(k, default) if d else default

    # ===== Verification fields =====
    pred_zone = llm_json.get("initial_zone")
    pred_bus = llm_json.get("initial_bus")
    pred_time = llm_json.get("initial_time")
    gt_zone = gt_json.get("initial_zone")
    gt_bus = gt_json.get("initial_bus")
    gt_time = gt_json.get("initial_time")

    # ===== Method text (readable) =====
    selector = gv(gt_json, "selector", "")
    m = re.search(r"z\s*>=\s*([\d.]+)", selector or "", flags=re.I)
    zth = m.group(1) if m else "6.0"

    method_readable = (
        "<strong>How we generate Ground Truth (and detect the anomaly)</strong><br>"
        "- <strong>Delta (Δ)</strong> means the voltage change for each bus between two consecutive timestamps.<br>"
        '- The time spacing is set by <code>--resample</code> (e.g., <code>1min</code> by default; use <code>5min</code> for 5-minute bins, or <code>""</code> to skip).<br>'
        "- We support three delta modes:<br>"
        "&nbsp;&nbsp;&nbsp;• <code>abs</code>: Δ<sub>t</sub> = x<sub>t</sub> − x<sub>t−1</sub> (absolute change)<br>"
        "&nbsp;&nbsp;&nbsp;• <code>pct</code>: Δ<sub>t</sub> = (x<sub>t</sub> − x<sub>t−1</sub>)/(x<sub>t−1</sub>+ε) × 100% (percent change)<br>"
        "&nbsp;&nbsp;&nbsp;• <code>z</code>: compute a <em>robust z-score</em> on the delta series using MAD.<br>"
        "- On each Δ(t), we compute a robust z-score with Median & MAD: "
        "<code>z(t) = |Δ(t) − median(Δ)| / (1.4826 × MAD)</code> (resistant to outliers/seasonality).<br>"
        f"- A timestamp is a <em>candidate anomaly</em> if its z-score passes the threshold <code>z ≥ {zth}</code>, "
        "no matter whether Δ came from <code>abs</code> or <code>pct</code>.<br>"
        "- If multiple timestamps in the <em>same</em> zone pass, keep that zone’s <em>earliest</em> passing time. "
        "Then compare across zones and pick the <em>global earliest</em>. Ties go to the smaller zone id.<br>"
        "- If <em>no</em> timestamp passes the threshold anywhere, we fall back to the single point with the "
        "<strong>largest Δ</strong> (over all buses and times). That point’s time and zone become the GT anomaly.<br>"
        "- The report’s <em>time tolerance</em> is measured in steps (e.g., with 1-minute resampling, 3 steps = ±3 minutes).<br>"
        f'- For transparency, we record the rule in <code>selector</code> (e.g., "{_html.escape(selector)}").'
    )

    # ===== Mini examples (include a concrete one aligned with the current GT) =====
    example_A = (
        "<strong>Example A — threshold satisfied → earliest passing time wins</strong><br>"
        "Inputs to the LLM (simplified zone lines):<br>"
        "<pre>Zone 1: t=2013-10-09 13:06:00  ->  zscore=7.10  Δ=0.10%\n"
        "Zone 1: t=2013-10-09 13:12:00  ->  zscore=9.90  Δ=52.46%\n"
        "Zone 3: t=2014-05-01 08:00:00  ->  zscore=6.05  Δ=0.00%</pre>"
        "All three pass the threshold (z≥6.0). We first keep Zone 1’s earliest passing time (13:06:00), "
        "then compare across zones: 13:06:00 (Z1) vs 2014-05-01 08:00:00 (Z3). "
        "<em>Global earliest</em> is <code>2013-10-09 13:06:00 @ Zone 1</code> → anomaly is that time in Zone 1."
    )

    # Use current GT to illustrate a concrete picked result
    chosen = (
        f"<code>{_html.escape(str(gt_time))} @ Zone {gt_zone}</code>"
        if (gt_zone and gt_time)
        else "<em>N/A</em>"
    )
    example_A_current = (
        "<div style='margin-top:8px;background:#f7f7f7;border:1px solid #e5e5e5;border-radius:10px;"
        "padding:8px 10px;'>"
        f"<strong>In this run</strong>, the picked anomaly is {chosen}.</div>"
    )

    example_B = (
        "<strong>Example B — nobody passes → fallback to max-Δ</strong><br>"
        "Inputs (z never reaches the threshold):<br>"
        "<pre>Zone 2: t=2013-10-09 15:00:00  ->  zscore=3.2  Δ=0.04%\n"
        "Zone 4: t=2013-10-10 10:46:00  ->  zscore=2.8  Δ=0.00%\n"
        "Zone 5: t=2013-10-09 17:05:00  ->  zscore=1.5  Δ=0.00%</pre>"
        "No candidate passes z≥6.0, so we scan all buses and times for the largest Δ; "
        "the time (and zone) of that max-Δ becomes the anomaly."
    )

    # ===== Run metadata & rolling log =====
    row = run_meta or {}
    csv_path = gv(row, "csv_path")
    if isinstance(rolling_df, pd.DataFrame):
        df_html = rolling_df.to_html(index=False)
    else:
        df_html = "<em>No runs logged.</em>"

    # ===== Page body =====
    summary_html = f"""
    <h2>LLM Global Summary</h2>
    <pre style="background:#111;color:#0f0;padding:12px;border-radius:8px;white-space:pre-wrap;">{_html.escape(json.dumps(llm_json, indent=2))}</pre>

    <h2>Ground Truth</h2>
    <pre style="background:#111;color:#0f0;padding:12px;border-radius:8px;white-space:pre-wrap;">{_html.escape(json.dumps(gt_json, indent=2))}</pre>

    <h2>Method (readable)</h2>
    <div style="background:#fff;border:1px solid #e5e5e5;border-radius:10px;padding:12px 14px;line-height:1.55;">
      {method_readable}
      <div style="height:10px"></div>
      {example_A}
      {example_A_current}
      <div style="height:10px"></div>
      {example_B}
    </div>

    <h2>Verification</h2>
    <table style="border-collapse:collapse;min-width:520px;">
      <tr>
        <td style="padding:6px 12px;border-bottom:1px solid #ddd;">Initial Zone</td>
        <td style="padding:6px 12px;border-bottom:1px solid #ddd;">model={pred_zone} | gt={gt_zone}</td>
        <td style="padding:6px 12px;border-bottom:1px solid #ddd;">{badge(ok_zone)}</td>
      </tr>
      <tr>
        <td style="padding:6px 12px;border-bottom:1px solid #ddd;">Initial Bus</td>
        <td style="padding:6px 12px;border-bottom:1px solid #ddd;">model={pred_bus} | gt={gt_bus}</td>
        <td style="padding:6px 12px;border-bottom:1px solid #ddd;">{badge(ok_bus)}</td>
      </tr>
      <tr>
        <td style="padding:6px 12px;border-bottom:1px solid #ddd;">Initial Time</td>
        <td style="padding:6px 12px;border-bottom:1px solid #ddd;">model={pred_time} | gt={gt_time}</td>
        <td style="padding:6px 12px;border-bottom:1px solid #ddd;">{badge(ok_time)}</td>
      </tr>
    </table>

    <h2>Run Metadata</h2>
    <table style="border-collapse:collapse;min-width:520px;">
      <tr><td style="padding:6px 12px;border-bottom:1px solid #ddd;">Model</td>
          <td style="padding:6px 12px;border-bottom:1px solid #ddd;">{gv(row, "model")}</td></tr>
      <tr><td style="padding:6px 12px;border-bottom:1px solid #ddd;">Embedding</td>
          <td style="padding:6px 12px;border-bottom:1px solid #ddd;">{gv(row, "embedding")}</td></tr>
      <tr><td style="padding:6px 12px;border-bottom:1px solid #ddd;">Seed</td>
          <td style="padding:6px 12px;border-bottom:1px solid #ddd;">{gv(row, "seed")}</td></tr>
      <tr><td style="padding:6px 12px;border-bottom:1px solid #ddd;">Grid Size</td>
          <td style="padding:6px 12px;border-bottom:1px solid #ddd;">plants={gv(row,"num_plants")}, substations={gv(row,"num_substations")}, loads={gv(row,"num_loads")}</td></tr>
      <tr><td style="padding:6px 12px;border-bottom:1px solid #ddd;">Timesteps</td>
          <td style="padding:6px 12px;border-bottom:1px solid #ddd;">{gv(row, "timestep")}</td></tr>
      <tr><td style="padding:6px 12px;border-bottom:1px solid #ddd;">Saved CSV</td>
          <td style="padding:6px 12px;border-bottom:1px solid #ddd;">{csv_path}</td></tr>
    </table>

    <h2>All Runs (rolling log)</h2>
    {df_html}
    """.strip()

    # ===== Outer shell =====
    page = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Power LLM vs Ground Truth</title>
  <style>
    body{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:24px;color:#111}}
    pre{{background:#111;color:#0f0;padding:16px;border-radius:12px;overflow:auto}}
    code{{background:#f2f2f2;border:1px solid #e6e6e6;border-radius:6px;padding:0 4px}}
  </style>
</head>
<body>
{summary_html}
</body>
</html>
""".strip()
    return page


def make_report(
    save_dir: str,
    delta: pd.DataFrame,
    zones: List[List[str]],
    llm_json: dict,
    z_thresh: float = 6.0,
    time_tol_steps: int = 3,
    gt_json: Optional[dict] = None,  # 允许外部传入 GT；否则自动生成
    run_meta: Optional[Dict] = None,  # 新增：运行元信息（用于 Run Metadata）
    rolling_df: Optional[pd.DataFrame] = None,  # 新增：All Runs (rolling log)
):
    os.makedirs(save_dir, exist_ok=True)
    # 1) 生成/保存 GT（或使用外部 GT）
    gt = gt_json or auto_ground_truth(delta, zones, z_thresh=z_thresh)
    debug_dump("real_output_llama3/_debug", "gt_auto.json", gt, is_json=True)
    debug_dump(
        "real_output_llama3/_debug",
        "delta_head.csv",
        delta.head(50).to_csv(index=False),
    )
    # 2) 清洗时间，仅用于比较/展示
    llm_clean = dict(llm_json) if llm_json else {}
    gt_clean = dict(gt) if gt else {}
    llm_clean["initial_time"] = _clean_time_str(llm_clean.get("initial_time"))
    gt_clean["initial_time"] = _clean_time_str(gt_clean.get("initial_time"))
    # 3) 比较
    ok_zone, ok_bus, ok_time = compare_llm_vs_gt(
        llm_clean, gt_clean, time_tolerance_steps=time_tol_steps
    )
    # 4) 富 HTML
    html_text = make_html_rich(
        llm_clean, gt_clean, ok_zone, ok_bus, ok_time, run_meta, rolling_df
    )
    out = os.path.join(save_dir, "report_house.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html_text)
    print(f"💾 Saved HTML report to: {out}")


# -------------------- deterministic pick from candidates --------------------
def pick_from_candidates(cands: List[Dict], z_thresh: float = 6.0):
    """不读文件，直接在内存 candidates 上做：过滤 z>=阈值，按(时间, zone, -z) 选最早。"""
    ok = []
    for c in cands:
        if float(c.get("z", -1e9)) >= z_thresh:
            t = _clean_time_str(c.get("time"))
            ok.append(
                {
                    "zone": int(c.get("zone")),
                    "time": t,
                    "z": float(c.get("z", 0.0)),
                    "delta": float(c.get("delta_pct", 0.0)),
                }
            )
    if not ok:
        return {
            "initial_zone": None,
            "initial_time": None,
            "zscore": None,
            "delta_pct": None,
        }
    ok.sort(key=lambda x: (str(x["time"]), x["zone"], -x["z"]))
    head = ok[0]
    return {
        "initial_zone": head["zone"],
        "initial_time": head["time"],
        "zscore": head["z"],
        "delta_pct": head["delta"],
    }


# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Path to CLEAN_HouseX.csv")
    ap.add_argument(
        "--resample",
        type=str,
        default="1min",
        help="Pandas offset alias (e.g., 1min, 5min, 15min). Use '' to skip.",
    )
    ap.add_argument(
        "--delta",
        type=str,
        default="pct",
        choices=["pct", "abs", "z"],
        help="Delta metric",
    )
    ap.add_argument("--num_plants", type=int, default=5)
    ap.add_argument("--num_substations", type=int, default=3)
    ap.add_argument("--num_loads", type=int, default=40)
    ap.add_argument(
        "--zones", type=int, default=8, help="Number of zones to split buses into"
    )
    ap.add_argument(
        "--ollama_model", type=str, default="llama3.1:8b", help="Ollama model name"
    )
    ap.add_argument("--save_dir", type=str, default="real_output_llama3")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument(
        "--seed", type=int, default=42, help="random seed for reproducibility"
    )

    # 可选：embedding 名称、滚动 CSV、All-Runs 日志（如果你的工程里已经有）
    ap.add_argument(
        "--embedding", type=str, default="sentence-transformers/all-MiniLM-L6-v2"
    )
    ap.add_argument(
        "--log_csv", type=str, default=""
    )  # 例如 run_output_xxx/results_log.csv
    ap.add_argument(
        "--all_runs_csv", type=str, default=""
    )  # 用于 All Runs(rolling log)

    args = ap.parse_args()
    set_seed(args.seed)

    resample = None if args.resample == "" else args.resample
    buses = load_clean_csv(args.csv, resample=resample)
    print(f"✅ Loaded {args.csv} with shape {buses.shape} (rows × buses).")
    print(f"📊 Number of buses (numeric columns): {buses.shape[1]}")

    delta = compute_delta(buses, mode=args.delta)
    bus_names = list(buses.columns)

    zones = split_zones(bus_names, args.zones)
    print("buses:", buses.shape, "delta:", delta.shape)
    assert list(delta.columns) == list(buses.columns), "delta列与buses不一致"
    assert delta.index.equals(buses.index), "delta索引与buses不一致"
    print("delta head:\n", delta.iloc[:3, :3])

    zone_lines, picks, zscore_df, candidates = build_zone_lines_from_z(
        df=buses,
        delta_df=delta,
        zones=zones,
        z_thresh=6.0,
        topk=3,
        prefer_earliest=True,
        use="delta",
    )

    # 保存候选（和给 LLM 的文本）
    debug_dir = os.path.join(args.save_dir, "_debug")
    os.makedirs(debug_dir, exist_ok=True)
    save_json(candidates, os.path.join(debug_dir, "candidates.json"))
    debug_dump(debug_dir, "zone_lines.txt", "\n".join(zone_lines))

    # 用 picks/zscore_df 生成“最早事件”GT
    best = min(picks, key=lambda x: x[1])  # 最早（索引最小）
    zi, j, z_at_j, delta_at_j = best
    t0 = zscore_df.index[j]
    bus_cols = list(delta.columns)  # 或者：list(buses.columns)
    bus_name = str(bus_cols[j]) if j < len(bus_cols) else None
    gt = {
        "initial_zone": int(zi),
        "initial_bus": None,  # 兼容旧格式：仍为 None
        "initial_bus_name": bus_name,  #  人类可读的通道名
        "initial_bus_idx": int(j),  # 备查：在 DataFrame 中的列索引
        "initial_time": t0.strftime("%Y-%m-%d %H:%M:%S"),
        "selector": f"z>={6.0} (earliest)",
        "z_at_event": float(z_at_j),
        "delta_at_event": float(delta_at_j),
    }

    idx = buses.index  # 或 delta.index
    print("start:", idx[0], "end:", idx[-1], "n_steps:", len(idx))
    if len(idx) > 1:
        inferred = pd.infer_freq(idx)
        print("inferred_freq:", inferred)  # 常见会得到 'T' (1min) 或 None
        span_days = (idx[-1] - idx[0]).total_seconds() / 86400
        print(f"span_days≈{span_days:.2f}")
    # 组装 Run Metadata（和你截图一致的字段名）
    run_meta = {
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.ollama_model,
        "embedding": args.embedding,
        "seed": args.seed,
        "num_plants": "N/A",  # 如果你有真实网格规模可在外部改这里
        "num_substations": "N/A",
        "num_loads": buses.shape[1],
        "timestep": len(delta.index),
        "csv_path": args.log_csv or "N/A",
        "span_days": span_days,
    }
    # All Runs(rolling log)
    rolling_df = None
    if args.all_runs_csv and os.path.exists(args.all_runs_csv):
        try:
            rolling_df = pd.read_csv(args.all_runs_csv)
        except Exception:
            rolling_df = None

    # -------------------- LLM stage --------------------
    if OLLAMA_OK:
        try:
            llm = Ollama(
                model=args.ollama_model,
                temperature=0.0,
                top_p=1.0,
                repeat_penalty=1.0,
                num_predict=512,
                seed=args.seed,
            )
            prompt = build_master_prompt(zone_lines)
            print("\n📦 Feeding zone findings to LLM...\n")
            raw = llm.invoke(prompt)
            debug_dump(debug_dir, "llm_raw.txt", raw)

            js = coerce_llm_json(raw)  # 解析到 dict
            # 兜底：使用 candidates 做“确定性覆盖”
            picked = pick_from_candidates(candidates, z_thresh=6.0)
            if picked["initial_zone"] is not None:
                js["initial_zone"] = picked["initial_zone"]
            if picked["initial_time"] is not None:
                js["initial_time"] = picked["initial_time"]
            js["initial_time_note"] = (
                f'{picked["initial_time"]}  ->  zscore = {picked["zscore"]:.2f}  Δ = {picked["delta_pct"]:.2f}%'
                if picked["initial_time"]
                else None
            )

            save_json(js, os.path.join(debug_dir, "llm_json_parsed.json"))
            print("\n🧠 LLM Global Summary (JSON):\n")
            print(json.dumps(js, indent=2))

            make_report(
                save_dir=args.save_dir,
                delta=delta,
                zones=zones,
                llm_json=js,
                z_thresh=6.0,
                time_tol_steps=3,
                gt_json=gt,  # 使用 picks 的 GT
                run_meta=run_meta,  # <<<<<<<<<< 新增：把元信息写进 HTML
                rolling_df=rolling_df,  # <<<<<<<< 新增：All Runs 表
            )
            return
        except Exception as e:
            print(f"⚠️ LLM unavailable or failed ({e}). Falling back to heuristic.")
    else:
        print("⚠️ Ollama not available; falling back to heuristic result.")

    # -------------------- Heuristic fallback --------------------
    parsed_times = []
    for zi, line in enumerate(zone_lines, start=1):
        tm = re.search(r"t\s*=\s*([^\s]+.*?)(?:\s*→|$)", line)
        busm = re.search(r"Bus\s+([^\s:]+)", line)
        t_str = tm.group(1).strip() if tm else str(zi)
        parsed_times.append((zi, t_str, busm.group(1) if busm else None))
    parsed_times.sort(key=lambda x: str(x[1]))
    z0, t0s, b0 = parsed_times[0]
    js = {
        "initial_zone": z0,
        "initial_bus": try_parse_int(b0),
        "initial_time": t0s,
        "propagation": None,
        "root_cause": "unknown",
        "recommendation": [],
    }
    print("\n🧮 Heuristic Global Summary (JSON):\n")
    print(json.dumps(js, indent=2))
    make_report(
        save_dir=args.save_dir,
        delta=delta,
        zones=zones,
        llm_json=js,
        z_thresh=6.0,
        time_tol_steps=3,
        gt_json=gt,
        run_meta=run_meta,
        rolling_df=rolling_df,
    )
    csv_path = os.path.join(args.save_dir, "results_log.csv")
    if os.path.exists(csv_path):
        df_all = pd.read_csv(csv_path)
        df_all = pd.concat([df_all, pd.DataFrame([run_meta])], ignore_index=True)
    else:
        df_all = pd.DataFrame([run_meta])
    df_all.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()
