# viz_app.py

import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
import networkx as nx
from main_demo import compute_mad_z

from main_demo import (
    build_graph_and_anomaly_scores,
)  # 刚刚我们在 main.py 里加的 helper


# ========== 一些小工具 ==========


def compute_anomaly_nodes(anomaly_scores, top_k=5, thresh=None):
    """根据分数选出需要高亮的异常节点。"""
    if not anomaly_scores:
        return []

    if thresh is not None:
        return [n for n, s in anomaly_scores.items() if s >= thresh]

    sorted_nodes = sorted(anomaly_scores.items(), key=lambda x: x[1], reverse=True)
    return [n for n, _ in sorted_nodes[:top_k]]


def draw_graph_html(G, highlight_nodes=None):
    from pyvis.network import Network
    import tempfile
    import os
    import networkx as nx
    import numpy as np

    highlight_nodes = set(highlight_nodes or [])

    # 1. 初始化
    net = Network(height="650px", width="100%", notebook=False, directed=False)

    # 2. 计算固定布局 (关键修改在这里)
    # k: 控制节点间距 (0.1~1.0)，值越大越分散
    # iterations: 迭代次数，多跑几次让布局更舒展
    # weight=None: 关键！忽略边的权重，防止强相关节点吸在一起变成一团
    pos = nx.spring_layout(G, seed=42, k=0.8, iterations=100, weight=None)

    # 3. 添加节点
    for n, data in G.nodes(data=True):
        label = data.get("label", str(n))
        is_hi = n in highlight_nodes

        size = 28 if is_hi else 18
        color = "#ff6b6b" if is_hi else "#666666"

        # 获取坐标
        x_raw, y_raw = pos[n]

        # 兜底防止 NaN
        if np.isnan(x_raw) or np.isnan(y_raw):
            x_val, y_val = 0.0, 0.0
        else:
            # 这里的倍数可以适当调大，比如 1200
            x_val = float(x_raw) * 1200
            y_val = float(y_raw) * 1200

        net.add_node(
            n,
            label=label,
            size=size,
            color=color,
            x=x_val,
            y=y_val,
            physics=False,
            title=str(label),
        )

    # 4. 添加边
    for u, v, edata in G.edges(data=True):
        net.add_edge(u, v, color="#cccccc")

    # 5. 关闭物理引擎
    net.toggle_physics(False)

    # 6. 生成 HTML
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.write_html(tmp_file.name, notebook=False)

    with open(tmp_file.name, "r", encoding="utf-8") as f:
        html = f.read()
    os.unlink(tmp_file.name)

    return html


def call_llm_with_graph(prompt, G, anomaly_scores):
    """
    占位：之后你把真实 LLM 调用塞进来。
    现在先用 demo：返回固定 reasoning + 两个异常节点。
    """
    reasoning = (
        "Reasoning Explaination: the LLM flags some buses whose power-change patterns look abnormal, "
        "for example nodes like 'Bus_1' and 'Bus_3' that show persistently high z-scores "
        "across multiple time windows."
    )

    # 如果你想更真实一点，可以按分数选 top-2
    if anomaly_scores:
        top2 = compute_anomaly_nodes(anomaly_scores, top_k=2)
    else:
        top2 = []

    return reasoning, top2


# ========== Streamlit 主体 ==========


def main():
    st.set_page_config(page_title="Power Network Anomaly Viz", layout="wide")

    st.title("Power Network Anomaly Visualization + LLM Reasoning")

    # ---- 侧边栏：相当于原来命令行参数 ----
    st.sidebar.header("Data & Method Settings")

    csv_path = st.sidebar.text_input(
        "CSV path",
        value="./real_data/CLEAN_Pecan_multiHouses.csv",  # 对应你之前的 --csv
    )
    resample = st.sidebar.text_input(
        "Resample (Pandas offset)",
        value="1min",  # 对应 --resample 1min
    )
    delta_mode = st.sidebar.selectbox(
        "Delta metric",
        options=["pct", "abs", "z"],
        index=0,  # 默认 "pct" 对应你之前的 --delta pct
    )
    num_zones = st.sidebar.number_input(
        "Number of zones",
        min_value=1,
        max_value=64,
        value=1,  # 对应 --zones 8
        step=1,
    )

    st.sidebar.markdown("---")
    top_k = st.sidebar.number_input(
        "Top-K nodes to highlight (by score)",
        min_value=1,
        max_value=50,
        value=5,
        step=1,
    )

    # ---- 左右两列：左图右控制 ----
    col1, col2 = st.columns([2, 1])

    # 初始化高亮节点
    if "highlight_nodes" not in st.session_state:
        st.session_state.highlight_nodes = []

    # 读取数据 + 构图 + 算 anomaly_scores
    with st.spinner("Loading data and building graph..."):
        try:
            G, anomaly_scores = build_graph_and_anomaly_scores(
                csv_path="./real_data/CLEAN_Pecan_multiHouses.csv",
                resample="1min",
                delta_mode="pct",
                k_neighbors=5,
                num_zones=8,
            )

            data_ok = True
        except Exception as e:
            data_ok = False
            col1.error(f"Failed to load / process data: {e}")
            G, anomaly_scores = nx.Graph(), {}

    with col1:
        st.subheader("Power Network")
        if data_ok:
            html = draw_graph_html(G, st.session_state.highlight_nodes)
            # 用 components.html，而不是 st.components.v1.html
            components.html(html, height=650, scrolling=True)
        else:
            st.info("Please fix the settings in the sidebar and rerun.")

    with col2:
        st.subheader("LLM Prompt")

        default_prompt = (
            "Given the power network structure and anomaly scores for each bus, "
            "identify the most anomalous nodes and explain your reasoning."
        )
        prompt = st.text_area("Prompt to LLM:", value=default_prompt, height=200)

        st.markdown("**GT anomalies (Top-K by score):**")
        if st.button("Show ground-truth anomalies"):
            if anomaly_scores:
                st.session_state.highlight_nodes = compute_anomaly_nodes(
                    anomaly_scores, top_k=top_k
                )
                st.rerun()
            else:
                st.warning("No anomaly scores available.")

        st.markdown("---")
        if st.button("Ask LLM and highlight result"):
            if not data_ok:
                st.warning("Data not loaded correctly. Please check CSV path, etc.")
            else:
                with st.spinner("Calling LLM (demo stub)..."):
                    reasoning, llm_nodes = call_llm_with_graph(
                        prompt, G, anomaly_scores
                    )

                st.session_state.highlight_nodes = llm_nodes

                st.subheader("LLM Reasoning")
                st.write(reasoning)
                st.success(
                    f"LLM marked {len(llm_nodes)} node(s) as anomalous: {llm_nodes}"
                )

                # 再渲染一遍带高亮的图
                html = draw_graph_html(G, st.session_state.highlight_nodes)
                components.html(html, height=650, scrolling=True)


if __name__ == "__main__":
    main()
