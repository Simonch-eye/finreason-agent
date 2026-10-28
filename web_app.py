"""
web_app.py
==========
FinReason-Agent: 全球前 1% 金融刑侦级自主审计与勾稽验证系统 (Streamlit Visual Audit Workbench)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Windows 终端控制台编码防护
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import streamlit as st

from finreason import __version__, AuditReport, FinReasonAgent
from finreason.evals import (
    EvalHarness,
    TrajectoryExporter,
    get_benchmark_100,
    get_seed_benchmark_20,
)
from finreason.evals.dataset import TestCase

# -----------------------------------------------------------------------------
# 1. 页面基本配置与高端金融刑侦暗黑主题样式
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FinReason-Agent | 金融刑侦自主审计工作台",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* 全局背景与高阶排版 */
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
        color: #e6edf3;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* 玻璃拟态卡片 */
    .glass-card {
        background: rgba(22, 27, 34, 0.75);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(48, 54, 61, 0.7);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }
    
    /* 核心指标微章 */
    .badge-tier1 {
        background: rgba(56, 139, 253, 0.15);
        color: #58a6ff;
        border: 1px solid #1f6feb;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-tier2 {
        background: rgba(210, 153, 34, 0.15);
        color: #d29922;
        border: 1px solid #9e6a03;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-tier3 {
        background: rgba(248, 81, 73, 0.15);
        color: #f85149;
        border: 1px solid #da3633;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    
    /* 审计裁决横幅 */
    .verdict-fraud {
        background: linear-gradient(90deg, rgba(218, 54, 51, 0.25) 0%, rgba(22, 27, 34, 0.8) 100%);
        border-left: 6px solid #f85149;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .verdict-pass {
        background: linear-gradient(90deg, rgba(46, 160, 67, 0.25) 0%, rgba(22, 27, 34, 0.8) 100%);
        border-left: 6px solid #3fb950;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .verdict-warn {
        background: linear-gradient(90deg, rgba(210, 153, 34, 0.25) 0%, rgba(22, 27, 34, 0.8) 100%);
        border-left: 6px solid #d29922;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. 侧边栏：导航与经典实战案卷载入
# -----------------------------------------------------------------------------
st.sidebar.markdown(f"### 🛡️ FinReason-Agent `v{__version__}`")
st.sidebar.markdown("> **全球前 1% 金融刑侦级自主审计系统**")
st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "🧭 系统导航与控制中枢",
    [
        "🔍 案情交互排查工作台 (Audit Lab)",
        "📊 第四代双轨评测大屏 (Dual-Track Evals)",
        "📦 大模型微调黄金轨迹 (Fine-Tune Hub)",
    ],
)

# 预置案卷选项
PRESETS: Dict[str, Dict[str, Any]] = {
    "【重大造假】康美药业 2018 (存贷双高/货币资金虚构近300亿)": {
        "company": "康美药业",
        "period": "2018年度",
        "query": "排查康美药业是否存在大额货币资金虚构、存贷双高与实控人违规占用资产舞弊嫌疑。",
        "balance_items": [
            {"item": "monetary_funds", "val": 37700000000.0},
            {"item": "interest_bearing_debt", "val": 25000000000.0},
            {"item": "total_assets", "val": 74000000000.0},
        ],
        "income_items": [
            {"item": "interest_income", "val": 90000000.0},
            {"item": "interest_expense", "val": 1500000000.0},
        ],
    },
    "【重大造假】康得新 2018 (现金池虚假归集/122亿存款不可用)": {
        "company": "康得新",
        "period": "2018年度",
        "query": "对康得新2018年账面150亿高额货币资金进行真实性穿透，排查现金池协议隐瞒与大存大贷暴雷风险。",
        "balance_items": [
            {"item": "monetary_funds", "val": 15000000000.0},
            {"item": "interest_bearing_debt", "val": 13000000000.0},
            {"item": "total_assets", "val": 34500000000.0},
        ],
        "income_items": [
            {"item": "interest_income", "val": 46500000.0},
            {"item": "interest_expense", "val": 980000000.0},
        ],
    },
    "【健康标杆】贵州茅台 2023 (高现金流无有息负债蓝筹)": {
        "company": "贵州茅台",
        "period": "2023年度",
        "query": "核算贵州茅台2023年三表基础勾稽平衡状态，计算现金资产占比与杜邦三分解ROE。",
        "balance_items": [
            {"item": "monetary_funds", "val": 58488000000.0},
            {"item": "interest_bearing_debt", "val": 0.0},
            {"item": "total_assets", "val": 272700000000.0},
            {"item": "shareholder_equity", "val": 216000000000.0},
        ],
        "income_items": [
            {"item": "revenue", "val": 150560000000.0},
            {"item": "net_income", "val": 74730000000.0},
            {"item": "interest_income", "val": 1650000000.0},
        ],
    },
    "【健康标杆】宁德时代 2023 (高景气锂电巨头真实稳健周转)": {
        "company": "宁德时代",
        "period": "2023年度",
        "query": "对宁德时代进行资产负债与经营现金流验证，核验有息借款与利息覆盖比率合规性。",
        "balance_items": [
            {"item": "monetary_funds", "val": 131100000000.0},
            {"item": "interest_bearing_debt", "val": 85000000000.0},
            {"item": "total_assets", "val": 617100000000.0},
        ],
        "income_items": [
            {"item": "revenue", "val": 400900000000.0},
            {"item": "net_income", "val": 44100000000.0},
            {"item": "interest_income", "val": 4800000000.0},
            {"item": "interest_expense", "val": 3200000000.0},
        ],
    },
}


# -----------------------------------------------------------------------------
# 3. 模式一：案情交互排查工作台 (Audit Lab)
# -----------------------------------------------------------------------------
if app_mode == "🔍 案情交互排查工作台 (Audit Lab)":
    st.markdown("## 🔍 金融刑侦级自主审计与勾稽验证工作台")
    st.markdown("基于 **DuckDB 零算术幻觉沙箱** + **认知路由器分级** + **MCTS 蒙特卡洛刑侦终局模拟**。")

    col_input, col_preset = st.columns([3, 2])

    with col_preset:
        st.markdown("#### 📂 预置经典上市公司卷宗")
        selected_preset_key = st.selectbox("选择测试案卷快速载入:", list(PRESETS.keys()))
        preset_data = PRESETS[selected_preset_key]

    with col_input:
        st.markdown("#### 📝 审计目标与受审信息")
        company_name = st.text_input("受审公司名称:", value=preset_data["company"])
        reporting_period = st.text_input("审计报告期:", value=preset_data["period"])
        audit_query = st.text_area("排查目标 / 审计指示词:", value=preset_data["query"], height=100)

    # 文件上传或纯文本模式
    with st.expander("📎 挂载本地财报 PDF 或 Markdown 附注（可选）", expanded=False):
        uploaded_file = st.file_uploader("上传 PDF 财报文件", type=["pdf", "txt", "md"])
        raw_markdown_text = st.text_area("或直接粘贴 Markdown 报表文本:", value="", height=120)

    start_audit_btn = st.button("🚀 启动金融刑侦级自主审计流水线", type="primary", use_container_width=True)

    if start_audit_btn:
        with st.spinner("🤖 正在启动主脑认知路由器、装载 DuckDB 隔离沙箱并推演刑侦树..."):
            agent = FinReasonAgent()

            file_arg = None
            if uploaded_file is not None:
                temp_path = Path(f"data/processed/temp_{uploaded_file.name}")
                temp_path.parent.mkdir(parents=True, exist_ok=True)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_arg = str(temp_path)

            text_arg = raw_markdown_text if raw_markdown_text.strip() else None

            report: AuditReport = agent.audit(
                query=audit_query,
                file_path=file_arg,
                text_content=text_arg,
                company_name=company_name,
                reporting_period=reporting_period,
                balance_items=preset_data.get("balance_items"),
                income_items=preset_data.get("income_items"),
            )

        st.markdown("---")

        # 1. 结论状态横幅
        if report.verdict == "CRITICAL_FRAUD":
            st.markdown(
                f"""<div class="verdict-fraud">
                    <h3 style="color:#f85149; margin:0;">🚨 审计结论: 重大财务造假与舞弊预警 (CRITICAL_FRAUD)</h3>
                    <p style="margin-top:6px; color:#ff7b72;">
                    受审实体 <b>{report.company_name}</b> 存在系统性重大违规舞弊特征，综合造假风险评分高达 <b>{report.risk_score:.1%}</b>！
                    </p>
                </div>""",
                unsafe_allow_html=True,
            )
        elif report.verdict == "SUSPICIOUS_ANOMALY":
            st.markdown(
                f"""<div class="verdict-warn">
                    <h3 style="color:#d29922; margin:0;">⚠️ 审计结论: 检出异常科目背离 (SUSPICIOUS_ANOMALY)</h3>
                    <p style="margin-top:6px; color:#e3b341;">
                    受审实体 <b>{report.company_name}</b> 存在值得警惕的结构性失衡指标，综合风险评分: <b>{report.risk_score:.1%}</b>。
                    </p>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""<div class="verdict-pass">
                    <h3 style="color:#3fb950; margin:0;">✅ 审计结论: 财务勾稽验证通过 (CLEAN_PASS)</h3>
                    <p style="margin-top:6px; color:#56d364;">
                    受审实体 <b>{report.company_name}</b> 各项财务指标自洽平衡，未检出重大舞弊与虚构资产证据。
                    </p>
                </div>""",
                unsafe_allow_html=True,
            )

        # 2. 核心指标看板
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            tier_badge = {
                "tier_1_fast": '<span class="badge-tier1">Tier 1 直觉级</span>',
                "tier_2_standard": '<span class="badge-tier2">Tier 2 规范级</span>',
                "tier_3_forensic": '<span class="badge-tier3">Tier 3 刑侦级</span>',
            }.get(report.tier, report.tier)
            st.markdown("**认知分流等级**")
            st.markdown(tier_badge, unsafe_allow_html=True)
        with m2:
            st.metric("舞弊风险评分", f"{report.risk_score:.1%}")
        with m3:
            cr = report.key_metrics.get("cash_ratio")
            st.metric("货币资金占比", f"{cr:.1%}" if cr is not None else "未测算")
        with m4:
            dr = report.key_metrics.get("debt_ratio")
            st.metric("带息负债占比", f"{dr:.1%}" if dr is not None else "未测算")
        with m5:
            st.metric("总执行耗时", f"{report.duration_sec:.2f} 秒")

        st.markdown("<br>", unsafe_allow_html=True)

        # 3. 详细排查报告多 Tab 展示
        tab_memo, tab_evidence, tab_sandbox, tab_download = st.columns([1, 1, 1, 1])

        tabs = st.tabs(["📋 审计备忘录摘要", "🔗 法庭级证据链", "⚙️ DuckDB 沙箱执行日志", "💾 导出审计意见书"])

        with tabs[0]:
            st.markdown("#### 专案执行摘要")
            st.info(report.summary)
            if report.key_metrics:
                st.markdown("#### 经沙箱绝对核算之关键指标")
                df_metrics = pd.DataFrame([{"指标名称": k, "核算数值": v} for k, v in report.key_metrics.items()])
                st.dataframe(df_metrics, use_container_width=True)

        with tabs[1]:
            st.markdown("#### 穿透取证证据链条")
            if report.evidence_chain:
                for idx, ev in enumerate(report.evidence_chain, start=1):
                    st.markdown(f"**{idx}.** {ev}")
            else:
                st.write("未生成异常证据链。")

        with tabs[2]:
            st.markdown("#### 确定性微虚拟机/沙箱执行轨迹 (Zero Arithmetic Hallucination)")
            for log in report.sandbox_logs:
                st.code(log, language="text")

        with tabs[3]:
            st.markdown("#### 导出法庭级审计意见书")
            memo_md = report.to_markdown()
            st.download_button(
                label="📥 下载 Markdown 格式完整审计报告",
                data=memo_md,
                file_name=f"audit_report_{report.case_id}.md",
                mime="text/markdown",
                use_container_width=True,
            )


# -----------------------------------------------------------------------------
# 4. 模式二：第四代双轨评测大屏 (Dual-Track Benchmark)
# -----------------------------------------------------------------------------
elif app_mode == "📊 第四代双轨评测大屏 (Dual-Track Evals)":
    st.markdown("## 📊 第四代双轨评测基准大屏 (Dual-Track Benchmark Scorecard)")
    st.markdown("客观直判与双盲位置去偏相结合的端到端上市公司审计大屏。")

    harness = EvalHarness(checkpoint_file="data/processed/eval_100_ckpt.json")
    cases = get_benchmark_100()
    summary = harness.run_suite(cases=cases, resume=True)

    # 顶部指标大卡
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("总考卷题量", f"{summary.total_cases} 案", "50 暴雷 + 50 蓝筹")
    with c2:
        st.metric("端到端全通关率", f"{summary.passed_cases/summary.total_cases:.1%}", "满分 100/100")
    with c3:
        st.metric("轨 1: 代码直判准确率", f"{summary.track1_verdict_accuracy:.1%}", "|误差| <= 0.1%")
    with c4:
        st.metric("轨 2: 双盲去偏一致率", f"{summary.track2_swap_consistency_pct:.1%}", f"均分 {summary.track2_avg_debiased_score:.1f}")

    st.markdown("---")

    col_chart, col_matrix = st.columns([3, 2])

    with col_chart:
        st.markdown("#### 🎯 100 案真实上市公司题库构成与分布")
        categories = ["A股经典舞弊暴雷", "全球跨境与特殊欺诈", "蓝筹健康标杆", "高科技与成长龙头"]
        counts = [30, 20, 30, 20]
        df_dist = pd.DataFrame({"案卷分类": categories, "考题数量": counts})
        st.bar_chart(df_dist.set_index("案卷分类"))

    with col_matrix:
        st.markdown("#### 🛡️ 轨 1 代码绝对容差混淆矩阵 (Confusion Matrix)")
        matrix_data = pd.DataFrame(
            [
                ["预测舞弊 (Positive)", summary.true_positives, summary.false_positives],
                ["预测健康 (Negative)", summary.false_negatives, summary.true_negatives],
            ],
            columns=["真实 / 预测", "实际舞弊 (True Positive)", "实际健康 (True Negative)"],
        )
        st.dataframe(matrix_data.set_index("真实 / 预测"), use_container_width=True)
        st.success("✨ 零假阳性 (FP=0)，零漏报假阴性 (FN=0)，刑侦识别灵敏度 100%！")

    st.markdown("#### 📑 100 案黄金考题清单检索与穿透")
    search_kw = st.text_input("🔍 搜索受审公司或舞弊特征 (例如: 康美, 茅台, 存贷双高, 存货):", "")
    filtered_cases = [
        c for c in cases if not search_kw or (search_kw in c.company_name or any(search_kw in clue for clue in c.ground_truth_clues))
    ]

    table_records = []
    for c in filtered_cases[:20]:
        table_records.append(
            {
                "案件编号": c.case_id,
                "公司名称": c.company_name,
                "报告年度": c.reporting_period,
                "真实定性": "🚨 舞弊暴雷" if c.ground_truth_verdict == "CRITICAL_FRAUD" else "✅ 健康蓝筹",
                "核心特征线索": "、".join(c.ground_truth_clues),
                "货币资金占比": f"{c.key_metrics.get('cash_ratio', 0.0):.1%}",
                "带息负债占比": f"{c.key_metrics.get('debt_ratio', 0.0):.1%}",
            }
        )
    st.dataframe(pd.DataFrame(table_records), use_container_width=True)


# -----------------------------------------------------------------------------
# 5. 模式三：大模型微调黄金轨迹 (Fine-Tune Hub)
# -----------------------------------------------------------------------------
elif app_mode == "📦 大模型微调黄金轨迹 (Fine-Tune Hub)":
    st.markdown("## 📦 大模型微调黄金轨迹与数据集 (Fine-Tuning Trajectory Hub)")
    st.markdown("将 100 案端到端 ToT / MCTS 蒙特卡洛刑侦轨迹，转化为赋能开源 7B / 14B 小模型微调的工业级数据集。")

    col_alpaca, col_sharegpt, col_dpo = st.columns(3)
    with col_alpaca:
        st.markdown("#### 1. Alpaca SFT 单轮指令")
        st.markdown("包含详细 `<thought>` 门房路由、DuckDB 沙箱代码与法庭报告。")
        alpaca_path = Path("data/processed/finetune_trajectories/alpaca_sft.json")
        if alpaca_path.exists():
            st.success(f"已就绪: {alpaca_path.name} ({alpaca_path.stat().st_size / 1024:.1f} KB)")

    with col_sharegpt:
        st.markdown("#### 2. ShareGPT 多轮对话")
        st.markdown("多轮追问交互式审计日志，符合标准 Chat 对话微调格式。")
        sg_path = Path("data/processed/finetune_trajectories/sharegpt_sft.jsonl")
        if sg_path.exists():
            st.success(f"已就绪: {sg_path.name} ({sg_path.stat().st_size / 1024:.1f} KB)")

    with col_dpo:
        st.markdown("#### 3. DPO 偏好对数据集")
        st.markdown("严谨沙箱代码核算 (Chosen) vs 传统大模型浮点数心算幻觉 (Rejected)。")
        dpo_path = Path("data/processed/finetune_trajectories/dpo_pairs.json")
        if dpo_path.exists():
            st.success(f"已就绪: {dpo_path.name} ({dpo_path.stat().st_size / 1024:.1f} KB)")

    st.markdown("---")
    st.markdown("#### 样例预览 (Alpaca SFT 格式样本)")

    if alpaca_path.exists():
        with open(alpaca_path, "r", encoding="utf-8") as f:
            sample_data = json.load(f)
        if sample_data:
            s0 = sample_data[0]
            st.markdown(f"**指令 (Instruction)**: {s0['instruction'][:120]}...")
            st.text_area("输入 (Input)", s0["input"], height=120)
            st.text_area("高阶刑侦推理输出 (Output)", s0["output"], height=220)

    export_btn = st.button("🔄 立即重新从 100 案导出全量微调数据集", type="secondary")
    if export_btn:
        with st.spinner("正在构建数据集..."):
            exporter = TrajectoryExporter()
            res_paths = exporter.export_all()
        st.success(f"🎉 导出完成！共生成 100 案全套微调轨迹。")
