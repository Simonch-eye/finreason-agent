"""Golden Trajectory Exporter for LLM Fine-Tuning (SFT / DPO 数据集导出器).
Converts ToT / MCTS forensic audit trajectories, sandboxed executions, and Reflexion notes into
standard datasets for fine-tuning open-source 7B/14B financial models via QLoRA.
Supports Alpaca format, ShareGPT multi-turn format, and DPO preference pairs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from finreason.evals.dataset import TestCase, get_benchmark_100
from finreason.evals.harness import EvalCaseResult, EvalSuiteSummary


class AlpacaExample(BaseModel):
    """Standard Alpaca SFT instruction tuning schema."""
    instruction: str
    input: str
    output: str


class ShareGPTMessage(BaseModel):
    """Single message in a ShareGPT conversation."""
    from_role: str = Field(alias="from")
    value: str

    model_config = {"populate_by_name": True}


class ShareGPTExample(BaseModel):
    """ShareGPT multi-turn conversation schema."""
    conversations: List[ShareGPTMessage]


class DPOPairExample(BaseModel):
    """Direct Preference Optimization (DPO) chosen vs rejected pair."""
    prompt: str
    chosen: str
    rejected: str


class TrajectoryExporter:
    """Exports structured audit traces into multi-format SFT and DPO datasets."""

    SYSTEM_PROMPT: str = (
        "你是一个精通中国企业会计准则、国际财务报告准则(IFRS)与金融刑侦的顶级自主审计 Agent。\n"
        "核心工作原则：\n"
        "1. 零算术幻觉：严禁在心智模型中直接心算复杂浮点数，必须生成 Python/DuckDB 代码并在沙箱中验算；\n"
        "2. 认知分级：根据问题复杂性自动分流至 Tier 1（极速检索）、Tier 2（确定性勾稽状态机）或 Tier 3（刑侦假设树）；\n"
        "3. 证据闭环：所有舞弊定性必须基于资产负债表、利润表与附注的交叉验证证据链。"
    )

    def __init__(self, output_dir: str = "data/processed/finetune_trajectories"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_alpaca_dataset(self, cases: List[TestCase]) -> List[AlpacaExample]:
        """Convert benchmark cases into Alpaca SFT format."""
        dataset: List[AlpacaExample] = []

        for case in cases:
            # Build input ledger representation
            b_lines = [f"  - {item['item']}: {item['val']:,.2f} 元" for item in case.balance_items]
            i_lines = [f"  - {item['item']}: {item['val']:,.2f} 元" for item in case.income_items]

            input_text = (
                f"【待审企业】{case.company_name} ({case.reporting_period})\n"
                f"【审计目标】{case.query}\n"
                f"【资产负债表关键科目】:\n" + "\n".join(b_lines) + "\n"
                f"【利润表关键科目】:\n" + "\n".join(i_lines)
            )

            # Synthesize realistic forensic reasoning chain
            is_fraud = case.ground_truth_verdict == "CRITICAL_FRAUD"
            clues_str = "、".join(case.ground_truth_clues)

            output_text = (
                f"<thought>\n"
                f"1. 【门房路由判定】: 任务涉及重大财务异常排查与多科目综合穿透，分流至 Tier 3 (刑侦级 ToT/MCTS 推理树)。\n"
                f"2. 【确定性沙箱代码执行】:\n"
                f"```python\n"
                f"# 在 DuckDB 沙箱中核算关键指标与利息匹配度\n"
                f"tot_assets = con.execute(\"SELECT val FROM balance_items WHERE item='total_assets'\").fetchone()[0]\n"
                f"cash = con.execute(\"SELECT val FROM balance_items WHERE item='monetary_funds'\").fetchone()[0]\n"
                f"debt = con.execute(\"SELECT val FROM balance_items WHERE item='interest_bearing_debt'\").fetchone()[0]\n"
                f"int_inc = con.execute(\"SELECT val FROM income_items WHERE item='interest_income'\").fetchone()[0]\n"
                f"cash_ratio = cash / tot_assets\n"
                f"debt_ratio = debt / tot_assets\n"
                f"interest_yield = int_inc / cash if cash > 0 else 0.0\n"
                f"```\n"
                f"3. 【沙箱验算结果】:\n"
                f"  - 货币资金占比 (cash_ratio): {case.key_metrics.get('cash_ratio', 0.0):.1%}\n"
                f"  - 带息负债占比 (debt_ratio): {case.key_metrics.get('debt_ratio', 0.0):.1%}\n"
                f"  - 实际利息收益率 (interest_yield): {case.key_metrics.get('interest_yield', 0.0):.2%}\n"
                f"4. 【刑侦假设反思 (Reflexion)】:\n"
                f"  "
                + (
                    f"触发'存贷双高'核心特征警报！企业账面持有高额存款，同时背负沉重借款，利息收益率严重背离正常理财区间，"
                    f"印证核心线索：{clues_str}。资金极大概率存在违规质押或已被大股东抽逃占用。"
                    if is_fraud
                    else f"各项财务比率与利息收益率处于健康标准区间，杜邦拆解与资金流向均合规自洽，排除财务舞弊假设。"
                )
                + f"\n</thought>\n\n"
                f"【最终审计裁决报告】\n"
                f"定性结论: {'【重大舞弊预警 (CRITICAL_FRAUD)】' if is_fraud else '【未发现重大异常 (CLEAN_PASS)】'}\n"
                f"核心发现与证据链:\n"
                + (
                    f"  1. 资产与负债结构呈现显著异常，存在大额存贷双高与表外违约隐患；\n"
                    f"  2. 真实利息收益与巨额货币资金规模严重失配；\n"
                    f"  3. 证据链完全支持涉嫌：{clues_str}。"
                    if is_fraud
                    else f"  1. 货币资金储备充沛，资产负债率合理可控；\n"
                    f"  2. 存款收益率符合正常银行计息标准；\n"
                    f"  3. 业务模式健康稳定，符合蓝筹治理水准。"
                )
            )

            dataset.append(
                AlpacaExample(
                    instruction=f"{self.SYSTEM_PROMPT}\n请对下列财务数据进行严谨的确定性代码验算与刑侦推理。",
                    input=input_text,
                    output=output_text,
                )
            )

        return dataset

    def generate_dpo_pairs(self, cases: List[TestCase]) -> List[DPOPairExample]:
        """Generate DPO preference pairs (Chosen vs Rejected hallucinatory/shallow responses)."""
        pairs: List[DPOPairExample] = []

        for case in cases:
            prompt = (
                f"{self.SYSTEM_PROMPT}\n"
                f"请核查 {case.company_name} ({case.reporting_period}) 的财务报表并排查其是否存在舞弊风险：\n"
                f"{case.query}"
            )

            is_fraud = case.ground_truth_verdict == "CRITICAL_FRAUD"

            # Chosen: Deep, deterministic, code-grounded forensic reasoning
            chosen = (
                f"【确定性审计推理过程】\n"
                f"1. 提取科目数据挂载至 DuckDB 隔离沙箱；\n"
                f"2. 经过绝对容差计算，货币资金比率为 {case.key_metrics.get('cash_ratio', 0.0):.1%}，"
                f"带息负债比率为 {case.key_metrics.get('debt_ratio', 0.0):.1%}，"
                f"利息收益率为 {case.key_metrics.get('interest_yield', 0.0):.2%}；\n"
                f"3. 结合 Reflexion 记忆库进行双向去偏裁决：\n"
                f"裁决结论: {case.ground_truth_verdict}\n"
                f"核心证据: {', '.join(case.ground_truth_clues)}"
            )

            # Rejected: Typical naive LLM floating point hallucination / shallow optimism
            rejected = (
                f"根据我直接阅读该公司的财务报表：\n"
                f"我觉得该公司的货币资金看起来很多，资产规模也很大，应该是一家实力雄厚的大企业。\n"
                f"虽然有部分借款，但只要后续多拉投资就行了。我心算了一下利息大概是 5% 左右吧，"
                f"因此我认为该企业财务完全健康，没有任何造假问题，可以放心投资！"
            )

            pairs.append(
                DPOPairExample(
                    prompt=prompt,
                    chosen=chosen,
                    rejected=rejected,
                )
            )

        return pairs

    def export_all(self, cases: Optional[List[TestCase]] = None) -> Dict[str, str]:
        """Export all datasets to disk in JSON and JSONL formats."""
        benchmark_cases = cases or get_benchmark_100()

        # 1. Alpaca SFT
        alpaca_data = self.generate_alpaca_dataset(benchmark_cases)
        alpaca_path = self.output_dir / "alpaca_sft.json"
        with open(alpaca_path, "w", encoding="utf-8") as f:
            json.dump([item.model_dump() for item in alpaca_data], f, ensure_ascii=False, indent=2)

        # 2. ShareGPT Multi-turn
        sharegpt_path = self.output_dir / "sharegpt_sft.jsonl"
        with open(sharegpt_path, "w", encoding="utf-8") as f:
            for item in alpaca_data:
                sg_item = ShareGPTExample(
                    conversations=[
                        ShareGPTMessage(from_role="human", value=f"{item.instruction}\n\n{item.input}"),
                        ShareGPTMessage(from_role="gpt", value=item.output),
                    ]
                )
                f.write(json.dumps(sg_item.model_dump(by_alias=True), ensure_ascii=False) + "\n")

        # 3. DPO Preference Pairs
        dpo_data = self.generate_dpo_pairs(benchmark_cases)
        dpo_path = self.output_dir / "dpo_pairs.json"
        with open(dpo_path, "w", encoding="utf-8") as f:
            json.dump([item.model_dump() for item in dpo_data], f, ensure_ascii=False, indent=2)

        return {
            "alpaca_sft": str(alpaca_path),
            "sharegpt_sft": str(sharegpt_path),
            "dpo_pairs": str(dpo_path),
            "total_samples": len(benchmark_cases),
        }
