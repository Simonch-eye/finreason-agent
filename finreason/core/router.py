"""Cognitive Router (门房认知路由器) with constrained decoding (Tier 1/2/3).
Directs financial queries to Fast-Path QA, Deterministic State Machine, or Forensic Tree (ToT/MCTS).
"""

from __future__ import annotations

import os
import re
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field

try:
    import instructor
    from openai import OpenAI
    HAS_OPENAI_INSTRUCTOR = True
except ImportError:
    HAS_OPENAI_INSTRUCTOR = False


class CognitiveTier(str, Enum):
    """Complexity tiers defined in MASTER_SPEC.md."""
    TIER_1_FAST = "tier_1_fast"          # Tier 1 直觉级: 简单混合检索 QA
    TIER_2_STANDARD = "tier_2_standard"  # Tier 2 规范级: 确定性状态机与单流沙箱验算
    TIER_3_FORENSIC = "tier_3_forensic"  # Tier 3 刑侦级: 深度推理刑侦树 (ToT/MCTS + Reflexion)


class FinancialTableType(str, Enum):
    """Core financial tables and report sections."""
    BALANCE_SHEET = "balance_sheet"          # 资产负债表
    INCOME_STATEMENT = "income_statement"    # 利润表
    CASH_FLOW = "cash_flow"                  # 现金流量表
    EQUITY_STATEMENT = "equity_statement"    # 所有者权益变动表
    FOOTNOTES = "footnotes"                  # 财务报表附注
    AUDIT_OPINION = "audit_opinion"          # 审计意见与关键审计事项
    GENERAL_INFO = "general_info"            # 公司概况与管理层讨论 (MD&A)


class RoutingDecision(BaseModel):
    """Structured routing decision output via constrained decoding."""
    tier: CognitiveTier = Field(
        description="The assigned cognitive tier (tier_1_fast, tier_2_standard, or tier_3_forensic)"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score in the routing decision (0.0 to 1.0)"
    )
    reasoning: str = Field(
        description="Brief forensic explanation of why this complexity level was assigned"
    )
    target_tables: List[FinancialTableType] = Field(
        default_factory=list,
        description="List of primary financial tables required to resolve the query"
    )
    deterministic_required: bool = Field(
        default=True,
        description="Whether Python/DuckDB sandboxed calculation is mandatory to avoid arithmetic hallucination"
    )
    sub_tasks: List[str] = Field(
        default_factory=list,
        description="Initial breakdown of sub-audit objectives or verification steps"
    )

    def is_forensic(self) -> bool:
        """Check if this query requires deep forensic tree exploration (Tier 3)."""
        return self.tier == CognitiveTier.TIER_3_FORENSIC

    def is_deterministic_standard(self) -> bool:
        """Check if this query follows standard formulaic state machine (Tier 2)."""
        return self.tier == CognitiveTier.TIER_2_STANDARD


ROUTER_SYSTEM_PROMPT = """你是一个世界前 1% 的金融刑侦级认知路由器（Cognitive Router）。
你的核心任务是根据用户的金融/财报审计需求，基于语法受限解码进行严格的三级认知复杂度分流（Cognitive Tiering）：

【复杂度分级标准】：
1. Tier 1 (直觉级 / tier_1_fast):
   - 特征：单一事实查找、基础财务定义、非计算性信息提取（如董事长姓名、注册地址、某一年度单一科目原始数值）。
   - 典型案例：“茅台2023年的营业收入是多少？”、“什么是商誉减值？”。
   - 是否需沙箱计算：通常为 False。

2. Tier 2 (规范级 / tier_2_standard):
   - 特征：标准公式计算、比率拆解、确定性勾稽验证、多表平衡核算、多期环比/同比变动率计算。
   - 典型案例：“计算茅台2023年ROE的杜邦三分解”、“核算资产负债表左右两边是否勾稽平衡”、“计算流动比率与资产负债率”。
   - 是否需沙箱计算：必须为 True（杜绝大模型心算与浮点数幻觉）。

3. Tier 3 (刑侦级 / tier_3_forensic):
   - 特征：舞弊排查、造假线索穿透、“存贷双高”异动排查、多期科目背离深层侦查、隐藏关联交易、虚增资产/利润排查、需要构建多假设推理树（ToT/MCTS）与反思回滚。
   - 典型案例：“排查康美药业是否存在大存大贷与资金占用嫌疑”、“深度分析应收账款与营收背离异动，排查虚假销售回流”。
   - 是否需沙箱计算：必须为 True。

必须以极其客观、专业的金融审计思维输出结构化决策。
"""


class CognitiveRouter:
    """The gatekeeper cognitive router that parses incoming queries into tiered execution plans."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        client: Optional[Any] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.config = self._load_config(config_path)
        self.model = model or self.config.get("llm", {}).get("router", {}).get("model", "gpt-4o-mini")
        self.client = client
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if self.client is None and HAS_OPENAI_INSTRUCTOR and self.api_key:
            raw_client = OpenAI(api_key=self.api_key)
            self.client = instructor.from_openai(raw_client)

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load default configuration from yaml if available."""
        if config_path and Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}

        # Fallback to default project config
        default_file = Path(__file__).resolve().parent.parent.parent / "configs" / "default.yaml"
        if default_file.exists():
            with open(default_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def route(self, query: str, context: Optional[Dict[str, Any]] = None) -> RoutingDecision:
        """Route a financial query to its optimal cognitive tier.

        Uses LLM constrained decoding via Instructor when API credentials are provided;
        falls back to deterministic expert heuristic rule engine when offline or testing.
        """
        query_clean = query.strip()
        if not query_clean:
            return RoutingDecision(
                tier=CognitiveTier.TIER_1_FAST,
                confidence=1.0,
                reasoning="Empty query defaulted to fast tier.",
                target_tables=[FinancialTableType.GENERAL_INFO],
                deterministic_required=False,
                sub_tasks=[],
            )

        # Attempt constrained LLM decoding if client is active
        if self.client is not None:
            try:
                decision = self.client.chat.completions.create(
                    model=self.model,
                    response_model=RoutingDecision,
                    messages=[
                        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                        {"role": "user", "content": f"请对以下金融财报审计任务进行复杂度认知路由分流：\n{query_clean}"},
                    ],
                    temperature=0.0,
                )
                return decision
            except Exception:
                # In case of network error or rate limit, fall back gracefully to heuristic engine
                pass

        # Deterministic financial rule engine fallback
        return self._heuristic_route(query_clean, context)

    def _heuristic_route(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> RoutingDecision:
        """Deterministic financial keyword, syntax, and intention classifier."""
        q = query.lower()

        # 1. Forensic Keywords (Tier 3)
        forensic_patterns = [
            r"舞弊", r"造假", r"暴雷", r"存贷双高", r"洗钱", r"资金占用", r"利益输送",
            r"虚增", r"虚构", r"刑侦", r"穿透", r"欺诈", r"掏空", r"违规担保",
            r"大存大贷", r"异常背离", r"排查.*嫌疑", r"康美", r"深度推理", r"假账",
            r"fraud", r"forensic", r"manipulation", r"anomaly", r"money laundering"
        ]
        is_forensic = any(re.search(p, q) for p in forensic_patterns)

        # 2. Standard Formulaic / Calculation Keywords (Tier 2)
        standard_patterns = [
            r"勾稽", r"平衡", r"杜邦", r"roe", r"比率", r"拆解", r"核算", r"计算",
            r"资产负债率", r"流动比率", r"速动比率", r"毛利率", r"周转率", r"变动率",
            r"现金流调节", r"恒等式", r"验算", r"三表", r"同比", r"环比",
            r"dupont", r"ratio", r"articulation", r"calculate", r"identity"
        ]
        is_standard = any(re.search(p, q) for p in standard_patterns)

        # Identify target financial tables
        tables: List[FinancialTableType] = []
        if any(k in q for k in ["资产", "负债", "所有者权益", "应收", "货币资金", "存货", "balance"]):
            tables.append(FinancialTableType.BALANCE_SHEET)
        if any(k in q for k in ["收入", "利润", "成本", "毛利", "费用", "净利润", "income"]):
            tables.append(FinancialTableType.INCOME_STATEMENT)
        if any(k in q for k in ["现金流", "经营活动", "投资活动", "筹资活动", "cash"]):
            tables.append(FinancialTableType.CASH_FLOW)
        if any(k in q for k in ["附注", "关联方", "明细", "notes"]):
            tables.append(FinancialTableType.FOOTNOTES)
        if any(k in q for k in ["审计意见", "保留意见", "关键审计事项", "opinion"]):
            tables.append(FinancialTableType.AUDIT_OPINION)

        if not tables:
            tables.append(FinancialTableType.GENERAL_INFO)

        # Classification precedence
        if is_forensic:
            sub_tasks = [
                "识别财报异动指标与科目背离线索",
                "生成舞弊假设树并检索跨期附注与现金流证据",
                "多维勾稽穿透与排查反思",
            ]
            return RoutingDecision(
                tier=CognitiveTier.TIER_3_FORENSIC,
                confidence=0.95,
                reasoning="检测到金融刑侦、舞弊嫌疑排查或深层科目异常背离需求，需启动 Tier 3 假设树搜索 (ToT/MCTS)。",
                target_tables=tables,
                deterministic_required=True,
                sub_tasks=sub_tasks,
            )

        if is_standard:
            sub_tasks = [
                "从财报中提取所需财务科目基础数值",
                "在 DuckDB 隔离沙箱中执行确定性计算公式",
                "比对并输出勾稽平衡与比率分析结果",
            ]
            return RoutingDecision(
                tier=CognitiveTier.TIER_2_STANDARD,
                confidence=0.92,
                reasoning="检测到公式化指标计算、三表勾稽平衡或杜邦比率拆解，走 Tier 2 确定性状态机沙箱验算。",
                target_tables=tables,
                deterministic_required=True,
                sub_tasks=sub_tasks,
            )

        # Tier 1: Fast Path Simple QA
        return RoutingDecision(
            tier=CognitiveTier.TIER_1_FAST,
            confidence=0.88,
            reasoning="单一事实查询或基础概念说明，无需复杂多表计算与刑侦穿透，走 Tier 1 极速通道。",
            target_tables=tables,
            deterministic_required=False,
            sub_tasks=["定位目标科目或段落", "直接提取事实返回给用户"],
        )
