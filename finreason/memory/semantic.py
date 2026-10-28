"""Semantic Memory (会计准则、行业基准与刑侦舞弊特征库).
Provides conceptual guidance, CAS/IFRS standards lookup, and known fraud pattern matching.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AccountingKnowledgeDoc(BaseModel):
    """Knowledge document representing an accounting rule, threshold, or fraud typology."""
    id: str = Field(description="Unique rule/doc identifier")
    title: str = Field(description="Title of standard or typology")
    category: str = Field(description="Category: standard, fraud_typology, ratio_benchmark")
    content: str = Field(description="Full text of accounting guidance or forensic indicators")
    keywords: List[str] = Field(default_factory=list, description="Core matching keywords")


DEFAULT_SEED_KNOWLEDGE: List[AccountingKnowledgeDoc] = [
    AccountingKnowledgeDoc(
        id="FRAUD_01",
        title="存贷双高（大存大贷）舞弊特征与排查指引",
        category="fraud_typology",
        keywords=["存贷双高", "大存大贷", "货币资金", "短期借款", "利息收入", "质押", "受限资金"],
        content=(
            "【存贷双高排查要点】：公司账面拥有巨额货币资金，同时维持高额带息负债（短期借款、应付债券），"
            "并承担高额利息支出。常见舞弊手法：大额银行存款已被控股股东暗中质押担保（受限未披露），"
            "或资金已被关联方挪用，账面资金虚假。排查指标：货币资金/总资产 > 25% 且 带息负债/总资产 > 25%；"
            "测算实际利息收益率（利息收入/平均货币资金），若明显低于基准活期/协定利率（如低于0.5%），高度疑似虚构。"
        ),
    ),
    AccountingKnowledgeDoc(
        id="FRAUD_02",
        title="在建工程长期不转固与资产虚增穿透",
        category="fraud_typology",
        keywords=["在建工程", "转固", "固定资产", "康美药业", "工程进度", "虚增资产"],
        content=(
            "【在建工程舞弊要点】：通过虚构工程项目与采购合同，将虚增的现金流以'工程款'名义支付出表，"
            "形成虚假在建工程。为避免计提折旧侵蚀利润，长期拖延竣工验收不转入固定资产。"
            "排查要点：核查工程预算与累计投入比、工程进度是否长期停滞（如超3年未转固）、"
            "借款费用资本化金额是否超标，穿透承包商是否为关联空壳公司。"
        ),
    ),
    AccountingKnowledgeDoc(
        id="FRAUD_03",
        title="应收账款与营业收入异常背离（虚构收入与回流）",
        category="fraud_typology",
        keywords=["应收账款", "营业收入", "坏账准备", "信用减值", "账龄", "体外循环"],
        content=(
            "【应收账款背离排查】：营业收入增长乏力甚至下滑，但应收账款余额与账龄大幅飙升；"
            "应收账款周转天数异动拉长。排查要点：审查第四季度或年末突击确认大额销售、"
            "前五大欠款方集中度异常、核查经营现金流是否与营收严重背离，防范通过供应商打款回流虚构销售闭环。"
        ),
    ),
    AccountingKnowledgeDoc(
        id="STD_01",
        title="企业会计准则第14号——收入（确认与计量原则）",
        category="standard",
        keywords=["收入准则", "履约义务", "主要责任人", "代理人", "总额法", "净额法"],
        content=(
            "【收入准则要点】：收入确认以'客户取得商品控制权'为核心。对涉及第三方参与的交易，"
            "必须严格区分企业是'主要责任人'（总额法确认）还是'代理人'（净额法确认）。"
            "若企业在转让前未承担存货风险、无自主定价权，采用总额法虚增流水的属于会计违规。"
        ),
    ),
    AccountingKnowledgeDoc(
        id="STD_02",
        title="企业会计准则第8号——资产减值与商誉减值测试",
        category="standard",
        keywords=["商誉减值", "资产减值", "洗大澡", "资产组", "减值准备"],
        content=(
            "【资产减值准则要点】：商誉至少应当在每年年度终了进行减值测试，且不得转回。"
            "重点排查并购重组高估值承诺期满后突发性计提巨额商誉减值（'财务洗大澡'），"
            "核实预测现金流假设、折现率选取的合理性与跨期一致性。"
        ),
    ),
]


class SemanticMemory:
    """Retrieval engine for accounting standards, benchmark ratios, and forensic fraud signatures."""

    def __init__(self, seed_defaults: bool = True):
        self.docs: Dict[str, AccountingKnowledgeDoc] = {}
        if seed_defaults:
            for doc in DEFAULT_SEED_KNOWLEDGE:
                self.add_document(doc)

    def add_document(self, doc: AccountingKnowledgeDoc) -> None:
        """Add or update a knowledge document in semantic memory."""
        self.docs[doc.id] = doc

    def get_document(self, doc_id: str) -> Optional[AccountingKnowledgeDoc]:
        """Fetch document by unique rule ID."""
        return self.docs.get(doc_id)

    def search(
        self,
        query: str,
        top_k: int = 3,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search knowledge docs using keyword matching and character n-gram relevance."""
        q = query.lower().strip()
        if not q:
            return []

        scored_docs: List[tuple[float, AccountingKnowledgeDoc]] = []

        for doc in self.docs.values():
            if category and doc.category != category:
                continue

            score = 0.0

            # 1. Direct keyword matching (highest precision)
            for kw in doc.keywords:
                if kw.lower() in q:
                    score += 5.0

            # 2. Title matching
            for token in re.findall(r"[\u4e00-\u9fa5]{2,4}|[A-Za-z0-9_]+", doc.title.lower()):
                if token in q:
                    score += 3.0

            # 3. Content matching with 2-gram overlap
            doc_text = doc.content.lower()
            query_ngrams = [q[i : i + 2] for i in range(len(q) - 1) if not q[i : i + 2].isspace()]
            for ng in query_ngrams:
                if len(ng) == 2 and ng in doc_text:
                    score += 0.2

            if score > 0:
                scored_docs.append((score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, doc in scored_docs[:top_k]:
            results.append({
                "id": doc.id,
                "title": doc.title,
                "category": doc.category,
                "content": doc.content,
                "score": score,
            })
        return results


    def format_guidance_prompt(self, query: str, top_k: int = 2) -> str:
        """Format matching rules into regulatory guidance prompt context."""
        hits = self.search(query, top_k=top_k)
        if not hits:
            return ""

        parts = ["【会计法规与刑侦经验基准 (Semantic Memory)】:"]
        for h in hits:
            parts.append(f"• 《{h['title']}》:\n  {h['content']}")
        return "\n".join(parts)
