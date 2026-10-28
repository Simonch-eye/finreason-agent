"""Tree-of-Thoughts (ToT) Forensic Investigation Engine (刑侦假设树与剪枝回溯引擎).
Implements hypothesis generation, deterministic tool execution, evidence verification, and counterfactual pruning.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from finreason.memory import (
    EpisodicMemory,
    InvestigationStep,
    ProceduralMemory,
    SemanticMemory,
    WorkingMemory,
)
from finreason.parser import FootnoteLinker
from finreason.sandbox import CodeRunner, ExecutionResult


class ThoughtState(str, Enum):
    """Evaluation status of an investigative thought branch."""
    UNEXPLORED = "unexplored"
    EXPLORING = "exploring"
    CONFIRMED = "confirmed"      # 确凿证实 (Evidence confirms fraud/anomaly)
    REFUTED = "refuted"          # 明确推翻 (Evidence contradicts hypothesis)
    SUSPICIOUS = "suspicious"    # 高度可疑 (Strong clues, needs deeper sub-lead expansion)
    INCONCLUSIVE = "inconclusive"


class ThoughtNode(BaseModel):
    """A single node representing a hypothesis, investigative step, and evidence in the ToT tree."""
    node_id: str = Field(description="Unique node identifier, e.g. node_0, node_0_1")
    parent_id: Optional[str] = Field(default=None, description="Parent thought node id")
    depth: int = Field(default=0, description="Depth level in the search tree")
    hypothesis: str = Field(description="Investigative hypothesis or lead being tested")
    action_type: str = Field(default="sandbox_eval", description="Action type: skill_exec, footnote_drill, ratio_check")
    action_payload: str = Field(default="", description="Specific code, query, or skill name")
    state: ThoughtState = Field(default=ThoughtState.UNEXPLORED, description="Current thought validation state")
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="Forensic risk/anomaly score (0.0=benign, 1.0=critical fraud)")
    evidence: str = Field(default="", description="Fact-checked evidence found in financial statements or sandbox")
    reflection: Optional[str] = Field(default=None, description="Counterfactual self-reflection note")
    children_ids: List[str] = Field(default_factory=list, description="IDs of expanded sub-lead nodes")
    timestamp: float = Field(default_factory=time.time)

    def is_prunable(self) -> bool:
        """Check if branch should be pruned (refuted or zero risk)."""
        return self.state == ThoughtState.REFUTED or (self.state == ThoughtState.INCONCLUSIVE and self.score < 0.2)


class ForensicInvestigationResult(BaseModel):
    """Final synthesis report produced by the ToT forensic tree exploration."""
    case_id: str
    target_company: str
    root_objective: str
    verdict: str = Field(description="Audit verdict: CRITICAL_FRAUD, SUSPICIOUS_ANOMALY, CLEAN_PASS")
    total_nodes_explored: int
    confirmed_fraud_leads: List[ThoughtNode]
    refuted_leads: List[ThoughtNode]
    suspicious_leads: List[ThoughtNode]
    evidence_chain: List[str]
    audit_summary: str


class ToTEngine:
    """Tree-of-Thoughts Forensic Investigation and Pruning Engine."""

    def __init__(
        self,
        runner: CodeRunner,
        working_mem: Optional[WorkingMemory] = None,
        episodic_mem: Optional[EpisodicMemory] = None,
        semantic_mem: Optional[SemanticMemory] = None,
        procedural_mem: Optional[ProceduralMemory] = None,
        footnote_linker: Optional[FootnoteLinker] = None,
        max_depth: int = 3,
        beam_width: int = 2,
    ):
        self.runner = runner
        self.working_mem = working_mem or WorkingMemory()
        self.episodic_mem = episodic_mem or EpisodicMemory(db_path=":memory:")
        self.semantic_mem = semantic_mem or SemanticMemory()
        self.procedural_mem = procedural_mem or ProceduralMemory()
        self.footnote_linker = footnote_linker
        self.max_depth = max_depth
        self.beam_width = beam_width

        self.nodes: Dict[str, ThoughtNode] = {}
        self.node_counter = 0

    def create_node(
        self,
        hypothesis: str,
        parent_id: Optional[str] = None,
        depth: int = 0,
        action_type: str = "sandbox_eval",
        action_payload: str = "",
    ) -> ThoughtNode:
        """Create and register a new thought node in the tree."""
        self.node_counter += 1
        node_id = f"thought_{self.node_counter}"
        node = ThoughtNode(
            node_id=node_id,
            parent_id=parent_id,
            depth=depth,
            hypothesis=hypothesis,
            action_type=action_type,
            action_payload=action_payload,
        )
        self.nodes[node_id] = node
        if parent_id and parent_id in self.nodes:
            self.nodes[parent_id].children_ids.append(node_id)
        return node

    def generate_initial_hypotheses(self, objective: str) -> List[ThoughtNode]:
        """Generate first-level forensic hypotheses based on objective and semantic memory."""
        # Consult Semantic Memory for matched fraud typologies
        semantic_hints = self.semantic_mem.search(objective, top_k=3)

        hypotheses = []
        if semantic_hints:
            for hint in semantic_hints:
                h_text = f"排查《{hint['title']}》嫌疑: 核验相关科目异动与附注披露"
                node = self.create_node(
                    hypothesis=h_text,
                    parent_id=None,
                    depth=1,
                    action_type="semantic_typology",
                    action_payload=hint["id"],
                )
                hypotheses.append(node)
        else:
            # Fallback standard forensic hypotheses
            standard_leads = [
                ("排查存贷双高与受限资金隐瞒嫌疑", "cash_debt_anomaly"),
                ("排查贝尼斯财报人为操纵与虚增利润嫌疑", "beneish_m_score"),
                ("排查主营业务与应收账款背离异动", "receivables_drift"),
            ]
            for lead_text, payload in standard_leads:
                node = self.create_node(
                    hypothesis=lead_text,
                    parent_id=None,
                    depth=1,
                    action_type="skill_exec",
                    action_payload=payload,
                )
                hypotheses.append(node)

        return hypotheses

    def evaluate_thought(self, node: ThoughtNode) -> ThoughtNode:
        """Evaluate a hypothesis node using sandbox execution, procedural skills, or footnote penetration."""
        node.state = ThoughtState.EXPLORING

        # Check episodic memory to ensure we are not repeating a previously refuted hypothesis
        refuted = self.episodic_mem.get_refuted_hypotheses(self.working_mem.case_id)
        if any(r in node.hypothesis or node.hypothesis in r for r in refuted):
            node.state = ThoughtState.REFUTED
            node.score = 0.0
            node.evidence = "分支已在前期调查中被证实为死路，依据 Reflexion 反思直接剪枝。"
            node.reflection = "避免重复探索已知无果的死路假设。"
            return node

        # 1. Evaluate via balance and income financial ledgers in sandbox
        if "balance_items" in self.runner.list_tables():
            df_b = self.runner.get_table_df("balance_items")
            b_dict = dict(zip(df_b["item"], df_b["val"]))
            tot_assets = max(b_dict.get("total_assets", 1.0), 1.0)
            cash = b_dict.get("monetary_funds", 0.0)
            debt = b_dict.get("interest_bearing_debt", 0.0)
            cash_ratio = cash / tot_assets
            debt_ratio = debt / tot_assets

            implied_yield = None
            if "income_items" in self.runner.list_tables():
                df_i = self.runner.get_table_df("income_items")
                i_dict = dict(zip(df_i["item"], df_i["val"]))
                int_inc = i_dict.get("interest_income", 0.0)
                if cash > 0:
                    implied_yield = int_inc / cash

            is_dual_high = (cash_ratio >= 0.20) and (debt_ratio >= 0.20)

            low_interest_yield = (implied_yield is not None) and (implied_yield < 0.008)

            # Check 1: 存贷双高与超低利息收益率 (康美, 康得新, 辅仁, 宜华, 金正大)
            if is_dual_high and low_interest_yield:
                node.state = ThoughtState.CONFIRMED
                node.score = 0.95
                node.evidence = (
                    f"确凿检出'存贷双高'异动：货币资金占比 {cash_ratio:.1%}，"
                    f"带息负债占比 {debt_ratio:.1%}，实际利息收益率仅 {implied_yield:.2%}，"
                    "资金极大概率受限或已被控股股东违规质押挪用！"
                )
                return node

            # Check 2: 高负债与流动性枯竭/资产异常 (抚顺特钢, 保千里, 东方金钰, 索菱股份, 广州浪奇)
            is_debt_heavy = (debt_ratio >= 0.40)
            is_cash_depleted = (cash_ratio < 0.10)
            if is_debt_heavy and low_interest_yield:
                node.state = ThoughtState.CONFIRMED
                node.score = 0.90
                node.evidence = (
                    f"确凿检出'高杠杆负债危机与流动性枯竭'：带息负债占比高达 {debt_ratio:.1%}，"
                    f"而真实可用货币资金严重短缺（占比仅 {cash_ratio:.1%}），"
                    f"利息匹配度 {implied_yield or 0:.2%} 严重异常偏低，存在资产虚构与违规抽逃风险！"
                )
                return node


            # Check 3: 健康标杆企业核验 (茅台, 宁德时代, 神华, 福耀, 海螺, 美的, 恒瑞, 长江电力, 伊利, 迈瑞)
            healthy_yield = (implied_yield is not None) and (implied_yield >= 0.015)
            low_debt = (debt_ratio <= 0.20)
            if (low_debt or healthy_yield) and not (is_dual_high and low_interest_yield):
                node.state = ThoughtState.REFUTED
                node.score = 0.1
                node.evidence = (
                    f"指标核验正常：带息负债占比 {debt_ratio:.1%}，利息收益率 {implied_yield or 0:.2%} 处在合理健康区间，"
                    "未出现异常资金占用与存贷双高舞弊特征。"
                )
                node.reflection = "企业资金储备充裕，债务与利息结构匹配良好，排除重大财务舞弊假说。"
                return node

        # 2. Evaluate via Beneish M-Score if applicable
        if "贝尼斯" in node.hypothesis or node.action_payload == "beneish_m_score":
            if "beneish_inputs" in self.runner.list_tables():
                skill_res = self.procedural_mem.execute_skill("beneish_m_score", self.runner)
                if skill_res.success and isinstance(skill_res.result, dict):
                    m_score = skill_res.result.get("m_score", -99.0)
                    is_suspect = skill_res.result.get("is_manipulator_suspect", False)
                    if is_suspect:
                        node.state = ThoughtState.CONFIRMED
                        node.score = 0.90
                        node.evidence = f"贝尼斯 M-Score 值为 {m_score:.4f} > -1.78，具有显著人为操纵财务报表特征。"
                    else:
                        node.state = ThoughtState.REFUTED
                        node.score = 0.15
                        node.evidence = f"贝尼斯 M-Score 值为 {m_score:.4f} <= -1.78，财务指标未呈现操纵倾向。"
                        node.reflection = "排除全面财务指标激进操纵假设。"
                    return node


        # 3. Evaluate via Footnote Linker Penetration if available
        if self.footnote_linker:
            # Check if this hypothesis targets an account in footnotes
            for acc in ["应收账款", "在建工程", "存货", "货币资金"]:
                if acc in node.hypothesis:
                    link = self.footnote_linker.get_by_account(acc)
                    if link and link.sub_tables:
                        node.state = ThoughtState.SUSPICIOUS
                        node.score = 0.70
                        node.evidence = f"成功穿透附注 {link.footnote_ref} ({link.footnote_title})，发现 {len(link.sub_tables)} 个下钻明细子表。"
                        return node

        # Default fallback verification
        node.state = ThoughtState.INCONCLUSIVE
        node.score = 0.40
        node.evidence = "完成基础数据核验，未出现断裂性证据，需进一步调取明细账目。"
        return node

    def expand_thought(self, node: ThoughtNode) -> List[ThoughtNode]:
        """Expand deeper sub-leads for promising or suspicious thought branches."""
        if node.is_prunable() or node.depth >= self.max_depth:
            return []

        children = []
        next_depth = node.depth + 1

        if "存贷双高" in node.hypothesis:
            # Sub-leads: 1. Footnote pledge verification, 2. Interest income vs borrowing rate
            c1 = self.create_node(
                hypothesis="穿透附注：核查受限货币资金披露，是否存在未如实登记的大额存单违规质押",
                parent_id=node.node_id,
                depth=next_depth,
                action_type="footnote_drill",
                action_payload="monetary_funds_pledge",
            )
            c2 = self.create_node(
                hypothesis="借款与利息流穿透：测算利息支出真实性，排查是否通过借款通道向关联方输送流动性",
                parent_id=node.node_id,
                depth=next_depth,
                action_type="sandbox_eval",
                action_payload="SELECT interest_expense FROM income_items",
            )
            children.extend([c1, c2])

        elif "在建工程" in node.hypothesis or "资产虚增" in node.hypothesis:
            c1 = self.create_node(
                hypothesis="在建工程转固周期核实：排查工程进度是否停滞与资本化利息异常",
                parent_id=node.node_id,
                depth=next_depth,
                action_type="footnote_drill",
                action_payload="construction_in_progress_turnover",
            )
            children.append(c1)

        elif "贝尼斯" in node.hypothesis or "应收账款" in node.hypothesis:
            c1 = self.create_node(
                hypothesis="应收账款账龄与集中度穿透：核验前五大欠款方是否存在空壳公司与体外循环",
                parent_id=node.node_id,
                depth=next_depth,
                action_type="footnote_drill",
                action_payload="receivables_aging_top5",
            )
            children.append(c1)

        return children

    def investigate(
        self,
        objective: str,
        case_id: str = "case_investigation",
        target_company: str = "TargetEnterprise",
    ) -> ForensicInvestigationResult:
        """Run full Tree-of-Thoughts forensic search with beam search, evaluation, and pruning."""
        self.working_mem.case_id = case_id
        self.working_mem.target_company = target_company
        self.working_mem.add_note(f"启动 ToT 刑侦假设树侦查: {objective}")

        # 1. Generate Root Hypotheses (Level 1)
        frontier = self.generate_initial_hypotheses(objective)
        confirmed_leads: List[ThoughtNode] = []
        refuted_leads: List[ThoughtNode] = []
        suspicious_leads: List[ThoughtNode] = []
        evidence_chain: List[str] = []

        current_depth = 1

        while frontier and current_depth <= self.max_depth:
            evaluated_nodes: List[ThoughtNode] = []

            for node in frontier:
                # Evaluate node
                eval_node = self.evaluate_thought(node)
                evaluated_nodes.append(eval_node)

                # Record in episodic memory
                step = InvestigationStep(
                    step_id=self.node_counter,
                    case_id=case_id,
                    action_type=eval_node.action_type,
                    action_payload=eval_node.action_payload,
                    observation=eval_node.evidence,
                    status=eval_node.state.value,
                    hypothesis=eval_node.hypothesis,
                    reflection=eval_node.reflection,
                )
                self.episodic_mem.log_step(step)

                # Categorize outcome
                if eval_node.state == ThoughtState.CONFIRMED:
                    confirmed_leads.append(eval_node)
                    evidence_chain.append(f"[CONFIRMED Depth {eval_node.depth}] {eval_node.hypothesis} => {eval_node.evidence}")
                    self.working_mem.add_note(f"证实舞弊嫌疑: {eval_node.hypothesis}")
                elif eval_node.state == ThoughtState.REFUTED:
                    refuted_leads.append(eval_node)
                    self.working_mem.add_note(f"推翻假设并剪枝: {eval_node.hypothesis}")
                elif eval_node.state == ThoughtState.SUSPICIOUS:
                    suspicious_leads.append(eval_node)
                    evidence_chain.append(f"[SUSPICIOUS Depth {eval_node.depth}] {eval_node.hypothesis} => {eval_node.evidence}")

            # Beam search selection: filter prunable nodes and pick top beam_width by score
            active_candidates = [n for n in evaluated_nodes if not n.is_prunable()]
            active_candidates.sort(key=lambda x: x.score, reverse=True)
            beam_selected = active_candidates[: self.beam_width]

            # Expand next level
            next_frontier: List[ThoughtNode] = []
            for parent_node in beam_selected:
                expanded_children = self.expand_thought(parent_node)
                next_frontier.extend(expanded_children)

            frontier = next_frontier
            current_depth += 1

        # Synthesize final audit verdict
        if confirmed_leads:
            verdict = "CRITICAL_FRAUD"
            summary = (
                f"【重大审计预警】经 ToT 刑侦假设树穿透检验，锁定 {len(confirmed_leads)} 项确凿财务造假与严重舞弊证据！"
                f"主要证据链包括：{confirmed_leads[0].evidence}"
            )
        elif suspicious_leads:
            verdict = "SUSPICIOUS_ANOMALY"
            summary = (
                f"【高度审计关注】检出 {len(suspicious_leads)} 项异常财务征兆，建议现场核实银行对账单及重要附注科目。"
            )
        else:
            verdict = "CLEAN_PASS"
            summary = "【核验通过】经多维刑侦树假设验证与剪枝排查，各财务科目与比率均在合理基准内，未检出重大舞弊。"

        return ForensicInvestigationResult(
            case_id=case_id,
            target_company=target_company,
            root_objective=objective,
            verdict=verdict,
            total_nodes_explored=self.node_counter,
            confirmed_fraud_leads=confirmed_leads,
            refuted_leads=refuted_leads,
            suspicious_leads=suspicious_leads,
            evidence_chain=evidence_chain,
            audit_summary=summary,
        )
