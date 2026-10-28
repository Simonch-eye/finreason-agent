"""Monte Carlo Tree Search (MCTS) Forensic Investigation Engine (蒙特卡洛刑侦树终局模拟与回传引擎).
Implements 4-stage MCTS (Selection, Expansion, Simulation/Rollout, Backpropagation) with UCB1/PUCT
for high-stakes financial fraud detection and deterministic sandbox verification.
"""

from __future__ import annotations

import math
import time
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
from finreason.sandbox import CodeRunner


class MCTSNodeState(BaseModel):
    """Snapshot state of an MCTS tree node for serialization and inspection."""
    node_id: str
    parent_id: Optional[str] = None
    action: str
    action_type: str = "sandbox_eval"
    visits: int = 0
    total_value: float = 0.0
    q_value: float = 0.0
    depth: int = 0
    state: str = "unexplored"
    evidence: str = ""
    children_ids: List[str] = Field(default_factory=list)


class MCTSNode:
    """A search tree node in the Monte Carlo Tree Search."""

    def __init__(
        self,
        node_id: str,
        action: str,
        parent: Optional[MCTSNode] = None,
        depth: int = 0,
        action_type: str = "sandbox_eval",
        action_payload: str = "",
    ):
        self.node_id = node_id
        self.action = action
        self.parent = parent
        self.depth = depth
        self.action_type = action_type
        self.action_payload = action_payload

        self.children: List[MCTSNode] = []
        self.visits: int = 0
        self.total_value: float = 0.0
        self.evidence: str = ""
        self.state: str = "unexplored"  # unexplored, confirmed, refuted, suspicious, inconclusive

    @property
    def q_value(self) -> float:
        """Mean action value Q(s, a)."""
        return self.total_value / self.visits if self.visits > 0 else 0.0

    def ucb1(self, c_param: float = 1.414) -> float:
        """Compute Upper Confidence Bound 1 (UCB1) score for action selection."""
        if self.visits == 0:
            return float("inf")  # Prioritize unvisited nodes
        if not self.parent or self.parent.visits == 0:
            return self.q_value
        exploitation = self.q_value
        exploration = c_param * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def is_fully_expanded(self, max_children: int = 3) -> bool:
        """Check if node has reached its branch capacity."""
        return len(self.children) >= max_children

    def best_child(self, c_param: float = 1.414) -> MCTSNode:
        """Select best child based on UCB1 metric."""
        if not self.children:
            raise ValueError("Cannot select from a node with no children.")
        return max(self.children, key=lambda c: c.ucb1(c_param))

    def to_state(self) -> MCTSNodeState:
        """Serialize node to Pydantic state model."""
        return MCTSNodeState(
            node_id=self.node_id,
            parent_id=self.parent.node_id if self.parent else None,
            action=self.action,
            action_type=self.action_type,
            visits=self.visits,
            total_value=round(self.total_value, 4),
            q_value=round(self.q_value, 4),
            depth=self.depth,
            state=self.state,
            evidence=self.evidence,
            children_ids=[c.node_id for c in self.children],
        )


class MCTSInvestigationResult(BaseModel):
    """Final synthesis report produced by the MCTS forensic search."""
    case_id: str
    target_company: str
    root_objective: str
    verdict: str = Field(description="Audit verdict: CRITICAL_FRAUD, SUSPICIOUS_ANOMALY, CLEAN_PASS")
    total_simulations: int
    total_nodes_created: int
    best_trajectory: List[MCTSNodeState]
    evidence_chain: List[str]
    audit_summary: str
    max_risk_q_score: float
    duration_ms: float


class MCTSEngine:
    """Monte Carlo Tree Search (MCTS) Engine for financial fraud detection."""

    def __init__(
        self,
        runner: CodeRunner,
        working_mem: Optional[WorkingMemory] = None,
        episodic_mem: Optional[EpisodicMemory] = None,
        semantic_mem: Optional[SemanticMemory] = None,
        procedural_mem: Optional[ProceduralMemory] = None,
        footnote_linker: Optional[FootnoteLinker] = None,
        exploration_constant: float = 1.414,
        max_simulations: int = 30,
        max_depth: int = 4,
    ):
        self.runner = runner
        self.working_mem = working_mem or WorkingMemory()
        self.episodic_mem = episodic_mem or EpisodicMemory(db_path=":memory:")
        self.semantic_mem = semantic_mem or SemanticMemory()
        self.procedural_mem = procedural_mem or ProceduralMemory()
        self.footnote_linker = footnote_linker
        self.c_param = exploration_constant
        self.max_simulations = max_simulations
        self.max_depth = max_depth

        self.node_counter: int = 0
        self.all_nodes: Dict[str, MCTSNode] = {}

    def _create_node(
        self,
        action: str,
        parent: Optional[MCTSNode] = None,
        depth: int = 0,
        action_type: str = "sandbox_eval",
        action_payload: str = "",
    ) -> MCTSNode:
        """Instantiate and index a search tree node."""
        self.node_counter += 1
        node_id = f"mcts_{self.node_counter}"
        node = MCTSNode(
            node_id=node_id,
            action=action,
            parent=parent,
            depth=depth,
            action_type=action_type,
            action_payload=action_payload,
        )
        self.all_nodes[node_id] = node
        if parent:
            parent.children.append(node)
        return node

    # ==========================================
    # 1. Selection (选择阶段)
    # ==========================================
    def select(self, root: MCTSNode) -> MCTSNode:
        """Traverse tree down to an unexpanded or leaf node using UCB1."""
        current = root
        while current.children and current.depth < self.max_depth:
            # If any child is unvisited, explore it immediately
            unvisited = [c for c in current.children if c.visits == 0]
            if unvisited:
                return unvisited[0]
            # Otherwise use UCB1 to pick the most promising child
            current = current.best_child(self.c_param)
        return current

    # ==========================================
    # 2. Expansion (扩展阶段)
    # ==========================================
    def expand(self, node: MCTSNode) -> List[MCTSNode]:
        """Expand candidate investigative sub-actions for the selected node."""
        if node.depth >= self.max_depth or node.children:
            return node.children

        next_depth = node.depth + 1
        candidates: List[tuple[str, str, str]] = []  # (action, action_type, payload)

        if node.depth == 0:
            # Root actions derived from objective and semantic memory
            candidates.append((
                "核验'存贷双高'与巨额货币资金真实收益率（排查违规质押与抽逃占用）",
                "sandbox_eval",
                "check_dual_high",
            ))
            candidates.append((
                "核算带息负债偿债压力与流动性枯竭风险（排查资产虚增与表外隐匿）",
                "sandbox_eval",
                "check_liquidity_crisis",
            ))
            candidates.append((
                "测算贝尼斯 M-Score 8维财务操纵指数（排查激进盈余管理）",
                "skill_exec",
                "beneish_m_score",
            ))
            candidates.append((
                "穿透重大在建工程项目附注（排查长期不转固与利息资本化造假）",
                "footnote_drill",
                "construction_in_progress",
            ))
        elif "存贷双高" in node.action:
            candidates.append((
                "下钻受限货币资金披露：核实银行承兑汇票保证金与违规质押担保",
                "footnote_drill",
                "monetary_funds_pledge",
            ))
            candidates.append((
                "对比借款利息支出与存款利息收入利率偏离度",
                "sandbox_eval",
                "interest_spread_analysis",
            ))
        elif "流动性" in node.action or "负债" in node.action:
            candidates.append((
                "核查短期借款续贷情况与应付票据逾期风险",
                "sandbox_eval",
                "debt_repayment_risk",
            ))
            candidates.append((
                "排查控股股东大比例股票质押风险与被动平仓警报",
                "footnote_drill",
                "shareholder_pledge",
            ))
        elif "在建工程" in node.action or "资产" in node.action:
            candidates.append((
                "核查在建工程转固进度与工程预算超支真实性",
                "footnote_drill",
                "construction_in_progress_turnover",
            ))
            candidates.append((
                "排查无形资产与研发支出资本化比例异常偏高",
                "sandbox_eval",
                "rd_capitalization_anomaly",
            ))
        else:
            candidates.append((
                "综合校验三表勾稽与杜邦指标拆解匹配度",
                "sandbox_eval",
                "dupont_reconciliation",
            ))

        new_children = []
        for action_text, a_type, payload in candidates[:3]:
            child = self._create_node(
                action=action_text,
                parent=node,
                depth=next_depth,
                action_type=a_type,
                action_payload=payload,
            )
            new_children.append(child)

        return new_children

    # ==========================================
    # 3. Simulation / Rollout (终局模拟与快速走子)
    # ==========================================
    def simulate(self, node: MCTSNode) -> float:
        """Perform deterministic sandbox validation or heuristic rollout, returning reward V in [0.0, 1.0].
        Reward interpretation:
          1.0 = Definite evidence of critical fraud / severe anomaly
          0.0 = Clear evidence of healthy benchmark / standard compliance
        """
        # Step A: Check episodic memory for previous refutation (counterfactual pruning)
        refuted = self.episodic_mem.get_refuted_hypotheses(self.working_mem.case_id)
        if any(r in node.action or node.action in r for r in refuted):
            node.state = "refuted"
            node.evidence = "依据 Reflexion 侦查记忆，此方向已被证伪，直接赋予零舞弊回报。"
            return 0.0

        # Step B: Sandbox deterministic calculation
        reward = 0.3  # Baseline neutral prior
        tables = self.runner.list_tables()

        if "balance_items" in tables:
            df_b = self.runner.get_table_df("balance_items")
            b_dict = dict(zip(df_b["item"], df_b["val"]))
            tot_assets = max(b_dict.get("total_assets", 1.0), 1.0)
            cash = b_dict.get("monetary_funds", 0.0)
            debt = b_dict.get("interest_bearing_debt", 0.0)
            cash_ratio = cash / tot_assets
            debt_ratio = debt / tot_assets

            implied_yield = None
            if "income_items" in tables:
                df_i = self.runner.get_table_df("income_items")
                i_dict = dict(zip(df_i["item"], df_i["val"]))
                int_inc = i_dict.get("interest_income", 0.0)
                if cash > 0:
                    implied_yield = int_inc / cash

            is_dual_high = (cash_ratio >= 0.20) and (debt_ratio >= 0.20)

            low_interest_yield = (implied_yield is not None) and (implied_yield < 0.008)

            # Condition 1: 存贷双高 + 超低利息收益率 (康美/康得新等经典雷)
            if is_dual_high and low_interest_yield:
                node.state = "confirmed"
                node.evidence = (
                    f"MCTS 终局模拟命中致命造假模式：货币资金占比 {cash_ratio:.1%}，"
                    f"带息负债占比 {debt_ratio:.1%}，利息收益率仅 {implied_yield:.2%}，高度确认巨额资金受限或被占用！"
                )
                return 0.98

            # Condition 2: 高负债高杠杆 + 流动性极度枯竭
            is_debt_heavy = (debt_ratio >= 0.40)
            if is_debt_heavy and low_interest_yield:
                node.state = "confirmed"
                node.evidence = (
                    f"MCTS 终局模拟命中高危杠杆异动：带息负债占比高达 {debt_ratio:.1%}，"
                    f"可用货币资金微薄（{cash_ratio:.1%}），利息收益率 {implied_yield or 0:.2%} 严重失真。"
                )
                return 0.92

            # Condition 3: 蓝筹稳健健康企业
            healthy_yield = (implied_yield is not None) and (implied_yield >= 0.015)
            low_debt = (debt_ratio <= 0.20)
            if (low_debt or healthy_yield) and not (is_dual_high and low_interest_yield):
                node.state = "refuted"
                node.evidence = (
                    f"MCTS 终局模拟确认健康底色：带息负债占比 {debt_ratio:.1%}，"
                    f"利息收益率 {implied_yield or 0:.2%}，各项资金指标均处于行业健康分位。"
                )
                return 0.05

        # Step C: Procedural skill execution if requested
        if "beneish_m_score" in node.action_payload or "贝尼斯" in node.action:
            if "beneish_inputs" in tables:
                skill_res = self.procedural_mem.execute_skill("beneish_m_score", self.runner)
                if skill_res.success and isinstance(skill_res.result, dict):
                    m_score = skill_res.result.get("m_score", -99.0)
                    if skill_res.result.get("is_manipulator_suspect", False):
                        node.state = "confirmed"
                        node.evidence = f"贝尼斯 M-Score 为 {m_score:.4f} > -1.78，大概率存在人为报表操纵。"
                        return 0.90
                    else:
                        node.state = "refuted"
                        node.evidence = f"贝尼斯 M-Score 为 {m_score:.4f} <= -1.78，指标未见明显操纵。"
                        return 0.10

        # Step D: Footnote linkage drill-down
        if self.footnote_linker and node.action_type == "footnote_drill":
            for acc in ["应收账款", "在建工程", "存货", "货币资金"]:
                if acc in node.action:
                    link = self.footnote_linker.get_by_account(acc)
                    if link and link.sub_tables:
                        node.state = "suspicious"
                        node.evidence = f"穿透附注 {link.footnote_ref} 获取到 {len(link.sub_tables)} 个明细表。"
                        return 0.70

        # Fallback inconclusive
        node.state = "inconclusive"
        node.evidence = "完成初步模拟核算，证据尚需多维度交叉印证。"
        return reward

    # ==========================================
    # 4. Backpropagation (价值回传阶段)
    # ==========================================
    def backpropagate(self, node: MCTSNode, reward: float):
        """Propagate simulation outcome value up to root."""
        current: Optional[MCTSNode] = node
        while current is not None:
            current.visits += 1
            current.total_value += reward
            current = current.parent

    # ==========================================
    # Core Search Loop (主搜索循环)
    # ==========================================
    def search(
        self,
        objective: str,
        case_id: str = "mcts_case",
        target_company: str = "TargetEnterprise",
    ) -> MCTSInvestigationResult:
        """Run full MCTS simulation loop for financial fraud investigation."""
        start_time = time.perf_counter()

        self.working_mem.case_id = case_id
        self.working_mem.target_company = target_company
        self.working_mem.add_note(f"启动 MCTS 蒙特卡洛刑侦模拟搜索: {objective}")

        # Initialize root node
        root = self._create_node(
            action=f"审计根任务: {objective}",
            depth=0,
            action_type="root_objective",
        )

        # MCTS Iteration Loop
        for sim_idx in range(self.max_simulations):
            # 1. Selection
            selected_node = self.select(root)

            # 2. Expansion
            if selected_node.visits > 0 and selected_node.depth < self.max_depth:
                children = self.expand(selected_node)
                if children:
                    selected_node = children[0]

            # 3. Simulation / Rollout
            reward = self.simulate(selected_node)

            # 4. Backpropagation
            self.backpropagate(selected_node, reward)

            # Log step in episodic memory
            step_reflection = (
                f"已被证伪: {selected_node.action}"
                if selected_node.state == "refuted"
                else None
            )
            step = InvestigationStep(
                step_id=sim_idx + 1,
                case_id=case_id,
                action_type=selected_node.action_type,
                action_payload=selected_node.action_payload,
                observation=selected_node.evidence,
                status=selected_node.state,
                hypothesis=selected_node.action,
                reflection=step_reflection,
            )
            self.episodic_mem.log_step(step)


        # Trace best trajectory from root
        best_trajectory_nodes: List[MCTSNode] = [root]
        current = root
        while current.children:
            # Pick most visited child (robust child strategy)
            most_visited = max(current.children, key=lambda c: (c.visits, c.q_value))
            best_trajectory_nodes.append(most_visited)
            current = most_visited

        # Evaluate final verdict based on maximum forensic risk observed
        max_q = max((n.q_value for n in self.all_nodes.values() if n.visits > 0), default=0.0)
        evidence_chain = [
            f"[{n.node_id}] {n.action} -> {n.evidence} (Q={n.q_value:.2f}, N={n.visits})"
            for n in best_trajectory_nodes
            if n.evidence
        ]

        if max_q >= 0.75:
            verdict = "CRITICAL_FRAUD"
            summary = (
                f"【重大舞弊审计预警】MCTS 蒙特卡洛终局模拟最高造假置信度达到 {max_q:.1%}。"
                f"对 {target_company} 的深入推演命中重大财务舞弊特征，检出核心异常证据链。"
            )
        elif max_q >= 0.50:
            verdict = "SUSPICIOUS_ANOMALY"
            summary = (
                f"【异常财务异动提示】MCTS 终局模拟置信度为 {max_q:.1%}，"
                f"{target_company} 存在多项值得高度怀疑的财务指标背离，建议重点调取底稿穿透。"
            )
        else:
            verdict = "CLEAN_PASS"
            summary = (
                f"【未发现重大财务异常】MCTS 终局模拟综合风险置信度仅为 {max_q:.1%}，"
                f"{target_company} 财务结构健康稳健，各项勾稽指标处于合规标准区间。"
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return MCTSInvestigationResult(
            case_id=case_id,
            target_company=target_company,
            root_objective=objective,
            verdict=verdict,
            total_simulations=self.max_simulations,
            total_nodes_created=len(self.all_nodes),
            best_trajectory=[n.to_state() for n in best_trajectory_nodes],
            evidence_chain=evidence_chain,
            audit_summary=summary,
            max_risk_q_score=round(max_q, 4),
            duration_ms=round(elapsed_ms, 2),
        )
