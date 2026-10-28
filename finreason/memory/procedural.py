"""Procedural Memory (现成审计代码技能注册表).
Maintains hardened, pre-audited Python routines for forensic accounting and financial analysis.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from finreason.sandbox import CodeRunner, ExecutionResult


class AuditSkill(BaseModel):
    """An executable financial analysis or forensic audit skill."""
    skill_id: str = Field(description="Unique skill slug (e.g. dupont_analysis, beneish_m_score)")
    name: str = Field(description="Human-readable skill name")
    description: str = Field(description="Audit objective and calculation summary")
    category: str = Field(description="Category: financial_ratio, forensic_audit, articulation_check")
    required_tables: List[str] = Field(default_factory=list, description="Tables required in DuckDB session")
    code_template: str = Field(description="Executable Python code snippet executed in sandbox")


# Pre-defined, hardened forensic and ratio skills
BUILTIN_SKILLS: List[AuditSkill] = [
    AuditSkill(
        skill_id="dupont_analysis",
        name="杜邦三分解净资产收益率 (DuPont 3-Way ROE)",
        category="financial_ratio",
        description="将净资产收益率拆解为销售净利率 × 资产周转率 × 权益乘数，并校验恒等关系。",
        required_tables=["financial_metrics"],
        code_template="""
df = con.execute("SELECT item, val FROM financial_metrics").fetchdf()
m = dict(zip(df['item'], df['val']))

net_margin = safe_ratio(m['net_income'], m['revenue'])
asset_turnover = safe_ratio(m['revenue'], m['total_assets'])
equity_multiplier = safe_ratio(m['total_assets'], m['shareholder_equity'])

roe = net_margin * asset_turnover * equity_multiplier
direct_roe = safe_ratio(m['net_income'], m['shareholder_equity'])

chk = verify_equation("DuPont_Identity", roe, direct_roe)
result = {
    "net_margin": net_margin,
    "asset_turnover": asset_turnover,
    "equity_multiplier": equity_multiplier,
    "roe": roe,
    "balanced": chk["balanced"]
}
print(f"DuPont Analysis: ROE={roe:.4f}, NetMargin={net_margin:.4f}, AssetTurnover={asset_turnover:.4f}")
""",
    ),
    AuditSkill(
        skill_id="cash_debt_anomaly",
        name="存贷双高舞弊异动指数 (Cash-Debt Anomaly Index)",
        category="forensic_audit",
        description="检测货币资金与带息负债'双高'异动，结合利息收益率背离排查受限资金隐瞒。",
        required_tables=["balance_items", "income_items"],
        code_template="""
b_df = con.execute("SELECT item, val FROM balance_items").fetchdf()
b = dict(zip(b_df['item'], b_df['val']))
i_df = con.execute("SELECT item, val FROM income_items").fetchdf()
inc = dict(zip(i_df['item'], i_df['val']))

cash_ratio = safe_ratio(b.get('monetary_funds', 0.0), b.get('total_assets', 1.0))
debt_ratio = safe_ratio(b.get('interest_bearing_debt', 0.0), b.get('total_assets', 1.0))

# 实际利息收益率
implied_deposit_rate = safe_ratio(inc.get('interest_income', 0.0), b.get('monetary_funds', 1.0))
# 实际借款利息率
implied_borrowing_rate = safe_ratio(inc.get('interest_expense', 0.0), b.get('interest_bearing_debt', 1.0))

# 存贷双高风险判定: 货币资金占比>20% 且 借款占比>20%
is_dual_high = (cash_ratio > 0.20) and (debt_ratio > 0.20)
# 利息收益异动: 利息收益率严重偏低(例如低于0.8%)
interest_anomaly = (implied_deposit_rate is not None) and (implied_deposit_rate < 0.008)

risk_level = "HIGH" if (is_dual_high and interest_anomaly) else ("MEDIUM" if is_dual_high else "LOW")

result = {
    "cash_ratio": cash_ratio,
    "debt_ratio": debt_ratio,
    "implied_deposit_rate": implied_deposit_rate,
    "implied_borrowing_rate": implied_borrowing_rate,
    "is_dual_high": is_dual_high,
    "interest_anomaly": interest_anomaly,
    "risk_level": risk_level
}
print(f"存贷双高排查: 风险等级={risk_level}, 现金占比={cash_ratio:.2%}, 负债占比={debt_ratio:.2%}")
""",
    ),
    AuditSkill(
        skill_id="beneish_m_score",
        name="贝尼斯财务舞弊 M-Score (Beneish M-Score)",
        category="forensic_audit",
        description="8变量财务操纵预测模型。若 M-Score > -1.78，提示企业存在极高财务粉饰与造假嫌疑。",
        required_tables=["beneish_inputs"],
        code_template="""
df = con.execute("SELECT var, val FROM beneish_inputs").fetchdf()
v = dict(zip(df['var'], df['val']))

dsri = v.get('dsri', 1.0) # 应收账款指数
gmi = v.get('gmi', 1.0)   # 毛利率指数
aqi = v.get('aqi', 1.0)   # 资产质量指数
sgi = v.get('sgi', 1.0)   # 销售增长指数
depi = v.get('depi', 1.0) # 折旧率指数
sgai = v.get('sgai', 1.0) # 销售管理费用指数
lvgi = v.get('lvgi', 1.0) # 财务杠杆指数
tata = v.get('tata', 0.0) # 应计总额与总资产比

m_score = (
    -4.84
    + 0.920 * dsri
    + 0.528 * gmi
    + 0.404 * aqi
    + 0.892 * sgi
    + 0.115 * depi
    - 0.172 * sgai
    + 4.037 * tata
    + 0.0327 * lvgi
)

is_manipulator_suspect = m_score > -1.78

result = {
    "m_score": m_score,
    "threshold": -1.78,
    "is_manipulator_suspect": is_manipulator_suspect,
    "components": {
        "dsri": dsri, "gmi": gmi, "aqi": aqi, "sgi": sgi,
        "depi": depi, "sgai": sgai, "tata": tata, "lvgi": lvgi
    }
}
print(f"Beneish M-Score: {m_score:.4f} (Suspect Fraud: {is_manipulator_suspect})")
""",
    ),
    AuditSkill(
        skill_id="altman_z_score",
        name="奥特曼破产风险 Z-Score (Altman Z-Score)",
        category="financial_ratio",
        description="用于预测制造型企业破产财务困境风险 (Z > 2.99 安全，Z < 1.81 破产高危)。",
        required_tables=["altman_inputs"],
        code_template="""
df = con.execute("SELECT var, val FROM altman_inputs").fetchdf()
v = dict(zip(df['var'], df['val']))

x1 = safe_ratio(v['working_capital'], v['total_assets'])
x2 = safe_ratio(v['retained_earnings'], v['total_assets'])
x3 = safe_ratio(v['ebit'], v['total_assets'])
x4 = safe_ratio(v['market_equity'], v['total_liabilities'])
x5 = safe_ratio(v['sales'], v['total_assets'])

z_score = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 0.999 * x5

zone = "SAFE" if z_score > 2.99 else ("GREY" if z_score >= 1.81 else "DISTRESS")

result = {
    "z_score": z_score,
    "zone": zone,
    "x1_working_capital": x1,
    "x2_retained_earnings": x2,
    "x3_ebit": x3,
    "x4_market_equity": x4,
    "x5_sales": x5
}
print(f"Altman Z-Score: {z_score:.4f} (Zone: {zone})")
""",
    ),
]


class ProceduralMemory:
    """Registry and execution orchestrator for hardened audit skills."""

    def __init__(self, seed_defaults: bool = True):
        self.skills: Dict[str, AuditSkill] = {}
        if seed_defaults:
            for s in BUILTIN_SKILLS:
                self.register_skill(s)

    def register_skill(self, skill: AuditSkill) -> None:
        """Register an audited skill into procedural memory."""
        self.skills[skill.skill_id] = skill

    def get_skill(self, skill_id: str) -> Optional[AuditSkill]:
        """Fetch skill by unique id."""
        return self.skills.get(skill_id)

    def has_skill(self, skill_id: str) -> bool:
        """Check if skill is registered in procedural memory."""
        return skill_id in self.skills


    def list_skills(self, category: Optional[str] = None) -> List[AuditSkill]:
        """List all available skills filtered optionally by category."""
        if category:
            return [s for s in self.skills.values() if s.category == category]
        return list(self.skills.values())

    def search_skills(self, query: str) -> List[AuditSkill]:
        """Search available skills matching a forensic requirement."""
        q = query.lower()
        matched = []
        for s in self.skills.values():
            if (q in s.skill_id.lower() or
                q in s.name.lower() or
                q in s.description.lower() or
                q in s.category.lower()):
                matched.append(s)
        return matched

    def execute_skill(
        self,
        skill_id: str,
        runner: CodeRunner,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Execute a registered skill in the deterministic sandbox."""
        skill = self.get_skill(skill_id)
        if not skill:
            return ExecutionResult(
                success=False,
                error=f"Skill '{skill_id}' not found in procedural memory.",
                tables=runner.list_tables(),
            )

        # Check required tables presence
        current_tables = runner.list_tables()
        missing = [t for t in skill.required_tables if t not in current_tables]
        if missing:
            return ExecutionResult(
                success=False,
                error=f"Cannot execute skill '{skill_id}': Missing required tables in DuckDB: {missing}",
                tables=current_tables,
            )

        return runner.execute_python(code=skill.code_template, context=context)
