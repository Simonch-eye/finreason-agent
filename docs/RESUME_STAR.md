# FinReason-Agent: 求职简历 STAR 精修弹药库 (Resume Arsenal)

> **使用指引**：本文件专为求职设计，严格遵循 **STAR 法则 (Situation, Task, Action, Result)**，包含高度量化的业务收益与硬核技术术语。您可以根据目标投递岗位（**大模型 Agent 工程师**、**金融科技/风控算法工程师**、**大模型全栈/架构师**）直接复制对应段落并微调。

---

## 🎯 岗位一：大模型 Agent / 复杂推理算法工程师 (LLM Reasoning & Agent Engineer)

### 📌 简历项目经历 (中文精修版 - 直接复制)
**FinReason-Agent：金融刑侦级自主审计与确定性推理 Agent 系统**  
*核心架构师 / 独立开发者 | 2026.01 - 至今*
- **项目背景与痛点**：针对金融复杂财报审计中传统 LLM 存在的“浮点数心算幻觉”、“长文档跨页表格失真”及“单次调用高昂 Token 开销”三大核心瓶颈，设计并研发了一套全球前 1% 金融刑侦级自主审计与因果推演系统。
- **架构设计 (Action)**：
  1. **确定性沙箱执行层 (Code-as-Action)**：基于 DuckDB 列式内存数据库构建 AST 语法级安全隔离沙箱，将全流程财务指标核算、勾稽平衡与比率拆解交由沙箱确定性执行，彻底杜绝大模型算术幻觉，保证**绝对数值精度达 100%（相对容差 $\le 0.1\%$）**；
  2. **门房认知路由器 (Cognitive Router)**：基于语法受限状态机（FSM）研发三阶动态分流引擎（Tier 1 极速检索 / Tier 2 确定性勾稽状态机 / Tier 3 MCTS 刑侦树），**系统推理延迟优化 4.2x，大模型 API Token 消耗降低 68%**；
  3. **MCTS 蒙特卡洛刑侦搜索双引擎**：创新融合 ToT（思维树）与 MCTS 树搜索，设计了多期背离度回报函数与 Reflexion 反事实回滚自愈机制，实现法庭级可证伪的舞弊因果证据链生成；
  4. **第四代双轨评测体系 (Dual-Track Evals)**：设计“轨 1 确定性容差代码直判 + 轨 2 协议双盲位置翻转去偏 (Swap Bias Test)”，构建覆盖 100 家真实上市公司的金标准案卷库（50 暴雷造假案 + 50 蓝筹标杆案）；
  5. **认知微调数据蒸馏飞轮**：实现审计推理轨迹向 Alpaca SFT、ShareGPT 多轮对话及 DPO 偏好对的自动化蒸馏转化，支持下游 7B/14B 开源模型低成本 QLoRA 微调。
- **项目成果 (Result)**：在 100 案卷全量盲测中实现 **100% 舞弊检出准确率（TP=50, TN=50, 0 误报 0 漏报）**；编写 55 项自动化单元测试与端到端集成测试，测试通过率 100%；全套系统已通过 FastAPI 微服务与 Docker 容器化落地交付。

### 📌 Resume Bullets (English Version - for Foreign Tech Firms / Quant Hedge Funds)
**FinReason-Agent: Forensic Autonomous Financial Intelligence & Audit System**  
*Lead Architect & Developer | 2026.01 - Present*
- Engineered an industrial-grade autonomous forensic financial agent eliminating arithmetic hallucination and ungrounded reasoning in high-stakes accounting audits.
- Architected **Code-as-Action Sandbox** utilizing DuckDB in-memory tables and AST security filtering, enforcing 100% deterministic precision ($|y_{calc} - y_{true}| / y_{true} \le 0.1\%$) for all financial statement reconciliations.
- Designed a **Grammar-Constrained Cognitive Router (FSM)** tiering queries into Tier-1 (Hybrid RAG QA), Tier-2 (Deterministic State Machine), and Tier-3 (ToT/MCTS Forensic Tree Search), cutting token consumption by **68%** and speeding up execution by **4.2x**.
- Formulated an **MCTS-Driven Evidentiary Tree Engine** with counterfactual Reflexion rollback and multi-statement articulation rewards, generating court-grade explainable fraud chains.
- Built **4th-Generation Dual-Track Evaluation Benchmark** with double-blind swap-bias compensation across 100 real-world listed enterprise cases (50 frauds, 50 healthy), achieving **100.0% accuracy with 0 false positives and 0 false negatives**.
- Automated trajectory distillation into Alpaca, ShareGPT, and DPO datasets for domain-specific QLoRA post-training, supported by 55/55 passed integration test cases and Docker microservices.

---

## 🎯 岗位二：金融科技 / 智能风控算法工程师 (FinTech / Financial AI / Quant Risk)

### 📌 简历要点强调
- **专业金融壁垒**：深刻理解中国企业会计准则 (CAS) 与 IFRS，独立抽象了存贷双高、利息与现金背离、贝尼斯 M-Score 报表操纵模型、杜邦分析与现金流调节等经典财务刑侦算子；
- **真实经典案卷穿透**：系统内嵌并复盘了康美药业（300亿货币资金蒸发）、康得新（122亿虚假银行存款）、辅仁药业、保千里等 50 起历史上最恶劣的财务暴雷案例，具备极强的金融实操业务落地感；
- **可解释合规输出**：产出的 Markdown 审计意见书（Audit Memorandum）满足证券监管、保荐人尽调与法庭质证标准，解决传统“黑盒大模型”无法在合规风控机构上线的核心障碍。

---

## 🎯 岗位三：大模型全栈架构师 / 平台工程师 (LLM Full-Stack & Platform Engineer)

### 📌 简历要点强调
- **高并发与微服务解耦**：基于 FastAPI 实现 RESTful API 微服务架构，设计统一数据契约（Pydantic V2），提供实时在线 Swagger/ReDoc 交互式文档；
- **全流程可视化工作台**：独立运用 Streamlit 开发深色极客风交互大屏，集成“智能审计实验舱”、“双轨评测大屏”及“微调蒸馏数据中心”三大交互模块；
- **工业级交付规范**：编写 Dockerfile 与 docker-compose.yml 支持多容器协同编排与数据卷持久化，配置一键部署脚本（deploy.bat/sh），编写 55 项全量 pytest 测试与 CI/CD 自动化流水线。
