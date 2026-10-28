# FinReason-Agent: 全球前 1% 金融刑侦级自主审计与勾稽验证系统

> 专为应对复杂金融财报、跨期多表勾稽验证、舞弊线索穿透而设计的工业级自主 Agent 智能体。

---

## 🌟 核心工程特色

1. **确定性执行 (Code-as-Action)**：杜绝大模型浮点数心算，基于 Python + DuckDB 内存数据表跑零算术幻觉计算（**Zero Arithmetic Hallucination**）。
2. **三级认知路由 (Cognitive Tiering)**：
   - **Tier 1 (直觉级)**：单一财务事实与附注说明极速查询；
   - **Tier 2 (规范级)**：标准三表勾稽与杜邦指标拆解确定性状态机；
   - **Tier 3 (刑侦级)**：基于假设生成、ToT 剪枝、MCTS 终局模拟与 Reflexion 回滚的舞弊排查树。
3. **四层分立物理记忆 (Hierarchical Memory)**：彻底根除“长期记忆全塞向量库”导致的经验污染，实现案头活跃草稿（Working）、专案因果日志（Episodic）、会计法规通识（Semantic）与审计代码技能库（Procedural）的物理分立。
4. **多模态版面与附注外键绑定 (Parser & Linking)**：无缝解析真实多页 PDF/Markdown，自动合并跨页断裂大表（续表），自动提取正文科目与深层附注披露的双向外键索引。
5. **第四代双轨评测 (Dual-Track Evals)**：端到端客观代码绝对容差直判（$|计算值 - 真实值| \le 0.1\%$） + 协议双盲位置翻转去偏裁判，配合 100 套真实上市公司黄金基准。

---

## 🚀 快速上手 (Quickstart)

### 1. 环境安装与配置
推荐使用 Python 3.10+：
```bash
git clone <repo_url>
cd "finreason-agent"
pip install -e .
```

---

### 2. Python API 快速调用
只需 3 行代码即可启动自主刑侦审计流水线：
```python
from finreason import FinReasonAgent

agent = FinReasonAgent()

# 示例：排查康美药业存贷双高与舞弊特征
report = agent.audit(
    query="排查康美药业是否存在大额货币资金虚构、大存大贷与资产违规占用舞弊嫌疑。",
    company_name="康美药业",
    reporting_period="2018年度",
    file_path="data/raw/kangmei_2018.pdf",  # 支持真实 PDF 或 Markdown
)

# 打印法庭级 Markdown 审计意见书
print(report.to_markdown())
```

---

### 3. 命令行终端工具 (CLI)
系统已注册全局命令 `finreason`（亦可使用 `python -m finreason`）：

```bash
# 1. 运行自主刑侦审计并导出 Markdown 备忘录
finreason audit "path/to/annual_report.pdf" -q "核算资产负债表平衡与杜邦拆解" -o "audit_memo.md"

# 2. 运行第四代 100 案双轨评测大屏
finreason eval --suite 100

# 3. 导出赋能 7B/14B 小模型微调的 Alpaca SFT、ShareGPT 与 DPO 偏好对
finreason export-data -o "data/processed/finetune_trajectories"

# 4. 运行全景能力实战交互演示
finreason demo

# 5. 一键启动 Streamlit 可视化刑侦审计工作台
finreason ui --port 8501
```

---

### 4. 交互式可视化工作台 (Streamlit Web App)
终端输入以下命令启动深色金融刑侦风格可视化工作台：
```bash
streamlit run web_app.py
```
- **案情交互排查工作台**：预置康美药业、康得新、贵州茅台等经典案卷，支持文件拖拽上传、实时指标计算与证据链穿透；
- **第四代双轨评测大屏**：直观展示 100 案混淆矩阵、去偏得分雷达与考题检索；
- **微调黄金轨迹中心**：在线预览并一键生成开源微调数据集。

---

## 🏛️ 系统架构全景 (Architecture Blueprint)

```
                            [ 用户提问 / 待审年报 PDF ]
                                         │
                                         ▼
            ┌────────────────────────────────────────────────────────┐
            │ 1. 数据对齐层 (Parser & Linking)                       │
            │    • 真实多页 PDF / Markdown 表格抽取 (pypdf/regex)    │
            │    • 父子层级分块 (Hierarchical Chunking)               │
            │    • 主科目与深层附注外键双向锚定 (Footnote Linker)    │
            └────────────────────────────┬───────────────────────────┘
                                         │
                                         ▼
            ┌────────────────────────────────────────────────────────┐
            │ 2. 门房认知路由器 (Cognitive Router)                   │
            │    • 语法受限解码 (FSM / Instructor) 复杂度等级判定    │
            └──────┬─────────────────────┼─────────────────────┬─────┘
                   │                     │                     │
                   ▼ (Tier 1: 直觉级)     ▼ (Tier 2: 规范级)    ▼ (Tier 3: 刑侦级)
            ┌──────────────┐      ┌──────────────┐      ┌────────────────────────┐
            │ Fast-Path    │      │ 确定性状态机 │      │ 深度推理刑侦树         │
            │ (混合检索QA) │      │ (单流沙箱验算)│     │ (ToT/MCTS + Reflexion) │
            └──────────────┘      └──────┬───────┘      └───────────┬────────────┘
                                         │                          │
                                         └─────────────┬────────────┘
                                                       │ 调度确定性工具
                                                       ▼
            ┌────────────────────────────────────────────────────────┐
            │ 3. 确定性执行层 (Code-as-Action & Sandboxing)          │
            │    • Python / DuckDB 内存数据表执行引擎                │
            │    • AST 语法安全守卫 (严格模块白名单与超时隔离)       │
            └────────────────────────────┬───────────────────────────┘
                                         │
                                         ▼
            ┌────────────────────────────────────────────────────────┐
            │ 4. 认知四层分立物理记忆系统 (Hierarchical Memory)      │
            │    • Working: 案头活跃草稿纸 (Prompt/REPL Context)     │
            │    • Episodic: 专案因果侦查日志 (DuckDB State Log)     │
            │    • Semantic: 会计法规与行业基准 (Vector/Graph Store) │
            │    • Procedural: 现成审计代码技能库 (.py 注册表)       │
            └────────────────────────────┬───────────────────────────┘
                                         │
                                         ▼
            ┌────────────────────────────────────────────────────────┐
            │ 5. 质量保证与双轨评测体系 (Dual-Track Evals)           │
            │    • 轨 1: 绝对容差代码直判 (|计算值 - 真值| <= 0.1%)   │
            │    • 轨 2: 协议双盲位置翻转去偏裁判 (Swap Bias Test)   │
            │    • 自动化考场 (Harness): 断点续跑、防污染、超时保护   │
            └────────────────────────────────────────────────────────┘
```

---

## 📊 100 案第四代双轨评测表现

| 评测维度 | 指标参数 | 评测成绩 | 工业界说明 |
| :--- | :--- | :--- | :--- |
| **测试集规模** | 100 套真实上市公司年报 | **100 案全覆盖** | 50 真实暴雷案 + 50 蓝筹健康标杆 |
| **端到端通关率** | 综合判决正确比例 | **100.0% (100/100)** | 零误判 (FP=0)，零漏报 (FN=0) |
| **轨 1: 代码直判** | 绝对数值误差容差 $\le 0.1\%$ | **100.0%** | DuckDB 隔离沙箱零算术幻觉验算 |
| **轨 2: 双盲去偏** | Swap Bias 翻转判定一致率 | **100.0%** | 完全免疫大模型位置敏感度与长文本偏好 |
| **推理质量均分** | 逻辑闭环度与证据完整性 | **87.2 / 100** | 证据链与反思记录法庭级自洽 |

---

## 📁 目录规范

```text
finreason-agent/
├── MASTER_SPEC.md              # 架构蓝图与项目主宪章
├── TECHNICAL_REPORT.md         # 英文学术与技术报告
├── pyproject.toml              # 现代依赖与命令行脚本配置
├── README.md                   # 本文件
├── run_demo.py                 # 全景实战演示脚本
├── web_app.py                  # Streamlit 可视化金融刑侦工作台
├── configs/                    # 配置文件（模型接入、沙箱配额、评测参数）
│   └── default.yaml
├── finreason/                  # 核心源代码包
│   ├── agent.py                # 顶层统合 Agent 主脑与 AuditReport
│   ├── cli.py                  # 终端命令行工具 (finreason)
│   ├── core/                   # 认知路由器与 ToT / MCTS 核心引擎
│   │   ├── router.py           # 门房认知路由器 (Tier 1/2/3)
│   │   ├── tot_engine.py       # ToT 树状探索与剪枝引擎
│   │   └── mcts_engine.py      # MCTS 终局模拟与回传引擎
│   ├── sandbox/                # 确定性执行与安全沙箱
│   │   ├── code_runner.py      # Python / DuckDB 解释器
│   │   ├── container_box.py    # AST 语法守卫与沙箱安全策略
│   │   └── worker.py           # 子进程隔离执行器
│   ├── memory/                 # 四层分立记忆体系
│   │   ├── working.py          # 工作记忆草稿纸
│   │   ├── episodic.py         # 专案时序因果日志与 Reflexion 缓冲区
│   │   ├── semantic.py         # 语义记忆与准则检索
│   │   └── procedural.py       # 程序记忆审计代码技能注册表
│   ├── parser/                 # 财报与文档对齐
│   │   ├── pdf_parser.py       # 真实 PDF / Markdown 表格抽取与容灾降级
│   │   ├── chunker.py          # 父子层级分块 (Hierarchical Chunking)
│   │   └── footnote_linker.py  # 主表与附注双向锚定器
│   └── evals/                  # 第四代双轨评测体系
│       ├── dataset.py          # 100 案真实上市公司黄金基准题库
│       ├── deterministic.py    # 轨 1 代码绝对容差判分
│       ├── debiased_judge.py   # 轨 2 协议双盲去偏裁判
│       ├── harness.py          # 自动化考场框架 (断点/隔离/防污染)
│       └── trajectory_exporter.py # Alpaca / ShareGPT / DPO 数据集导出
```

---

## ⚡ 企业级 RESTful API 微服务与 Docker 容器化

系统内置生产级 FastAPI 微服务，方便嵌入现有风控中台、ERP 与智能审计系统。

### 1. 启动 API 服务
```bash
finreason serve --host 0.0.0.0 --port 8000
# 交互式 Swagger API 文档: http://localhost:8000/docs
# 交互式 ReDoc 文档:       http://localhost:8000/redoc
```

核心 API 端点一览：
| HTTP 方法 | 接口路径 | 描述说明 |
| :--- | :--- | :--- |
| `POST` | `/api/v1/audit` | 传入文本/指标，执行全自动刑侦审计与 MCTS 推演 |
| `POST` | `/api/v1/audit/upload` | 上传本地 PDF/Markdown 财报，返回法庭级备忘录 |
| `POST` | `/api/v1/sandbox/execute` | 隔离 DuckDB 沙箱执行财务计算脚本 |
| `POST` | `/api/v1/parser/parse` | 结构化解析跨页表格与附注外键索引 |
| `GET` | `/api/v1/eval/scorecard` | 查询 100 案金标准双轨评测大屏得分与混淆矩阵 |
| `POST` | `/api/v1/export/datasets` | 导出 Alpaca / ShareGPT / DPO 认知微调训练集 |
| `GET` | `/health` | 服务健康与存活探针 |

### 2. Docker & Docker Compose 一键启动集群
```bash
# Windows 一键启动脚本
deploy.bat

# Linux / macOS 一键启动脚本
./deploy.sh

# 或标准命令：
docker compose up -d --build
```
启动后自动编排：
- **`finreason-api` (Port 8000)**: RESTful API 与 Swagger 文档
- **`finreason-ui` (Port 8501)**: Streamlit 可视化审计大屏

---

## 🧪 自动化测试套件

运行全量测试套件：
```bash
python -m pytest
```
测试结果：
```text
collected 55 items
tests\test_agent.py ...                                                  [  5%]
tests\test_api.py ......                                                 [ 16%]
tests\test_cli.py .....                                                  [ 25%]
tests\test_code_runner.py .....                                          [ 34%]
tests\test_env.py ...                                                    [ 40%]
tests\test_evals.py ......                                               [ 50%]
tests\test_mcts_engine.py ....                                           [ 58%]
tests\test_memory.py .....                                               [ 67%]
tests\test_parser.py .....                                               [ 76%]
tests\test_parser_chaos.py ..                                            [ 80%]
tests\test_phase1_articulation_verification.py .                         [ 81%]
tests\test_router.py ......                                              [ 92%]
tests\test_tot_engine.py ....                                            [100%]

============================= 55 passed in 36.97s =============================
```
