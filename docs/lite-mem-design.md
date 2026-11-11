# lite-mem 设计文档

> 状态：设计稿（2026-11-11）。按第 12 节的阶段实施；某个阶段改变了行为，就在同一次提交里更新本文。

## 1. 概述

lite-mem 是一个**可读、可改、可以单独部署**的轻量记忆引擎，有两个用途：

1. **学习**：把一段对话变成长期记忆的每一步，都是这个仓库里自己写的代码，而且能量化比较。
   - 每一步：抽取事实、判断新旧事实的关系（新增、更新、删除、重复）、存储、混合检索、整合、遗忘。
   - 对照对象是 mem0：同一组评测数据上，比较 lite-mem 和 mem0 2.2.1 各自的表现。
2. **通用记忆层**：
   - 可以作为 Python 库使用，也可以作为独立服务运行（`litemem serve`，或直接用 Docker 镜像）。
   - 任何 agent 或应用都能直接用它记住用户。
   - [llm-steward](https://github.com/ArthurGao/llm-steward) 是第一个使用方：llm-steward 负责治理，lite-mem 负责记忆本身。

和 [lite-rag](https://github.com/ArthurGao/lite-rag) 是姊妹项目。lite-rag 管文档知识（切块、检索、回答），lite-mem 管关于"人"的事实（谁、在哪、喜欢什么、变了什么）。两者的工程约定一致：同样的 Protocol 风格、PostgreSQL + pgvector、namespace 隔离、可切换的 LLM、评测驱动。

### 来源

lite-mem 的第一版逻辑来自 llm-steward 已经在用的代码（2026-11-11 起，llm-steward 自己能为不会抽取的引擎抽取记忆）：

- 抽取：`backend/src/llm_steward/pipeline/extract.py`，`prompts/extract_facts.j2`。
- 冲突消解：`pipeline/conflict.py`，`prompts/resolve_conflict.j2`。

lite-mem 把这两步变成引擎内部的完整流水线，补上存储、历史、检索、整合和遗忘，并且可以脱离 llm-steward 独立使用。

### 目标

- **默认零成本**：Ollama 提供 embedding（`bge-m3`）和 LLM（`qwen3:8b`），存储用 PostgreSQL + pgvector。
- **库和服务用同一套核心**：`MemoryEngine` 是唯一的实现，HTTP 服务只是在它外面包一层。
- **每个环节都可插拔**：`Extractor`、`Resolver`、`Embedder`、`Store`、`LLM` 都是 Protocol，换实现只需要改配置。
- **记忆有历史**：每一条记忆的每一次变化（新增、更新、删除、合并）都追加进历史表，记下原因、由哪个模型做出、时间。支持"某个时间点时这个人的记忆是什么"（`as_of`）。
- **可度量**：自带评测，覆盖抽取、更新是否正确、检索召回、问答正确率。同一组数据在不同配置下、以及和 mem0 之间并排比较。
- **LLM 可切换**：`ollama`、`openai`（任何 OpenAI 兼容端点，包括 llm-gateway）、`anthropic`，只改配置；没有 fallback 链。

### 非目标（v1）

- **不做治理**：审核、终端用户确认、可信度分级、GDPR 流程（按用户彻底删除除外，见 §4）交给 llm-steward 这类上层系统。
- **不做文档知识库**：那是 lite-rag 的事。
- **v1 不做知识图谱**：实体和关系图放在 S10 之后再考虑（见 §12）。
- **不做分布式**：一个 PG 实例就够。目标规模是每个 namespace 百万级记忆、每个用户千级记忆。

## 2. 数据流

```
写入（infer=true）：
  messages → Extractor（LLM，extract_facts.j2）→ 候选事实[]
           → 对每条事实：混合检索同一 scope 的相关记忆 top-n
           → Resolver（LLM，resolve_conflict.j2）→ ADD / UPDATE / DELETE / NOOP（代码校验目标 id）
           → Store（一个事务：记忆行 + 向量 + 倒排 + 历史事件）→ MemoryEvent[]
写入（infer=false）：  事实原样 → （可选：去重）→ Store
检索：  query → Embedder → 向量 top-n ┐
        query → 分词(jieba) → BM25 top-n ┴→ RRF → 权重（重要性 × 时间衰减 × 置信度）→ top-k MemoryHit
维护（定时）：整合（相近记忆合并、旧的 episodic 归纳成 semantic）· 遗忘（过期、衰减到阈值以下）
```

## 3. 组件与接口

所有接口都是 `typing.Protocol`；测试用 fake 实现，不依赖网络和模型。

| 接口 | 方法 | 默认实现 | 其他实现 |
| --- | --- | --- | --- |
| `Extractor` | `extract(messages, scope) -> list[FactDraft]` | `LlmExtractor`（`extract_facts.j2`，结构化输出） | `RuleExtractor`（测试与无模型时：每条 user 消息一条事实） |
| `Resolver` | `resolve(new, related) -> Resolution` | `LlmResolver`（`resolve_conflict.j2`） | `NoopResolver`（只新增，用来和 mem0 2.2.1 的行为对照） |
| `Embedder` | `embed(texts) -> vectors`，`dim`，`model` | Ollama `bge-m3` | fastembed（本地 MiniLM）、OpenAI 兼容 |
| `Store` | 见 §4 | `PgStore` | `SqliteStore`（sqlite-vec + FTS5，S7b，嵌入式） |
| `LLM` | `structured(role, system, user, schema)` | OpenAI 兼容（含 Ollama `/v1`） | Anthropic（extra） |

- **角色（role）**：`extract`、`resolve`、`consolidate`。配置把角色映射到 provider 和模型；任何一次 LLM 调用都只说角色，不写模型名。
- **出处**：每条由模型产生的记忆或变更，都带 `provider:model`（写进历史事件）。
- **代码决定**（同 llm-steward 的做法）：
  - 模型给出的 UPDATE/DELETE 目标不在候选里，不执行，按 ADD 处理；
  - 抽取结果要去空白、去重，并截断条数和长度。

### 记忆的形状

| 字段 | 说明 |
| --- | --- |
| `id` | UUIDv7，由 lite-mem 生成 |
| `namespace` | 必填，隔离单位（见 §4） |
| `user_id` / `agent_id` / `session_id` | scope；都为空表示 namespace 内共享 |
| `content` | 一条原子事实，短句 |
| `type` | `semantic`（事实）、`episodic`（发生过的事）、`procedural`（偏好的做法）、`profile`（带 `key` 的当前值） |
| `key` | 只用于 `profile`：同一 scope 下一个 key 只有一个当前值，新值取代旧值 |
| `importance`、`confidence` | 0–1；抽取时由模型给出初值，代码夹到范围内 |
| `valid_from` / `valid_to`、`superseded_by` | 生效区间；更新时旧记忆写 `valid_to` 和 `superseded_by`，**不删除** |
| `source` | 来源（调用方给的 episode id、消息 id），可选 |
| `metadata` | 调用方的键值，可以过滤（等值） |
| `access_count`、`last_accessed_at` | 检索命中时更新，用于衰减和遗忘 |

历史表 `memory_events` 只追加：`(memory_id, op, before, after, reason, produced_by, at)`。op 有：

- `ADD`：新记忆；
- `CORRECT`：原地更正措辞，id 不变（同一事实写得更好）；
- `SUPERSEDE`：事实变了，新 id 取代旧行，旧行写 `valid_to` 和 `superseded_by`（profile key 的新值、冲突消解的 UPDATE 决定都落成这个）；
- `DELETE`：退役；
- `MERGE`：合并（S7）；
- `EXPIRE`：过期（S7）。

所有写操作都经过引擎的同一个入口 `_apply`（新增、取代、更正、退役），都记 `produced_by`；整批在一个事务里。

## 4. 存储与 id

### PostgreSQL 后端（默认）

- 自己的 schema `lite_mem`（和 lite-rag 的 `lite_rag`、llm-steward 的 `steward` 可以共用一个 PG 实例，各用各的 schema）。
- 表：
  - `memories`；
  - `memory_vectors_<dim>`：一种维度一张表，HNSW cosine 索引；
  - `memory_postings`：自写 BM25 的倒排，jieba 分词；
  - `memory_events`；
  - `namespaces`：记录 embedding 模型、维度和分词器，打开时不一致就拒绝；
  - `api_keys`：服务端用。
- **一次写入一个事务**：记忆行、向量、倒排、历史事件同时成功或同时回滚。这是 mem0 2.2.1 没有的保证（它的历史在单独的 SQLite 文件里）。
- 迁移用自己的 `litemem migrate`，和 lite-rag 的做法一致。

### Namespace（强制）

- 引擎绑定一个 namespace（如 `app` 或 `steward/<namespace_id>`）。所有读写都带着它，没有能跨 namespace 的 API。
- 按用户彻底删除：`erase(user_id)` 删掉这个用户的全部记忆、向量、倒排和历史。这是唯一的硬删除，为 GDPR 类请求准备。

### 通用规则

- **只失效、不删除**：取代、删除、合并、过期都只写 `valid_to` 和历史，`as_of` 能回看。
- **更正对 `as_of` 追溯生效**：`CORRECT` 改的是同一事实的措辞，`as_of` 看到的是更正后的文字；改前的文字在历史里。事实本身变了要用取代，不要用更正。
- **索引绑定模型**：换 embedding 模型要新建 namespace 并重建（`litemem reindex`），不能就地替换。
- **幂等**：调用方可以带 `idempotency_key`（例如 episode id）。同一个 key 的写入只执行一次，重试会返回第一次的结果。mem0 2.2.1 没有这个保证，llm-steward 需要它。

## 5. 检索

- 混合检索：向量 top-n 和 BM25 top-n 用 RRF 融合，再乘以下几个权重：
  - 重要性：`1 + boost × importance`；
  - 时间衰减：按类型的半衰期，episodic 默认 30 天，其余不衰减；
  - 置信度：低于 0.5 的乘以 0.8。
- 过滤：scope（自己的，加上共享的）、类型、metadata 等值，`as_of`（某个时刻生效的版本）。
- 结果 `MemoryHit(id, content, type, score, rel, scope, valid_from, valid_to)`。分数只在 lite-mem 内部可比。
- 冲突消解用的"相关记忆"也走同一条检索，只是限定在**完全相同的 scope**内，并且只取当前生效的记忆。过滤条件有两种模式：`visible`（这个 scope 能看到的：自己的、没指定 session 时含它的所有 session、可选含共享）和 `exact`（恰好这个 scope，session 也要一致，不含共享）。
- 向量检索在过滤后的行上精确计算距离（先物化该 scope 可见的行，再算余弦），返回 (id, 相似度)；按 scope 检索时一个 scope 只有千级记忆，精确算很便宜，也不会因为别的 scope 有更近的向量而漏召回。HNSW 索引留给以后不按 scope 的查询。
- BM25 的契约：PgStore 和 `keyword.py` 的公式逐项相等（只在 PG 上测）；其他后端（S7b 的 SQLite/FTS5）只需要是"有排序的关键词匹配"，共享的合约测试只查排序性质。

## 6. 整合与遗忘（S7）

- **合并**：同一 scope 内相似度超过阈值（默认 0.92，用 eval 定）的语义记忆，由 `consolidate` 角色合成一条。被合并的旧记忆写 `superseded_by`，历史事件记为 MERGE。
- **归纳**：一个用户较早的 episodic 记忆（按时间窗口）归纳成 semantic 记忆，例如"每周五都问退款进度"。
- **遗忘**：
  - 到了 `expires_at` 的记忆过期，历史记为 EXPIRE；
  - 衰减权重低于阈值、且长期没被检索命中的 episodic 记忆过期。阈值可配置，默认关闭。
- 这些都是离线任务（`litemem maintain`，或服务内的定时任务），不在写入路径上。

## 7. HTTP 服务（`litemem serve`）

- FastAPI（extra `[server]`），默认端口 `127.0.0.1:8710`。
- API key 鉴权，key 按 glob 限定 namespace。没有 key 时，服务只能绑定回环地址。
- 错误用 Problem Details；写接口支持 `Idempotency-Key`。
- 路由：
  - `POST /v1/memories:add`（messages + scope + infer）；
  - `POST /v1/memories:search`；
  - `GET|PATCH|DELETE /v1/memories/{id}`；
  - `GET /v1/memories/{id}/history`；
  - `GET /v1/memories?scope…`；
  - `DELETE /v1/users/{user_id}`（彻底删除）；
  - `GET /health`。
- Dockerfile 和 compose 示例（服务 + PG，Ollama 在宿主机上）；CI 构建镜像（llm-steward 的教训：不在 CI 里构建，镜像坏了也没人发现）。

## 8. 评测

- **数据**：自己写的合成对话集，中英混合。
  - 一个虚构用户的若干轮会话，里面包含：新事实、改变的事实（搬家、换工作）、重复说的事实、闲聊、敏感信息（不该被记住）。
  - 每段会话后标注"此时应该记住的事实集合"和若干问题（及标准答案）。
  - 语料自己写，可以公开。
- **指标**：
  - 抽取的准确率和召回率（与标注事实比较，用 LLM 评判加人工抽查）；
  - 更新是否正确（该取代的取代了、该保留的保留了）；
  - 检索 recall@k；
  - 问答正确率；
  - 每次写入的模型调用次数和延迟。
- **对照**：
  - lite-mem 的各种配置：只新增 vs 消解冲突、不同的抽取提示词、不同模型；
  - mem0 2.2.1（同一份数据，同一个本地模型）。
- `litemem eval` 输出对比表。真实模型只用本地 Ollama（`-m ollama`，可选开启），付费 API 只在 owner 提供 key 并同意后才跑。结果记进 KB。
- 以后可以加 LoCoMo 这类公开基准（需要真实模型，先问 owner）。

## 9. 配置

pydantic-settings，前缀 `LITEMEM_`：

- 数据库 URL、namespace；
- embedding provider 和模型；
- 每个角色对应的 LLM provider 和模型；
- 检索权重和半衰期；
- 整合和遗忘的阈值；
- 服务端口；
- API key 的密钥。

密钥只从环境变量读取；仓库里只提交 `.env.example`。

## 10. 与 llm-steward 集成（S9）

llm-steward 新增 `LiteMemProvider`，实现它的 `MemoryProvider` Protocol，支持两种接入方式：直接 import 库，或调用 HTTP 服务。两种方式都要跑 llm-steward 的记忆合约测试。

| llm-steward 调用 | lite-mem 实现 |
| --- | --- |
| `add(messages, scope, infer)` | `add`，带 `idempotency_key`（批次里最后一个 episode 的 id），返回 `MemoryEvent[]`（ADD / SUPERSEDE / DELETE，带 `produced_by`） |
| `search(query, scope, top_k, filters)` | `search` |
| `get / update / delete / list` | `get`、`update`（= CORRECT，id 不变）、`delete`、`list_memories` |
| `health()` | 存储可用，且 embedding 服务可用 |

- **能力**：`{extract, resolves_conflicts, update, delete, metadata_filter}`。
- **llm-steward 那边要改的**：现在的 `ExtractionService` 只把引擎的 ADD 事件写进镜像（mem0 2.2.1 只会新增）。引擎自己会消解冲突时，要把以下事件映射成治理操作，让历史仍在 llm-steward 里：
  - SUPERSEDE（`replaced` 是被取代的记忆）映射成 `SupersedeMemory`；
  - DELETE 映射成 `DeleteMemory`。
- **另一个选择**：llm-steward 用 lite-mem 时关掉引擎内的冲突消解，只用抽取，继续走 llm-steward 自己的冲突补偿。见 §13。
- **治理规则仍由 llm-steward 保证**：
  - 检索结果要经过它的安全网；
  - 它的保留 key（`steward.*`）不会推给引擎；
  - 按用户删除时调 `erase`。
- 接入后，llm-steward 自己的抽取（`STEWARD_MEMORY_LLM`）对 lite-mem 不再生效：lite-mem 有 `extract` 能力，按能力判断会走引擎自己的抽取。

## 11. 项目结构

```
lite-mem/
  pyproject.toml              # uv；extras: [server] [fastembed] [anthropic] [sqlite]
  Dockerfile  deploy/compose.yaml
  src/lite_mem/
    config.py models.py engine.py            # MemoryEngine：库和服务的唯一核心
    extract.py resolve.py consolidate.py     # 流水线各步（Protocol + LLM 实现 + 规则实现）
    embedding.py keyword.py fusion.py llm.py # 与 lite-rag 同样的写法，各自独立，不互相依赖
    stores/ base.py pg.py sqlite.py          # Store Protocol + 后端
    migrations/                              # litemem migrate
    server/ app.py auth.py schemas.py        # FastAPI（extra [server]）
    cli.py prompts/*.j2
  tests/  unit/（fakes）  integration/（PG：独立的测试 schema lite_mem_test）  -m ollama 可选
  eval/   合成对话、标注、报告；mem0 对照脚本
  docs/lite-mem-design.md
```

## 12. 阶段与验收

每个阶段结束时，ruff、ruff format、mypy、pytest 都要通过，每个新测试至少见过一次失败（red）。然后做阶段收尾：

1. 对整个阶段的 diff 跑 `/simplify` 并应用修改；
2. 用 Codex 只读审查同一个 diff；
3. 修掉成立的问题（bug 先写失败的测试），不成立的写下理由；
4. 重跑全部检查。

见 `CLAUDE.md`。收尾完成之前不开始下一阶段。

| 阶段 | 内容 | 验收 |
| --- | --- | --- |
| S0 环境 | 确认 Ollama 上有 `bge-m3`、`qwen3:8b`；共享 PG 上建库或 schema `lite_mem` 和角色；用文档核对 psycopg、pgvector、jieba、FastAPI、OpenAI/Anthropic SDK 的版本和 API，并锁定；建项目 KB `projects/lite-mem/` | 版本写进 `pyproject.toml`；`psql` 能连上，`vector` 扩展可用；KB 建好并通过校验 |
| S1 骨架 | uv、ruff、mypy、pytest、配置、fakes（FakeLLM、FakeEmbedder）、CLI 骨架、`litemem migrate`、CI（含镜像构建的占位） | `litemem --help` 可用；迁移在测试 schema 上执行通过；CI 绿 |
| S2 存储与直接写入 | `PgStore`；`add(infer=false)`、`get`、`list_memories`、`update`（更正）、`delete`、`erase`；历史事件；`as_of`；profile key 取代；幂等键 | 一次写入一个事务（故障注入时整体回滚）；取代后旧版本在 `as_of` 下仍可见；`erase` 之后该用户什么都不剩；同一幂等键只写一次 |
| S3 检索 | 向量 + 自写 BM25（jieba）+ RRF；重要性、衰减、置信度权重；scope、类型、metadata 过滤 | BM25 与手算一致；融合顺序正确；共享记忆和自己的记忆都能找到，别人的找不到；`-m ollama` 真实往返 |
| S4 抽取与冲突消解 | `Extractor`、`Resolver`（从 llm-steward 移植，提示词放进 `prompts/`）；`add(infer=true)` 完整流水线；事件带出处 | FakeLLM 测试：北京→上海是 UPDATE（旧的失效）、重复是 NOOP、越界目标按 ADD；整批一个事务；重试（同一幂等键）不会再调模型 |
| S5 评测 | 合成对话集与标注、`litemem eval`、mem0 对照脚本 | 用本地 Ollama 跑出对比表：只新增 vs 消解、两版抽取提示词、lite-mem vs mem0；结果记进 KB |
| S6 HTTP 服务 | FastAPI、API key、Problem Details、`Idempotency-Key`、Dockerfile、compose、CI 构建镜像 | 未授权 401，越权 namespace 403；`docker compose up` 后完成 add → search → history；库和 API 结果一致 |
| S7 整合与遗忘 | 合并、归纳、过期、衰减遗忘；`litemem maintain` | FakeLLM 测试：相似记忆合并成一条且历史为 MERGE；过期的检索不到但 `as_of` 可见；用 eval 定合并阈值 |
| S7b SQLite 后端（可选） | `SqliteStore`（sqlite-vec + FTS5），嵌入式运行 | 同一套 Store 合约测试在 PG 和 SQLite 上都通过 |
| S8 收尾 | 整个引擎的 `/simplify` 和 Codex 审查，修复，重跑全部检查 | 审查结论记录在案 |
| S9 接入 llm-steward | `LiteMemProvider`（库和 HTTP 两种）；llm-steward 的事件映射（UPDATE、DELETE）；合约测试；更新 llm-steward 设计文档 | llm-steward 的记忆合约测试在两种接入方式下都通过；手动端到端：对话 → 记忆 → 更新 → 检索 |
| S10 记忆图谱（以后） | 实体和关系的抽取与查询 | 先写设计，再看 eval 的收益决定做不做 |

### 阶段记录

- **S0（2026-11-12）**：共享 PG 上建了库 `lite_mem`（角色 `lite_mem`，`vector` 0.8.6）；Ollama 上已有 `bge-m3:567m`、`qwen3:8b`；版本沿用 lite-rag 已在用的（psycopg 3.3.8、openai 3.46.0、anthropic 2.6.0、fastapi 0.162.2 等）；项目 KB `projects/lite-mem/` 建好。环境阶段没有代码，不做 /simplify 和 Codex。
- **S1（2026-11-12）**：骨架、配置、迁移（`litemem migrate`）、fakes、CI。
  - /simplify 后：每个角色可单独指定 provider、端点和模型（`LITEMEM_LLM__ROLES={"resolve": {"provider": "anthropic", "model": …}}`，换了 provider 就不沿用默认的端点和 key）；LLM 接口改收 payload 字典，由客户端序列化；namespace 记录分词器；只留一个 `errors.redact()`（整段连接串按 libpq 解析，其余按模式），在 CLI 出口统一脱敏；namespace 和 schema 用 pydantic 注解类型校验。没做：错误类先保留（S2 起就用到）；迁移"已是最新"时的往返次数、测试每次重建 schema（迁移少时无所谓，S2 再看）。
  - 自己发现的 bug：schema 字段带别名 `schema` 时，pydantic-settings 从不带前缀的 `SCHEMA` 读，测试设置的 `LITEMEM_SCHEMA` 没生效，迁移测试实际跑在了应用 schema 上。字段改名 `db_schema`（`LITEMEM_DB_SCHEMA`），不用别名；应用 schema 当时是空的，已删除重建。
  - Codex：P2 拒绝 `public` 和系统 schema（配置和 `migrate()` 两处都查）——修了；P2 错误原文可能带连接信息——由 CLI 出口统一脱敏，测试在边界制造一条带密码的错误来验证；P1"`conn.execute` 不接受 bytes"——不成立：psycopg 3 的查询参数接受 `bytes`，本阶段的测试实际执行了迁移，lite-rag 同样这么用。
- **S2（2026-11-12）**：迁移 0002（`memories`、`memory_events`、`idempotency`），`PgStore`，`MemoryEngine` 的直接写入、取代、更正、退役、彻底删除、历史、`as_of`、幂等。
  - /simplify 后：操作命名定为 ADD / CORRECT / SUPERSEDE / DELETE（冲突消解的 UPDATE 决定落成 SUPERSEDE，S9 映射跟着改）；所有写操作走一个 `_apply`，都带 `produced_by`，S3 只需在这里加向量和倒排；幂等先只读查（`replay`），再在事务里认领，S4 重试就不会再调模型；`superseded_by` 加索引；`current_profile` 的条件和唯一索引的表达式一致、列表查询按字段拼条件，能用上索引；`history` 一次查询；事件序列化交给 pydantic；`MEMORY_TYPES` 从 Literal 推出。没做：一次写入多条时的往返合并（pipeline、批量插入）——批次小，等 eval 量出瓶颈再做；只读查询不开事务；`Fact` 改用 pydantic 校验；`last_accessed_at` 留给 S3（检索命中时更新）。
  - Codex：High 幂等键被另一个用户复用会拿到前一个用户的结果——改为拒绝（Conflict）；High 同一 profile key 第一次并发写会撞唯一索引——按 (namespace, scope, key) 加事务级 advisory lock；Medium 空字符串和 NULL 在唯一索引里相同——拒绝空的 scope id；Medium `as_of` 看到更正后的措辞——按设计（更正追溯生效，见 §4），写进文档。每条修复都先有失败的测试，并做了变异检查。
  - 自己发现的：取代时先关旧行（腾出唯一的"当前值"位置）再插新行，`superseded_by` 外键要延迟到提交时检查（`DEFERRABLE INITIALLY DEFERRED`）。
- **S3（2026-11-12）**：迁移 0003（倒排 `memory_postings`、`memories.terms`；每个维度一张向量表，绑定 namespace 时建）；jieba 分词与 BM25（SQL 里算，和公式逐项相等）、RRF、按重要性、类型半衰期、置信度加权；`Filters`（scope、共享、类型、metadata、`as_of`）；Ollama 和 OpenAI 兼容两种 embedder；写入时向量在事务外算好，倒排和向量与记忆行同一个事务；检索命中计数。`-m ollama` 真实往返通过（含中文问、英文记忆的跨语言查询）。
  - /simplify 后：`Filters` 加 `exact` 模式（S4 用）；检索拆出内部的 `_rank`（S4 复用，不算访问、不重复 embedding）；向量检索返回相似度（S7 合并用）；已带向量的操作不再重算；更正只校验一次；维度缓存放在 `PgStore`；向量用 `float8[]::vector` 传；建向量表前先查是否已存在（避免每次构造引擎都跑 DDL）。没做：BM25 统计每次查询现算（十万级以上再加缓存）；检索时的命中计数是一次写（每秒数百次检索时再改成批量）；一次检索 4–5 次往返、一次写入每条记忆约 4 次往返（批量导入或远程库时再合并）；OpenAI embedder 改用官方 SDK（两种 embedder 共用一套 HTTP 代码更简单）；半衰期的 key 用 `MemoryType` 类型（只是外观）。
  - Codex：P1 带过滤的 HNSW 检索在别的 scope 有大量更近向量时可能漏召回——改为在 scope 内精确检索，正确性不再取决于规划器；回归测试在本地没能复现出原问题（scope 窄时规划器本来就走精确路径），如实记录；P2 半衰期可以是 0 或负数——改为必须大于 0（先红后绿）。

## 13. 待确认

- **llm-steward 接入时冲突消解归谁**（S9 前定）：
  - 引擎消解，llm-steward 映射事件：更简单，能力完整；
  - llm-steward 消解，引擎只抽取：治理层能看到每一次判断，但多一次往返。
- **合并阈值、遗忘阈值的默认值**：S5、S7 用 eval 定。
- **和 lite-rag 共用代码**（embedding、BM25、LLM 层）还是各自独立：v1 各自独立（两个仓库都在学习期，接口还在变）；稳定后再考虑抽出一个共享小包。
- **License**：仓库建立时用 Apache-2.0（与 llm-steward 一致），owner 可以改。
