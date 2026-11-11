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
写入（infer=false）：  每条 user 消息原样（过敏感信息筛查）→ Store；不去重（要去重靠 infer=true 或 maintain 的合并）
写入（remember）：    调用方给的一条事实（类型、key、重要性、过期时间）→ Store
检索：  query → Embedder → 向量 top-n ┐
        query → 分词(jieba) → BM25 top-n ┴→ RRF → 权重（重要性 × 时间衰减 × 置信度）→ top-k MemoryHit
维护（定时）：整合（相近记忆合并、旧的 episodic 归纳成 semantic）· 遗忘（过期、衰减到阈值以下）
```

## 3. 组件与接口

所有接口都是 `typing.Protocol`；测试用 fake 实现，不依赖网络和模型。

| 接口 | 方法 | 默认实现 | 其他实现 |
| --- | --- | --- | --- |
| `Extractor` | `extract(messages) -> Extraction`（事实 + 出处） | `LlmExtractor`（`extract_facts.j2`，结构化输出） | `testing.RuleExtractor`（测试：每条 user 消息一条事实） |
| `Resolver` | `resolve(new, related) -> Resolution`（含 `remainder`） | `LlmResolver`（`resolve_conflict.j2`） | `AddOnlyResolver`（只新增，用来和 mem0 2.2.1 的行为对照） |
| `Consolidator` | `merge(memories) -> MergeDecision`，`summarise(episodes) -> Summary[]` | `LlmConsolidator`（`consolidate_merge.j2`、`consolidate_summarise.j2`） | — |
| `Embedder` | `embed(texts) -> vectors`，`dim`，`model` | Ollama `bge-m3` | OpenAI 兼容 |
| `Store` | 见 §4 | `PgStore` | `SqliteStore`（sqlite-vec + 自己的倒排，S7b，嵌入式） |
| `LLM` | `structured(role, system, payload, schema) -> (结果, 出处)` | OpenAI 兼容（含 Ollama `/v1`） | Anthropic（extra） |

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
  - `memory_vectors_<dim>`：一种维度一张表（没有 ANN 索引：检索在 scope 内精确算，见 §5；S8 起不再建 HNSW，迁移 0004 删掉旧的）；
  - `memory_postings`：自写 BM25 的倒排，jieba 分词；
  - `memory_events`；
  - `namespaces`：记录 embedding 模型、维度和分词器，打开时不一致就拒绝；
  - `idempotency`：幂等键和它第一次写入的结果。
  - HTTP 服务的 API key 不在库里：在配置里（`LITEMEM_SERVER__API_KEYS`，只存 sha256，见 §7）。
- **一次写入一个事务**：记忆行、向量、倒排、历史事件同时成功或同时回滚。这是 mem0 2.2.1 没有的保证（它的历史在单独的 SQLite 文件里）。
- 迁移用自己的 `litemem migrate`，和 lite-rag 的做法一致。

### SQLite 后端（S7b，可选，extra `sqlite`）

- `LITEMEM_DATABASE_URL=sqlite:///<文件>`：一个文件、不要服务，适合单机和嵌入式使用。打开时建表（`PRAGMA user_version` 记版本），`litemem migrate` 也可以。
- 表和 PG 一样（`memories`、`memory_events`、`memory_postings`、`namespaces`、`idempotency`），向量放一张 `memory_vectors`（float32 blob）。约束也一样：profile 当前值的部分唯一索引、`valid_to >= valid_from`、`superseded_by` 外键延迟到提交时检查。
- 向量：sqlite-vec 的 `vec_distance_cosine`，在过滤后的行里精确排序（和 PG 一样：先过滤再算距离；`vec0` 虚拟表先取最近的 k 条再过滤，会漏掉本 scope 的记忆，llm-steward 的 native store 就是这样）。
- 关键词：不用 FTS5——它的 bm25 是另一套公式，两个 store 的排序会不同。用和 PG 一样的倒排表，分数用 `keyword.py` 的函数算，两个 store 排序相同。
- 事务：每个线程一个连接，WAL，所有事务都 `BEGIN IMMEDIATE`，写者由数据库串行；PG 的行锁和 advisory lock 在这里不需要。读也要等写者（`busy_timeout` 默认 10 秒）：单机够用，并发写多就用 PG。
- 时间存成固定宽度的 UTC ISO 文本（文本序就是时间序）；不带时区的时间在引擎入口就被拒绝。metadata 过滤和 PG 一样是 jsonb 包含（`@>` 的规则：嵌套对象只要包含、数组只要有、`true` 不等于 `1`），用一个注册进 SQLite 的函数实现。

### Namespace（强制）

- 引擎绑定一个 namespace（如 `app` 或 `steward/<namespace_id>`）。所有读写都带着它，没有能跨 namespace 的 API。
- 按用户彻底删除：`erase(user_id)` 删掉这个用户的全部记忆、向量、倒排和历史。这是唯一的硬删除，为 GDPR 类请求准备。

### 通用规则

- **只失效、不删除**：取代、删除、合并、过期都只写 `valid_to` 和历史，`as_of` 能回看。
- **更正对 `as_of` 追溯生效**：`CORRECT` 改的是同一事实的措辞，`as_of` 看到的是更正后的文字；改前的文字在历史里。事实本身变了要用取代，不要用更正。
- **索引绑定模型**：换 embedding 模型要新建 namespace，把记忆重新写进去（没有 `litemem reindex` 命令：v1 的做法是新 namespace 加重写），不能就地替换。
- **上限**（库、CLI、HTTP 一致，S8）：一次写入最多 200 条消息、每条最多 2 万字；`list_memories` 可分页（`limit` 最多 1 万，HTTP 默认 1000）；`top_k` 最多 100；一个服务进程最多打开 `LITEMEM_SERVER__MAX_NAMESPACES`（默认 1000）个 namespace。
- **敏感信息**：内容、metadata 里的文字值、事件的理由都过同一套筛查（`lite_mem.sensitive`）。
- **幂等**：调用方可以带 `idempotency_key`（例如 episode id）。同一个 key 的写入只执行一次，重试会返回第一次的结果。mem0 2.2.1 没有这个保证，llm-steward 需要它。

## 5. 检索

- 混合检索：向量 top-n 和 BM25 top-n 用 RRF 融合，再乘以下几个权重：
  - 重要性：`1 + boost × importance`；
  - 时间衰减：按类型的半衰期，episodic 默认 30 天，其余不衰减；
  - 置信度：低于 0.5 的乘以 0.8。
- 过滤：scope（自己的，加上共享的：user、agent、session 都为空的才是共享）、类型、metadata（jsonb 包含）、`as_of`（某个时刻生效的版本）。
- 结果 `MemoryHit(memory, score, relevance)`：`memory` 是整条记忆；`relevance` 是融合后的相关度，`score` 再乘了权重。分数只在 lite-mem 内部可比。
- 冲突消解用的"相关记忆"也走同一条检索，只是限定在**完全相同的 scope**内，并且只取当前生效的记忆。过滤条件有两种模式：`visible`（这个 scope 能看到的：自己的、没指定 session 时含它的所有 session、可选含共享）和 `exact`（恰好这个 scope，session 也要一致，不含共享）。
- 向量检索在过滤后的行上精确计算距离（先物化该 scope 可见的行，再算余弦），返回 (id, 相似度)；按 scope 检索时一个 scope 只有千级记忆，精确算很便宜，也不会因为别的 scope 有更近的向量而漏召回。ANN 索引（HNSW）留给以后不按 scope 的查询，到时再建。
- BM25 的契约：两个 store 都和 `keyword.py` 的公式逐项相等（PgStore 在 SQL 里算，SqliteStore 用 `keyword.py` 的函数在 Python 里算），同一套测试在两个 store 上都查。

## 6. 整合与遗忘（S7）

- **"当前"只有一个定义**：没有被关闭（`valid_to` 为空），并且没过 `expires_at`（`Memory.holds_at`；读查询在 SQL 里用同一条规则）。过了期的记忆立刻检索不到，`as_of` 仍能看到它到期之前的样子；`maintain` 再把它正式关闭，`valid_to` 取到期的时间，历史记 EXPIRE。写入时 `expires_at` 已过的，直接拒绝。
- **合并**：同一 scope、同一类型（`semantic`、`procedural`；`profile` 永不合并，新值取代旧值）内，向量相似度不低于阈值的记忆分成组，由 `consolidate` 角色判断是不是**同一个事实**（换了说法，或一条更精确），是就写成一句；说的是不同的事（两个城市、一个变化），就不合并。合并结果是一条新记忆，来源各写 `superseded_by`，每一行都记 MERGE。新记忆的字段由一条规则决定（`_inherit`）：最早那条的类型和 metadata、最高的重要性和置信度、所有来源都会过期时才过期（取最晚的）。合并结果和其他派生事实一样过敏感信息筛查。阈值默认 0.92，S7 用 eval 定（见 §12）。
- **归纳**（默认关闭）：一个 scope 里早于 N 天的 episodic 记忆，由 `consolidate` 角色归纳成 semantic 事实，每个事实列出它总结的 episode（只能是给它的、每个只用一次），被总结的 episode 以 MERGE 关闭。
- **遗忘**（默认关闭）：会衰减的类型（配了半衰期的，默认 episodic）里，搜索权重低于阈值、并且 N 天没被检索命中的记忆，以 EXPIRE 关闭。
- **写入时的复合事实**：消解器判 UPDATE / DELETE 时，如果旧记忆说了几件事、只变了一部分，给出 `remainder`（仍然成立的部分），代码把它另存一条（S6 验收发现："住 Wellington，是护士" 被搬家取代，"是护士"跟着丢了）。
- 这些都是离线任务：`litemem maintain`、`POST /v1/maintain`，或调用方自己的定时任务；不在写入路径上。模型调用在事务外，写入时在锁下复查（来源仍然有效、仍在同一个 scope），变了的组跳过，下次再看。`maintain` 和 `add` 一样返回事件。

## 7. HTTP 服务（`litemem serve`）

- FastAPI（extra `[server]`），默认端口 `127.0.0.1:8710`。
- API key 鉴权，key 按 glob 限定 namespace。没有 key 时，服务只能绑定回环地址。
- 错误用 Problem Details；写接口支持 `Idempotency-Key`。
- 路由：
  - `POST /v1/memories:add`（messages + scope + infer，`Idempotency-Key`）；
  - `POST /v1/memories`（`remember`：一条事实原样写入，`Idempotency-Key`；S8，给 S9 推送治理过的记忆用）；
  - `POST /v1/memories:search`；
  - `GET|PATCH|DELETE /v1/memories/{id}`；
  - `GET /v1/memories/{id}/history`；
  - `GET /v1/memories?scope…&limit&offset`（分页）；
  - `POST /v1/maintain[?user_id]`（整合与遗忘，返回事件）；
  - `DELETE /v1/users/{user_id}`（彻底删除）；
  - `GET /health`（只查数据库）、`GET /health?deep=true`（再做一次 embedding）。
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
  - DELETE 映射成 `DeleteMemory`；
  - EXPIRE 也会出现在 `add` 的结果里：profile key 的旧值已经过期时，写新值会先把它记成 EXPIRE；
  - 复合事实的 `remainder` 以 ADD 返回，理由是 "kept from <旧 id>"。
- **推送治理过的记忆**用 `remember`（库，或 `POST /v1/memories`），幂等键用 llm-steward 的记忆 id：`add(infer=false)` 遇到敏感信息会静默丢掉、没有幂等键时重试会重复写。
- **另一个选择**：llm-steward 用 lite-mem 时关掉引擎内的冲突消解，只用抽取，继续走 llm-steward 自己的冲突补偿。见 §13。
- **离线变化（S7）**：合并和过期不是 llm-steward 的调用触发的。llm-steward 自己定时调用 `maintain`（库或 `POST /v1/maintain`），把返回的事件映射成治理操作：
  - MERGE（每个被合并的来源一条，`replaced` 是来源）映射成 `SupersedeMemory`（来源 → 新记忆）；
  - EXPIRE 映射成按到期时间失效。
  这样它的镜像不会留着已关闭的来源，也不会因为找不到外部引用而丢掉合并出来的新记忆。不在 llm-steward 控制下跑 `maintain`（比如运营方自己的 cron），镜像就会落后——S9 定要不要加一个按游标拉取事件的接口。
- **治理规则仍由 llm-steward 保证**：
  - 检索结果要经过它的安全网；
  - 它的保留 key（`steward.*`）不会推给引擎；
  - 按用户删除时调 `erase`。
- 接入后，llm-steward 自己的抽取（`STEWARD_MEMORY_LLM`）对 lite-mem 不再生效：lite-mem 有 `extract` 能力，按能力判断会走引擎自己的抽取。

## 11. 项目结构

```
lite-mem/
  pyproject.toml              # uv；extras: [server] [anthropic] [sqlite] [eval]
  Dockerfile  deploy/compose.yaml
  src/lite_mem/
    config.py types.py engine.py factory.py  # MemoryEngine：库和服务的唯一核心；factory 按配置组装
    extract.py resolve.py consolidate.py     # 流水线各步（Protocol + LLM 实现）
    embedding.py keyword.py fusion.py llm.py # 与 lite-rag 同样的写法，各自独立，不互相依赖
    sensitive.py errors.py ids.py db.py      # 敏感信息、错误与脱敏、UUIDv7、PG 迁移
    stores/ base.py pg.py sqlite.py          # Store Protocol + 两个后端
    migrations/                              # litemem migrate（PG）
    server/ app.py auth.py schemas.py        # FastAPI（extra [server]）
    eval.py eval_mem0.py                     # litemem eval（mem0 对照：extra [eval]）
    cli.py testing.py prompts/*.j2           # testing：给测试和下游项目用的 fake
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
| S5b 敏感信息 | 写入前由代码识别密码、卡号、证件号等（规则 + 可选的模型判断），按配置拒绝或打码；事件记下原因 | 合成集里的银行卡密码不再成为记忆；正常数字（电话分机、年份）不误伤；eval 重跑 |
| S6 HTTP 服务 | FastAPI、API key、Problem Details、`Idempotency-Key`、Dockerfile、compose、CI 构建镜像 | 未授权 401，越权 namespace 403；`docker compose up` 后完成 add → search → history；库和 API 结果一致 |
| S7 整合与遗忘 | 合并、归纳、过期、衰减遗忘；`litemem maintain`；复合事实被取代时不丢没变的部分（S6 发现） | FakeLLM 测试：相似记忆合并成一条且历史为 MERGE；过期的检索不到但 `as_of` 可见；用 eval 定合并阈值 |
| S7b SQLite 后端（可选） | `SqliteStore`（sqlite-vec + 自己的倒排），嵌入式运行 | 同一套 Store 合约测试在 PG 和 SQLite 上都通过 |
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
- **S4（2026-11-12）**：`extract`、`resolve` 两个角色（提示词 `extract_facts.j2`、`resolve_conflict.j2`，从 llm-steward 移植并扩展：抽取同时给出类型和重要性）；OpenAI 兼容（含 Ollama `/v1`）和 Anthropic（强制工具）两种客户端，按角色路由；`add(infer=true)` 完整流水线；`litemem add/search/list/history/erase`（erase 要 `--yes`）；`factory.open_engine`。`-m ollama` 用 qwen3:8b 跑通"住北京 → 搬到上海"。
  - 真实模型暴露的问题（S5 量化）：qwen3:8b 不遵守"用对方的语言写"（中文输入写成英文事实）；一句话抽成两条近义事实；一次往返 100–300 秒（思考模式）。处理：语言由代码判断后显式传给模型（`LANGUAGE`）；同批事实互相比对（见下）；思考模式开关留给 S5 实测。
  - Codex：High 模型调用失败时把上游报错原文放进 `ModelError`（可能带 key、授权头、带凭据的 URL）——改为只给错误类型，原异常留作 cause；Medium 空回复（没有 choice、没有工具块）抛 `IndexError`——改为 `ModelError`；Medium 同批两条事实指向同一条旧记忆，第二条降级为新增，留下矛盾的当前事实——改为逐条消解，本批已接受的事实作为候选（批内临时 id）交给后面的事实，后者可以判 NOOP 或在批内取代前者并继承它对旧记忆的取代，指向已被本批取代的旧记忆时改为指向取代它的那条；Medium 同一幂等键的并发请求都会调一遍模型——不改：只写一次，多花的是成本，杜绝它要跨模型调用持锁，不值得。
  - /simplify 后：引擎核对目标必须在候选里且仍是当前，每次降级都写原因（消解器只解析 id）——测试证明它挡住了"消解器指向同 namespace 里别的用户的记忆"；两条写路径共用 `_commit`；客户端只注入一个、两种 vendor 共用回复校验、Anthropic 客户端按目标缓存（原来两个角色会共用第一个）；提示词名、消解方式（`llm` / `add_only`）、相关记忆条数可配置（`LITEMEM_PIPELINE__*`），S5 直接切换；配置给定向量维度就不再探测；CLI 只开一个连接；内部查相关记忆时少取候选。没做：改用 SDK 的 `parse()`（手写的 strict schema 对现有模型够用，换了要连测试替身一起改）；并行调用消解（逐条消解需要顺序）；一次调用消解整批（S5 作为另一种实现对比）；出处拆成"抽取者 / 决定者"两个字段、返回结果里带 NOOP（S9 定事件契约时再看）；思考模式开关（S5 实测）。
- **S5（2026-11-12）**：合成对话集 `eval/dataset.json`（三个虚构的人：中文、英文、中英混杂；11 段对话、19 条应记住、7 条应忘掉、11 个问题），`litemem eval`（lite-mem 的 variant：`resolve`、`add_only`、`terse_prompt`、`thinking`；`--mem0` 加 mem0 2.2.1 对照；`--out` 存 JSON），每个角色可设 `reasoning`（以 `reasoning_effort` 发出；qwen3 关掉思考后一次调用约 20 秒 → 1 秒，Ollama `/v1` 不认 `think:false`）。
  - 结果（qwen3:8b + bge-m3，本地 Ollama，跑一次，19/7/11 条检查；文件 `eval/results/2026-11-12-s5.json`，详细分析在 KB `projects/lite-mem/reference/eval-results.md`）：

    | 系统 | 记住 | 该忘没忘 | hit@3 | 记忆数 | 秒/次写入 |
    | --- | --- | --- | --- | --- | --- |
    | resolve（关思考） | 16/19 | 4/7 | 8/11 | 21 | 9.9 |
    | add_only（关思考） | 18/19 | 6/7 | 10/11 | 25 | 3.1 |
    | terse_prompt（关思考） | 18/19 | 2/7 | 11/11 | 15 | 7.6 |
    | thinking（开思考） | 19/19 | 1/7 | 11/11 | 18 | 46.9 |
    | mem0 2.2.1 | 19/19 | 7/7 | 11/11 | 22 | 33.2 |

    结论：冲突消解是"改了的事实不留旧值"的来源（mem0 7/7 全留，lite-mem 开思考只留 1/7）；qwen3 关掉思考时消解会出错（把"住在上海"判成"住在北京"的重复而丢掉）；简短的抽取提示词对不思考的模型更好；所有系统（含 mem0）都存下了银行卡密码；NOOP 不留事件，查不到事实为什么丢了。默认值不改（一次运行、19 条检查不足以换默认），terse 提示词待第二次运行确认。
  - 由此加入的后续：写入前在代码里过滤敏感信息（密码、卡号），不靠提示词；NOOP 决定记在写入结果里（S9 定事件契约时）；S7 加整合后重跑。
  - 自己发现的：mem0 2.2.1 用 ollama embedder 时要 `ollama` 包但没声明依赖，第一次长跑在 lite-mem 跑完 12 分钟后才在 mem0 初始化时崩溃、结果全丢——eval 依赖里加上 `ollama==0.6.3`，并改为先初始化 mem0 再跑（测试覆盖）；CI 的测试库密码是 `postgres`，脱敏后的 `postgresql://postgres:***@` 总会包含它，"输出里没有密码"的断言必然失败——CI 换成不会在输出里巧合出现的密码（`363b770`）。
  - Codex：P1 stale 评分只要记忆里提到新值就放过旧值（"住 Auckland 和 Wellington" 不算 stale）——`unless` 改为"旧值已过去"的措辞（moved from、以前、used to），先红后绿；P1 mem0 对照按固定的 user id 在固定的 `mem0_eval` 里删数据，会删到别人的——每次运行用自己的 collection 和 history 目录，结束时只删自己的；P2 `get_all` 上限 1000、`MEM0_DIR` 复用旧 history——随上一条一起解决；P2 角色的 `reasoning` 清不回端点默认——加 `"default"`。
  - /simplify 后：mem0 对照的模型、端点、维度改从 `Settings` 读（原来写死，"同一个本地模型"只是巧合）；`--out` 记录数据集 sha256、git sha、各系统的设置（不含数据库 URL，密钥遮盖）和结果，KB 里的数字能对回到那次运行；`Report` 带 k、比例共用一个函数、CLI 复用 `_engine`。没做（留给 S7，那时 `maintain` 的接口才定）：对话带时间、`System` 加 maintain 钩子、`Report` 加合并数和每次写入的模型调用数；评分拆成可替换的 `Scorer`（关键词 / LLM 评判，接 LoCoMo 时需要）；variant 放进数据文件、命令行覆盖设置（阈值扫描时需要）；按用户并发跑（取决于 Ollama 是否并行）。
- **S5b（2026-11-12）**：`sensitive.py` 用规则在代码里识别密码 / PIN / 验证码（关键词后面带数字的值，可以被空格分成几段）、银行卡号（卡组织号段 + Luhn 校验，前面标着订单、运单、发票的除外）、身份证号（校验位）、常见 API key；`LITEMEM_PIPELINE__SENSITIVE` = `drop`（默认）| `mask` | `keep`。引擎自己得出的事实（抽取结果、`infer=false` 原样存的消息）遇到敏感内容就丢掉或打码；调用方明确给的事实（`remember`、`update`）在 `drop` 下返回 `InvalidRequest`，让调用方知道没存；所有事件的 reason 一律打码（reason 是自由文本，不拒绝）；日志只记类型和数量。真实模型重跑（`eval/results/2026-11-12-s5b.json`，`add_only` 和 `terse_prompt`）：银行卡密码不再成为记忆，其余检查和 S5 持平。
  - 取舍：不带数字的密码（"密码是 tiger"）识别不到——要识别就会误伤"forgot her password"这类普通事实；识别靠规则，不靠模型判断（模型判断可以以后作为 `mask` 的补充，不能替代规则）。
  - Codex：High 带空格的值（"PIN 是 4829 1038"）漏掉——值后面允许跟最多三段带数字的片段；High `update` 的 reason 原样进历史——改在 `_apply` 里给所有事件的 reason 打码，模型写的消解理由也覆盖；Medium 符合 Luhn 的订单号被当成卡号——加卡组织号段，并排除前面标着订单 / 运单 / 发票的数字；Medium 重叠检查是平方复杂度——改为按字符标记，测试用两万处匹配限时 5 秒（平方实现要七秒以上）。每条都先有失败的测试。
  - /simplify：自己做（diff 约 150 行生产代码）：`keep` 的判断收进一个 `_secrets` 帮助函数；`Policy` 类型在配置里复用，不重复写 Literal。
- **S6（2026-11-12）**：FastAPI 服务（`litemem serve`，extra `server`，`127.0.0.1:8710`）。API key 写在配置里（`LITEMEM_SERVER__API_KEYS`，只存 sha256，`litemem key new` 生成，token 只显示一次），按 glob 限定 namespace（请求头 `X-LiteMem-Namespace`，默认 `LITEMEM_NAMESPACE`）；没配 key 时只服务直连的回环客户端，`serve` 拒绝绑定其他地址；错误一律 Problem Details，上游错误只记（脱敏的）日志不返回，校验错误不回显输入；`Idempotency-Key`；`/health` 只查数据库（给频繁的探针），`/health?deep=true` 再做一次 embedding（S9 的 provider 用）。存储、embedding、LLM 客户端只建一次（`factory.Components`），每个 namespace 一个轻量引擎。Dockerfile（非 root）、`deploy/compose.yaml`（自带 PG，Ollama 在宿主机，`.env` 整个传进容器）、CI 构建并启动镜像。验收：`docker compose up` 后用 curl 走了 401 / 403、add → search → history、同一幂等键重放，容器里用库的 CLI 读到同样的结果；用真实 qwen3 写入"搬到 Auckland"，判成 SUPERSEDE（91 秒）。
  - 验收中发现：被取代的是一条复合事实（"住 Wellington，是护士"），整条失效后"护士"也跟着没了。事实要拆得更原子，或取代时保留没变的部分——放进 S7。
  - /simplify 后：共享组件（原来每个 namespace 一套 HTTP 客户端、一次维度探测，而且建引擎时持有全局锁，第一次用某个 namespace 会卡住所有请求）——改为 `Components` + 按 namespace 加锁、已建好的不加锁读；健康检查不经过 namespace，Docker 每 10 秒的探针不再做 embedding（原来会让 Ollama 永远不卸载模型）；Store 加 `ping()`；去掉自定义的 `Problem` 异常，改用 `HTTPException`；namespace 检查改成一条直线（缺 namespace 一律 400）；compose 用 `env_file` 把 `.env` 全部传进去（原来只列了几个变量，其他设置在容器里悄悄不生效）；CI 不只构建镜像，还启动它；测试里的假引擎收成一个 helper。没做：服务只开已存在的 namespace（S9 里 llm-steward 的 namespace 是动态的，key 的 glob 就是授权边界）；连接池大小可配（事务很短，这个规模下只会排队）；Docker / CI 层缓存（收益小）。
  - Codex：High 没配 key 时信任直接对端，本机的反向代理会让所有请求看起来都来自回环——没配 key 时，带 `Forwarded` / `X-Forwarded-For` / `X-Real-IP` 的请求一律拒绝，要放到代理后面就必须配 key；Medium 未预料的异常绕过 Problem Details——加兜底 handler，返回通用 500，日志脱敏。都先有失败的测试（变异检查：去掉对应代码，测试变红）。
  - 提交后的自动安全审查：没配 key 时，本机浏览器里的网页可以借 DNS rebinding 把请求打到 `127.0.0.1:8710`——没配 key 时还要求 Host 是回环名字（`localhost`、`127.0.0.1`、`[::1]`），并拒绝带 `Origin` 的请求（浏览器发的）。
- **S7（2026-11-12）**：过期（读查询按引擎时钟排除，`maintain` 以到期时间记 EXPIRE）、合并（`consolidate` 角色，同一 scope、同一类型）、归纳和遗忘（默认关闭）、复合事实的 `remainder`、`litemem maintain` 和 `POST /v1/maintain`、`maintain` 返回事件、eval 的 maintain 钩子；规则见 §6，llm-steward 的映射见 §10。
  - 合并阈值：在最终代码上跑了基线 + 0.80 / 0.85 / 0.90 / 0.95 四个阈值（`eval/results/2026-11-12-s7-merge.json`），**四个都没有合并任何一条**——合成集里的重复在写入时已被消解器判成 NOOP，扫描定不了阈值。改为直接看 `add_only`（不消解、重复都留着）记忆两两之间的 bge-m3 相似度：唯一一对真重复（"每天早上喝一杯绿茶" / "每天喝绿茶"）0.946；最高的非重复全是冲突或变化（Auckland / Wellington 0.848、准备 / 通过 AWS 考试 0.829、北京 / 上海 0.773）。所以默认 0.92：高于所有冲突对，低于真重复；阈值放到 0.80–0.85 会把冲突送给模型判断"是否同一件事"，判错就把矛盾合成一条。证据很薄（只有一对真重复），换模型或有了真实数据要重看。
  - 这次扫描的另一个用处：五个配置实际行为相同（都没合并），它们之间的差异就是同一配置重跑的噪声——该忘没忘 2–4/7、hit@3 8–10/11、每次写入 7.9–73.9 秒。一次运行差一两条不能下结论。
  - 自己发现的：写入时 `expires_at` 已过的记忆让 `maintain` 违反 `valid_to >= valid_from` 检查、整次运行失败（CLI 测试撞出来的）——改为写入时拒绝，关闭时间也不早于 `valid_from`；eval 遇到 Ollama 偶发的 500（今天 12 次，S7 之前就有）整次中断、结果全丢——单次调用失败记一次 error，继续跑。
  - /simplify 后：Merge 并进 Add（`replaces` 是 0..n 个 id，`closes_as` 是 SUPERSEDE 或 MERGE），新记忆的构造只有一处（`_new_memory`），派生事实继承哪些字段只有一条规则（`_inherit`；原来三处各不相同，合并或拆出 remainder 会把会过期的记忆变成永久的）；派生事实的筛查只有一个入口（`_screened`）；`_embedded` 泛型化，去掉断言；`now` 每次 `maintain` 只取一次；遗忘的类型改为"配了半衰期的类型"；提示词里的记忆列表只有一个写法（`prompts.listed`）；合并提示词只允许"同一事实"，不再把不同的事实拼成一条（会重新制造 S6 的复合事实问题）；eval 的 Variant 显式带 maintain 开关，计数改名 `closed`。没做：每条记忆一次相似查询改成一条 SQL 自连接（离线、每个用户几秒，慢了再改）；把 maintenance 移出 engine.py（大量用到引擎的私有方法，拆开更糟）；按游标拉取事件的接口（S9 定）。
  - Codex：High 合并写入时只做了不加锁的检查、没核对来源仍在同一 scope、两次 `maintain` 并发会把同一组合并两次——改为写入事务里按 id 顺序加锁、复查（仍有效、同一 scope），不符合就跳过这一组；High profile 可以被配置成参与合并、丢掉 key——配置拒绝 `profile`，合并按类型分组；Medium 已过期但未关闭的记忆仍能被 `update` 等写操作当作当前——"当前"统一为 `Memory.holds_at`，profile 的旧值过期后写新值记 EXPIRE 而不是 SUPERSEDE；Medium 写入时已过期的 `expires_at`——拒绝；Medium 过期时未加锁——加锁复查。都先有失败的测试（变异检查）。不改：`keep` 策略下合并结果可能带出秘密——`keep` 是运营方明确的选择，模型生成的合并也照它办；性能——见上。
- **S7b（2026-11-12）**：`SqliteStore`（extra `sqlite`，sqlite-vec 0.1.9），`LITEMEM_DATABASE_URL=sqlite:///<文件>` 选用；做法见 §4。验收：测试的 `store` fixture 参数化成 PG 和 SQLite，所有 store 层的测试（写入、取代、幂等、并发、故障注入、检索、BM25 与 `keyword.py` 逐项相等、过期、合并、敏感信息）在两个 store 上都跑；只依赖 PG 本身的（HNSW 扫描、CLI 和 eval 的 schema、psycopg 连接）用 `pg_store`。另有 SQLite 的端到端测试（真实 factory 和 CLI，只换 embedding）。
  - 自己发现的：SQLite 下测试偶发失败——`uuid7()` 在同一毫秒内的两个 id 顺序随机，而列表在时间相同时按 id 排序；PG 每次写入都有网络往返，很少落在同一毫秒，所以一直没暴露。`uuid7()` 改成单调（RFC 9562 §6.2 方法 3：毫秒 + 74 位计数器，同一毫秒或时钟回拨时加一），先有失败的测试；修完 SQLite 那一半连跑八次都绿。
  - /simplify 后：scope 隔离规则（`own_rows`）、namespace 绑定检查（`check_binding`）、列清单（`MEMORY_COLUMNS`）两个 store 共用；SQLite 带共享记忆时改成两次索引查找（`EXPLAIN QUERY PLAN` 证实原来的 `ns AND (own OR shared)` 会扫整个 namespace）；合并分组时每次相似查询一个短事务（SQLite 每个事务都持写锁，一整轮分组会让所有读写排队）；uuid7 简化成（毫秒，计数器）；测试的 helper 收拢、类型标注改成 `Store`；CLI 和 factory 共用 `is_sqlite`。没做：只读事务用 `BEGIN DEFERRED`（要改 Store 接口，合并分组改成短事务后收益很小）；`erase` 里清 `superseded_by` 的那句（公开 API 不会产生跨用户的链接，取代只在同一 scope 内，两个 store 都只是防御）。
  - Codex：P1 SQLite 的 metadata 过滤是逐键相等，PG 是 jsonb 包含（`@>`：嵌套对象只要包含、数组只要有、`true` 不等于 `1`）——SQLite 注册一个和 jsonb 包含规则相同的函数，同一个测试在两个 store 上查；P2 `close()` 会关掉别的线程正在用的连接——等进行中的事务结束再关，之后的事务报"已关闭"；P2 不带时区的时间在 SQLite 按本机时区、在 PG 按会话时区——引擎入口拒绝（`as_of`、`expires_at`）；P2 uuid7 在时间戳上限会回绕——报错（公元 10889 年）。都先有失败的测试（变异检查）。
- **S8（2026-11-12）**：整个引擎的收尾审查（Codex 一份、/simplify 两份，都读整个仓库，找只有放在一起才看得出的问题）。
  - Codex：High 共享记忆的条件只看 user 和 agent 为空，只属于某个 session 的记忆会被所有 session 看到——共享要求 session 也为空（两个 store 共用一条规则）；High metadata 不过敏感信息筛查，密码放在 metadata 里会被存下来——metadata 里每个文字值都过筛（`drop` 拒绝、`mask` 打码）；Medium glob key 可以无限新建 namespace——一个服务进程最多打开 `max_namespaces` 个（默认 1000），超了 403；Medium 列表没有上限——`list_memories` 分页（库和 HTTP 一样），HTTP 默认 1000；Medium 请求大小没有上限——一次写入最多 200 条消息、每条 2 万字，在引擎里检查（库、CLI、HTTP 一致），HTTP 的 schema 也写上；Medium/Low 文档和代码不一致——改文档。都先有失败的测试。不改：`POST /v1/maintain` 对整个 namespace 返回的事件不设上限——这是运维调用，S9 按用户调用（§13）。
  - /simplify（altitude）：HNSW 索引自 S3 起没有查询用到、却是每次写入的成本——不再建，迁移 0004 删掉旧的，维度上限从 HNSW 的 2000 放到 `vector` 的 16000（先有失败的测试）；HTTP 缺 `remember`——加 `POST /v1/memories`，S9 推送治理过的记忆用它（带幂等键）；每个 namespace 一个引擎的缓存从 server 挪到 `factory.Engines`，S9 的进程内 provider 也用它；`/v1/maintain` 直接返回引擎的 `MaintenanceReport`（原来手抄了一个一样的 schema）。
  - /simplify（reuse / simplification）：`_write` 只收 Add、去掉没用的参数；`_closing` 去掉没用的参数；提示词文件名只在 `prompts` 定义一次；两个 LLM 客户端共用一个基类（客户端缓存、user 消息），Anthropic 的出处也用配置里的 provider；`RuleExtractor` 只有测试在用，挪到 `testing.py`；抽取的长度上限改名 `MAX_EXTRACTED_CHARS`，和引擎的 `MAX_FACT_CHARS` 分开；`FakeLLM.roles` 没人用，删掉；几处过时的注释；测试里重复的 `_current`、`_memory` 收进共享 helper。没做：engine.py（约 850 行）不拆——只有几个纯校验函数是干净的接缝，维护和推断都依赖引擎的私有方法，拆开会把唯一的写入口 `_apply` 散到几个文件；`Components` 的默认值和 `MemoryEngine` 的重复（只有一个测试 helper 用到默认值）；`MemoryEngine.health` 保留（S9 的进程内 provider 要用）；两个 store 里方言不同的 SQL 不抽象（由同一套测试保证一致）；测试里各文件自己的 `ADA` 常量保留（一行，读起来更直接）。
  - **留到 S8b**：BM25 按 scope 统计。现在每次检索、每条待消解的事实都要按整个 namespace 算一遍统计量和文档频率，开销随 namespace 增长，而不是随用户；S9 里一个 llm-steward namespace 就是一个 lite-mem namespace，里面有很多用户。改成先物化 scope 可见的行再算（和向量检索一样），排序会变，按规则要真实模型 eval 前后对比。
  - **留到 S9**：事件契约（时间、scope、NOOP、出处拆分）、离线变化怎么到达 llm-steward、延迟和租约——都要和 llm-steward 那边一起改，见 §13。

## 13. 待确认

- **llm-steward 接入时冲突消解归谁**（S9 前定）：
  - 引擎消解，llm-steward 映射事件：更简单，能力完整；
  - llm-steward 消解，引擎只抽取：治理层能看到每一次判断，但多一次往返。
- **合并阈值、遗忘阈值的默认值**：合并 0.92（S7，证据薄，见 §12）；遗忘默认关闭，等有长期使用的数据再定。
- **和 lite-rag 共用代码**（embedding、BM25、LLM 层）还是各自独立：v1 各自独立（两个仓库都在学习期，接口还在变）；稳定后再考虑抽出一个共享小包。
- **License**：仓库建立时用 Apache-2.0（与 llm-steward 一致），owner 可以改。
- **S9 开始前要定的（S8 审查汇总）**：
  - 事件契约：`MemoryEvent` 加时间（EXPIRE 按到期时间失效需要它）和 scope（整个 namespace 的 `maintain` 返回的事件要能对回用户）；NOOP 是否放进写入结果（S4、S5 都推到了这里）；出处是否拆成"抽取者 / 决定者"。和 llm-steward 的 `MemoryEvent`（现在只有 ADD / UPDATE / DELETE、没有 `replaced`）一起改，只改一次。
  - 离线变化怎么到达 llm-steward：它自己按用户调 `maintain`（推荐），还是加按游标拉事件的接口（`memory_events` 的 id 在并发写时提交顺序不保证，游标要留安全余量）。
  - llm-steward 的 `MemoryProvider.add` 没有幂等键，`ExtractionService` 只镜像 ADD。
  - 延迟和租约：一次 `add(infer=true)` 是一次抽取加每条事实一次消解，本地 qwen3 3–10 秒（不思考）到 90 秒（思考）；llm-steward 的任务租约 60 秒。要么一次调用消解整批（S4 起推迟的另一种 Resolver，要 eval 验证），要么给 lite-mem 的任务单独的租约。
  - BM25 按 scope 统计（见 §12 S8）。
