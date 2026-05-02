# GM3 in nvdamas: Local/Global Graph Memory, Retrieval, Routing, Prompt Rendering, and TextGrad Optimization

本文档整理当前 `nvdamas` 中 `graph_memory3` 的真实实现链路。它不是独立接管 `nvdamas` workflow 的新系统，而是在 **保持现有 AutoGen / solver / env 流程不变** 的前提下，把 GM2 的 graph memory backend 重新组织成更稳定的 prompt routing，并可选地通过 TextGrad / TextLoss 对 memory prompt 本身做迭代优化。

主要代码位置：

- [`mas/memory/mas_memory/graph_memory3.py`](/bigdata/xenial/nvdamas/mas/memory/mas_memory/graph_memory3.py)
- [`mas/memory/mas_memory/graph_memory2.py`](/bigdata/xenial/nvdamas/mas/memory/mas_memory/graph_memory2.py)
- [`mas/memory/mas_memory/gm2_backend/graph_types.py`](/bigdata/xenial/nvdamas/mas/memory/mas_memory/gm2_backend/graph_types.py)
- [`mas/memory/mas_memory/gm2_backend/construction_graph.py`](/bigdata/xenial/nvdamas/mas/memory/mas_memory/gm2_backend/construction_graph.py)
- [`mas/memory/mas_memory/gm2_backend/phasee_retrieval.py`](/bigdata/xenial/nvdamas/mas/memory/mas_memory/gm2_backend/phasee_retrieval.py)
- [`mas/memory/mas_memory/gm2_backend/phasee_routing.py`](/bigdata/xenial/nvdamas/mas/memory/mas_memory/gm2_backend/phasee_routing.py)
- [`scripts/medmcqa/eval_collab_domain_adaptation.py`](/bigdata/xenial/nvdamas/scripts/medmcqa/eval_collab_domain_adaptation.py)
- [`tasks/run.py`](/bigdata/xenial/nvdamas/tasks/run.py)

---

## 1. GM3 总览

GM3 的定位可以概括成：

```text
GM2 graph backend
  + GM3 routing / prompt shaping
  + optional TextGrad prompt optimization
  = GM3
```

GM3 的目标不是扩大 memory 数量，而是让已经构造出来的 local / global graph memory 更好地进入当前决策 prompt：

1. local graph 负责当前状态 grounding。
2. global graph 负责抽象 workflow / phase knowledge。
3. source priors 只做弱 search context。
4. TextGrad 可以在不改 workflow 的前提下，优化 GM3 memory prompt 的文本形态。

最重要的一点是：

```text
GM3 不接管 solver，不改 env.step()，不替换 AutoGen workflow。
它只改“memory 如何被组织成 prompt”。
```

---

## 2. 数据结构层

GM3 复用 GM2 backend 的图结构，不重新定义 episode graph。

### 2.1 LocalGraphMemory

定义在 [`graph_types.py`](/bigdata/xenial/nvdamas/mas/memory/mas_memory/gm2_backend/graph_types.py)：

```python
@dataclass(slots=True)
class LocalGraphMemory:
    agent_id: str
    nodes_by_signature: dict[str, GraphNode] = field(default_factory=dict)
    edges_by_signature: dict[str, GraphEdge] = field(default_factory=dict)
    candidates: dict[str, PromotionCandidate] = field(default_factory=dict)
    rules_by_id: dict[str, MemoryRule] = field(default_factory=dict)
    artifacts_by_id: dict[str, MemoryArtifact] = field(default_factory=dict)
    episode_ids: list[str] = field(default_factory=list)
```

含义：

- `nodes_by_signature`
  - 局部 episode graph 的状态节点、动作节点、failure 节点等。
- `edges_by_signature`
  - 节点之间的 temporal / causes / advances_to / fails_under 连接。
- `candidates`
  - 从 episode graph 诱导出来、尚未完全晋升的模式。
- `rules_by_id`
  - 规则级 memory，描述某个 progress_state 下的 precondition / workflow / closure / blocked / repair。
- `artifacts_by_id`
  - 更高层的可检索 memory，通常包含 prototype / rule / reflection。
- `episode_ids`
  - 已并入该 local memory 的 episode 列表。

### 2.2 GlobalGraphMemory

```python
@dataclass(slots=True)
class GlobalGraphMemory:
    candidates: dict[str, PromotionCandidate] = field(default_factory=dict)
    rules_by_id: dict[str, MemoryRule] = field(default_factory=dict)
    artifacts_by_id: dict[str, MemoryArtifact] = field(default_factory=dict)
    promoted_batches: list[str] = field(default_factory=list)
```

含义：

- `candidates`
  - 从多个 local memory 中挑出来的可迁移候选。
- `rules_by_id`
  - 全局规则。
- `artifacts_by_id`
  - 全局 artifact。
- `promoted_batches`
  - 已经做过 promotion 的批次名称。

### 2.3 GraphNode / GraphEdge

仍然是 GM2 backend 中的基本图元：

- `GraphNode`
  - `node_id`
  - `node_type`
  - `signature`
  - `payload`
  - `stats`
- `GraphEdge`
  - `src`
  - `dst`
  - `edge_type`
  - `signature`
  - `payload`
  - `stats`

`stats` 中最常见的字段是：

- `support`
- `positive`
- `negative`
- `stalled`
- `confidence`

### 2.4 PromotionCandidate

```python
@dataclass(slots=True)
class PromotionCandidate:
    candidate_id: str
    candidate_type: CandidateType
    summary: str
    structure: dict[str, Any]
    source_episode_ids: set[str] = field(default_factory=set)
    source_scenes: set[str] = field(default_factory=set)
    positive: int = 0
    negative: int = 0
    stalled: int = 0
    utility: float = 0.0
```

它是 local memory 中诱导出来的“待晋升模式”：

- `PRECONDITION`
- `FAILURE`
- `WORKFLOW`
- `REPAIR`

### 2.5 MemoryRule

```python
@dataclass(slots=True)
class MemoryRule:
    rule_id: str
    rule_type: RuleType
    summary: str
    task_family: str = ""
    goal_arity: int = 1
    progress_state: str = ""
    goal_roles: dict[str, str] = field(default_factory=dict)
    condition: dict[str, Any] = field(default_factory=dict)
    effect: dict[str, Any] = field(default_factory=dict)
    source_episode_ids: set[str] = field(default_factory=set)
    source_scenes: set[str] = field(default_factory=set)
    stats: RuleStats = field(default_factory=RuleStats)
    specificity: float = 0.0
    conflict: float = 0.0
```

规则的意义是把 episode graph 压成更可迁移的状态-动作知识。

### 2.6 MemoryArtifact

```python
@dataclass(slots=True)
class MemoryArtifact:
    artifact_id: str
    kind: ArtifactKind
    summary: str
    anchor: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)
    level: str = "local"
    source_episode_ids: set[str] = field(default_factory=set)
    source_scenes: set[str] = field(default_factory=set)
    stats: ArtifactStats = field(default_factory=ArtifactStats)
    specificity: float = 0.0
    conflict: float = 0.0
```

GM3 的 routing 主要读取 artifact / rule / candidate 的摘要和统计量，而不是直接读原始历史轨迹。

### 2.7 MemoryQuery

GM3 retrieve 的查询对象：

```python
@dataclass(slots=True)
class MemoryQuery:
    goal: str
    scene_id: str | None = None
    current_stage: str | None = None
    location: str | None = None
    progress_hint: str = ""
    progress_state: str = ""
    task_family: str = ""
    goal_roles: dict[str, str] = field(default_factory=dict)
    required_count: int = 1
    held_relevant_count: int = 0
    placed_relevant_count: int = 0
    remaining_relevant_count: int = 0
    destination_reached: bool = False
    goal_object_matches_visible: bool = False
    admissible_actions: tuple[CanonicalAction, ...] = ()
    desired_types: tuple[CandidateType, ...] = ()
    failure_label: str | None = None
    keywords: tuple[str, ...] = ()
    belief: dict[str, Any] = field(default_factory=dict)
    dynamic_context: dict[str, Any] = field(default_factory=dict)
```

它会把当前 env 状态、目标约束和动态上下文统一成一个 retrieval query。

---

## 3. Memory 构造流程

GM3 的 memory 构造仍然是 GM2 的构造链路。

### 3.1 从轨迹到 EpisodeRecord

ALFWorld 轨迹会先通过 adapter 转成结构化 episode：

```text
history / traj
  -> ALFWorldAdapter.episode_from_history(...)
  -> EpisodeRecord
```

`EpisodeRecord` 包含：

- `agent_id`
- `scene_id`
- `task_id`
- `goal`
- `steps`
- `metadata`

每个 `EpisodeStep` 包含：

- `state`
- `action`
- `next_state`
- `feedback`
- `subgoal`

`StateSummary` 里会记录：

- `location`
- `visible_objects`
- `open_containers`
- `closed_containers`
- `held_objects`
- `searched_locations`
- `admissible_verbs`
- `workflow_stage`
- `raw_observation`

`CanonicalAction` 会把动作规范化成统一字符串，例如：

```text
go(target=fridge_1)
open(container=cabinet_2)
take(object=apple_1,source=countertop_1)
move(object=apple_1,destination=fridge_1)
```

---

### 3.2 EpisodeGraphBuilder

`EpisodeGraphBuilder.build(episode)` 会把轨迹变成图：

- `STATE`
- `ACTION`
- `SUBGOAL`
- `FAILURE`

以及边：

- `TEMPORAL`
- `CAUSES`
- `ADVANCES_TO`
- `FAILS_UNDER`

本质上是：

```text
state -> action -> next_state
action -> delta / subgoal / failure
```

这一步把“执行过程”变成“可积累的图 memory”。

---

### 3.3 LocalGraphMaintainer.update

局部 memory 的增量更新在：

```python
LocalGraphMaintainer.update(local_memory, episode_graph, episode)
```

它会做四类事：

1. 累计 graph 节点和边的统计量。
2. 从 episode graph 中诱导 `PromotionCandidate`。
3. 从 candidate/rule 中生成 `MemoryRule`。
4. 生成 `MemoryArtifact`，供后续 retrieval / promotion 使用。

#### 局部 graph 的统计

同一 signature 的 node/edge 会累积：

- `support`
- `positive`
- `negative`
- `stalled`

这样 local memory 不是“若干条完整 episode 存档”，而是“跨 episode 的统计图”。

#### candidate 诱导的类型

常见 candidate 包括：

- `workflow`
- `precondition`
- `failure`
- `repair`
- `closure`
- `anti_pattern`

#### rule 诱导的类型

常见 rule 包括：

- `precondition`
- `blocked`
- `repair`
- `workflow`
- `closure`

#### artifact 诱导的类型

最重要的 artifact 类别：

- `scene_relation`
- `source_type_prior`
- `plan`
- `reflection`
- `rule`

---

### 3.4 source_relation / source_type_prior

GM3 里最关键的局部/全局迁移经验，很多都来自这两类：

#### scene_relation

当成功 episode 中出现：

```text
take <target> from <source_instance>
```

会诱导出：

- `pattern_kind = "scene_relation"`
- `relation_kind = "object_location_prior"`
- `source_role`
- `source_base`
- `source_instance`
- `goal_signature`

它表达的是：

```text
这个 target 在这个布局里，往往和某个具体 source instance 有关联。
```

#### source_type_prior

当 `source_base` 不是具体编号，而是抽象 source type 时，还会诱导：

- `pattern_kind = "source_type_prior"`
- `source_base`
- `source_role`
- `goal_object`

它表达的是：

```text
某类 source type 在相似任务中更值得先试。
```

这部分是 GM3 / GM2 中最接近“迁移能力”的经验。

---

### 3.5 GlobalPromoter.promote

`GlobalPromoter` 会把多个 local memory 中合格的 candidate / rule / artifact 提升为 global memory。

关键点：

- 需要足够 support / confidence / coverage。
- `scene_relation` 过于具体的内容通常不会直接上 global。
- `source_type_prior` 会被视为更抽象、更可迁移的模式。
- `REFLECTION` 通常不直接进入 global。

global memory 最后会落到：

- `global_memory.json`
- `summary.json`

---

## 4. Retrieve 流程

GM3 的 retrieve 在 [`graph_memory3.py`](/bigdata/xenial/nvdamas/mas/memory/mas_memory/graph_memory3.py) 中完成。

### 4.1 `_retrieve_external_prompt_payload`

简化流程如下：

```text
query = build query from current env state
local_memory = current scene local graph memory
global_memory = shared global graph memory
bundle = external_retriever.retrieve(query, local_memory, global_memory)
route = _render_gm3_textloss_evidence(...)
planner_notes = ["### GM3 MEMORY DECISION SUMMARY\n" + prompt]
return planner_notes to nvdamas
```

代码骨架：

```python
def _retrieve_external_prompt_payload(self, **kargs) -> dict[str, list[str]]:
    query = self._build_external_query(**kargs)
    owner_scene = self._resolve_external_owner_scene(...)
    local_memory = ...
    global_memory = ...
    bundle = self._external_retriever.retrieve(query, local_memory, global_memory)
    route = self._render_gm3_textloss_evidence(...)
    planner_notes = ["### GM3 MEMORY DECISION SUMMARY\n" + str(route["prompt"]).strip()]
    return {"planner_notes": planner_notes, ...}
```

### 4.2 `gm2_settings`

GM3 继承 GM2 的 memory 开关：

- `base`
- `local_only`
- `global_only`
- `local_plus_global`

它们决定：

- 只用 local
- 只用 global
- 二者一起用
- 或基本不用 memory

---

## 5. Routing 流程

GM3 routing 的目标不是“把所有 memory 都塞进 prompt”，而是：

1. 从 memory 中产生几个 slot。
2. 按当前状态打 text-loss。
3. 按 phase 组装出更短、更可执行的 memory block。

### 5.1 GM3 的候选 slot

GM3 当前主要构造这些 slots：

- `phase_policy`
- `global_workflow`
- `local_grounding`
- `source_roles`
- `failure_avoidance`

它们分别代表：

- `phase_policy`
  - 当前 macro phase 下该做什么。
- `global_workflow`
  - 任务级抽象 workflow。
- `local_grounding`
  - 当前 scene 的局部图 grounding。
- `source_roles`
  - search phase 下的 source-type prior。
- `failure_avoidance`
  - 失败规避、反例提醒。

### 5.2 `_gm3_section_textloss`

每个 section 会被粗略计算一个 loss。

主要维度：

- `phase_fit`
- `transfer_value`
- `grounding_value`
- `actionability`
- `noise`
- `wrong_phase`

直觉上：

- phase 对了，loss 下降。
- 能映射到当前 admissible action，loss 下降。
- 太具体、错 phase、太长，loss 上升。

### 5.3 `_gm3_select_composed_sections`

GM3 不只是按 loss 排序，而是按角色组装：

```text
phase_policy
  -> local_grounding
  -> source_roles (search phase 才更重要)
  -> global_workflow
  -> failure_avoidance
```

也就是说：

- phase 是锚。
- local 负责当前状态落地。
- global 负责抽象迁移。
- source_roles 只在 search phase 起作用。
- failure_avoidance 防重复踩坑。

---

## 6. Prompt 构造流程

最终注入 workflow 的是一个固定短槽位的 GM3 memory decision summary，而不是完整 graph object。

### 6.1 `_gm3_render_decision_summary`

这一步会把 selected sections 渲染成统一的 7 行短摘要：

```text
Current phase: <search target / acquire visible target / process held target / deliver held target>.
Current state: target=<object>; tool=<tool|none>; destination=<destination>; held=<...>; visible=<...>; searched=<...>
Local memory: <current-state local graph grounding or no reliable local grounding>
Global memory: <transferable workflow/source-role hint or no reliable transferable workflow>
Failure memory: <similar failure warning or no specific failure warning>
Next priority: <one concrete current priority or safe fallback>
Confidence / caveat: <why to trust or treat as advisory>
```

这个格式是当前 GM3 的核心约束：短、稳定、槽位固定。它避免了早期版本把 local/global/failure evidence 展开成多段长 prompt，降低对 solver 原始任务指令的 attention 干扰。

### 6.2 local grounding 的渲染

local grounding 常见输出：

```text
local graph links target_object=cup to source `countertop 3`; currently admissible grounding is `go to countertop 3`.
```

它的目的不是“泛泛提示方向”，而是直接告诉模型：

- 当前 target 和某个 source instance / source base 的局部图联系是什么。
- 这个联系在当前 admissible action 里怎么落地。

在 7 行 summary 中，它会被压到：

```text
Local memory: local graph links target_object=cup to source `countertop 3`; currently admissible grounding is `go to countertop 3`.
Next priority: go to countertop 3
Confidence / caveat: high; local grounding maps to a current admissible action.
```

### 6.3 source_roles 的渲染

source_roles 会把 source type prior 组织成一个 queue，例如：

```text
try countertop before broad search; current admissible queue: `go to countertop 1` -> `go to countertop 2` -> `go to countertop 3`.
Evidence: local exact evidence support=3, confidence=0.88.
```

它只在 search phase 起作用，而且有两个原则：

- 只做弱 search context，不覆盖 local grounding。
- 已经拿到目标后，不再继续强调 source search。

### 6.4 phase-specific priority

`Next priority` 的生成顺序是：

1. 如果已经持有目标物体，优先找 tool process action 或 destination delivery action。
2. 如果目标物体可见，优先 `take <target> ...`。
3. 如果处于 search phase，只允许 local grounding / source_roles 中**已经映射到当前 admissible action** 的候选进入 `Next priority`。
4. 如果 memory 只有 `prefer source type ... when actionable` 这类弱 source prior，它只能进入 `Local memory` / `Global memory` / `Confidence / caveat`，不能进入 `Next priority`。
5. 如果没有可靠 memory action，明确回退到 current observation / admissible actions，或者直接不注入 GM3 summary。

因此 GM3 现在不是“每步讲很多 memory”，而是把 memory 压成一个可执行优先级和一个 caveat。

### 6.4.1 Summary emit gate

GM3 不再只要有 phase/source evidence 就注入 summary。当前实现会先判断是否值得说话：

```text
允许注入 GM3 summary 的条件：

1. local/source memory 能映射到当前 admissible action
2. visible target / held target 有 concrete next action
3. failure memory 匹配当前 exhausted search 状态，且能产生一个由 memory 支持的当前可执行替代 search action
4. global workflow 非空，且已经通过 task/phase 过滤

否则：
  summary_suppressed = true
  summary_emit_reason = only_weak_or_non_actionable_memory
```

这个 gate 的目的，是避免把弱 evidence 当成强决策建议。例如：

```text
prefer source type countertop when it becomes actionable
```

这类内容可以作为 context，但不能单独触发 GM3 prompt，也不能成为 `Next priority`。

failure-only summary 还有一条更严格的边界：

```text
如果只知道“已经搜了很多 shelf/drawer/cabinet”，但当前没有 memory 支持的别的可执行 source action，
GM3 不注入 summary。
```

这时 debug reason 是：

```text
failure_only_without_memory_supported_alternative
```

GM3 会先从 local/global graph artifacts 里计算 `source_base -> evidence_score`：

```text
成功 source_type_prior / scene_relation/object_location_prior  -> 加分
searched_empty / 失败占优的同类 evidence                      -> 扣分或排除
已经 exhausted 的 source base                            -> 不能作为替代动作
tool / destination 本身                                  -> 不作为 search source
```

只有当某个 source base 有正向 memory score，并且当前 admissible actions 里存在可映射动作时，例如：

```text
go to coffeetable 1
go to sofa 1
open safe 1
```

GM3 才会注入 summary，并把替代动作写入 `Next priority`。这时 reason 是：

```text
failure_memory_has_supported_alternative_action
```

这个规则避免 GM3 只输出“别继续搜 X”这种不可执行提醒，也避免从 admissible action 里随便挑一个没搜过的位置。failure memory 只负责排除；下一步 source 必须由成功或相关 graph memory 支持。

### 6.4.2 SourceEvidenceTable

GM3 现在不会把 `source_roles`、`failure_avoidance`、`global_workflow` 各自独立交给 prompt。它会先构造一个内部证据表：

```text
SourceEvidenceTable:
  source_base
  source_role
  evidence_level = exact / same_family / role
  positive_score
  negative_score
  exhausted_penalty
  final_score
  admissible_actions
  evidence snippets
```

证据来源包括：

```text
local/global source_type_prior
local/global scene_relation/object_location_prior
local/global scene_relation/searched_empty
current exhausted_locations
current admissible_actions
```

分级原则：

```text
exact:
  same target_object + same task_family，最强。

same_family:
  different target_object but same task_family，中等迁移。

role:
  only source_role/source_base pattern，弱证据，只能做 context。
```

只有 `exact` 或 `same_family` 且 `final_score` 过阈值，并且能映射到当前 admissible action，才允许进入 `Next priority`。这一步解决了两个问题：

```text
1. 不再过早 top-k 截断 source evidence；
2. 不再把多个 retrieve/candidate section 直接塞进 prompt，而是先统一仲裁。
```

同时修复了 source action 映射：`go to cabinet 1` 会先解析成 command target `cabinet 1`，再归一化成 `cabinet`，而不是错误地变成 `go to cabinet`。

### 6.4.3 SourceEvidenceTable 是唯一 source 仲裁器

`source_roles` 不再从 raw artifact rows 直接渲染。所有 source prompt 都必须来自 `SourceEvidenceTable`：

```text
raw source artifacts
-> SourceEvidenceTable 聚合正负分
-> final_score > 0 的 source 才能渲染
-> final_score <= 0 的 source 不能进入 prompt
```

这避免了“表里认为 cabinet 是负分，但旧 source_roles 仍然推荐 cabinet”的冲突。

当正向 source_base 本身当前不可执行时，GM3 会做一个受控的 source_role fallback：

```text
tvstand / coffeetable -> support_surface
如果当前没有 tvstand/coffeetable action，
可从当前 admissible actions 中选择不同 source_base 的 support_surface 动作。
```

fallback 有两个限制：

```text
1. 只在原 source_base final_score > 0 时触发；
2. 每个 source_base 只取一个动作，避免 shelf 1 -> shelf 2 -> ... 这种长扫队列。
```

### 6.5 重复 prompt 签名

GM3 仍然记录 `prompt_signature`，并在 debug 中标记 `prompt_repeated`，但当前实现不再走旧的 “compact repeat prompt” 另一套渲染路径。原因是固定 7 行 summary 本身已经足够短，再维护第二套重复 prompt 会增加不一致和调试成本。

---

## 7. TextGrad / TextLoss 优化流程

这是 GM3 当前最重要的新部分之一。

### 7.1 目标

TextGrad 优化的对象不是 solver prompt，也不是 env workflow，而只是：

```text
GM3 MEMORY DECISION SUMMARY body
```

也就是：

```text
Current phase: ...
Current state: ...
Local memory: ...
Global memory: ...
Failure memory: ...
Next priority: ...
Confidence / caveat: ...
```

这个 block 的内容会被当成一个可优化变量。

### 7.2 相关配置

当前实现支持：

- `gm3_use_textgrad`
- `gm3_textgrad_engine`
- `gm3_textgrad_max_iters`
- `gm3_textgrad_pass_threshold`
- `gm3_textgrad_max_calls_per_episode`

### 7.3 TextGrad 代码入口

核心实现位于 [`graph_memory3.py`](/bigdata/xenial/nvdamas/mas/memory/mas_memory/graph_memory3.py)。

构造入口在 `_render_gm3_textloss_evidence(...)`：

```python
summary = self._gm3_render_decision_summary(...)
optimized = self._gm3_textgrad_optimize_prompt(
    draft_prompt=summary,
    query=query,
    selected=selected,
    routed=routed,
    admissible=admissible,
    visible=visible,
    held=held,
    exhausted=exhausted,
    step_index=step_index,
)
if optimized.get("debug"):
    routed["textgrad_prompt_optimization"] = optimized["debug"]
summary = str(optimized.get("prompt") or summary)
```

也就是说：

1. 先生成 deterministic draft。
2. 本地 gate 判断是否值得调用 TextGrad。
3. 如果调用 TextGrad，它只能重写这个 7 行 summary。
4. 优化失败或质量检查不过，就回退 deterministic draft。
5. 最终 prompt 仍然是 GM3 的 memory block。

### 7.4 `_gm3_textgrad_optimize_prompt`

这个函数是 GM3 的 TextGrad 闭环主体。其设计意图是：

- 把 draft prompt 作为 `tg.Variable`
- 用 `TextLoss` 评价 prompt 是否适合当前 ALFWorld 决策
- 用 `TGD` 重写 variable
- 用 judge 再检查一遍
- pass 或达到阈值后停止

代码骨架：

```python
def _gm3_textgrad_optimize_prompt(...):
    draft = str(draft_prompt or "").strip()
    if not self._gm3_use_textgrad:
        return {"prompt": draft, "debug": debug}
    if not self._gm3_textgrad_engine:
        return {"prompt": draft, "debug": debug}

    context = self._gm3_textgrad_context(...)
    engine = tg.get_engine(self._gm3_textgrad_engine)
    variable = tg.Variable(
        draft,
        requires_grad=True,
        role_description="GM3 memory decision summary injected into an ALFWorld agent",
    )
    loss = tg.TextLoss(self._gm3_textgrad_loss_prompt(context), engine=engine)
    optimizer = tg.TGD([variable], engine=engine, constraints=[...])
    judge = tg.BlackboxLLM(engine=engine, system_prompt=self._gm3_textgrad_judge_system_prompt())

    for iteration in range(max_iters + 1):
        current = self._gm3_sanitize_optimized_prompt(variable.get_value(), fallback=draft)
        judge_payload = self._gm3_textgrad_judge_payload(context=context, prompt=current)
        judge_output = judge(tg.Variable(judge_payload, requires_grad=False, role_description="GM3 prompt judge input"))
        parsed = self._gm3_parse_textgrad_judge(...)
        if parsed["pass"] or parsed["score"] >= threshold:
            break
        optimizer.zero_grad()
        objective = loss(variable)
        objective.backward()
        optimizer.step()

    final_prompt = self._gm3_sanitize_optimized_prompt(best_prompt, fallback=draft)
    quality_issue = self._gm3_optimized_prompt_quality_issue(final_prompt, draft_prompt=draft, query=query)
    if quality_issue:
        final_prompt = draft
    return {"prompt": final_prompt, "debug": debug}
```

### 7.4.1 TextGrad gate

TextGrad 不是每步都跑。当前 gate 的基本原则是：

- visible target 且已有 concrete priority：不跑。
- held target 且已有 process/delivery priority：不跑。
- search phase 有 actionable local grounding：不跑。
- 没有 local/global/failure memory 信号：不跑。
- 搜索卡住、只有泛化 memory、缺少具体 priority、或 wrong-phase prompt 风险较高时才跑。

这样做的目的不是抗拒 TextGrad，而是避免把已经正确的短 summary 送去昂贵地重写。TextGrad 的职责是修正“不知道怎么组合 local/global/failure memory”的场景，不是重写每一步 already-good prompt。

### 7.5 `_gm3_textgrad_context`

这个函数把 TextGrad 优化所需上下文打包成文本：

- 当前任务状态
- 当前阶段
- 目标 object / tool / destination
- held / visible / exhausted
- admissible_actions
- 已选中的 local/global graph evidence
- routing/loss diagnostics

它就是 TextGrad 的“输入上下文”。

### 7.6 `_gm3_textgrad_loss_prompt`

这是 TextLoss 的主要约束说明，核心要求包括：

1. phase-correct
2. local graph grounding
3. global 只提供抽象 workflow/source-role guidance
4. 抑制弱/错 phase/重复/噪声 memory
5. 对 seen / unseen 都有帮助
6. prompt 要简洁且 action-oriented
7. 必须使用固定 7 个字段：
   - `Current phase`
   - `Current state`
   - `Local memory`
   - `Global memory`
   - `Failure memory`
   - `Next priority`
   - `Confidence / caveat`

### 7.7 judge system prompt

```python
@staticmethod
def _gm3_textgrad_judge_system_prompt() -> str:
    return (
        "You are a strict judge for a GM3 memory decision summary injected into an ALFWorld agent. "
        "Evaluate whether the prompt will likely help the next action without harming generalization. "
        "Return ONLY compact JSON with keys: pass, score, issues, rewrite_instruction. "
        "score is 0..1. pass should be true only if the prompt is phase-correct, concise, "
        "uses local/global memory appropriately, and avoids noisy or concrete global location transfer. "
        "A prompt with unresolved placeholders, decorative symbols, markdown emphasis, more than seven lines, "
        "or a missing concrete next priority must not pass."
    )
```

### 7.8 judge payload

```python
@staticmethod
def _gm3_textgrad_judge_payload(*, context: str, prompt: str) -> str:
    return (
        "Decision context:\n"
        f"{context}\n\n"
        "GM3 memory decision summary to judge:\n"
        f"{prompt}\n\n"
        "Judge requirements:\n"
        "- It must use these fields: Current phase, Current state, Local memory, Global memory, Failure memory, Next priority, Confidence / caveat.\n"
        "- It must be seven lines or fewer.\n"
        "- If target is held, the prompt must prioritize process/delivery and ignore source search.\n"
        "- If target is visible, it must prioritize taking the matching target.\n"
        "- If in search phase, it may use source/action priorities only when they are task-relevant.\n"
        "- Local graph evidence should ground current actions; global evidence should stay abstract and transferable.\n"
        "- Penalize broad generic advice, wrong object/tool/destination, concrete global scene positions, and long noisy text.\n"
        "Return JSON only."
    )
```

### 7.9 judge JSON 解析

```python
def _gm3_parse_textgrad_judge(self, text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    parsed: dict[str, Any] = {"pass": False, "score": 0.0, "issues": [], "rewrite_instruction": raw[:500]}
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                parsed.update(data)
        except Exception:
            pass
    ...
    return parsed
```

### 7.10 prompt sanitize

TextGrad 产生的输出会经过清理：

```python
def _gm3_sanitize_optimized_prompt(self, value: Any, *, fallback: str) -> str:
    text = str(value or "").strip()
    if not text:
        return str(fallback or "").strip()
    text = re.sub(r"^```(?:\w+)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(r"^#+\s*GM3 (?:CURRENT MEMORY PRIORITY|MEMORY DECISION SUMMARY)\s*", "", text, flags=re.IGNORECASE)
    lines = [line.rstrip() for line in text.splitlines()]
    compact = "\n".join(lines[:7]).strip()
    if len(compact) > 1000:
        compact = compact[:1000].rsplit("\n", 1)[0].strip()
    return compact or str(fallback or "").strip()
```

这一步的作用是：

- 去掉 markdown fences
- 去掉重复 header
- 控制为最多 7 行 / 约 1000 字符以内
- 防止 TextGrad 输出太长破坏 prompt

### 7.11 optimized prompt quality check

TextGrad 的最终输出还会经过 `_gm3_optimized_prompt_quality_issue(...)`：

- 空输出回退。
- code fence / `<think>` 回退。
- `[goal_destination]` 这类 unresolved placeholder 回退。
- emoji / markdown emphasis 回退。
- 超过 7 行回退。
- 缺少固定 7 个字段回退。
- target / destination 明显缺失或未绑定回退。

这一步是为了保证 TextGrad 只优化 memory summary 的表达和筛选，而不是引入新噪声。

---

## 8. Debug / 落盘字段

GM3 新增了独立 debug trace：

- `gm3_debug_trace.jsonl`

同时也会进入 GM2 的 debug 通道。

### 8.1 事件类型

常见事件：

- `task_start`
- `retrieve`
- `gm3_retrieve`
- `textgrad_prompt_optimization`
- `textgrad_prompt_optimization_error`
- `action_hook_observe`
- `env_feedback`

### 8.2 retrieve 记录

`retrieve` / `gm3_retrieve` 常记录：

- `query`
- `setting`
- `owner_scene`
- `local_memory_counts`
- `global_memory_counts`
- `bundle`
- `gm3_textloss`
- `rendered_prompt`
- `rendered_prompt_sections`

### 8.3 TextGrad 记录

`textgrad_prompt_optimization` 常记录：

- `enabled`
- `engine`
- `max_iters`
- `max_calls_per_episode`
- `pass_threshold`
- `route_key`
- `draft_prompt`
- `context`
- `iterations`
- `final_reason`
- `best_score`
- `final_prompt`

每个 iteration 里会有：

- `prompt`
- `judge_raw`
- `judge`

---

## 9. GM3 的执行入口和参数

### 9.1 入口脚本

GM3 仍然通过：

- [`scripts/medmcqa/eval_collab_domain_adaptation.py`](/bigdata/xenial/nvdamas/scripts/medmcqa/eval_collab_domain_adaptation.py)

来启用。

### 9.2 关键参数

`eval_collab_domain_adaptation.py` 里新增/沿用的相关参数：

- `--mas_memory graph_memory3`
- `--gm2_dynamic_graph`
- `--gm2_retrieval_mode graph_policy`
- `--gm2_settings local_plus_global`
- `--gm2_enable_overlay`
- `--gm3_use_textgrad`
- `--gm3_textgrad_engine`

### 9.3 `tasks/run.py`

`tasks/run.py` 也同步挂了 GM3 参数，方便统一入口运行。

---

## 10. 一句话理解当前 GM3

GM3 当前的主思路可以概括为：

```text
local graph = 当前状态怎么落地
global graph = 抽象 workflow / source-role 怎么迁移
TextGrad = 把这两者组合成更合适的当前 prompt
```

它不是“增加更多 memory”，而是“让现有 memory 更会说话”。

---

## 11. 仍然值得继续看的点

如果后面还要继续调 GM3，最值得盯住的 debug 项是：

- `gm3_textloss.routing_matrix`
- `gm3_textloss.selected`
- `textgrad_prompt_optimization.iterations`
- `textgrad_prompt_optimization.final_prompt`
- `source_priority_queue`
- `rendered_prompt_sections.planner_notes`

这些字段基本能把“为什么 memory 被选中 / 为什么没被选中 / 为什么 prompt 变长 / 为什么 solver 仍然走偏”解释清楚。
