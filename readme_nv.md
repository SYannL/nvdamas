# GraphMemory2 in nvdamas: Episode Graph, Local/Global Memory, Retrieval, Routing, and Action Repair

本文档整理当前 `nvdamas` 中 `graph_memory2` 的真实实现链路。重点不是原版 GraphMemory2 的独立 `run_eval.py` 控制流，而是当前项目里如何在 **保持 nvdamas / AutoGen workflow 不变** 的条件下，把 GraphMemory2 的 graph memory 引入到每个 ALFWorld episode 的决策 prompt 和可选 action repair 中。

主要代码位置：

- `mas/memory/mas_memory/graph_memory2.py`
- `mas/memory/mas_memory/gm2_backend/alfworld_adapter.py`
- `mas/memory/mas_memory/gm2_backend/construction_graph.py`
- `mas/memory/mas_memory/gm2_backend/retrieval_graph.py`
- `mas/memory/mas_memory/gm2_backend/routing_graph.py`
- `tasks/mas_workflow/autogen/autogen.py`
- `scripts/medmcqa/eval_collab_domain_adaptation.py`

## 1. 总体链路

当前完整流程可以概括为：

```text
ALFWorld episode history
  -> ALFWorldAdapter.episode_from_history(...)
  -> EpisodeRecord
  -> EpisodeGraphBuilder.build(...)
  -> EpisodeGraph
  -> LocalGraphMaintainer.update(...)
  -> LocalGraphMemory
  -> GlobalPromoter.promote(...)
  -> GlobalGraphMemory

每个决策 step:
  current env state/history/admissible actions
  -> MemoryQuery
  -> QueryBasedRetriever.retrieve(...)
  -> SupportBundle
  -> routing weights + selected support items
  -> graph_policy / lightweight renderer
  -> planner_notes/action_constraints/repair_hints
  -> nvdamas format_prompt_payload(...)
  -> solver.response(...)
  -> env.process_action(...)
  -> optional graph_memory2.repair_action(...)
  -> env.step(action)
```

也就是说，GraphMemory2 在当前框架中有两个作用：

1. **跨 episode memory 构造与检索**：从已完成任务累计 graph memory，用于后续任务。
2. **step-level prompt / repair 辅助**：每一步根据当前状态重新 query、retrieve、routing，然后注入 planner notes；必要时做非常窄的 deterministic action repair。

nvdamas 的 MAS workflow 没有被 GraphMemory2 接管。solver 仍然是 AutoGen workflow 中的 solver agent，环境交互仍然走 `env.step()`。

## 2. Episode 如何变成 Episode Graph

入口在 `graph_memory2.py` 的 `save_task_context(...)`：

```text
AutoGen task done
  -> meta_memory.save_task_context(...)
  -> export ALFWorld current_history to temporary history json
  -> ALFWorldAdapter.episode_from_history(...)
  -> EpisodeGraphBuilder.build(...)
```

### 2.1 EpisodeRecord

`ALFWorldAdapter.episode_from_history(...)` 会把 ALFWorld 运行历史转换成 `EpisodeRecord`。

一个 `EpisodeRecord` 包含：

- `agent_id`
- `scene_id`
- `task_id`
- `goal`
- `metadata`
- `steps: list[EpisodeStep]`

每个 `EpisodeStep` 包含：

- `state: StateSummary`
- `action: CanonicalAction`
- `next_state: StateSummary`
- `feedback: StepFeedback`
- `subgoal`

其中 `StateSummary` 是对环境状态的结构化表示，例如：

- 当前 location
- visible objects
- held objects
- searched locations
- admissible verbs
- workflow stage
- raw observation

`CanonicalAction` 则会把动作规范化成类似：

```text
go(target=fridge_1)
open(container=cabinet_2)
take(object=apple_1,source=countertop_1)
move(object=apple_1,destination=fridge_1)
```

这一步非常关键：后面的 graph 不直接依赖 LLM 原始自然语言，而是依赖 adapter 解析出的状态、动作、反馈。

### 2.2 EpisodeGraphBuilder

`EpisodeGraphBuilder.build(episode)` 把 `EpisodeRecord` 转成 `EpisodeGraph`。

当前 episode graph 的核心节点类型：

```text
STATE
ACTION
SUBGOAL
FAILURE
```

核心边类型：

```text
TEMPORAL     state -> action -> next_state
CAUSES       action -> delta
ADVANCES_TO  action -> subgoal
FAILS_UNDER  action -> failure
```

每一步会产生：

```text
state:<state.signature>
action:<action.canonical_str>
state:<next_state.signature>
```

并连接：

```text
state -> action
action -> next_state
```

如果这一步改变了状态，还会生成 delta 节点：

```text
action -> delta
```

如果这一步推进了某个 subgoal：

```text
action -> subgoal
```

如果失败：

```text
action -> failure
```

因此 episode graph 不是完整文本轨迹，而是一个结构化的状态-动作-反馈图。

## 3. Episode Graph 如何累计成 Local Memory

入口：

```text
LocalGraphMaintainer.update(local_memory, episode_graph, episode)
```

`LocalGraphMemory` 的结构是：

```text
nodes_by_signature
edges_by_signature
candidates
rules_by_id
artifacts_by_id
episode_ids
```

### 3.1 图结构累计

`LocalGraphMaintainer.update(...)` 会把本 episode 的 nodes/edges 合并进 local memory。

合并依据是 signature：

- 同一个 state signature 会累计统计信息。
- 同一个 action transition 会累计 support/positive/negative/stalled。
- 成功 episode 会给相关 node/edge 增加 positive。
- 失败或 stalled step 会增加 negative/stalled。

这意味着 local memory 是一个跨 episode 的同 domain 图，不是简单保存 N 条完整 episode。

### 3.2 Candidate induction

Local memory 会从 episode 中诱导 `PromotionCandidate`，类型包括：

```text
WORKFLOW
PRECONDITION
FAILURE
REPAIR
```

典型 candidate：

- workflow pattern：多个 subgoal 的顺序。
- precondition：例如在 container 位置时 open(container) 有用。
- holding target before move：拿着目标物后 move 到目标容器有用。
- failure pattern：某 action 在某 failure_label 下失败。
- repair pattern：失败后，下一步成功动作可以作为 repair。
- closure pattern：成功 episode 尾部几步形成的完成模式。
- anti-pattern：重复无进展动作导致 stall。

这些 candidate 先留在 local memory 中，后续可被 local retrieval 使用，也可被 promote 到 global。

### 3.3 Rule induction

Local memory 还会诱导 `MemoryRule`，类型包括：

```text
PRECONDITION
BLOCKED
REPAIR
WORKFLOW
CLOSURE
```

规则会带上：

- `task_family`
- `goal_arity`
- `progress_state`
- `goal_roles`
- `condition`
- `effect`
- `stats`

例如：

```text
Precondition: when carry_target, prefer move(target_object, goal_destination).
Blocked: during search, avoid take(non_target_object).
Repair: after no_effect, try open(container).
Closure: when carrying the target, prefer move(target_object, goal_destination).
```

这些规则比原始 episode 更抽象，也更适合检索。

### 3.4 Artifact induction

`MemoryArtifact` 是更可读、可检索的记忆单元。当前主要有：

```text
PROTOTYPE
RULE
REFLECTION
```

重要 artifact 类型：

1. **Plan template**

成功 episode 会尝试抽象出流程骨架：

```text
Plan: go(target=support_surface)
  -> take(object=target_object,source=support_surface)
  -> go(target=goal_destination)
  -> move(object=target_object,destination=goal_destination)
```

如果是 heat/cool/clean，会插入 process tool：

```text
take target
-> go tool
-> heat/cool/clean target
-> go destination
-> move target
```

two-object 任务会附加：

```text
repeat acquire and deliver for remaining target_object
```

2. **Scene relation**

成功拿到目标物时，会记录：

```text
Scene relation: in this layout, target_object was found at source_instance (source_role).
```

搜索过但没找到时，会记录：

```text
Scene relation: in this layout, target_object was not found after searching source_instance.
```

这个 artifact 对 **local same-scene / same-domain** 很有用，但对 **global/unseen** 容易成为噪声。所以当前 routing/render 中把它当成 local grounding，而不是 global transferable plan。

3. **Reflection**

失败或 stall 会形成反思：

```text
Reflection: during search, wrong target object tends to derail progress.
```

并保留：

- trigger action
- avoid patterns
- repair patterns

这些会进入 local cautions / repair hints。

## 4. Local 如何 Promote 成 Global Memory

入口：

```text
GlobalPromoter.promote(global_memory, local_memories)
```

当前 global memory 结构：

```text
candidates
rules_by_id
artifacts_by_id
promoted_batches
```

Global promotion 的目标是把 local 中更可迁移的结构提升到 global：

- 高置信 workflow
- 高置信 precondition
- closure / plan
- 抽象 rule
- 部分 artifact

### 4.1 Global promotion 的抽象化

promotion 时会做 abstraction，去掉过强场景绑定字段，例如：

```text
goal_signature
layout_id
source_instance
source_base
graph_refs
state_ref
next_state_ref
target_object
```

这样 global memory 理论上应该保留：

```text
task family / progress state / abstract role / action pattern
```

而不是：

```text
apple_1 was found at countertop_2
```

### 4.2 当前需要特别注意的噪声

即使 promotion 会抽象字段，artifact 的 summary 仍可能携带文本噪声，例如：

```text
Scene relation: in this layout, target_object was found at shelf_1.
```

这类信息在 local 中有帮助，但 global/unseen 中容易误导。因此当前 `graph_policy` renderer 增加了 `_is_concrete_scene_hint(...)`，在 prompt 组装时压掉 global concrete location hint。

## 5. Train/Eval 中 Memory 如何变化

### 5.1 Train 阶段

每个 train episode 结束后：

```text
save_task_context(...)
  -> build episode graph
  -> update local memory
  -> promote current local memory into shared global memory
  -> persist local_*.json / global_memory.json
```

在 collaborative domain adaptation 脚本中，A/B domain 顺序训练。当前 `graph_memory2` 配置带有 `gm2_shared_global_dir`，所以 local A 和 local B 可以共同维护同一个 global memory。

这对应用户之前说的 strict online 思路：

```text
A train episode -> update A local -> update shared global
B train episode -> update B local -> update shared global
```

也就是说，global 不只是训练结束后才有；在动态模式下，每个 episode 后也会尝试更新 shared global。

脚本末尾仍有一次 final global rebuild：

```text
rebuild_graph_memory2_global_from_locals(...)
```

它的作用是把 local A/B 重新汇总成最终 global artifact，主要是收尾一致性检查和落盘。它不是 eval 时在线学习。

### 5.2 Eval 阶段

eval 使用：

```text
gm2_freeze_memory=True
```

因此 eval episode 不会把结果写回 train memory。

每个 eval episode 内仍会有 overlay / current history，用来构造当前 step 的 query 和状态判断，但不会跨 eval episode 累积成新的 local/global memory。

换句话说：

```text
eval task 2 不应该使用 eval task 1 新产生的跨 episode memory。
```

它使用的是 train 后已有的 local/global memory，加上当前 episode 内部状态。

## 6. Workflow 中如何 Retrieve

retrieve 发生在 `tasks/mas_workflow/autogen/autogen.py`。

### 6.1 episode 开始前 retrieve

任务开始时：

```python
payload = self.meta_memory.retrieve_prompt_payload(
    query_task=task_main,
    env_ref=env,
    task_config=task_config,
    step_index=0,
)
```

返回内容包括：

```text
execution_patterns
insights
planner_notes
action_constraints
repair_hints
```

这些被 `format_prompt_payload(...)` 拼进 solver prompt。

### 6.2 每一步重新 retrieve

因为 `graph_memory2` 初始化时设置：

```python
self.refresh_each_step = True
```

所以每个 step 都会重新调用：

```python
retrieve_prompt_payload(..., step_index=i+1)
```

这样 GM2 能根据当前：

- observation
- current history
- held objects
- visible objects
- searched/exhausted locations
- admissible actions
- progress_state

动态生成新的 `MemoryQuery`，而不是只在 episode 开头查一次。

## 7. MemoryQuery 是如何构造的

入口：

```text
graph_memory2._build_external_query(...)
```

它从当前 env 中读取：

- `goal_instruction`
- `resolved_gamefile`
- `current_history`
- `initial_observation`
- `last_admissible_commands`

然后通过 `ALFWorldAdapter` 得到：

- `scene_id`
- 当前 state
- 当前 location
- canonical admissible actions
- goal roles
- task family
- progress state
- held relevant count
- placed relevant count
- remaining relevant count
- destination reached
- visible target match

`MemoryQuery` 的关键字段：

```text
goal
scene_id
current_stage
location
progress_state
task_family
goal_roles
required_count
held_relevant_count
placed_relevant_count
remaining_relevant_count
destination_reached
goal_object_matches_visible
admissible_actions
desired_types
failure_label
belief
dynamic_context
```

其中 `dynamic_context` 记录：

```text
visible_objects
held_objects
searched_locations
visited_locations
inspected_locations
exhausted_locations
search_attempt_counts
delivered_instances
last_failure
layout_id
```

这就是 retrieve 的 query，不是只用 task string，也不是直接拿完整 episode 做相似度。

## 8. Retrieve 返回的是什么

当前 `--
 graph_policy` 走：

```text
QueryBasedRetriever(top_k=5)
```

实际 retrieve 入口在：

```text
retrieval_graph.py
```

它会对 local/global memory 中不同结构分别打分：

1. **local graph facts**

来自 local graph 的 state node / current anchor，例如：

```text
Local state fact: current location / visible target / held object / searched location
```

2. **anchor subgraph context**

从当前 query 找 local graph 中相关 state/action transition：

```text
graph_anchor
graph_transition
```

3. **local artifacts**

local 的 prototype/rule/reflection/scene_relation。

4. **local rules**

local rules_by_id 中的 precondition/workflow/closure/blocked/repair。

5. **global artifacts**

global promoted artifacts。

6. **global rules**

global promoted rules。

最终返回 `SupportBundle`，里面不是一个单一 list，而是分槽位：

```text
fact_items
relation_items
plan_items
global_task_plan_items
workflow_items
precondition_items
repair_items
closure_items
reflection_items
local_graph_items
local_promoted_items
global_promoted_items
local_items
global_items
blocked_actions
suggested_actions
workflow_hints
warnings
routing_weights
routing_decisions
local_graph_contribution
local_promoted_contribution
global_promoted_contribution
fused_support_items
```

所以 retrieve 返回的不是“完整 episode”，而是从 graph/rule/artifact 中根据当前 query 选出的若干结构化 support item。

## 9. Routing 机制

当前 routing 有两层。

### 9.1 retrieval_graph 内部的 phase-aware routing

`retrieval_graph.py` 会根据当前阶段判断：

```text
search_like
process_like
deliver_like
```

然后给不同来源不同 prior：

```text
local_graph
local_promoted
global_promoted
```

例如：

- search 阶段更偏 local graph grounding。
- process 阶段更重 process/precondition/tool guidance。
- deliver 阶段更重 closure / goal destination。
- task_plan 更允许 global。
- fact/relation/repair/blocked 更偏 local。

之后生成：

```text
routing_weights[slot] = {local, global, none}
routing_decisions[slot] = local/global/mixed/none
```

slot 包括：

```text
fact
relation
task_plan
plan
workflow
precondition
repair
closure
blocked
```

### 9.2 routing_graph.route_bundle

`routing_graph.py` 中的 `route_bundle(...)` 也支持按 slot 选择 local/global/mixed。

核心思想是：

```text
对每个 slot 计算 local_items 和 global_items 的 utility/risk/repair value
再 softmax 成 local/global/none 权重
再用 _select_mixed_items 选出最终 SupportItem
```

它会输出：

```text
bundle.routing_weights
bundle.routing_decisions
bundle.blocked_actions
bundle.suggested_actions
bundle.workflow_hints
bundle.local_graph_contribution
bundle.local_promoted_contribution
bundle.global_promoted_contribution
bundle.fused_support_items
```

当前 `graph_policy` 最后不是简单把所有 support item 拼起来，而是在 render 层进一步做 role-aware filtering。

## 10. graph_policy 如何组装给决策模型

入口：

```text
graph_memory2._render_graph_policy_evidence(...)
```

它把 `SupportBundle` 变成 `planner_notes` 中的：

```text
### GM2 MEMORY EVIDENCE
...
```

当前 graph_policy 的角色划分：

### 10.1 Global task skeleton

global 只允许承担抽象迁移流程：

```text
Global task skeleton (abstract transfer only; no scene-specific locations):
- Plan: go(target=support_surface) -> take(...) -> go(target=goal_destination) -> move(...)
```

它会过滤掉具体场景提示：

```text
found at shelf_1
was found at drawer_2
located at ...
in this layout ...
Scene relation ...
```

原因：这些信息对 local 有价值，但对 global/unseen 是高风险噪声。

### 10.2 Persistent global task plan

对于 two-object / search_second 这类任务，global skeleton 会在 episode 内缓存：

```text
Persistent global task plan
```

目的：避免完成第一个 object 后，step-level retrieve 找不到 global evidence，就把高层流程完全丢掉。

### 10.3 Local current-state grounding

local graph 负责当前状态证据：

```text
Local current-state grounding:
- current visible object
- held object
- searched/exhausted location
- local scene relation
- current graph transition
```

这些信息必须和当前 task/object/destination/progress/admissible actions 有一定相关性。

### 10.4 Local domain guidance

local promoted rules/artifacts 提供 domain 内经验：

```text
Local domain guidance:
- precondition
- workflow
- closure
```

它比 global 更可以携带 domain-specific 经验，但仍会过滤掉过强的具体场景位置。

### 10.5 Local cautions

failure/reflection/repair 进入：

```text
Local cautions:
- wrong target object tends to derail progress
- avoid premature destination focus
```

这类信息主要用于提醒，不直接替换 solver action。

### 10.6 Memory arbitration

如果 local/global 冲突，当前 prompt 会明确加入：

```text
Memory arbitration:
- Suppressed N global concrete/location-specific hint(s); global memory is used as task skeleton only.
- Suppressed local evidence with weak task/current-state relevance.
- If memory sources disagree, prefer evidence matching the target object, goal destination, current progress, and admissible actions.
```

这就是当前 conflict handling 的核心：

```text
global = abstract transferable skeleton
local = current-state grounding
task/admissible alignment = final priority
```

## 11. Prompt 是如何进入 AutoGen solver 的

`retrieve_prompt_payload(...)` 返回：

```text
planner_notes
action_constraints
repair_hints
execution_patterns
insights
```

AutoGen workflow 里会调用：

```python
format_prompt_payload(
    {
        "reference_cases": few_shots,
        "execution_patterns": successful_shots,
        "insights": roles_rules.get(solver.profile, raw_rules),
        "planner_notes": planner_notes,
        "action_constraints": action_constraints,
        "repair_hints": repair_hints,
    },
    self.meta_memory.summarize(),
)
```

因此 GM2 的 graph evidence 最终是作为 prompt 的一部分给 solver，而不是直接替换 solver。

日志中可以看到：

```text
=== Prompt With Memory/Insights ===
[detail][step N] solver.input:
...
### GM2 MEMORY EVIDENCE
...
```

## 12. Action Hook 和 Repair

workflow 中，solver 输出后流程是：

```text
solver.response(...)
-> env.process_action(...)
-> memory.repair_action(...)
-> env.step(action)
```

hook 位置在：

```text
tasks/mas_workflow/autogen/autogen.py
```

调用：

```python
repair_fn = getattr(self.meta_memory, "repair_action", None)
if callable(repair_fn):
    action = repair_fn(
        raw_response=action_raw,
        processed_action=action,
        env_ref=env,
        task_config=task_config,
        step_index=i + 1,
    )
```

这个 hook 对其他 memory 不起作用，只有 memory 实现了 `repair_action(...)` 才会触发。

### 12.1 graph_policy 下当前允许的 repair

当前 `graph_policy` 下 repair 是保守的，不做完整 action rerank。

允许：

1. **official042 delivery syntax repair**

当 official042 环境需要：

```text
move X to Y
```

而 solver 还在使用旧式：

```text
put X in/on Y
```

或者已经拿着目标物站在目标位置但还在做无效动作时，会选择当前 admissible 的 delivery action。

2. **wrong-object guard**

如果任务目标是 `cup`，solver 却要：

```text
take mug 1 from shelf 1
```

且当前 admissible 中存在正确目标对象动作，会替换为正确对象动作；否则转成 harmless thought，避免明显错拿。

3. **已是具体可执行动作则保留**

如果 solver 已经输出 concrete ALFWorld action，graph_policy 不会用 memory 另选动作。

### 12.2 当前明确禁止的 repair

当前禁止把自由文本 `think:` 强行投影成动作：

```text
think: I should look for the apple...
```

不会被自动改成：

```text
take creditcard 1 from shelf 1
```

日志标记：

```text
no_freeform_action_projection
```

这是为了避免 memory/action hook 越权，把 solver 的思考误解析成错误动作。

## 13. lightweight、graph_policy、phasee_action 的区别

### lightweight

特点：

```text
GM2 retrieve
-> 较保守的 local/global evidence
-> 简短 planner_notes
-> 不做强 action policy
```

优点是干扰少；缺点是 graph/local/global 的角色区分不强。

### graph_policy

特点：

```text
GM2 retrieve
-> graph/rule/artifact 分槽位
-> routing weights
-> global skeleton / local grounding / local domain / cautions
-> conflict arbitration
-> conservative deterministic repair
```

这是当前重点优化的模式。

### phasee_action / hybrid_repair

会引入更多 PhaseE-style state policy / safe extraction 逻辑。

风险是它更接近 action-level controller，可能影响 nvdamas workflow 的公平性。因此当前 graph_policy 没有把完整 PhaseE prompt/action loop 接管进来。

## 14. 当前 conflict 处理原则

当前最重要的原则：

```text
global memory 不应该携带具体位置决策。
local memory 可以携带当前场景位置，但必须和当前任务/状态/动作相关。
当 local/global 冲突时，优先相信当前 observation、admissible actions、目标物、目标容器、progress_state。
```

具体例子：

### 14.1 global 里的 scene relation

如果 global 出现：

```text
target_object was found at shelf_1
```

当前 graph_policy 会压掉它，因为这是 layout-specific，不适合 unseen。

### 14.2 local 里的 scene relation

如果 local 出现：

```text
target_object was found at shelf_1
```

并且当前 query 中目标物、当前场景、可执行动作都相关，则可以作为 local grounding。

### 14.3 abstract global plan

如果 global 出现：

```text
find target -> take target -> process target -> go destination -> move target
```

它会作为 skeleton 进入 prompt，尤其对 cross-domain / unseen transfer 有帮助。

## 15. 当前还有哪些问题

当前 GM2 已经实现：

- dynamic episode graph build
- local graph memory accumulate
- strict online shared global promotion
- local/global retrieve
- routing weights
- graph_policy prompt role separation
- conflict arbitration
- conservative action hook / repair

但还需要继续优化：

1. **routing weights 还不是完整的 learned controller**

目前是 heuristic routing，不是训练出的 policy。

2. **global/local 融合仍主要体现在 prompt 结构**

虽然已经有 slot-level routing 和 render-level arbitration，但最终还是通过 prompt 影响 solver，不是强制 action rerank。

3. **scene relation 的 local/global 边界仍需持续检查**

local scene relation 对 seen/local 有帮助，但如果误进入 global prompt，会伤害 unseen。

4. **global skeleton 的粒度需要控制**

太抽象则没有帮助，太具体则变噪声。

5. **action hook 必须保持窄**

一旦把 `think:` 或弱自然语言意图强行投影成动作，就可能出现错误动作覆盖 solver 的问题。

## 16. 调试建议

看一个 run 是否真的用了 GM2 graph_policy，可以 grep：

```bash
grep -n \
  "GM2 MEMORY EVIDENCE\|Graph policy memory routing\|Routing weights\|Global task skeleton\|Local current-state grounding\|Memory arbitration\|memory.repair_action\|no_freeform_action_projection" \
  logs/.../total_task.log | head -200
```

重点看：

```text
Routing weights
Global task skeleton
Local current-state grounding
Memory arbitration
solver.output.processed
memory.repair_action
env.done / env.won
```

如果看到 global 中大量具体位置：

```text
found at shelf 1
located at drawer 2
in this layout
```

说明 global/local 信息隔离还有问题。

如果看到：

```text
think: ... -> take wrong_object ...
```

说明 action hook 又越权了，需要回到 `no_freeform_action_projection` 的保守策略。

## 17. 一句话总结

当前 nvdamas 中的 GraphMemory2 不是替换 solver 的独立 agent loop，而是：

```text
用 episode graph 构造 local/global graph memory；
每一步根据当前状态 query graph memory；
用 routing 把 local graph、local domain、global skeleton 分开；
把筛选后的 evidence 注入 prompt；
只在极窄场景做 deterministic action repair。
```

这是为了在不破坏 nvdamas workflow 公平性的前提下，让 GM2 的 graph-based local/global memory 真正参与决策。

## 18. 当前保存版本：official042 / v3_s / graph_policy

本节记录当前要提交的版本状态，方便后续从 git commit 回到同一套实验配置。

### 18.1 ALFWorld 数据与 subset

当前推荐的 ALFWorld game root 是 official 0.4.2 下载得到的完整目录：

```text
/workspace/run_alf/ALFWORLD_DATA/alfworld_official_042/json_2.1.1
```

`data/alfworld/collab_subsets/v3_s` 由：

```bash
python scripts/alfworld/materialize_collab_subsets_v3_s.py --clean
```

从 `data/alfworld/collab_subsets/v3` 截断生成。该脚本只读取 `v3`，只写入 `v3_s`，不会修改 `v3`。

当前 `v3_s` 计数为：

```text
bathroom__train.json          100
bathroom__valid_seen.json      20
bathroom__valid_unseen.json    19
bedroom__train.json           100
bedroom__valid_seen.json       20
bedroom__valid_unseen.json     20
kitchen__train.json           100
kitchen__valid_seen.json       20
kitchen__valid_unseen.json     20
living__train.json            100
living__valid_seen.json        17
living__valid_unseen.json      11
```

这些数量反映 official042 数据中当前可 materialize 的任务数量；不是每个 domain 的 valid split 都一定能达到 20。

### 18.2 当前 graph_policy 的重点实现

当前 `--gm2_retrieval_mode graph_policy` 的目标是让 GraphMemory2 的 local/global graph 以不同角色参与决策：

```text
global graph memory:
  只作为 abstract task skeleton / transferable workflow 使用。
  不把 found-at / layout-specific source location 当作 unseen 决策依据。

local graph memory:
  作为 current-state grounding 和 source-priority evidence 使用。
  只在目标物、任务类型、当前可执行动作能够对齐时，给出 source/action priority。

action hook:
  只保留保守 deterministic repair。
  包括 official042 put->move delivery 修复、wrong-object guard、以及 graph_policy 下的 source-priority search assist。
  不把自由文本 think 强行投影成动作。
```

和前面版本相比，当前版本补了几个关键点：

1. **nested memory loading**

   eval-only 或复制历史 run memory 时，会正确读取类似：

   ```text
   .../memory/graph_memory2/local/kitchen/graph_memory2/local_kitchen.json
   .../memory/graph_memory2/global/graph_memory2/global_memory.json
   ```

   这避免了“复制了 memory 但实际没有加载”的问题。

2. **target-object source filtering**

   local source hint 只允许来自真正和目标物相关的 `take/found/scene_relation` evidence，避免把同场景共现对象误当成目标物位置。

3. **direct local scene relation assist**

   对 graph_policy，local graph 中的 `object_location_prior` 可以在当前 admissible actions 中映射成候选 source action，例如：

   ```text
   go to countertop 3
   open cabinet 5
   examine shelf 1
   ```

   这个逻辑只在 `graph_policy` 下启用，不影响 `lightweight`、`phasee_policy`、`graph_policy_quality` 等其他 mode。

4. **保守 action hook**

   如果 solver 已经输出具体可执行动作，graph_policy 通常不覆盖。
   如果 solver 输出 `think:`，不会做自由文本动作投影；最多在明确 search intent 且 local graph 有高置信 source candidate 时做非常窄的 search assist。

### 18.3 当前推荐 full run

```bash
cd /workspace/nvdamas
export OPENAI_API_BASE=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=EMPTY

nohup python scripts/medmcqa/eval_collab_domain_adaptation.py \
  --dataset_family alfworld \
  --alfworld_group_a kitchen \
  --alfworld_group_b living \
  --alfworld_subset_dir data/alfworld/collab_subsets/v3_s \
  --alfworld_eval_split valid_seen,valid_unseen \
  --alfworld_game_root /workspace/run_alf/ALFWORLD_DATA/alfworld_official_042/json_2.1.1 \
  --mas_type autogen \
  --mas_memory graph_memory2 \
  --reasoning io \
  --model qwen32b-api \
  --max_trials 30 \
  --batch_size 1 \
  --run_id gm2_graph_policy_nestedload_directsource_full \
  --reset_memory \
  --gm2_dynamic_graph \
  --gm2_retrieval_mode graph_policy \
  --gm2_settings local_plus_global \
  --gm2_enable_overlay \
  --scenarios inner_ours_A_on_A,inner_ours_B_on_B,cross_baseline_A_on_B,cross_baseline_B_on_A,cross_ours_A_on_B,cross_ours_B_on_A \
  > /workspace/nvdamas/L_gm2_graph_policy_nestedload_directsource_full.log 2>&1 &
```

### 18.4 宿主和 container 同步

代码修改通常先在宿主：

```text
/bigdata/xenial/nvdamas
```

再同步到 container：

```bash
docker cp /bigdata/xenial/nvdamas/mas/memory/mas_memory/graph_memory2.py \
  mcma-recover:/workspace/nvdamas/mas/memory/mas_memory/graph_memory2.py

docker cp /bigdata/xenial/nvdamas/scripts/medmcqa/eval_collab_domain_adaptation.py \
  mcma-recover:/workspace/nvdamas/scripts/medmcqa/eval_collab_domain_adaptation.py

docker cp /bigdata/xenial/nvdamas/tasks/run.py \
  mcma-recover:/workspace/nvdamas/tasks/run.py
```

如果需要同步宿主的 subset 到 container：

```bash
docker cp /bigdata/xenial/nvdamas/data/alfworld/collab_subsets \
  mcma-recover:/workspace/nvdamas/data/alfworld/
```

如果需要把 container 的 subset 拷回宿主，方向相反：

```bash
docker cp mcma-recover:/workspace/nvdamas/data/alfworld/collab_subsets \
  /bigdata/xenial/nvdamas/data/alfworld/
```
