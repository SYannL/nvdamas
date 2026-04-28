# ALFWorld 运行流程说明

这份 README 只聚焦本项目里 `alfworld` 相关的运行链路，目标是回答三个问题：

1. `alfworld` 从哪里进来。
2. 一条任务是怎么被加载、提示、执行、记忆和记录的。
3. 协作式 domain adaptation 实验是怎么组织起来的。

## 1. 先看结论

本项目里 `alfworld` 有两条主线：

- 单任务/常规评测主线：`tasks/run.py`
- 协作式记忆评测主线：`scripts/medmcqa/eval_collab_domain_adaptation.py`

如果只想理解“模型如何在 ALFWorld 里执行一步一步动作”，看第一条就够了；如果还想理解 “local memory / global memory / A-B domain 协作训练与评测”，再看第二条。

## 2. 目录与角色

### 核心代码

- `tasks/run.py`
  - 通用任务入口，负责装配 environment、MAS、memory、LLM、日志和执行循环。
- `tasks/envs/alfworld_env.py`
  - 对 `alfworld` 官方 `AlfredTWEnv` 的项目内封装。
- `tasks/envs/__init__.py`
  - 注册 `alfworld` 的任务列表、环境类和 recorder。
- `tasks/prompts/alfworld_prompt.py`
  - ALFWorld 系统提示词与 few-shot。
- `tasks/mas_workflow/autogen/autogen.py`
  - 默认 `autogen` MAS 的调度逻辑，真正执行 “生成动作 -> 环境反馈 -> 写入记忆 -> 继续下一步”。
- `scripts/medmcqa/eval_collab_domain_adaptation.py`
  - 复用上面的基础能力，做双域协作训练、全局记忆构建和多场景评测。

### 数据与配置

- `tasks/env_configs/alfworld_config.yaml`
  - ALFWorld 环境配置，包含 split、logic 路径、env 类型、默认 action space 等。
- `data/alfworld/alfworld_tasks_suffix.json`
  - 默认单任务运行使用的任务列表。
- `data/alfworld/collab_subsets/v*/`
  - 协作实验使用的固定子集。
- `data/alfworld/download_text_alfworld.py`
  - 下载 text-only ALFWorld 数据。
- `scripts/alfworld/materialize_collab_subsets.py`
  - 生成固定协作子集。

## 3. 运行前需要满足什么

### 3.1 Python 与包

项目代码里大量脚本默认写的是 `python`，但我当前检查到这个环境里只有 `python3`。如果你的机器也一样，运行命令时请把 `python` 换成 `python3`，或者自行配置 shell alias。

此外，当前工作区里 `alfworld` 包还没有安装；如果不安装，`tasks/envs/__init__.py` 会在加载 `AlfworldEnv` 时失败。

最少需要：

- `python3`
- `alfworld`
- `openai`
- `python-dotenv`
- 以及 `alfworld` 依赖的 TextWorld / fast-downward 相关环境

### 3.2 API 环境变量

`mas/__init__.py` 会自动读取根目录 `.env`，`mas/llm.py` 直接从环境变量中取：

- `OPENAI_API_BASE`
- `OPENAI_API_KEY`

建议自己新建 `.env`，不要把真实 key 写进版本控制：

```bash
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_API_KEY=your_key_here
```

### 3.3 ALFWorld 数据

当前仓库下已经有：

- `data/alfworld/logic/`
- `data/alfworld/alfworld_tasks_suffix.json`
- `data/alfworld/collab_subsets/`

但 `data/alfworld/json_2.1.1/` 这类运行时需要的文本数据目录未必齐全。缺数据时可执行：

```bash
python3 data/alfworld/download_text_alfworld.py
```

它会补齐：

- `json_2.1.1/train`
- `json_2.1.1/valid_seen`
- `json_2.1.1/valid_unseen`
- `logic/alfred.pddl`
- `logic/alfred.twl2`

## 4. 单任务运行主线

## 4.1 总流程图

```text
tasks/run.py
  -> 读取 tasks/configs.yaml
  -> build_task()
     -> 读取 tasks/env_configs/alfworld_config.yaml
     -> get_env('alfworld')
     -> get_recorder('alfworld')
     -> get_task('alfworld') 或载入 --alfworld_tasks_json
     -> get_mas(...)
  -> build_mas()
     -> 构造 GPTChat / Reasoning / Memory
     -> MAS.build_system(...)
  -> run_task()
     -> 对每个 task:
        -> env.set_env(task_config)
        -> 取 few-shot + system prompt
        -> mas.schedule(task_config)
           -> LLM 生成动作
           -> env.step(action)
           -> memory.move_memory_state(...)
           -> done 后 env.feedback()
           -> memory.save_task_context()/backward()
     -> recorder 写日志
```

## 4.2 入口：`tasks/run.py`

这是统一入口。对 `alfworld` 来说，它主要做 5 件事：

1. 解析参数，例如 `--task alfworld --mas_type autogen --mas_memory selectivemem --model ...`
2. 创建工作目录
   - memory: `./.db/...`
   - log: `./logs/...`
3. 调 `build_task()` 构造 `TaskManager`
4. 调 `build_mas()` 装配 LLM / reasoning / memory / MAS
5. 调 `run_task()` 逐题执行

其中 ALFWorld 额外支持两个很关键的参数：

- `--alfworld_tasks_json`
  - 用一个 JSON 数组替换默认任务列表
- `--alfworld_max_tasks`
  - 对外部任务列表截断，方便 smoke test

### 4.3 任务列表从哪来

默认情况下，`tasks/envs/__init__.py` 会读取：

- `data/alfworld/alfworld_tasks_suffix.json`

并把每一条样本转成项目内部统一格式：

- `task`
- `env_kwargs.gamefile`
- `task_type`
- `env_name`

这里的 `task_type` 不是原始 ALFWorld 类型名，而是 few-shot 用的短标签：

- `put`
- `clean`
- `heat`
- `cool`
- `examine`
- `puttwo`

如果你传了 `--alfworld_tasks_json`，则 `tasks/run.py` 会绕过默认列表，直接读取你给的 JSON 文件。这个 JSON 的字段要求与协作子集格式保持一致，至少包含：

- `env_kwargs.gamefile`
- `env_name`
- `task_type`
- `task` 或 `goal_instruction`

## 4.4 环境如何被封装

`tasks/envs/alfworld_env.py` 做了几层关键处理：

### 1. 固定使用 text-only 环境

只支持：

- `env.type: AlfredTWEnv`

也就是本项目当前只走文字版 ALFWorld，不走 THOR 视觉环境。

### 2. 设置 ALFWORLD_DATA

环境变量 `ALFWORLD_DATA` 会默认指向：

- `data/alfworld/_runtime_cache`

这样运行时缓存落在仓库内，而不是依赖用户家目录缓存。

### 3. `set_env(task_config)`

每换一道题，都会：

- 从 `task_config["env_kwargs"]["gamefile"]` 取出具体 `game.tw-pddl`
- 把 `main_env.game_files` 设成这个单文件列表
- reset 环境
- 生成当前题的 `task_main` 和 `task_description`

如果任务条目里没有完整 `task`，但有 `goal_instruction`，它会自动把初始 observation 和 goal instruction 拼起来。

### 4. `step(action)`

这里不是直接把 LLM 输出原样丢给环境，而是先做一次规整：

- 去掉编号前缀，比如 `1. take ...`
- 去掉列表符号，比如 `- think: ...`
- 去掉多余的 `OK.`

然后再执行 `env.step([action])`。

项目对奖励的处理也做了二次封装：

- `think:` 或 `Nothing happens.` 会给一个负反馈
- 真正成功与否主要看 `info['won']`
- `done=True` 不等于成功，可能只是步数耗尽

最终成功判断在 `feedback()` 里统一收口：

- `won=True` 才算成功

## 4.5 Prompt 怎么进来

Prompt 分两层：

### 系统提示词

来自：

- `tasks/prompts/alfworld_prompt.py`

它约束了 ALFWorld 可输出的动作格式，例如：

- `take a from b`
- `go to a`
- `open a`
- `put a in/on b`
- `clean/heat/cool/use`
- `think: ...`

还额外强调了几条运行策略，例如：

- 放入 receptacle 之前要先确认它是否 open
- 如果收到 `Nothing happens.`，不要重复同一动作

### Few-shot

同样来自 `tasks/prompts/alfworld_prompt.py`，由 `get_task_few_shots()` 按 `task_type` 选择。

对 ALFWorld，当前逻辑会优先取：

- `react_<task_type>_2`
- `react_<task_type>_0`

再按照 `tasks/configs.yaml` 里的 `few_shots_num` 截断。当前 `alfworld` 默认是 `1` 个 few-shot。

## 4.6 MAS 怎么执行一题

以默认 `autogen` 为例，`tasks/mas_workflow/autogen/autogen.py` 的 `schedule()` 是核心。

每题开始时它会：

1. `env.reset()`
2. `meta_memory.init_task_context(task_main, task_description)`
3. 从 memory 里检索：
   - 历史成功轨迹
   - insights
4. 用 `format_task_prompt_with_insights()` 把以下内容拼成 user prompt：
   - few-shot
   - memory few-shot
   - insights
   - 当前 task description

之后进入 step loop：

1. solver agent 基于 prompt 生成动作
2. 如果检测到动作连续重复，触发 `ground_truth` agent 兜底
3. 调 `env.step(action)`
4. 把 `action + observation + reward` 写入 memory
5. 如果 `done`，跳出循环

循环结束后：

1. 调 `env.feedback()` 取最终成败
2. `meta_memory.save_task_context(...)`
3. `meta_memory.backward(final_done)`

因此，真正的一题执行闭环是：

```text
检索记忆 -> 组 prompt -> LLM 产生命令 -> 环境返回 observation -> 记忆更新 -> 最终总结入库
```

## 4.7 日志与输出落哪

### 单任务运行

- memory
  - `./.db/<model>/<task>/<mas_type>/memory/<mas_memory>/...`
- log
  - `./logs/<task>/<mas_type>/memory/<mas_memory>/<model>/...`

如果传了 `--run_id`，会继续往下分子目录。

### 常见额外产物

- `selectivemem`
  - 可能会写本地/全局图 JSON
- `g-memory`
  - 运行结束后可能会导出 insight-query provenance

## 5. 协作式 ALFWorld 评测主线

这条线在：

- `scripts/medmcqa/eval_collab_domain_adaptation.py`

尽管脚本名里有 `medmcqa`，它已经把 `alfworld` 当成 `dataset_family` 的一个正式分支来处理。

## 5.1 这个脚本做什么

它想回答的问题是：

- 在 A 域学到的局部记忆，对 A 域自己有多大帮助？
- A/B 两个域的记忆抽象成 global memory 后，能不能提高 A/B 各自的表现？
- 跨域直接迁移 local memory 是否有效？

对 ALFWorld，域的定义不是学科，而是任务簇：

- `kitchen_statechange`
  - 厨房中的 heat / cool / clean
- `home_search`
  - living / bedroom / bathroom 中的 put / puttwo / examine

## 5.2 任务子集怎么来

脚本默认从固定子集目录读取：

- `data/alfworld/collab_subsets/v1`

但仓库里也已经有 `v2` / `v3` / `v3_ss` 等版本。已有 smoke 脚本使用的是 `v2`：

- `scripts/medmcqa/run_alfworld_smoke.sh`

固定子集的读取入口是：

- `load_alfworld_collab_tasks()`

它会分别读：

- `<group>__train.json`
- `<group>__valid_unseen.json`

如果你想重新生成固定子集，可以看：

- `scripts/alfworld/materialize_collab_subsets.py`

它会：

1. 遍历 `data/alfworld/json_2.1.1/<split>`
2. 按 scene domain 和 task family 过滤
3. 检查 `game.tw-pddl` 是否可加载
4. 生成稳定、可复现的 JSON 子集

## 5.3 协作训练怎么跑

对 `alfworld + selectivemem`，训练阶段是一个两遍流程：

### Pass 1: Local-only

分别在 A 域、B 域上跑任务，构建各自的 local graph / local memory。

### Merge: 构建 Global Graph

每个 batch 结束后，把 A/B 的本地图持久化，再合并成一个 global graph。

### Pass 2: Local + Global

在同一批任务上再次执行，但这次在 local retrieval 基础上，再挂载 global insights。

也就是说，对 `selectivemem` 来说，训练不是简单的一次扫数据，而是：

```text
batch 内 A_local / B_local
  -> 持久化 local graph
  -> merge 成 global graph
  -> A_local+global / B_local+global 再跑一遍
```

另外，ALFWorld 在这条路径里还做了两个特殊处理：

- 只允许 `--tool_mode search`
- 每题可放进 subprocess 跑，避免 TextWorld/PDDL 偶发崩溃把主进程带死

这个 subprocess worker 就是：

- `_run_one_alfworld_task_worker()`

## 5.4 评测怎么组织

训练完后，脚本会冻结 memory，按 scenario 输出结果。

对 ALFWorld，默认 6 个 scenario：

- `inner_baseline_A_on_A`
- `inner_baseline_B_on_B`
- `inner_ours_A_on_A`
- `inner_ours_B_on_B`
- `cross_baseline_A_on_B`
- `cross_baseline_B_on_A`

含义可以粗略理解为：

- `inner_baseline`
  - 只用本域 local memory
- `inner_ours`
  - local memory + global insights
- `cross_baseline`
  - 直接拿另一个域的 local memory 来测跨域迁移

结果会写到：

- `./reports/collab/<timestamp>/<dataset_family>_collab_domain_eval.json`
- `./reports/collab/<timestamp>/<dataset_family>_collab_domain_eval.md`
- `./reports/collab/<timestamp>/<dataset_family>_collab_train_metrics.json`
- `./reports/collab/<timestamp>/<dataset_family>_collab_train_metrics.md`
- `./reports/collab/<timestamp>/<dataset_family>_collab_memory_nl.md`

## 6. 常用命令

## 6.1 跑单任务 ALFWorld

使用默认任务列表：

```bash
python3 tasks/run.py \
  --task alfworld \
  --mas_type autogen \
  --mas_memory selectivemem \
  --reasoning io \
  --model gpt-4o-mini \
  --max_trials 30 \
  --verbose
```

只跑自定义任务子集：

```bash
python3 tasks/run.py \
  --task alfworld \
  --mas_type autogen \
  --mas_memory selectivemem \
  --reasoning io \
  --model gpt-4o-mini \
  --alfworld_tasks_json data/alfworld/collab_subsets/v2/kitchen_statechange__train.json \
  --alfworld_max_tasks 5
```

## 6.2 跑协作式 smoke test

仓库自带脚本是：

- `scripts/medmcqa/run_alfworld_smoke.sh`

但如果当前环境没有 `python` 命令，建议直接执行等价的 `python3` 命令：

```bash
python3 scripts/medmcqa/eval_collab_domain_adaptation.py \
  --dataset_family alfworld \
  --alfworld_group_a kitchen_statechange \
  --alfworld_group_b home_search \
  --alfworld_subset_dir data/alfworld/collab_subsets/v2 \
  --max_train 10 \
  --max_eval 6 \
  --mas_type autogen \
  --mas_memory selectivemem \
  --reasoning io \
  --model gpt-4o-mini \
  --batch_size 2 \
  --scenarios all
```

## 7. 读代码时建议的顺序

如果你是第一次接手这部分代码，推荐按这个顺序读：

1. `tasks/run.py`
   - 先看参数和主循环，建立总流程。
2. `tasks/envs/__init__.py`
   - 看任务、环境、recorder 是怎么注册的。
3. `tasks/envs/alfworld_env.py`
   - 看 `set_env / step / feedback` 的真实语义。
4. `tasks/prompts/alfworld_prompt.py`
   - 看动作格式约束和 few-shot 来源。
5. `tasks/mas_workflow/autogen/autogen.py`
   - 看每一步 prompt、动作、memory 更新怎么串起来。
6. `scripts/medmcqa/eval_collab_domain_adaptation.py`
   - 再看协作训练和评测是如何在主链路之上搭起来的。

## 8. 常见坑

- 没装 `alfworld`
  - `tasks/envs/__init__.py` 在 import `AlfworldEnv` 时就会失败。
- 没有 `data/alfworld/json_2.1.1/`
  - 能看到任务 JSON，但实际找不到 `game.tw-pddl`。
- `.env` 没配
  - `mas/llm.py` 会直接因为缺少 `OPENAI_API_KEY` / `OPENAI_API_BASE` 报错。
- `python` 命令不存在
  - 仓库里不少脚本默认写 `python`，当前机器上更稳妥的是 `python3`。
- `done=True` 不代表成功
  - 本项目里对 ALFWorld 的最终成功判定看的是 `info['won']`，这一点和很多简单脚本不一样。
- TextWorld/PDDL 个别样本会崩
  - 协作脚本专门加了 subprocess worker 来隔离这类问题。

## 9. 一句话总结

这个项目的 `alfworld` 实现不是“直接把模型接到环境上”这么简单，而是把它包装成了一个统一任务框架中的一类 environment：

- `tasks/run.py` 负责装配
- `AlfworldEnv` 负责执行
- `alfworld_prompt.py` 负责动作约束
- `autogen.py` 负责调度与记忆闭环
- `eval_collab_domain_adaptation.py` 负责双域协作训练与评测

如果你只想先跑通，优先确认三件事：`alfworld` 包已安装、`data/alfworld/json_2.1.1` 已下载、`.env` 已配置。
