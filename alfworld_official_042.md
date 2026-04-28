# ALFWorld official_042 数据与 nvdamas 适配流程

本文记录在 `nvdamas` 项目中使用 ALFWorld official 0.4.2 下载数据的完整流程，以及为什么需要修改 `tasks/envs/alfworld_env.py` 适配层。

## 目标

原始 `json_2.1.1` 数据中，部分 `valid_unseen` 的 TextWorld PDDL game 会在当前 ALFWorld/TextWorld 环境中加载失败，例如：

```text
KeyError: 'val1'
```

典型坏样本：

```text
valid_unseen/look_at_obj_in_light-Box-None-FloorLamp-219/...
```

为了避免修改 ALFWorld/TextWorld 原生库，改用官方 0.4.2 下载器生成的完整数据目录：

```text
/workspace/run_alf/ALFWORLD_DATA/alfworld_official_042
```

并在本项目 wrapper 层兼容 official_042 的动作语法。

## 重要结论

不要只下载并直接使用：

```text
json_2.1.3_tw-pddl.zip
```

这个 zip 只包含一部分 `game.tw-pddl`，不是完整 ALFWorld 数据目录。官方完整目录需要由 `alfworld-download` 组合生成，最终仍然会包含：

```text
json_2.1.1/
```

但其中的 `game.tw-pddl` 来自更新后的 official 0.4.2 数据。

## 在 container 内下载 official_042

宿主机没有安装 `alfworld` 时，不需要在宿主机安装。直接使用 container 中已有的 ALFWorld 环境下载：

```bash
docker exec -it mcma-recover bash

mkdir -p /workspace/run_alf/ALFWORLD_DATA/alfworld_official_042
export ALFWORLD_DATA=/workspace/run_alf/ALFWORLD_DATA/alfworld_official_042

alfworld-download --force-download --force
```

下载完成后检查：

```bash
du -sh /workspace/run_alf/ALFWORLD_DATA/alfworld_official_042
find /workspace/run_alf/ALFWORLD_DATA/alfworld_official_042/json_2.1.1 -name traj_data.json | wc -l
find /workspace/run_alf/ALFWORLD_DATA/alfworld_official_042/json_2.1.1 -name initial_state.pddl | wc -l
find /workspace/run_alf/ALFWORLD_DATA/alfworld_official_042/json_2.1.1 -name game.tw-pddl | wc -l
```

当前实测：

```text
2.3G    /workspace/run_alf/ALFWORLD_DATA/alfworld_official_042
traj_data.json      7080
initial_state.pddl  7080
game.tw-pddl        4027
```

`game.tw-pddl` 少于 `traj_data.json` 是正常的。official_042 并没有给所有 trajectory 提供 TextWorld game。

## 设置项目数据软链接

进入 container：

```bash
docker exec -it mcma-recover bash
cd /workspace/nvdamas
```

设置 API：

```bash
export OPENAI_API_BASE=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=EMPTY
```

设置数据软链接：

```bash
cd /workspace/nvdamas/data/alfworld
rm -rf json_2.1.1
ln -s /workspace/run_alf/ALFWORLD_DATA/alfworld_official_042/json_2.1.1 json_2.1.1

cd /workspace/nvdamas
readlink -f data/alfworld/json_2.1.1
```

期望输出：

```text
/workspace/run_alf/ALFWORLD_DATA/alfworld_official_042/json_2.1.1
```

注意：软链接通常不需要每次重新进入 container 后重建。只有在重新拷贝代码覆盖 `data/alfworld/json_2.1.1`、重建 container、或手动删掉软链接后才需要重新设置。

## 重新生成 subset

旧的 `data/alfworld/collab_subsets/v3_s` 不能直接用于 official_042，因为旧 subset 里可能包含 official_042 没有 `game.tw-pddl` 的样本。

在 container 内，确认软链接指向 official_042 后重新生成：

```bash
cd /workspace/nvdamas

python scripts/alfworld/materialize_collab_subsets_v3.py --splits train,valid_unseen
python scripts/alfworld/materialize_collab_subsets_v3_s.py --clean
```

当前 official_042 实测生成结果：

```text
v3:
kitchen train        1696
living train          477
bedroom train         734
bathroom train        646
kitchen valid_unseen   77
living valid_unseen    11
bedroom valid_unseen   27
bathroom valid_unseen  19

v3_s:
kitchen train         100
living train          100
bedroom train         100
bathroom train        100
kitchen valid_unseen   20
living valid_unseen    11
bedroom valid_unseen   20
bathroom valid_unseen  19
```

`living valid_unseen` 只有 11 个不是脚本错误，而是 official_042 中 living `valid_unseen` 实际可用的 `game.tw-pddl` 只有 11 个。

检查当前 subset：

```bash
python - <<'PY'
import json
from pathlib import Path

base = Path("data/alfworld/collab_subsets/v3_s")
for name in [
    "kitchen__train.json",
    "living__train.json",
    "kitchen__valid_unseen.json",
    "living__valid_unseen.json",
]:
    rows = json.loads((base / name).read_text())
    print(name, len(rows))
    print("  first:", rows[0]["env_kwargs"]["gamefile"])
PY
```

`living__valid_unseen.json` 应该是：

```text
11
data/alfworld/json_2.1.1/valid_unseen/pick_two_obj_and_place-KeyChain-None-Safe-219/...
```

不应该再以 `FloorLamp` 开头。

## 为什么修改 `tasks/envs/alfworld_env.py`

`tasks/envs/alfworld_env.py` 不是 ALFWorld 原生代码，而是本项目为 G-Memory/MCMA 集成 ALFWorld 写的适配层。它负责：

```text
1. 调用 ALFWorld 原生 AlfredTWEnv
2. 包装 reset / step / feedback 接口
3. 解析任务类型
4. 清洗模型输出动作
5. 统计成功率
6. 为 GraphMemory2 导出 episode history
```

official_042 的 `game.tw-pddl` 使用了新版放置动作，例如：

```text
move bowl 1 to shelf 2
```

而本项目 prompt、few-shot examples 和旧 memory 常引导模型输出旧式动作：

```text
put bowl 1 in/on shelf 2
```

在 official_042 中，旧式动作会返回：

```text
Nothing happens.
```

因此需要在 wrapper 层做动作兼容，而不是修改 ALFWorld/TextWorld 原生库。

## 当前 `alfworld_env.py` 的关键修改

文件：

```text
tasks/envs/alfworld_env.py
```

关键点：

```text
1. `_extract_action()` 支持 `move ` 开头的动作。
2. `step()` 在执行前调用 `_adapt_action_to_admissible()`。
3. `_adapt_action_to_admissible()` 只在当前 admissible commands 中确实存在目标命令时进行转换。
4. 恢复/维护 `current_history` 和 `last_admissible_commands`，供 GraphMemory2 动态构图使用。
5. 提供 `export_gm2_history()`，让 GraphMemory2 online accumulate episode history。
```

动作转换规则：

```text
put X in/on Y -> move X to Y
put X in Y    -> move X to Y
put X on Y    -> move X to Y
```

但只有当当前环境的 `admissible_commands` 中存在对应的：

```text
move X to Y
```

时才转换。否则保留原动作。

这保证了旧 `json_2.1.1` 中如果仍然使用 `put ... in/on ...`，不会被盲目改坏。

检查代码是否在 container 中生效：

```bash
cd /workspace/nvdamas
grep -n "move " tasks/envs/alfworld_env.py | head
```

应该能看到类似：

```text
179:            "move ",
213:        """Map legacy ALFWorld placement syntax to official_042 move syntax.
230:        candidate = f"move {obj} to {dest}"
```

## 从宿主同步代码到 container

如果在宿主 `/bigdata/xenial/nvdamas` 修改了代码，需要同步到 container：

```bash
cd /bigdata/xenial/nvdamas

tar \
  --exclude='.git' \
  --exclude='.db' \
  --exclude='logs' \
  --exclude='__pycache__' \
  -cf - . | docker exec -i mcma-recover tar -C /workspace/nvdamas -xf -
```

如果只同步 `alfworld_env.py`：

```bash
docker cp /bigdata/xenial/nvdamas/tasks/envs/alfworld_env.py \
  mcma-recover:/workspace/nvdamas/tasks/envs/alfworld_env.py
```

同步后检查：

```bash
docker exec -it mcma-recover bash
cd /workspace/nvdamas

export OPENAI_API_BASE=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=EMPTY

python -m py_compile tasks/envs/alfworld_env.py
grep -n "move " tasks/envs/alfworld_env.py | head
```

## 每次重新进入 container 后的检查

每次新开 shell 都需要重新 export API：

```bash
cd /workspace/nvdamas

export OPENAI_API_BASE=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=EMPTY
```

然后检查三件事：

```bash
readlink -f data/alfworld/json_2.1.1

python - <<'PY'
import json
from pathlib import Path
rows = json.loads(Path("data/alfworld/collab_subsets/v3_s/living__valid_unseen.json").read_text())
print(len(rows))
print(rows[0]["env_kwargs"]["gamefile"])
PY

grep -n "move " tasks/envs/alfworld_env.py | head
```

期望：

```text
/workspace/run_alf/ALFWORLD_DATA/alfworld_official_042/json_2.1.1
11
data/alfworld/json_2.1.1/valid_unseen/pick_two_obj_and_place-KeyChain-None-Safe-219/...
```

并且 `grep` 能看到 `move` 相关逻辑。

## Smoke test

小样本测试：

```bash
cd /workspace/nvdamas

export OPENAI_API_BASE=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=EMPTY

python scripts/medmcqa/eval_collab_domain_adaptation.py \
  --dataset_family alfworld \
  --alfworld_group_a kitchen \
  --alfworld_group_b living \
  --alfworld_subset_dir data/alfworld/collab_subsets/v3_s \
  --alfworld_game_root /workspace/run_alf/ALFWORLD_DATA/alfworld_official_042/json_2.1.1 \
  --mas_type autogen \
  --mas_memory graph_memory2 \
  --reasoning io \
  --model qwen32b-api \
  --max_trials 30 \
  --batch_size 1 \
  --max_train 2 \
  --max_eval 2 \
  --run_id gm2_official042_movefix_resubset_smoke \
  --reset_memory \
  --gm2_dynamic_graph \
  --gm2_retrieval_mode lightweight \
  --gm2_settings local_plus_global \
  --gm2_enable_overlay \
  --scenarios inner_ours_A_on_A,inner_ours_B_on_B
```

当前 smoke test 实测：

```text
train kitchen: 2/2 success
train living:  2/2 success
eval kitchen:  1/2 success
eval living:   0/2 success
```

这个结果说明：

```text
1. official_042 数据路径可用。
2. `move` 动作兼容生效。
3. train episode 可以成功并写入 GraphMemory2 local memory。
4. 不再出现 `Unable to find game` / `KeyError: val1`。
```

eval 没有全对不代表接口失败。这个 smoke 的训练样本只有 2+2，而且 train/eval 任务类型不同：

```text
train kitchen: cool bowl -> shelf
eval kitchen:  heat apple -> fridge

train living:  put book -> sofa
eval living:   put two keychain -> safe
```

因此 smoke 主要用于验证流程，不用于评估最终效果。

## 中等规模 probe

建议正式跑之前先做 probe：

```bash
python scripts/medmcqa/eval_collab_domain_adaptation.py \
  --dataset_family alfworld \
  --alfworld_group_a kitchen \
  --alfworld_group_b living \
  --alfworld_subset_dir data/alfworld/collab_subsets/v3_s \
  --alfworld_game_root /workspace/run_alf/ALFWORLD_DATA/alfworld_official_042/json_2.1.1 \
  --mas_type autogen \
  --mas_memory graph_memory2 \
  --reasoning io \
  --model qwen32b-api \
  --max_trials 30 \
  --batch_size 1 \
  --max_train 20 \
  --max_eval 5 \
  --run_id gm2_official042_movefix_probe20 \
  --reset_memory \
  --gm2_dynamic_graph \
  --gm2_retrieval_mode lightweight \
  --gm2_settings local_plus_global \
  --gm2_enable_overlay \
  --scenarios inner_ours_A_on_A,inner_ours_B_on_B
```

## 正式运行

确认 probe 正常后运行完整 `v3_s`：

```bash
nohup python scripts/medmcqa/eval_collab_domain_adaptation.py \
  --dataset_family alfworld \
  --alfworld_group_a kitchen \
  --alfworld_group_b living \
  --alfworld_subset_dir data/alfworld/collab_subsets/v3_s \
  --alfworld_game_root /workspace/run_alf/ALFWORLD_DATA/alfworld_official_042/json_2.1.1 \
  --mas_type autogen \
  --mas_memory graph_memory2 \
  --reasoning io \
  --model qwen32b-api \
  --max_trials 30 \
  --batch_size 1 \
  --run_id gm2_kitchen_living_official042_movefix_mt30 \
  --reset_memory \
  --gm2_dynamic_graph \
  --gm2_retrieval_mode lightweight \
  --gm2_settings local_plus_global \
  --gm2_enable_overlay \
  --scenarios inner_ours_A_on_A,inner_ours_B_on_B \
  > /workspace/nvdamas/L_FALL_GM2_official042_movefix.log 2>&1 &
```

查看日志：

```bash
tail -f /workspace/nvdamas/L_FALL_GM2_official042_movefix.log
```

## 注意事项

1. 新开 shell 后必须重新设置：

```bash
export OPENAI_API_BASE=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=EMPTY
```

否则会出现：

```text
TypeError: str expected, not NoneType
```

2. 如果从宿主整目录同步代码到 container，可能会覆盖 container 中重新生成的 `v3_s`。同步后需要重新检查：

```bash
python - <<'PY'
import json
from pathlib import Path
rows = json.loads(Path("data/alfworld/collab_subsets/v3_s/living__valid_unseen.json").read_text())
print(len(rows))
print(rows[0]["env_kwargs"]["gamefile"])
PY
```

如果显示 20 且开头是 `FloorLamp`，说明 subset 被旧版本覆盖了，需要重新执行 materialize。

3. official_042 的结果不能和旧 `json_2.1.1 + 旧 v3_s` 做严格一一对比，因为可运行 `game.tw-pddl` 集合发生了变化，尤其是 living `valid_unseen` 从旧 subset 的 20 个变为 official_042 可用的 11 个。

