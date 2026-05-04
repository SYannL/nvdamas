from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_HF_REPO_ID = "XaiverZ/MemSkill"
DEFAULT_CHECKPOINT_NAME = "alfworld_controller.pt"


@dataclass
class MemSkillAssetReport:
    checkpoint_path: str = ""
    models_dir: str = ""
    hf_download: dict[str, Any] | None = None
    ppo_training: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "models_dir": self.models_dir,
            "hf_download": self.hf_download,
            "ppo_training": self.ppo_training,
        }


def resolve_models_dir(repo_root: Path, value: str | None) -> Path:
    raw = str(value or "Models/memskill").strip() or "Models/memskill"
    path = Path(raw)
    if not path.is_absolute():
        path = repo_root / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_checkpoint_path(repo_root: Path, models_dir: Path, value: str | None) -> Path:
    raw = str(value or "").strip()
    if raw:
        path = Path(raw)
        if not path.is_absolute():
            path = repo_root / path
        return path
    return models_dir / DEFAULT_CHECKPOINT_NAME


def is_controller_checkpoint(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        import torch

        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
    except Exception:
        return False
    return isinstance(payload, dict) and isinstance(payload.get("controller_state_dict"), dict)


def find_controller_checkpoint(root: Path) -> Path | None:
    if root.is_file():
        return root if is_controller_checkpoint(root) else None
    if not root.exists():
        return None
    candidates: list[Path] = []
    for pattern in ("*.pt", "*.pth", "*.bin"):
        candidates.extend(root.rglob(pattern))
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    for path in candidates:
        if is_controller_checkpoint(path):
            return path
    return None


def copy_checkpoint(src: Path, dst: Path, *, force: bool = False) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not force:
        return dst
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return dst


@contextmanager
def _original_memskill_context(repo_path: Path):
    old_cwd = os.getcwd()
    old_argv = list(sys.argv)
    old_path = list(sys.path)
    repo_str = str(repo_path)
    try:
        if repo_str in sys.path:
            sys.path.remove(repo_str)
        sys.path.insert(0, repo_str)
        os.chdir(repo_str)
        yield
    finally:
        os.chdir(old_cwd)
        sys.argv = old_argv
        sys.path[:] = old_path


def _api_key_list(value: str) -> list[str]:
    raw = str(value or "").strip()
    if not raw:
        return ["dummy"]
    pieces = [item.strip() for item in raw.replace(",", " ").split() if item.strip()]
    return pieces or [raw]


def _set_if_present(obj: Any, name: str, value: Any) -> None:
    if hasattr(obj, name):
        setattr(obj, name, value)


def _append_cli_arg(cmd: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    cmd.extend([flag, str(value)])


def _append_cli_bool(cmd: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        cmd.append(flag)


def download_hf_checkpoint(
    *,
    repo_id: str,
    filename: str,
    models_dir: Path,
    checkpoint_path: Path,
    force: bool = False,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "requested": True,
        "repo_id": repo_id,
        "filename": filename,
        "checkpoint_path": str(checkpoint_path),
        "status": "not_started",
    }
    if checkpoint_path.exists() and is_controller_checkpoint(checkpoint_path) and not force:
        report.update({"status": "already_present"})
        return report

    download_dir = models_dir / "hf" / repo_id.replace("/", "__")
    download_dir.mkdir(parents=True, exist_ok=True)

    selected: Path | None = None
    try:
        from huggingface_hub import hf_hub_download, snapshot_download

        if filename:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(download_dir),
                local_dir_use_symlinks=False,
            )
            selected = Path(downloaded)
        else:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(download_dir),
                local_dir_use_symlinks=False,
            )
            selected = find_controller_checkpoint(download_dir)
        report["method"] = "huggingface_hub"
    except Exception as hub_exc:
        report["huggingface_hub_error"] = str(hub_exc)
        cmd = [
            "huggingface-cli",
            "download",
            repo_id,
            "--local-dir",
            str(download_dir),
            "--local-dir-use-symlinks",
            "False",
        ]
        if filename:
            cmd.insert(3, filename)
        try:
            completed = subprocess.run(cmd, check=False, text=True, capture_output=True)
            report["method"] = "huggingface-cli"
            report["stdout_tail"] = completed.stdout[-2000:]
            report["stderr_tail"] = completed.stderr[-2000:]
            if completed.returncode != 0:
                report.update({"status": "failed", "returncode": completed.returncode})
                return report
            selected = find_controller_checkpoint(download_dir if not filename else download_dir / filename)
        except Exception as cli_exc:
            report.update({"status": "failed", "huggingface_cli_error": str(cli_exc)})
            return report

    if selected is None or not is_controller_checkpoint(selected):
        report.update(
            {
                "status": "failed",
                "error": "download completed but no file containing controller_state_dict was found",
                "download_dir": str(download_dir),
            }
        )
        return report
    copy_checkpoint(selected, checkpoint_path, force=True)
    report.update({"status": "downloaded", "source_checkpoint": str(selected)})
    return report


def generate_alfworld_offline_data(
    *,
    repo_path: Path,
    output_path: Path,
    split: str,
    force: bool = False,
) -> dict[str, Any]:
    report = {"requested": True, "output_path": str(output_path), "split": split, "status": "not_started"}
    if output_path.exists() and not force:
        report["status"] = "already_present"
        return report
    script = repo_path / "alfworld_replay.py"
    if not script.is_file():
        report.update({"status": "failed", "error": f"missing {script}"})
        return report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(script), "--split", split, "--output", str(output_path)]
    completed = subprocess.run(cmd, cwd=str(repo_path), check=False, text=True, capture_output=True)
    report.update(
        {
            "status": "ok" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "cmd": cmd,
            "stdout_tail": completed.stdout[-2000:],
            "stderr_tail": completed.stderr[-2000:],
        }
    )
    return report


def train_original_memskill_ppo(
    *,
    repo_path: Path,
    checkpoint_path: Path,
    offline_data: Path,
    model: str,
    designer_model: str,
    api_base: str,
    api_key: str,
    save_dir: Path,
    out_file: Path,
    device: str,
    state_encoder: str = "Qwen/Qwen3-Embedding-0.6B",
    op_encoder: str = "Qwen/Qwen3-Embedding-0.6B",
    retriever: str = "contriever",
    inner_epochs: int,
    outer_epochs: int,
    batch_size: int,
    encode_batch_size: int,
    ppo_epochs: int,
    action_top_k: int,
    mem_top_k: int,
    mem_top_k_eval: int,
    reward_metric: str,
    enable_designer: bool,
    train_backend: str,
    wandb_mode: str,
    alfworld_eval_query_source: str,
    alfworld_pair_a_min: int,
    alfworld_pair_a_max: int,
    alfworld_pair_b_size: int,
    alfworld_pair_same_type_prob: float,
    alfworld_pair_chunk_size: int,
    alfworld_pair_max_steps: int,
    alfworld_pair_b_workers: int,
    designer_freq: int,
    new_action_bias_steps: int,
    stage_reward_fraction: float,
    designer_reflection_cycles: int,
    designer_max_changes: int,
    designer_failure_window_epochs: int,
    designer_failure_pool_size: int,
    designer_new_skill_hint: bool,
    extra_args: str,
    force: bool = False,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "requested": True,
        "repo_path": str(repo_path),
        "checkpoint_path": str(checkpoint_path),
        "status": "not_started",
    }
    if checkpoint_path.exists() and is_controller_checkpoint(checkpoint_path) and not force:
        report["status"] = "already_present"
        return report
    main_py = repo_path / "main.py"
    if not main_py.is_file():
        report.update({"status": "failed", "error": f"missing original MemSkill main.py: {main_py}"})
        return report
    if not offline_data.is_file():
        report.update({"status": "failed", "error": f"missing ALFWorld offline data: {offline_data}"})
        return report

    backend = str(train_backend or "embedded").strip().lower()
    if backend == "embedded":
        embedded_report = train_embedded_memskill_ppo(
            repo_path=repo_path,
            checkpoint_path=checkpoint_path,
            offline_data=offline_data,
            model=model,
            designer_model=designer_model,
            api_base=api_base,
            api_key=api_key,
            save_dir=save_dir,
            out_file=out_file,
            device=device,
            inner_epochs=inner_epochs,
            outer_epochs=outer_epochs,
            batch_size=batch_size,
            encode_batch_size=encode_batch_size,
            ppo_epochs=ppo_epochs,
            action_top_k=action_top_k,
            mem_top_k=mem_top_k,
            mem_top_k_eval=mem_top_k_eval,
            reward_metric=reward_metric,
            enable_designer=enable_designer,
            wandb_mode=wandb_mode,
            alfworld_eval_query_source=alfworld_eval_query_source,
            alfworld_pair_a_min=alfworld_pair_a_min,
            alfworld_pair_a_max=alfworld_pair_a_max,
            alfworld_pair_b_size=alfworld_pair_b_size,
            alfworld_pair_same_type_prob=alfworld_pair_same_type_prob,
            alfworld_pair_chunk_size=alfworld_pair_chunk_size,
            alfworld_pair_max_steps=alfworld_pair_max_steps,
            alfworld_pair_b_workers=alfworld_pair_b_workers,
            designer_freq=designer_freq,
            new_action_bias_steps=new_action_bias_steps,
            stage_reward_fraction=stage_reward_fraction,
            designer_reflection_cycles=designer_reflection_cycles,
            designer_max_changes=designer_max_changes,
            designer_failure_window_epochs=designer_failure_window_epochs,
            designer_failure_pool_size=designer_failure_pool_size,
            designer_new_skill_hint=designer_new_skill_hint,
            extra_args=extra_args,
            force=force,
        )
        embedded_report.setdefault("backend", "embedded")
        return embedded_report
    if backend != "subprocess":
        report.update({"status": "failed", "error": f"unknown MemSkill PPO train backend: {train_backend}"})
        return report

    save_dir.mkdir(parents=True, exist_ok=True)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(main_py),
        "--dataset",
        "alfworld",
        "--alfworld-offline-data",
        str(offline_data),
        "--alfworld-eval-query-source",
        alfworld_eval_query_source,
        "--alfworld-pair-a-min",
        str(alfworld_pair_a_min),
        "--alfworld-pair-a-max",
        str(alfworld_pair_a_max),
        "--alfworld-pair-b-size",
        str(alfworld_pair_b_size),
        "--alfworld-pair-b-workers",
        str(alfworld_pair_b_workers),
        "--alfworld-pair-same-type-prob",
        str(alfworld_pair_same_type_prob),
        "--alfworld-pair-chunk-size",
        str(alfworld_pair_chunk_size),
        "--alfworld-pair-max-steps",
        str(alfworld_pair_max_steps),
        "--model",
        model,
        "--designer-model",
        designer_model or model,
        "--api",
        "--api-base",
        api_base,
        "--api-key",
        api_key,
        "--retriever",
        retriever,
        "--state-encoder",
        state_encoder,
        "--op-encoder",
        op_encoder,
        "--inner-epochs",
        str(inner_epochs),
        "--outer-epochs",
        str(outer_epochs),
        "--batch-size",
        str(batch_size),
        "--encode-batch-size",
        str(encode_batch_size),
        "--session-mode",
        "fixed-length",
        "--ppo-epochs",
        str(ppo_epochs),
        "--action-top-k",
        str(action_top_k),
        "--mem-top-k",
        str(mem_top_k),
        "--mem-top-k-eval",
        str(mem_top_k_eval),
        "--reward-metric",
        reward_metric,
        "--designer-freq",
        str(designer_freq),
        "--new-action-bias-steps",
        str(new_action_bias_steps),
        "--stage-reward-fraction",
        str(stage_reward_fraction),
        "--designer-reflection-cycles",
        str(designer_reflection_cycles),
        "--designer-max-changes",
        str(designer_max_changes),
        "--designer-failure-window-epochs",
        str(designer_failure_window_epochs),
        "--designer-failure-pool-size",
        str(designer_failure_pool_size),
        "--device",
        device,
        "--wandb-run-name",
        "nvdamas-memskill-alfworld",
        "--save-dir",
        str(save_dir),
        "--out-file",
        str(out_file),
    ]
    if enable_designer:
        cmd.append("--enable-designer")
    if designer_new_skill_hint:
        cmd.append("--designer-new-skill-hint")
    if extra_args.strip():
        cmd.extend(shlex.split(extra_args))

    env = os.environ.copy()
    env.setdefault("OPENAI_API_BASE", api_base)
    env.setdefault("OPENAI_API_KEY", api_key)
    env.setdefault("WANDB_MODE", wandb_mode or "disabled")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    completed = subprocess.run(cmd, cwd=str(repo_path), env=env, check=False, text=True)
    report.update({"backend": "subprocess", "cmd": cmd, "returncode": completed.returncode})
    if completed.returncode != 0:
        report["status"] = "failed"
        return report
    trained = find_controller_checkpoint(save_dir)
    if trained is None:
        report.update({"status": "failed", "error": f"no controller checkpoint found in {save_dir}"})
        return report
    copy_checkpoint(trained, checkpoint_path, force=True)
    report.update({"status": "trained", "source_checkpoint": str(trained)})
    return report


def train_embedded_memskill_ppo(
    *,
    repo_path: Path,
    checkpoint_path: Path,
    offline_data: Path,
    model: str,
    designer_model: str,
    api_base: str,
    api_key: str,
    save_dir: Path,
    out_file: Path,
    device: str,
    state_encoder: str = "Qwen/Qwen3-Embedding-0.6B",
    op_encoder: str = "Qwen/Qwen3-Embedding-0.6B",
    retriever: str = "contriever",
    inner_epochs: int,
    outer_epochs: int,
    batch_size: int,
    encode_batch_size: int,
    ppo_epochs: int,
    action_top_k: int,
    mem_top_k: int,
    mem_top_k_eval: int,
    reward_metric: str,
    enable_designer: bool,
    wandb_mode: str,
    alfworld_eval_query_source: str,
    alfworld_pair_a_min: int,
    alfworld_pair_a_max: int,
    alfworld_pair_b_size: int,
    alfworld_pair_same_type_prob: float,
    alfworld_pair_chunk_size: int,
    alfworld_pair_max_steps: int,
    alfworld_pair_b_workers: int,
    designer_freq: int,
    new_action_bias_steps: int,
    stage_reward_fraction: float,
    designer_reflection_cycles: int,
    designer_max_changes: int,
    designer_failure_window_epochs: int,
    designer_failure_pool_size: int,
    designer_new_skill_hint: bool,
    extra_args: str,
    force: bool = False,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "requested": True,
        "backend": "embedded",
        "repo_path": str(repo_path),
        "checkpoint_path": str(checkpoint_path),
        "offline_data": str(offline_data),
        "status": "not_started",
    }
    if checkpoint_path.exists() and is_controller_checkpoint(checkpoint_path) and not force:
        report["status"] = "already_present"
        return report

    save_dir.mkdir(parents=True, exist_ok=True)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    old_env = {
        key: os.environ.get(key)
        for key in ("OPENAI_API_BASE", "OPENAI_API_KEY", "WANDB_MODE", "TOKENIZERS_PARALLELISM")
    }
    os.environ.setdefault("OPENAI_API_BASE", api_base)
    os.environ.setdefault("OPENAI_API_KEY", api_key)
    os.environ.setdefault("WANDB_MODE", wandb_mode or "disabled")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        with _original_memskill_context(repo_path):
            sys.argv = [str(repo_path / "main.py")]
            from src.config import AgenticMemoryConfig, get_agentic_memory_args  # type: ignore
            from src.trainer import get_trainer  # type: ignore
            from main import load_dataset, set_seed, split_data  # type: ignore

            original_args = get_agentic_memory_args()

            # Apply the same defaults as MemSkill's ALFWorld training script,
            # but keep every value overridable from the nvdamas CLI.
            original_args.dataset = "alfworld"
            original_args.data_file = str(offline_data)
            original_args.alfworld_offline_data = str(offline_data)
            original_args.model = model
            original_args.designer_model = designer_model or model
            original_args.api = True
            original_args.api_base = api_base
            original_args.api_key = _api_key_list(api_key)
            original_args.retriever = retriever
            original_args.state_encoder = state_encoder
            original_args.op_encoder = op_encoder
            original_args.inner_epochs = inner_epochs
            original_args.outer_epochs = outer_epochs
            original_args.batch_size = batch_size
            original_args.encode_batch_size = encode_batch_size
            original_args.session_mode = "fixed-length"
            original_args.ppo_epochs = ppo_epochs
            original_args.action_top_k = action_top_k
            original_args.mem_top_k = mem_top_k
            original_args.mem_top_k_eval = mem_top_k_eval
            original_args.reward_metric = reward_metric
            original_args.device = device
            original_args.enable_designer = enable_designer
            original_args.save_dir = str(save_dir)
            original_args.out_file = str(out_file)
            original_args.wandb_run_name = "nvdamas-memskill-alfworld"
            original_args.alfworld_eval_query_source = alfworld_eval_query_source
            original_args.alfworld_pair_a_min = alfworld_pair_a_min
            original_args.alfworld_pair_a_max = alfworld_pair_a_max
            original_args.alfworld_pair_b_size = alfworld_pair_b_size
            original_args.alfworld_pair_same_type_prob = alfworld_pair_same_type_prob
            original_args.alfworld_pair_chunk_size = alfworld_pair_chunk_size
            original_args.alfworld_pair_max_steps = alfworld_pair_max_steps
            original_args.alfworld_pair_b_workers = alfworld_pair_b_workers
            original_args.designer_freq = designer_freq
            original_args.new_action_bias_steps = new_action_bias_steps
            original_args.stage_reward_fraction = stage_reward_fraction
            original_args.designer_reflection_cycles = designer_reflection_cycles
            original_args.designer_max_changes = designer_max_changes
            original_args.designer_failure_window_epochs = designer_failure_window_epochs
            original_args.designer_failure_pool_size = designer_failure_pool_size
            original_args.designer_new_skill_hint = designer_new_skill_hint

            if extra_args.strip():
                sys.argv = [str(repo_path / "main.py")]
                default_args_obj = get_agentic_memory_args()
                sys.argv = [str(repo_path / "main.py"), *shlex.split(extra_args)]
                extra_args_obj = get_agentic_memory_args()
                for key, value in vars(extra_args_obj).items():
                    if value != getattr(default_args_obj, key, None):
                        _set_if_present(original_args, key, value)
                original_args.dataset = "alfworld"
                original_args.data_file = str(offline_data)
                original_args.alfworld_offline_data = str(offline_data)
                original_args.save_dir = str(save_dir)
                original_args.out_file = str(out_file)

            config = AgenticMemoryConfig()
            config.update_from_args(original_args)
            set_seed(int(getattr(original_args, "seed", 42)))

            data = load_dataset(str(offline_data), "alfworld")
            train_data, _, _ = split_data(data, "alfworld")
            trainer = get_trainer(original_args, config)
            trainer.train(train_data)
            trainer.save_checkpoint("final")
    except SystemExit as exc:
        report.update({"status": "failed", "error": f"original MemSkill parser exited with {exc}"})
        return report
    except Exception as exc:
        report.update({"status": "failed", "error": str(exc)})
        return report
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    trained = find_controller_checkpoint(save_dir)
    if trained is None:
        report.update({"status": "failed", "error": f"no controller checkpoint found in {save_dir}"})
        return report
    copy_checkpoint(trained, checkpoint_path, force=True)
    report.update({"status": "trained", "source_checkpoint": str(trained), "save_dir": str(save_dir)})
    return report


def prepare_memskill_assets(args: Any, *, repo_root: Path) -> MemSkillAssetReport:
    models_dir = resolve_models_dir(repo_root, getattr(args, "memskill_models_dir", "Models/memskill"))
    checkpoint_path = resolve_checkpoint_path(repo_root, models_dir, getattr(args, "memskill_checkpoint_path", ""))
    setattr(args, "memskill_checkpoint_path", str(checkpoint_path))
    report = MemSkillAssetReport(checkpoint_path=str(checkpoint_path), models_dir=str(models_dir))

    if bool(getattr(args, "memskill_auto_download_checkpoint", False)):
        report.hf_download = download_hf_checkpoint(
            repo_id=str(getattr(args, "memskill_hf_repo_id", DEFAULT_HF_REPO_ID) or DEFAULT_HF_REPO_ID),
            filename=str(getattr(args, "memskill_hf_filename", "") or ""),
            models_dir=models_dir,
            checkpoint_path=checkpoint_path,
            force=bool(getattr(args, "memskill_force_checkpoint", False)),
        )

    if bool(getattr(args, "memskill_train_ppo_from_scratch", False)):
        repo_path = Path(str(getattr(args, "memskill_ppo_repo_path", "") or "")).expanduser()
        if not repo_path.is_absolute():
            repo_path = (repo_root / repo_path).resolve()
        if not (repo_path / "main.py").is_file():
            for candidate in (Path("/workspace/MemSkill-main"), Path("/bigdata/xenial/MemSkill-main")):
                if (candidate / "main.py").is_file():
                    repo_path = candidate
                    break
        offline_raw = str(getattr(args, "memskill_ppo_offline_data", "") or "").strip()
        offline_data = Path(offline_raw) if offline_raw else repo_path / "data" / "alfworld_train_offline.json"
        if not offline_data.is_absolute():
            offline_data = (repo_path / offline_data).resolve()
        training_report: dict[str, Any] = {}
        if bool(getattr(args, "memskill_generate_offline_data", False)):
            training_report["offline_generation"] = generate_alfworld_offline_data(
                repo_path=repo_path,
                output_path=offline_data,
                split=str(getattr(args, "memskill_offline_split", "train") or "train"),
                force=bool(getattr(args, "memskill_force_offline_data", False)),
            )
        save_dir_raw = str(getattr(args, "memskill_ppo_save_dir", "") or "").strip()
        save_dir = Path(save_dir_raw) if save_dir_raw else repo_path / "checkpoints" / "nvdamas_alfworld"
        if not save_dir.is_absolute():
            save_dir = (repo_path / save_dir).resolve()
        out_file = save_dir / "nvdamas_memskill_training_results.json"
        training_report["training"] = train_original_memskill_ppo(
            repo_path=repo_path,
            checkpoint_path=checkpoint_path,
            offline_data=offline_data,
            model=str(getattr(args, "memskill_ppo_model", "") or getattr(args, "model", "")),
            designer_model=str(getattr(args, "memskill_ppo_designer_model", "") or getattr(args, "model", "")),
            api_base=str(getattr(args, "memskill_ppo_api_base", "") or os.environ.get("OPENAI_API_BASE", "")),
            api_key=str(getattr(args, "memskill_ppo_api_key", "") or os.environ.get("OPENAI_API_KEY", "dummy")),
            save_dir=save_dir,
            out_file=out_file,
            device=str(getattr(args, "memskill_ppo_train_device", "") or getattr(args, "memskill_ppo_device", "") or "cuda"),
            state_encoder=str(getattr(args, "memskill_state_encoder", "") or "Qwen/Qwen3-Embedding-0.6B"),
            op_encoder=str(getattr(args, "memskill_op_encoder", "") or "Qwen/Qwen3-Embedding-0.6B"),
            retriever=str(getattr(args, "memskill_ppo_retriever", "") or "contriever"),
            inner_epochs=int(getattr(args, "memskill_ppo_inner_epochs", 100)),
            outer_epochs=int(getattr(args, "memskill_ppo_outer_epochs", 10)),
            batch_size=int(getattr(args, "memskill_ppo_batch_size", 16)),
            encode_batch_size=int(getattr(args, "memskill_ppo_encode_batch_size", 8)),
            ppo_epochs=int(getattr(args, "memskill_ppo_epochs", 2)),
            action_top_k=int(getattr(args, "memskill_action_top_k", None) or 3),
            mem_top_k=int(getattr(args, "memskill_ppo_mem_top_k", 20)),
            mem_top_k_eval=int(getattr(args, "memskill_ppo_mem_top_k_eval", 20)),
            reward_metric=str(getattr(args, "memskill_ppo_reward_metric", "llm_judge") or "llm_judge"),
            enable_designer=bool(getattr(args, "memskill_ppo_enable_designer", False)),
            train_backend=str(getattr(args, "memskill_ppo_train_backend", "embedded") or "embedded"),
            wandb_mode=str(getattr(args, "memskill_ppo_wandb_mode", "disabled") or "disabled"),
            alfworld_eval_query_source=str(
                getattr(args, "memskill_ppo_alfworld_eval_query_source", "objective") or "objective"
            ),
            alfworld_pair_a_min=int(getattr(args, "memskill_ppo_pair_a_min", 40)),
            alfworld_pair_a_max=int(getattr(args, "memskill_ppo_pair_a_max", 60)),
            alfworld_pair_b_size=int(getattr(args, "memskill_ppo_pair_b_size", 5)),
            alfworld_pair_same_type_prob=float(getattr(args, "memskill_ppo_pair_same_type_prob", 1.0)),
            alfworld_pair_chunk_size=int(getattr(args, "memskill_ppo_pair_chunk_size", 2048)),
            alfworld_pair_max_steps=int(getattr(args, "memskill_ppo_pair_max_steps", 50)),
            alfworld_pair_b_workers=int(getattr(args, "memskill_ppo_pair_b_workers", 80)),
            designer_freq=int(getattr(args, "memskill_ppo_designer_freq", 1)),
            new_action_bias_steps=int(getattr(args, "memskill_ppo_new_action_bias_steps", 25)),
            stage_reward_fraction=float(getattr(args, "memskill_ppo_stage_reward_fraction", 0.25)),
            designer_reflection_cycles=int(getattr(args, "memskill_ppo_designer_reflection_cycles", 3)),
            designer_max_changes=int(getattr(args, "memskill_ppo_designer_max_changes", 3)),
            designer_failure_window_epochs=int(getattr(args, "memskill_ppo_designer_failure_window_epochs", 100)),
            designer_failure_pool_size=int(getattr(args, "memskill_ppo_designer_failure_pool_size", 2000)),
            designer_new_skill_hint=bool(getattr(args, "memskill_ppo_designer_new_skill_hint", False)),
            extra_args=str(getattr(args, "memskill_ppo_extra_args", "") or ""),
            force=bool(getattr(args, "memskill_force_checkpoint", False)),
        )
        report.ppo_training = training_report
    return report
