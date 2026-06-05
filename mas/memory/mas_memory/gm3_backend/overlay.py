from __future__ import annotations

from dataclasses import dataclass

from .types import GM3EpisodeState


@dataclass
class GM3OnlineEpisodeBuilder:
    """Track lightweight per-episode state for dynamic prompt augmentation."""

    state: GM3EpisodeState

    @classmethod
    def from_task(cls, task_main: str, task_description: str = "") -> "GM3OnlineEpisodeBuilder":
        return cls(state=GM3EpisodeState(task_main=task_main, task_description=task_description or ""))

    def update(self, action: str, observation: str) -> None:
        clean_action = str(action or "").strip()
        clean_obs = str(observation or "").strip()
        if clean_action:
            self.state.recent_actions.append(clean_action)
            self.state.recent_actions = self.state.recent_actions[-5:]
        if clean_obs:
            self.state.recent_observations.append(clean_obs)
            self.state.recent_observations = self.state.recent_observations[-3:]
        self.state.step_count += 1

    @staticmethod
    def _is_fever_tool_miss(text: str) -> bool:
        lowered = str(text or "").lower()
        return any(
            marker in lowered
            for marker in (
                "could not find",
                "cannot find",
                "searcherrors",
                "jsondecodeerror",
                "pageerror",
                "disambiguationerror",
                "last page searched was not found",
            )
        )

    def planner_notes(self, *, task_family: str = "", progress_state: str = "") -> list[str]:
        notes: list[str] = [f"[GM3Overlay] Current episode step count: {self.state.step_count}."]
        is_fever = str(task_family or "").lower().startswith("fever")
        progress = str(progress_state or "").strip().lower()
        hide_actions = is_fever and progress in {"ready_finish", "done"}
        if self.state.recent_actions and not hide_actions:
            notes.append(
                "[GM3Overlay] Recent actions: " + " | ".join(self.state.recent_actions[-3:]) + "."
            )
        if self.state.recent_observations:
            last_obs = self.state.recent_observations[-1].replace("\n", " ").strip()
            if len(last_obs) > 220:
                last_obs = last_obs[:217] + "..."
            if is_fever and self._is_fever_tool_miss(last_obs):
                notes.append(
                    "[GM3Overlay] Latest tool status: search/page miss; this is not factual evidence for the FEVER label."
                )
            elif is_fever and progress in {"ready_finish", "done"}:
                notes.append(f"[GM3Overlay] Latest factual evidence: {last_obs}")
            else:
                notes.append(f"[GM3Overlay] Latest observation: {last_obs}")
        if is_fever and progress in {"ready_finish", "done"}:
            notes.append(
                "[GM3Overlay] FEVER final-label guard: action names and failed searches are operational history, not evidence."
            )
        return notes
