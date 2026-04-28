from __future__ import annotations

from dataclasses import dataclass

from .types import GM2EpisodeState


@dataclass
class GM2OnlineEpisodeBuilder:
    """Track lightweight per-episode state for dynamic prompt augmentation."""

    state: GM2EpisodeState

    @classmethod
    def from_task(cls, task_main: str, task_description: str = "") -> "GM2OnlineEpisodeBuilder":
        return cls(state=GM2EpisodeState(task_main=task_main, task_description=task_description or ""))

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

    def planner_notes(self) -> list[str]:
        notes: list[str] = [f"[GM2Overlay] Current episode step count: {self.state.step_count}."]
        if self.state.recent_actions:
            notes.append(
                "[GM2Overlay] Recent actions: " + " | ".join(self.state.recent_actions[-3:]) + "."
            )
        if self.state.recent_observations:
            last_obs = self.state.recent_observations[-1].replace("\n", " ").strip()
            if len(last_obs) > 220:
                last_obs = last_obs[:217] + "..."
            notes.append(f"[GM2Overlay] Latest observation: {last_obs}")
        return notes
