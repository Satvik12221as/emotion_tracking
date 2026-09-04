"""Priority Rules and Decision Action definitions."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class AssistanceAction:
    """Action object dispatched by the Decision Engine."""

    triggered: bool = False
    action_type: str = "NONE"       # SEATING_LEFT, SEATING_RIGHT, DISTANCE_FAR, DISTANCE_CLOSE, POSTURE_LEAN_LEFT, POSTURE_LEAN_RIGHT, POSTURE_SLOUCH, CONFUSION_ASSIST, MULTI_USER_ALERT, NONE
    priority: int = 99              # 1 (Highest) to 99 (Lowest)
    title: str = ""
    message: str = ""
    reasons: List[str] = field(default_factory=list)

    def explain(self) -> str:
        """Returns structured human-readable explanation (Principle 7)."""
        if not self.triggered:
            return "NO ASSISTANCE REQUIRED"

        lines = [
            "========================================",
            "ASSISTANCE TRIGGERED",
            f"Action: {self.title} [Priority {self.priority}]",
            f"Message: {self.message}",
            "Reasons:",
        ]
        for r in self.reasons:
            lines.append(f"  - {r}")
        lines.append("========================================")
        return "\n".join(lines)
