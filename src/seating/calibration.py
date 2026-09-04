"""Interactive Seating & Distance Calibration Tool."""

import logging
from pathlib import Path
from typing import Dict, Any

import yaml
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)


class SystemCalibrator:
    """Utility to measure and save user calibration thresholds to config.yaml."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self.config = load_config(config_path)

    def update_seating_zones(self, left_boundary: float, right_boundary: float) -> bool:
        """Updates left and right seating boundaries."""
        if not (0.0 < left_boundary < right_boundary < 1.0):
            logger.error(f"Invalid seating boundaries: left={left_boundary}, right={right_boundary}")
            return False

        if "seating" not in self.config:
            self.config["seating"] = {}

        self.config["seating"]["left_boundary"] = round(float(left_boundary), 3)
        self.config["seating"]["right_boundary"] = round(float(right_boundary), 3)
        return self._save()

    def update_distance_thresholds(self, too_close_ratio: float, too_far_ratio: float) -> bool:
        """Updates distance face ratio thresholds."""
        if not (0.0 < too_far_ratio < too_close_ratio < 1.0):
            logger.error(f"Invalid distance ratios: too_far={too_far_ratio}, too_close={too_close_ratio}")
            return False

        if "distance" not in self.config:
            self.config["distance"] = {}

        self.config["distance"]["too_close_face_ratio"] = round(float(too_close_ratio), 3)
        self.config["distance"]["too_far_face_ratio"] = round(float(too_far_ratio), 3)
        return self._save()

    def _save(self) -> bool:
        """Writes configuration back to YAML file."""
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(self.config, f, default_flow_style=False)
            logger.info(f"Updated calibration saved to {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save calibration to {self.config_path}: {e}")
            return False
