"""
Model Manager Module

Model versioning and persistence.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Model versioning and persistence.

    Features:
    - Version management
    - Model saving/loading
    - Metadata tracking
    - Rollback support
    """

    def __init__(self, model_dir: str = "models"):
        """
        Initialize model manager.

        Args:
            model_dir: Directory for model storage
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.versions_file = self.model_dir / "versions.json"
        self.versions = self._load_versions()

    def _load_versions(self) -> Dict[str, Any]:
        """Load versions metadata."""
        if self.versions_file.exists():
            with open(self.versions_file, "r") as f:
                return json.load(f)
        return {}

    def _save_versions(self) -> None:
        """Save versions metadata."""
        with open(self.versions_file, "w") as f:
            json.dump(self.versions, f, indent=2)

    def save_version(
        self,
        model: torch.nn.Module,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save model version.

        Args:
            model: Model to save
            model_name: Name of the model
            metadata: Additional metadata

        Returns:
            Version ID
        """
        # Create version ID
        timestamp = datetime.now().isoformat()
        version_id = f"{model_name}_v{len(self.versions) + 1}_{timestamp}"

        # Save model
        model_path = self.model_dir / f"{version_id}.pt"
        torch.save(model.state_dict(), model_path)

        # Save metadata
        version_info = {
            "version_id": version_id,
            "model_name": model_name,
            "timestamp": timestamp,
            "model_path": str(model_path),
        }

        if metadata:
            version_info.update(metadata)

        self.versions[version_id] = version_info
        self._save_versions()

        logger.info(f"Saved model version: {version_id}")
        return version_id

    def load_version(
        self,
        model: torch.nn.Module,
        version_id: str,
    ) -> torch.nn.Module:
        """
        Load model version.

        Args:
            model: Model to load into
            version_id: Version ID to load

        Returns:
            Loaded model
        """
        if version_id not in self.versions:
            raise ValueError(f"Version not found: {version_id}")

        model_path = self.versions[version_id]["model_path"]
        model.load_state_dict(torch.load(model_path))

        logger.info(f"Loaded model version: {version_id}")
        return model

    def list_versions(self, model_name: Optional[str] = None) -> list:
        """
        List all versions.

        Args:
            model_name: Filter by model name (all if None)

        Returns:
            List of version IDs
        """
        versions = list(self.versions.keys())

        if model_name:
            versions = [v for v in versions if model_name in v]

        return versions

    def get_latest_version(self, model_name: str) -> Optional[str]:
        """
        Get latest version of model.

        Args:
            model_name: Model name

        Returns:
            Latest version ID or None
        """
        versions = self.list_versions(model_name)
        return versions[-1] if versions else None

    def rollback(self, model: torch.nn.Module, version_id: str) -> torch.nn.Module:
        """
        Rollback to previous version.

        Args:
            model: Model to rollback
            version_id: Version to rollback to

        Returns:
            Rolled back model
        """
        return self.load_version(model, version_id)

    def get_version_info(self, version_id: str) -> Dict[str, Any]:
        """Get version information."""
        if version_id not in self.versions:
            raise ValueError(f"Version not found: {version_id}")

        return self.versions[version_id]
