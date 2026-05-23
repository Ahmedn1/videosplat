from __future__ import annotations

"""
Resolve paths to external algorithm backends.

Priority order for each backend:
  1. Environment variable (VIDEOSPLAT_4DGS_DIR, VIDEOSPLAT_STG_DIR, …)
  2. Config file: ~/.videosplat/config.json
  3. Built-in defaults (~/4DGaussians, ~/SpacetimeGaussians, …)
"""

import json
import os
from pathlib import Path

_CONFIG_FILE = Path.home() / ".videosplat" / "config.json"

_DEFAULTS = {
    "backend_dir": str(Path.home() / "4DGaussians"),
    "stg_dir":     str(Path.home() / "SpacetimeGaussians"),
    "gflow_dir":   str(Path.home() / "Gaussian-Flow"),
    "rotor_dir":   str(Path.home() / "4D-Rotor-Gaussians"),
    "mast3r_dir":  str(Path.home() / "mast3r"),
}

_ENV_VARS = {
    "backend_dir": "VIDEOSPLAT_4DGS_DIR",
    "stg_dir":     "VIDEOSPLAT_STG_DIR",
    "gflow_dir":   "VIDEOSPLAT_GFLOW_DIR",
    "rotor_dir":   "VIDEOSPLAT_ROTOR_DIR",
    "mast3r_dir":  "VIDEOSPLAT_MAST3R_DIR",
}

_TRAIN_SCRIPTS = {
    "backend_dir": "train.py",
    "stg_dir":     "train.py",
    "gflow_dir":   "train.py",
    "rotor_dir":   "train.py",   # 4D-Rotor uses ns-train; presence check is advisory
}


def _load_file() -> dict:
    if _CONFIG_FILE.exists():
        try:
            return json.loads(_CONFIG_FILE.read_text())
        except Exception:
            return {}
    return {}


def _get_dir(key: str) -> Path:
    if v := os.environ.get(_ENV_VARS[key]):
        return Path(v)
    cfg = _load_file()
    # legacy key migration: old "stg_dir" used to mean 4DGS backend
    if key == "backend_dir" and not cfg.get("backend_dir") and cfg.get("stg_dir"):
        return Path(cfg["stg_dir"])
    if v := cfg.get(key):
        return Path(v)
    return Path(_DEFAULTS[key])


def get_backend_dir() -> Path:
    """Path to 4DGaussians repo."""
    return _get_dir("backend_dir")


def get_stg_dir() -> Path:
    """Path to SpacetimeGaussians repo."""
    return _get_dir("stg_dir")


def get_gflow_dir() -> Path:
    """Path to Gaussian-Flow repo."""
    return _get_dir("gflow_dir")


def get_rotor_dir() -> Path:
    """Path to 4D-Rotor-Gaussians repo."""
    return _get_dir("rotor_dir")


def get_mast3r_dir() -> Path:
    """Path to MASt3R repo (clone of github.com/naver/mast3r)."""
    return _get_dir("mast3r_dir")


def save(
    backend_dir: Path | None = None,
    stg_dir: Path | None = None,
    gflow_dir: Path | None = None,
    rotor_dir: Path | None = None,
    mast3r_dir: Path | None = None,
) -> None:
    _CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = _load_file()
    data.pop("stg_dir_legacy", None)   # clean up any truly old keys
    if backend_dir is not None:
        data["backend_dir"] = str(backend_dir)
    if stg_dir is not None:
        data["stg_dir"] = str(stg_dir)
    if gflow_dir is not None:
        data["gflow_dir"] = str(gflow_dir)
    if rotor_dir is not None:
        data["rotor_dir"] = str(rotor_dir)
    if mast3r_dir is not None:
        data["mast3r_dir"] = str(mast3r_dir)
    _CONFIG_FILE.write_text(json.dumps(data, indent=2))


def validate(algo: str = "4dgs") -> list[str]:
    """Return a list of configuration issues for the chosen algorithm."""
    issues = []
    if algo == "4dgs":
        d = get_backend_dir()
        if not (d / "train.py").exists():
            issues.append(
                f"4DGaussians not found at {d}\n"
                "  Clone: git clone https://github.com/hustvl/4DGaussians\n"
                "  Set:   videosplat config --backend-dir /path/to/4DGaussians\n"
                "  Build: cd 4DGaussians && git submodule update --init --recursive\n"
                "         pip install submodules/depth-diff-gaussian-rasterization "
                "--no-build-isolation"
            )
    elif algo == "stg":
        d = get_stg_dir()
        if not (d / "train.py").exists():
            issues.append(
                f"SpacetimeGaussians not found at {d}\n"
                "  Clone: git clone https://github.com/oppo-us-research/SpacetimeGaussians\n"
                "  Set:   videosplat config --stg-dir /path/to/SpacetimeGaussians"
            )
    elif algo == "gaussian-flow":
        d = get_gflow_dir()
        # Gaussian-Flow (Pointrix) uses launch.py; original uses train.py
        if not (d / "launch.py").exists() and not (d / "train.py").exists():
            issues.append(
                f"Gaussian-Flow not found at {d}\n"
                "  Clone: git clone https://github.com/NJU-3DV/Gaussian-Flow\n"
                "  Set:   videosplat config --gflow-dir /path/to/Gaussian-Flow"
            )
    elif algo == "4d-rotor":
        import shutil
        if not shutil.which("ns-train"):
            issues.append(
                "nerfstudio not found on PATH (required for 4d-rotor).\n"
                "  Install: pip install nerfstudio\n"
                "  Then:    pip install -e /path/to/4D-Rotor-Gaussians\n"
                "  Set:     videosplat config --rotor-dir /path/to/4D-Rotor-Gaussians"
            )
    elif algo == "mast3r":
        d = get_mast3r_dir()
        ckpt_dir = d / "checkpoints"
        if not d.exists():
            issues.append(
                f"MASt3R not found at {d}\n"
                "  Clone: git clone --recursive https://github.com/naver/mast3r ~/mast3r\n"
                "  Set:   videosplat config --mast3r-dir /path/to/mast3r"
            )
        elif not any(ckpt_dir.glob("*.pth")):
            issues.append(
                f"No MASt3R checkpoint (.pth) found in {ckpt_dir}\n"
                "  Download from: https://github.com/naver/mast3r#usage\n"
                "  Place .pth file in: ~/mast3r/checkpoints/"
            )
    return issues
