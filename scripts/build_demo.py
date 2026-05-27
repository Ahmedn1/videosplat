#!/usr/bin/env python3
"""Assemble the static Netlify demo under docs/scenes/ from trained outputs.

For each scene: convert one representative keyframe PLY → a compact `.splat`
(via ply_to_splat) and copy its rendered MP4, then write docs/scenes.json that
docs/index.html (gallery) and docs/viewer.html (single-splat viewer) read.

Run from the repo root:  python scripts/build_demo.py
"""
import json
import shutil
from pathlib import Path

from ply_to_splat import ply_to_splat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DOCS = ROOT / "docs"

# (id, title, output_dir, keyframe, video_file, video_label, metric, blurb)
SCENES = [
    ("piano", "Piano — 3-camera casual capture",
     "piano_4dgs_frontier", "keyframe_0010.ply", "walkthrough_audio.mp4", "Walkthrough + audio",
     "held-out-time PSNR 28.0 · MASt3R + nerfies",
     "Three unsynced phones (one top-down), reconstructed via audio-sync → MASt3R → "
     "nerfies 4DGS. Free-viewpoint of a piano performance from a casual capture."),
    ("aist", "AIST Breakdance — 9-camera synced ring",
     "aist_breakdance_e6", "keyframe_0010.ply", "orbit.mp4", "360° orbit",
     "held-out-VIEW PSNR 15.13→16.50 · MASt3R + nerfies",
     "A dancer in a white cyclorama, 9 synced cameras in a 360° ring. Novel-view eval "
     "(hold out 2 of 9 cams); restoring opacity-reset fixed a floater collapse."),
    ("coffee_martini", "Coffee Martini — N3V 18-camera rig",
     "coffee_martini_final_50k", "keyframe_0010.ply", "orbit.mp4", "360° orbit",
     "test PSNR 28.08 · COLMAP + 4DGaussians",
     "Neural-3D-Video plenoptic kitchen scene. COLMAP calibration → 4DGaussians dynerf; "
     "a focal-units fix and full-camera coverage lifted PSNR 13 → 28 (published-level)."),
    ("flame_steak", "Flame Steak — N3V (SpacetimeGaussians)",
     "flame_steak_6cam_2x", "keyframe_0010.ply", "path_video.mp4", "Camera path",
     "N3V · SpacetimeGaussians (fast calib)",
     "Neural-3D-Video kitchen scene reconstructed with the SpacetimeGaussians backbone "
     "(per-frame COLMAP, ~1h vs 24h) — the fast-iteration algo path."),
    ("basketball", "Basketball — Dynamic3DGaussians benchmark",
     "basketball_10cam", "keyframe_0010.ply", "basketball_orbit.mp4", "360° orbit",
     "12-cam benchmark · pipeline validation",
     "The Dynamic3DGaussians multi-camera benchmark (synchronized rig) used as the "
     "Phase-0 validation that the end-to-end pipeline works before custom captures."),
]


def main():
    (DOCS / "scenes").mkdir(parents=True, exist_ok=True)
    manifest = []
    for sid, title, d, kf, vid, vlabel, metric, blurb in SCENES:
        sdir = DOCS / "scenes" / sid
        sdir.mkdir(parents=True, exist_ok=True)
        entry = {"id": sid, "title": title, "metric": metric, "blurb": blurb}

        ply = OUT / d / "viewer" / "frames" / kf
        if ply.exists():
            n = ply_to_splat(str(ply), str(sdir / "scene.splat"))
            entry["splat"] = f"scenes/{sid}/scene.splat"
            sz = (sdir / "scene.splat").stat().st_size / 1e6
            print(f"[{sid}] splat: {n:,} splats, {sz:.1f} MB")
        else:
            print(f"[{sid}] WARN no keyframe at {ply} — splat skipped")

        src_vid = OUT / d / vid
        if src_vid.exists():
            shutil.copy2(src_vid, sdir / vid)
            entry["video"] = f"scenes/{sid}/{vid}"
            entry["video_label"] = vlabel
            print(f"[{sid}] video: {vid} ({src_vid.stat().st_size/1e6:.1f} MB)")
        else:
            print(f"[{sid}] WARN no video at {src_vid} — video skipped")

        # camera framing for the single-splat viewer (from the scene's scene_meta)
        sm_path = OUT / d / "viewer" / "scene_meta.json"
        if sm_path.exists():
            sm = json.loads(sm_path.read_text())
            if sm.get("scene_center"):
                entry["center"] = sm["scene_center"]
            if sm.get("scene_radius"):
                entry["radius"] = sm["scene_radius"]
            cams = sm.get("cameras") or []
            if cams:
                c = cams[len(cams) // 2]
                entry["cam_pos"] = c.get("pos")
                fwd = c.get("forward")
                if c.get("pos") and fwd:
                    p = c["pos"]
                    entry["cam_lookat"] = [p[0] + fwd[0], p[1] + fwd[1], p[2] + fwd[2]]

        manifest.append(entry)

    (DOCS / "scenes.json").write_text(json.dumps(manifest, indent=2))
    total = sum(f.stat().st_size for f in DOCS.rglob("*") if f.is_file()) / 1e6
    print(f"\ndocs/scenes.json written ({len(manifest)} scenes). docs total ≈ {total:.0f} MB")


if __name__ == "__main__":
    main()
