from __future__ import annotations
"""
Casual multi-camera capture → 4DGaussians (nerfies/HyperNeRF) dataset.

For captures that the dynerf path CANNOT handle: heterogeneous cameras (different
resolution/intrinsics, incl. portrait), MOVING cameras, and unsynced clips. This
is the productionised form of the research on the 3-cam piano capture
(see .lab/ on branch research/piano-moving-cams).

Pipeline (orchestrated by `videosplat casual` in cli.py):
  audio-sync → MASt3R calibration (per-camera intrinsics; per-FRAME poses for
  moving cams via keyframe-slerp interpolation; person-confidence-downweighted so
  the moving subject doesn't corrupt pose) → letterbox to a common resolution →
  write nerfies format → (train / bake --configs / export handled by the caller).

Why each piece (validated empirically):
  * MASt3R not COLMAP: SIFT can't bridge wide baselines (e.g. a top-down view).
  * nerfies reader: the only 4DGaussians mode with per-frame poses + per-camera
    intrinsics + per-frame time + multiple cameras.
  * conf-DOWNWEIGHT (not black-out) the person: black-out also removes static
    structure next to them (keys/hands) and wrecks the pose solve.
  * temporal density is the dominant quality lever.
"""
import json, subprocess, tempfile, wave, contextlib, os, sys
from pathlib import Path
import numpy as np
import cv2
from rich.console import Console

console = Console()

# The nerfies/HyperNeRF reader (scene/hyper_loader.py) uses image[0]'s shape for
# FOV, so all frames must share one working resolution; portrait cams get
# letterboxed into it. 1024x576 (16:9) was the validated sweet spot.
DEFAULT_W, DEFAULT_H = 1024, 576


# ── audio sync ────────────────────────────────────────────────────────────────

def audio_sync_offsets(video_paths: list[Path], ffmpeg: str, sr: int = 8000) -> list[float]:
    """Per-clip start offset (seconds) vs the first clip, via onset-envelope
    cross-correlation. Robust for transient-rich audio (piano, claps, speech)."""
    def load(mp4):
        t = tempfile.mktemp(suffix=".wav")
        subprocess.run([ffmpeg, "-y", "-i", str(mp4), "-ac", "1", "-ar", str(sr), "-vn", t],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with contextlib.closing(wave.open(t, "rb")) as w:
            a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32)
        os.remove(t)
        a = a - a.mean(); return a / (np.abs(a).max() + 1e-9)
    def env(x):
        w = int(0.02 * sr); e = np.convolve(np.abs(x), np.ones(w) / w, "same"); return e - e.mean()
    ref = load(video_paths[0]); ea = env(ref)
    offs = [0.0]
    for v in video_paths[1:]:
        eb = env(load(v)); n = 1 << int(np.ceil(np.log2(len(ea) + len(eb))))
        c = np.fft.irfft(np.fft.rfft(ea, n) * np.conj(np.fft.rfft(eb, n)), n)
        c = np.concatenate([c[-(len(eb) - 1):], c[:len(ea)]])
        offs.append((np.argmax(c) - (len(eb) - 1)) / sr)
    return offs


# ── person masking (for pose-confidence downweighting) ──────────────────────────

def _person_masker(score: float = 0.7, dilate: int = 9):
    import torch
    from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT).eval().cuda()
    def mask(img_bgr):
        import torch as _t
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        t = _t.from_numpy(rgb).permute(2, 0, 1).float().div(255).unsqueeze(0).cuda()
        with _t.no_grad(): o = model(t)[0]
        keep = (o["labels"] == 1) & (o["scores"] > score)
        m = o["masks"][keep, 0].cpu().numpy()
        if len(m) == 0: return np.zeros(img_bgr.shape[:2], bool)
        u = cv2.dilate((m.max(0) > 0.5).astype(np.uint8), np.ones((dilate, dilate), np.uint8))
        return u > 0
    return model, mask


def _letterbox(img, W, H):
    hs, ws = img.shape[:2]; s = min(W / ws, H / hs)
    nw, nh = int(round(ws * s)), int(round(hs * s))
    o = np.zeros((H, W, 3), np.uint8); px, py = (W - nw) // 2, (H - nh) // 2
    o[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return o


# ── main builder ────────────────────────────────────────────────────────────────

def build_casual_nerfies_dataset(
    video_dir: Path, out_dir: Path, *, ffmpeg: str, mast3r_dir: Path,
    moving_cams: tuple[int, ...] = (), n_time: int = 300, n_keyframes: int = 18,
    seg: tuple[float, float] | None = None, W: int = DEFAULT_W, H: int = DEFAULT_H,
    fps: float = 30.0, mast3r_size: int = 512, mast3r_niter: int = 400,
    mask_person: bool = True, mask_downweight: float = 1.0, mask_dilate: int = 9,
    mask_score: float = 0.7, init_conf_thr: float = 1.5, max_init_pts: int = 100_000,
    static_calib_frames: int = 3, edge_thr_modular: int = 24,
    holdout_cams: tuple[int, ...] = (), audio_sync: bool = True,
) -> dict:
    """Write a 4DGaussians nerfies dataset from casual multi-cam videos. Every
    knob is a parameter (the CLI `videosplat casual` exposes them all).

    moving_cams:     indices of cameras that move → per-FRAME poses (keyframe
                     MASt3R + slerp/lerp interp). Others get one averaged static pose.
    n_time:          temporal density (dominant quality lever).
    n_keyframes:     MASt3R keyframes per moving cam (interpolated to n_time).
    seg:             (start,end) sec in cam0's clock; default = full overlap.
    mast3r_size:     MASt3R inference resolution (lower = less VRAM, worse calib).
    mast3r_niter:    global-alignment iterations.
    mask_person:     enable person conf-downweighting in the pose solve.
    mask_downweight: strength in [0,1]; 1.0 = fully ignore the person's pixels
                     (conf→1.0), 0.0 = no downweight. (Black-out is intentionally
                     NOT offered — it removes static structure and wrecks poses.)
    mask_dilate:     person-mask dilation px. mask_score: detector score threshold.
    init_conf_thr:   MASt3R confidence cutoff for the init point cloud.
    max_init_pts:    cap on init-cloud points.
    edge_thr_modular: switch to the lower-VRAM Modular optimizer above this image count.
    Returns dict with n_images / per-camera focals / trajectory stats.
    """
    mask_downweight = float(min(max(mask_downweight, 0.0), 1.0))
    for p in (str(mast3r_dir), str(mast3r_dir / "dust3r")):
        if p not in sys.path: sys.path.insert(0, p)
    import torch
    from scipy.spatial.transform import Rotation, Slerp
    from mast3r.model import AsymmetricMASt3R
    from mast3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.utils.image import load_images
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    from plyfile import PlyData, PlyElement

    vids = sorted(video_dir.glob("cam*.mp4"))
    ncam = len(vids)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rgb" / "2x").mkdir(parents=True, exist_ok=True)
    (out_dir / "camera").mkdir(parents=True, exist_ok=True)
    cal = out_dir / "_cal"; cal.mkdir(exist_ok=True)

    console.print(f"  [dim]casual: {ncam} cams, moving={moving_cams or 'none'}, {n_time} timesteps[/]")
    if audio_sync:
        offs = audio_sync_offsets(vids, ffmpeg)
        console.print(f"  [dim]audio sync offsets (s): {[round(o,2) for o in offs]}[/]")
    else:
        offs = [0.0] * ncam
        console.print("  [dim]audio sync DISABLED (hardware-synced rig) → offsets=0[/]")
    durs = []
    for v in vids:
        c = cv2.VideoCapture(str(v)); durs.append(c.get(cv2.CAP_PROP_FRAME_COUNT) / fps); c.release()
    if seg is None:
        seg = (3.0, min(durs) - 3.0)

    def grab(ci, tau):
        c = cv2.VideoCapture(str(vids[ci])); c.set(cv2.CAP_PROP_POS_FRAMES, int(round((tau + offs[ci]) * fps)))
        ok, fr = c.read(); c.release(); return _letterbox(fr, W, H) if ok else None

    # ---- calibration set: static cams (3 frames) + moving cams (n_keyframes over clip) ----
    cal_paths, cal_tag, cal_masks = [], [], []
    masker = mask = None
    if mask_person: masker, mask = _person_masker(mask_score, mask_dilate)
    def add_cal(ci, tau):
        im = grab(ci, tau)
        cal_masks.append(mask(im) if mask else np.zeros(im.shape[:2], bool))
        p = cal / f"c{ci}_{tau:07.2f}.jpg"; cv2.imwrite(str(p), im)
        cal_paths.append(str(p)); cal_tag.append((ci, float(tau)))
    for ci in range(ncam):
        if ci in moving_cams:
            for tau in np.linspace(seg[0], seg[1], n_keyframes): add_cal(ci, float(tau))
        else:
            for tau in np.linspace(seg[0], min(seg[1], seg[0] + 27), static_calib_frames): add_cal(ci, float(tau))
    if masker is not None:
        del masker; torch.cuda.empty_cache()

    # ---- MASt3R global alignment (Modular = lower VRAM for larger sets) ----
    model = AsymmetricMASt3R.from_pretrained(str(sorted((mast3r_dir / "checkpoints").glob("*.pth"))[0])).to("cuda").eval()
    images = load_images(cal_paths, size=mast3r_size, verbose=False)
    pairs = make_pairs(images, scene_graph="complete", symmetrize=True)
    out = inference(pairs, model, "cuda", batch_size=1, verbose=False)
    del model; torch.cuda.empty_cache()
    mode = (GlobalAlignerMode.ModularPointCloudOptimizer if len(images) > edge_thr_modular
            else GlobalAlignerMode.PointCloudOptimizer)
    sc = global_aligner(out, "cuda", mode=mode, verbose=False)
    # downweight person pixels in the alignment confidence (keep full image).
    # strength s: conf_person ← 1.0 + (conf-1.0)*(1-s); s=1 → conf=1.0 (ignored).
    if mask_person and mask_downweight > 0:
        import torch.nn.functional as F
        s = mask_downweight
        def rmask(n, shape):
            mb = torch.from_numpy(cal_masks[n].astype("float32"))[None, None]
            return F.interpolate(mb, size=tuple(shape), mode="nearest")[0, 0].bool()
        def dw(conf, m):
            conf.data[m] = 1.0 + (conf.data[m] - 1.0) * (1.0 - s)
        for ij in list(sc.str_edges):
            i, j = map(int, ij.split("_"))
            dw(sc.conf_i[ij], rmask(i, sc.conf_i[ij].shape).to(sc.conf_i[ij].device))
            dw(sc.conf_j[ij], rmask(j, sc.conf_j[ij].shape).to(sc.conf_j[ij].device))
        for n in range(len(sc.im_conf)):
            dw(sc.im_conf[n], rmask(n, sc.im_conf[n].shape).to(sc.im_conf[n].device))
    sc.compute_global_alignment(init="mst", niter=mast3r_niter, schedule="cosine", lr=0.01)
    P = sc.get_im_poses().detach().cpu().numpy()
    focals = sc.get_focals().detach().cpu().numpy().flatten()
    pts, confs = sc.get_pts3d(), sc.get_conf()
    ms_h, ms_w = int(images[0]["true_shape"][0][0]), int(images[0]["true_shape"][0][1])
    fscale = max(W, H) / max(ms_w, ms_h)

    taus = np.linspace(seg[0], seg[1], n_time)
    static_pose, static_focal, mov_R, mov_C, mov_focal = {}, {}, {}, {}, {}
    for ci in range(ncam):
        idx = [k for k, t in enumerate(cal_tag) if t[0] == ci]
        if ci in moving_cams:
            kf_taus = np.array([cal_tag[k][1] for k in idx]); kf_P = P[idx]
            kf_R = Rotation.from_matrix(kf_P[:, :3, :3]); cl = np.clip(taus, kf_taus[0], kf_taus[-1])
            mov_R[ci] = Slerp(kf_taus, kf_R)(cl).as_matrix()
            mov_C[ci] = np.stack([np.interp(cl, kf_taus, kf_P[:, a, 3]) for a in range(3)], 1)
            mov_focal[ci] = float(np.median(focals[idx])) * fscale
        else:
            static_pose[ci] = P[idx].mean(0); static_focal[ci] = float(np.mean(focals[idx])) * fscale

    parts = [mov_C[c] for c in mov_C]
    if static_pose:
        parts.append(np.stack([static_pose[c][:3, 3] for c in static_pose]))
    allC = np.concatenate(parts)
    cen = allC.mean(0); rad = float(np.linalg.norm(allC - cen, axis=1).max()) or 1.0; ssc = 1.0 / rad

    xyz, rgb = [], []
    for d_, pt, cf in zip(images, pts, confs):
        m = cf.detach().cpu().numpy().reshape(-1) > init_conf_thr
        xyz.append(pt.detach().cpu().numpy().reshape(-1, 3)[m])
        rgb.append(((d_["img"][0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8).reshape(-1, 3)[m])
    xyz = (np.concatenate(xyz) - cen) * ssc; rgb = np.concatenate(rgb)
    if len(xyz) > max_init_pts:
        s = np.random.default_rng(0).choice(len(xyz), max_init_pts, replace=False); xyz, rgb = xyz[s], rgb[s]

    def camjson(Rw2c, C, f):
        return {"orientation": Rw2c.tolist(), "position": C.tolist(), "focal_length": f,
                "principal_point": [W / 2., H / 2.], "image_size": [W, H], "skew": 0.,
                "pixel_aspect_ratio": 1., "radial_distortion": [0., 0., 0.], "tangential_distortion": [0., 0.]}

    ids, meta = [], {}
    for ti, tau in enumerate(taus):
        for ci in range(ncam):
            if ci in moving_cams:
                C = (mov_C[ci][ti] - cen) * ssc; Rw2c = mov_R[ci][ti].T; f = mov_focal[ci]
            else:
                c2w = static_pose[ci]; C = (c2w[:3, 3] - cen) * ssc; Rw2c = c2w[:3, :3].T; f = static_focal[ci]
            im = grab(ci, float(tau)); iid = f"cam{ci}_t{ti:03d}"
            cv2.imwrite(str(out_dir / "rgb" / "2x" / f"{iid}.png"), im)
            json.dump(camjson(Rw2c, C, f), open(out_dir / "camera" / f"{iid}.json", "w"))
            ids.append(iid); meta[iid] = {"camera_id": ci, "warp_id": ti, "appearance_id": ci, "time_id": ti}

    # Eval split. Default = held-out-TIME (every 10th timestep). If holdout_cams
    # is given, hold out those entire CAMERAS instead → the 4DGS eval becomes a
    # true held-out-VIEW (novel-view) PSNR rather than novel-time.
    if holdout_cams:
        val = [i for i in ids if meta[i]["camera_id"] in holdout_cams]
    else:
        val = [i for i in ids if int(i.split("t")[1]) % 10 == 9]
    json.dump(meta, open(out_dir / "metadata.json", "w"))
    json.dump({"ids": ids, "train_ids": [i for i in ids if i not in val], "val_ids": val}, open(out_dir / "dataset.json", "w"))
    json.dump({"near": 0.1, "far": 3.0, "scale": 1.0, "center": [0., 0., 0.]}, open(out_dir / "scene.json", "w"))
    v = np.zeros(len(xyz), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx', 'f4'), ('ny', 'f4'),
                                  ('nz', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    v['x'], v['y'], v['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    v['red'], v['green'], v['blue'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    PlyData([PlyElement.describe(v, 'vertex')]).write(str(out_dir / "points3D_downsample2.ply"))
    console.print(f"  [green]nerfies dataset: {len(ids)} imgs ({n_time} ts × {ncam} cams), "
                  f"moving={moving_cams or 'none'} → {out_dir}[/]")
    return {"n_images": len(ids), "n_cams": ncam, "static_focal": static_focal,
            "mov_focal": mov_focal, "moving_cams": list(moving_cams), "n_time": n_time}


def export_casual_viewer(out_dir: Path, *, label: str = "scene", fps: float | None = None) -> dict:
    """Assemble the browser viewer for a casual nerfies model: build scene_meta
    camera markers from the per-image camera/*.json (the casual path has no COLMAP
    sparse/), install the viewer HTML, convert baked keyframes. Run AFTER training+bake."""
    import shutil
    from videosplat.pipeline.export import export_keyframes
    meta = json.load(open(out_dir / "metadata.json"))
    by_cam: dict[int, list[str]] = {}
    for iid, m in meta.items():
        by_cam.setdefault(m["camera_id"], []).append(iid)
    for c in by_cam: by_cam[c].sort(key=lambda i: int(i.split("t")[1]))

    def marker(iid, name):
        c = json.load(open(out_dir / "camera" / f"{iid}.json"))
        R = np.array(c["orientation"])
        return {"pos": c["position"], "forward": R[2].tolist(), "up": (-R[1]).tolist(), "name": name}
    markers = []
    for c, lst in sorted(by_cam.items()):
        markers.append(marker(lst[0], f"cam{c}_a"))
        if len(lst) > 2:   # moving cam → also mid + end markers
            markers += [marker(lst[len(lst) // 2], f"cam{c}_mid"), marker(lst[-1], f"cam{c}_end")]
    cam0 = json.load(open(out_dir / "camera" / f"{by_cam[sorted(by_cam)[0]][0]}.json"))

    vdir = out_dir / "viewer"; vdir.mkdir(exist_ok=True)
    static = Path(__file__).resolve().parents[1] / "viewer" / "static"
    for a in static.iterdir():
        if a.is_file(): shutil.copy2(a, vdir / a.name)
    idx = vdir / "index.html"
    if idx.exists(): idx.write_text(idx.read_text().replace("{{SCENE_NAME}}", label.title()))

    n_kf = len(list((out_dir / "model" / "keyframes").glob("keyframe_*.ply")))
    return export_keyframes(
        keyframes_dir=out_dir / "model" / "keyframes", viewer_dir=vdir, sparse_dir=None,
        label=label, n_cameras=len(by_cam), fps=fps or max(0.5, round(n_kf / 107.0, 3)),
        image_height=int(cam0["image_size"][1]), image_width=int(cam0["image_size"][0]),
        focal_length=float(cam0["focal_length"]), extra_cameras=markers,
    )
