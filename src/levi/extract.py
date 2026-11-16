import base64
from io import BytesIO
import json
from pathlib import Path
from typing import Any
import av
import numpy as np
import pandas as pd
from PIL import Image

_dataset_root: Path | None = None
_parquet: pd.DataFrame | None = None
_episode_to_file_stem: dict[int, str] = {}
_tasks: dict[int, str] = {}


def open_set(path: str | Path) -> None:
  """Load a LeRobot dataset from its root folder or any subfolder/parquet inside it."""
  global _dataset_root, _parquet, _episode_to_file_stem, _tasks

  target = Path(path).resolve()
  if not target.exists():
    raise FileNotFoundError(f"Dataset path does not exist: {target}")

  # Walk up to find the dataset root containing 'data/' or 'videos/'
  search_dirs = [target, *target.parents] if target.is_dir() else list(target.parents)
  _dataset_root = None
  for candidate in search_dirs:
    if (candidate / "data").is_dir() or (candidate / "videos").is_dir():
      _dataset_root = candidate
      break

  if _dataset_root is None:
    _dataset_root = target if target.is_dir() else target.parent

  # Collect parquet files from single file, chunk folder, or full dataset
  if target.is_file():
    parquet_files = [target]
  elif target != _dataset_root and list(target.rglob("*.parquet")):
    parquet_files = sorted(target.rglob("*.parquet"))
  else:
    data_dir = _dataset_root / "data"
    parquet_files = sorted(
        data_dir.rglob("*.parquet")
        if data_dir.is_dir()
        else _dataset_root.rglob("*.parquet")
    )

  if not parquet_files:
    raise FileNotFoundError(f"No .parquet files found under {target}")

  dfs: list[pd.DataFrame] = []
  _episode_to_file_stem = {}

  for pf in parquet_files:
    df = pd.read_parquet(pf)
    for ep_id in df["episode_index"].unique():
      _episode_to_file_stem[int(ep_id)] = pf.stem
    dfs.append(df)

  _parquet = pd.concat(dfs, ignore_index=True)
  _tasks = _load_tasks(_dataset_root)


def _load_tasks(root: Path | None) -> dict[int, str]:
  """Read task descriptions from meta/tasks.parquet (LeRobot v3) or meta/tasks.jsonl (v2)."""
  if root is None:
    return {}

  tasks_parquet = root / "meta" / "tasks.parquet"
  if tasks_parquet.exists():
    tdf = pd.read_parquet(tasks_parquet)
    if "task_index" in tdf.columns:
      return {int(row["task_index"]): str(idx) for idx, row in tdf.iterrows()}
    if "task" in tdf.columns:
      return {int(i): str(row["task"]) for i, row in tdf.iterrows()}

  tasks_jsonl = root / "meta" / "tasks.jsonl"
  if tasks_jsonl.exists():
    mapping: dict[int, str] = {}
    with tasks_jsonl.open("r", encoding="utf-8") as f:
      for line in f:
        if line.strip():
          rec = json.loads(line)
          mapping[int(rec["task_index"])] = str(rec["task"])
    return mapping

  return {}


def _require_df() -> pd.DataFrame:
  if _parquet is None:
    raise RuntimeError("Dataset not loaded. Call open_set(path) first.")
  return _parquet


def get_episodes() -> list[int]:
  df = _require_df()
  return sorted(int(ep) for ep in df["episode_index"].unique())


def get_episode_trajectories(episode_index: int) -> dict[str, Any]:
  """Extract timestamps and per-joint time-series for a given episode."""
  df = _require_df()
  ep_df = df[df["episode_index"] == episode_index].sort_values("frame_index")

  if ep_df.empty:
    raise ValueError(f"Episode {episode_index} not found in dataset.")

  actions = np.stack(ep_df["action"].to_numpy())
  states = np.stack(ep_df["observation.state"].to_numpy())
  timestamps = np.round(ep_df["timestamp"].to_numpy(), 3).tolist()
  frame_indices = ep_df["frame_index"].astype(int).tolist()
  global_indices = ep_df["index"].astype(int).tolist()
  num_joints = states.shape[1]

  return {
      "episode_index": episode_index,
      "timestamps": timestamps,
      "frame_indices": frame_indices,
      "global_indices": global_indices,
      "num_joints": num_joints,
      "state_joints": {
          f"joint_{i}": np.round(states[:, i], 4).tolist()
          for i in range(num_joints)
      },
      "action_joints": {
          f"joint_{i}": np.round(actions[:, i], 4).tolist()
          for i in range(num_joints)
      },
  }


def find_episode_videos(episode_index: int) -> dict[str, Path]:
  """Locate all camera MP4 files corresponding to an episode inside videos/."""
  if _dataset_root is None:
    return {}

  videos_dir = _dataset_root / "videos"
  if not videos_dir.is_dir():
    return {}

  cameras: dict[str, Path] = {}
  ep_filename = f"episode_{episode_index:06d}.mp4"
  chunk_stem = _episode_to_file_stem.get(episode_index)

  for cam_dir in sorted(videos_dir.rglob("observation.images.*")):
    if not cam_dir.is_dir():
      continue

    cam_name = cam_dir.name.replace("observation.images.", "")

    if (cam_dir / ep_filename).exists():
      cameras[cam_name] = cam_dir / ep_filename
    elif chunk_stem:
      matches = list(cam_dir.rglob(f"{chunk_stem}.mp4"))
      if matches:
        cameras[cam_name] = matches[0]

  return cameras


def get_video_resolution(video_path: Path) -> tuple[int, int, int] | None:
  """Return (H, W, C) for a video stream."""
  try:
    with av.open(str(video_path)) as container:
      stream = container.streams.video[0]
      return (int(stream.height), int(stream.width), 3)
  except Exception:
    return None


def get_video_frame_b64(video_path: Path, frame_idx: int) -> str | None:
  """Decode a single AV1/MP4 frame using PyAV and return a base64 JPEG string."""
  try:
    with av.open(str(video_path)) as container:
      stream = container.streams.video[0]
      fps = float(stream.average_rate or 30)
      target_ts = int((frame_idx / fps) / stream.time_base)

      container.seek(target_ts, stream=stream, backward=True, any_frame=False)

      best_frame = None
      for frame in container.decode(video=0):
        if frame.pts is not None and frame.pts >= target_ts:
          best_frame = frame
          break
        best_frame = frame

      if best_frame is None:
        return None

      img: Image.Image = best_frame.to_image()
      buf = BytesIO()
      img.save(buf, format="JPEG", quality=85)
      b64 = base64.b64encode(buf.getvalue()).decode("ascii")
      return f"data:image/jpeg;base64,{b64}"
  except Exception as e:
    print(f"Frame decode error ({video_path.name} @ {frame_idx}): {e}")
    return None


def get_episode_metadata(episode_index: int) -> dict[str, Any]:
  """Return task instruction along with episode-level and per-timeframe tensor shapes."""
  df = _require_df()
  ep_df = df[df["episode_index"] == episode_index].sort_values("frame_index")

  if ep_df.empty:
    raise ValueError(f"Episode {episode_index} not found in dataset.")

  num_frames = len(ep_df)
  sample_state = np.asarray(ep_df["observation.state"].iloc[0])
  sample_action = np.asarray(ep_df["action"].iloc[0])

  task_idx = (
      int(ep_df["task_index"].iloc[0]) if "task_index" in ep_df.columns else 0
  )
  task_text = _tasks.get(task_idx, f"Task #{task_idx} (no description found)")

  timeframe_shapes: dict[str, tuple[int, ...]] = {
      "observation.state": tuple(sample_state.shape),
      "action": tuple(sample_action.shape),
  }
  episode_shapes: dict[str, tuple[int, ...]] = {
      "observation.state": (num_frames, *sample_state.shape),
      "action": (num_frames, *sample_action.shape),
  }

  for cam_name, video_path in find_episode_videos(episode_index).items():
    res = get_video_resolution(video_path)
    if res is not None:
      timeframe_shapes[f"image.{cam_name}"] = res
      episode_shapes[f"image.{cam_name}"] = (num_frames, *res)

  return {
      "task_index": task_idx,
      "task_instruction": task_text,
      "num_frames": num_frames,
      "timeframe_shapes": timeframe_shapes,
      "episode_shapes": episode_shapes,
  }
