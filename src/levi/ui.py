from nicegui import ui
from . import extract


def ui_graph(episode_index: int = 0, source: str = "state") -> None:
  """Render observation camera feeds, scrubbed joint trajectories, and bottom metadata footer."""
  data = extract.get_episode_trajectories(episode_index)
  meta = extract.get_episode_metadata(episode_index)

  timestamps = data["timestamps"]
  frame_indices = data["frame_indices"]
  global_indices = data["global_indices"]
  max_pos = len(timestamps) - 1

  videos = extract.find_episode_videos(episode_index)
  is_chunked_video = any(p.name.startswith("file-") for p in videos.values())

  # --- 1. Top Camera Strip ---
  cam_images: dict[str, ui.interactive_image] = {}
  with ui.row().classes("w-full justify-center gap-4 mb-2"):
    if not videos:
      ui.label("No observation videos found in dataset videos/ folder.").classes(
          "text-sm text-gray-500 italic"
      )
    else:
      for cam_name in videos:
        with ui.card().classes("p-2 items-center"):
          ui.label(cam_name).classes("text-xs font-mono font-bold mb-1")
          cam_images[cam_name] = ui.interactive_image().classes(
              "w-72 h-54 object-contain rounded bg-black"
          )

  # --- 2. Timeline Scrubber ---
  with ui.row().classes("w-full items-center gap-4 px-2"):
    frame_label = ui.label().classes("text-xs font-mono w-48")
    scrubber = (
        ui.slider(min=0, max=max_pos, value=0, step=1)
        .props("label-always")
        .classes("flex-1")
    )

  # --- 3. Joint Trajectories Graph with Vertical Scrubber Cursor ---
  series: list[dict] = []

  if source in ("state", "both"):
    for joint_name, values in data["state_joints"].items():
      series.append({
          "name": (
              f"{joint_name} (state)" if source == "both" else joint_name
          ),
          "type": "line",
          "data": values,
          "showSymbol": False,
      })

  if source in ("action", "both"):
    for joint_name, values in data["action_joints"].items():
      series.append({
          "name": (
              f"{joint_name} (action)" if source == "both" else joint_name
          ),
          "type": "line",
          "data": values,
          "showSymbol": False,
          "lineStyle": {"type": "dashed"} if source == "both" else {},
      })

  if series:
    series[0]["markLine"] = {
        "symbol": "none",
        "animation": False,
        "label": {"show": False},
        "lineStyle": {"color": "#ef4444", "width": 2},
        "data": [{"xAxis": 0}],
    }

  chart = ui.echart({
      "title": {
          "text": f"Episode {episode_index} — Joint Trajectories",
          "top": 4,
          "left": "center",
          "textStyle": {"color": "#e5e7eb"},
      },
      "tooltip": {"trigger": "axis"},
      "legend": {
          "top": 38,
          "type": "scroll",
          "textStyle": {"color": "#d1d5db"},
      },
      "grid": {
          "left": "4%",
          "right": "4%",
          "bottom": "15%",
          "top": 80,
      },
      "xAxis": {
          "type": "category",
          "name": "Time (s)",
          "data": timestamps,
      },
      "yAxis": {
          "type": "value",
          "name": "Angle / Position",
          "scale": True,
      },
      "dataZoom": [
          {"type": "inside", "start": 0, "end": 100},
          {"type": "slider", "start": 0, "end": 100},
      ],
      "series": series,
  }).classes("w-full h-96")

  # --- 4. Sync Handler ---
  def update_scrub(pos: int) -> None:
    pos = max(0, min(pos, max_pos))
    ep_frame = frame_indices[pos]
    video_frame = global_indices[pos] if is_chunked_video else ep_frame

    frame_label.text = f"Frame: {ep_frame} / {frame_indices[-1]} ({timestamps[pos]}s)"

    for cam_name, video_path in videos.items():
      src = extract.get_video_frame_b64(video_path, video_frame)
      if src:
        cam_images[cam_name].set_source(src)

    if chart.options["series"]:
      chart.options["series"][0]["markLine"]["data"] = [{"xAxis": pos}]
      chart.update()

  scrubber.on_value_change(lambda e: update_scrub(int(e.value)))

  # --- 5. Page Footer (Task Instruction & Differentiated Shapes) ---
  def format_shapes(shapes: dict[str, tuple[int, ...]]) -> str:
    return "  |  ".join(f"{k}: {v}" for k, v in shapes.items())

  with ui.element("footer").classes(
      "w-full mt-4 p-4 rounded-lg bg-gray-900 text-gray-200 border"
      " border-gray-700"
  ):
    with ui.column().classes("w-full gap-3"):
      with ui.row().classes("w-full items-center gap-2"):
        ui.badge(f"TASK #{meta['task_index']}", color="emerald").classes(
            "font-mono"
        )
        ui.label(meta["task_instruction"]).classes(
            "text-sm font-medium text-white"
        )

      ui.separator().classes("bg-gray-700")

      with ui.row().classes("w-full justify-between items-center gap-4"):
        with ui.row().classes("items-center gap-2"):
          ui.badge("SINGLE TIMEFRAME SHAPE (t)", color="blue").classes(
              "font-mono"
          )
          ui.label(format_shapes(meta["timeframe_shapes"])).classes(
              "text-xs font-mono text-blue-300"
          )

        with ui.row().classes("items-center gap-2"):
          ui.badge(
              f"EPISODE OVERALL SHAPE (T={meta['num_frames']})", color="purple"
          ).classes("font-mono")
          ui.label(format_shapes(meta["episode_shapes"])).classes(
              "text-xs font-mono text-purple-300"
          )

  # Render frame 0 immediately
  update_scrub(0)
