import argparse
import nicegui
from nicegui import ui
from . import __version__

from .ui import *
from .extract import *

def main():
    parser = argparse.ArgumentParser(description="LeRobot Dataset Visualizer")
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to dataset"
    )
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument(
        "--version", action="version", version=f"{__version__}"
    )

    args = parser.parse_args()
    print(f"Starting visualizer for {args.dataset} on port {args.port}...")

    extract.open_set(args.dataset)
    episodes = extract.get_episodes()

    @ui.page("/")
    def index():
        selected_ep = {"value": episodes[0]}
        selected_src = {"value": "both"}

        with ui.row().classes("w-full items-center gap-4 p-4"):
            ui.select(
                options=episodes,
                value=episodes[0],
                label="Episode",
                on_change=lambda e: (
                    selected_ep.update(value=e.value),
                    graph_area.refresh(),
                ),
            ).classes("w-40")

            ui.toggle(
                {"state": "State", "action": "Action", "both": "State vs Action"},
                value="both",
                on_change=lambda e: (
                    selected_src.update(value=e.value),
                    graph_area.refresh(),
                ),
            )

        @ui.refreshable
        def graph_area():
            with ui.card().classes("w-full p-4"):
                ui_graph(
                    episode_index=selected_ep["value"], source=selected_src["value"]
                )

        graph_area()

    ui.run(
        port = args.port,
        title = "levi",
        reload = False,
        show = True,
        dark = True,
        show_welcome_message=False
    )

if __name__ == "__main__":
    main()
