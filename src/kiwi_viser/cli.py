"""Customer-side entry points; authentication uses standard Hugging Face settings."""
from __future__ import annotations

import argparse
from pathlib import Path


def gaussian_limit(value: str) -> int | None:
    if value.lower() == "all":
        return None
    try:
        limit = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("use 'all' or a positive integer") from error
    if limit < 1:
        raise argparse.ArgumentTypeError("use 'all' or a positive integer")
    return limit


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore scene-aligned bimanual demonstrations.")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download a customer release from Hugging Face")
    download.add_argument("--repo-id", required=True)
    download.add_argument("--revision", default="v1")
    download.add_argument("--output", type=Path, default=Path(".data/dataset"))
    download.add_argument("--scene", help="Download one scene, for example scene_001")
    inspect = commands.add_parser("inspect", help="List scenes and episodes from local metadata")
    inspect.add_argument("root", type=Path)
    view = commands.add_parser("view", help="Start a local browser viewer")
    view.add_argument("root", type=Path)
    view.add_argument("--scene")
    view.add_argument("--episode", type=int, default=0, help="Scene-local episode index")
    view.add_argument("--host", default="127.0.0.1")
    view.add_argument("--port", type=int, default=8080)
    view.add_argument("--max-gaussians", type=gaussian_limit, default=None, metavar="all|N",
                      help="Render all Gaussians by default; set a positive cap for faster preview")
    view.add_argument("--render", choices=["splats", "points"], default="splats")
    args = parser.parse_args()
    try:
        if args.command == "download":
            from huggingface_hub import HfApi, snapshot_download
            # Resolve the mutable tag once so the entire snapshot has one revision.
            revision = HfApi().repo_info(args.repo_id, repo_type="dataset", revision=args.revision).sha
            prefix = f"lerobot/{args.scene}/**" if args.scene else "lerobot/**"
            path = snapshot_download(args.repo_id, repo_type="dataset", revision=revision,
                                     local_dir=args.output, allow_patterns=["README.md", prefix],
                                     max_workers=4)
            from .dataset import discover_datasets
            scenes = discover_datasets(Path(path))
            if not scenes:
                raise ValueError("No scene datasets found; check the release layout and --scene.")
            print(f"Downloaded {len(scenes)} scene(s) at commit {revision} to {Path(path).resolve()}")
        elif args.command == "inspect":
            from .dataset import discover_datasets
            for scene in discover_datasets(args.root):
                print(f"{scene.name}: {scene.world_frame}, {scene.fps:g} Hz")
                for episode in scene.episodes:
                    print(f"  {episode.index}: {episode.episode_id} | {episode.task} | "
                          f"{episode.length:,} frames | {episode.length / scene.fps:.1f} s")
        else:
            if not 1 <= args.port <= 65535:
                parser.error("--port must be between 1 and 65535")
            from .app import Viewer
            Viewer(args.root, scene_name=args.scene, episode_index=args.episode,
                   host=args.host, port=args.port, max_gaussians=args.max_gaussians,
                   render=args.render).run()
    except KeyboardInterrupt:
        pass
    except Exception as error:
        parser.exit(1, f"kiwi-viser: {error}\n")


if __name__ == "__main__":
    main()
