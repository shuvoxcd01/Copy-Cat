"""Watch a recorded TRex Runner episode, to check what a demonstration captured.

    python examples/trex_runner/replay_demonstrations.py            # latest episode
    python examples/trex_runner/replay_demonstrations.py <episode>  # a named one

Needs OpenCV: pip install 'copy-cat[replay]'.
"""

import argparse

import config
from copy_cat.storage.trajectory_store import TrajectoryStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("episode", nargs="?", help="episode name; defaults to the latest")
    parser.add_argument("--scale", type=float, default=2.0, help="window scale factor")
    args = parser.parse_args()

    store = TrajectoryStore(config.RECORDINGS_ROOT)
    episodes = store.list()

    if not episodes:
        raise SystemExit(
            f"No episodes in {str(config.RECORDINGS_ROOT)!r}; record one with "
            "record_demonstrations.py first."
        )

    episode = args.episode or episodes[-1]
    print("Stored episodes:", episodes)
    print("Playing:", episode)

    # record_demonstrations.py writes 4-deep grayscale stacks, so the stack is drawn frame
    # by frame; scale up, because 84x84 tiles are tiny on a screen.
    store.play(name=episode, scale=args.scale, is_framestacked=True)


if __name__ == "__main__":
    main()
