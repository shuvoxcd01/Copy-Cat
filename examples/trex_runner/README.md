# TRex Runner example

Teach a policy to play Chrome's offline dinosaur game (open `chrome://dino`) by imitating you.

## Setup

```bash
pip install -e '.[examples]'   # from the repo root; pulls in torch, tqdm and OpenCV
```

On macOS, give the terminal you run these scripts from these permissions:
- **Screen Recording**, so frames can be captured.
- **Input Monitoring**, so the toggle key and your key presses can be read.
- **Accessibility**, so `run_policy.py` and `probe_keypress.py` can press keys.

## Workflow

Run every script from the repo root, for example `python examples/trex_runner/record_demonstrations.py`.

| Step | Script | What it does |
|---|---|---|
| 1 | `record_demonstrations.py` | Press **9** to start, play with **space**, then press **9** to stop. The episode is preprocessed and saved under `examples/trex_runner/recordings/`. Repeat to record more episodes. |
| 2 | `replay_demonstrations.py [episode]` | Optional. Shows what an episode captured. Plays the latest episode unless you name one. |
| 3 | `train_policy.py` | Trains a CNN policy on every recorded episode and saves it to `examples/trex_runner/trained_policy.pt`. |
| 4 | `run_policy.py` | Focus the game and press **9**. The policy plays until you press **9** again. |
| - | `probe_keypress.py` | Troubleshooting. Sends five spaces with no policy involved, to check that key presses reach the game. |

All shared settings live in `config.py`: keys, toggle key, fps, monitor, paths and the preprocessing pipeline. Recording and playing both use `config.build_pipeline()`, so the policy always sees observations shaped the way it was trained on.
