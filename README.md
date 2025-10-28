# CARLA Simulation Utilities

## Introduction
`scripts/autopilot_simulation.py` provides a convenient way to launch an autopilot scenario in CARLA. The script connects to a running CARLA server, spawns vehicles with autopilot enabled, and continuously records their trajectories. Each run produces structured trajectory logs that can be inspected or visualised for further analysis.

## Prerequisites
- A running CARLA 0.10 server accessible from the machine executing the script.
- Python dependencies:
  - `carla`
  - `matplotlib` (optional, enables trajectory plotting)

## Usage
Run the autopilot simulation script with the desired connection and scenario parameters:

```bash
python scripts/autopilot_simulation.py \
  --host 127.0.0.1 \
  --port 2000 \
  --vehicles 25 \
  --duration 120 \
  --output-dir runs/example \
  --log-level INFO \
  --plot-trajectories
```

Key options:
- `--host` / `--port`: CARLA server address (default: `127.0.0.1:2000`).
- `--vehicles`: Number of autopilot vehicles to spawn (default: 10).
- `--duration`: Duration of the simulation in seconds (default: 60).
- `--output-dir`: Directory for generated outputs (default: `outputs`).
- `--log-level`: Logging verbosity (`DEBUG`, `INFO`, `WARNING`, etc.).
- `--plot-trajectories`: Enable generation of trajectory plots (requires `matplotlib`).
- `--no-save-json`: Disable JSON export if you only need CSV logs.

Refer to `python scripts/autopilot_simulation.py --help` for the full list of flags.

## Outputs
By default, results are saved inside the specified `--output-dir` (or `outputs` if omitted):
- `trajectories.csv`: Tabular log of sampled vehicle positions, orientations, and velocities over time.
- `trajectories.json`: JSON representation of the recorded trajectories (omit with `--no-save-json`).
- `trajectories.png`: Optional plot generated when `--plot-trajectories` is provided.

Each run creates or reuses the output directory; existing files with the same names are overwritten unless the script provides rotation options. Review the generated CSV/JSON files to process trajectory data programmatically, and use the plot to quickly inspect vehicle paths.
