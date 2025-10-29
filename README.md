# CARLA Simulation Utilities

## Introduction
`scripts/autopilot_simulation.py` provides a convenient way to launch an autopilot scenario in CARLA. The script connects to a running CARLA server, spawns vehicles with autopilot enabled, and continuously records their trajectories. Each run produces structured trajectory logs that can be inspected or visualised for further analysis.

`scripts/vehicle_state_stream.py` exposes a lightweight CLI for watching the world state in an existing CARLA simulation. It assigns a stable identifier to every `vehicle.*` actor discovered in the world and writes a CSV row per vehicle for each frame received from the simulator.

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

## Streaming vehicle states to CSV
Use the vehicle state streaming tool to monitor the traffic in a running simulation:

```bash
python scripts/vehicle_state_stream.py \
  --host 127.0.0.1 \
  --port 2000 \
  --interval 0.5 \
  --output vehicle_states.csv
```

Key options:
- `--host` / `--port`: CARLA server address to connect to.
- `--timeout`: Maximum time to wait for the client connection to establish (seconds).
- `--interval`: Optional delay (seconds) between snapshots to throttle logging.
- `--output`: Destination for the CSV log (`-` for `stdout`, default).

Each CSV row includes the frame index reported by CARLA, the stable ID assigned by the script, the original CARLA actor ID, the vehicle blueprint, as well as its world-space location and rotation. Because the header is emitted once at startup and the writer flushes after every frame, the command can be safely redirected to a file or piped into another process.

## Outputs
By default, results are saved inside the specified `--output-dir` (or `outputs` if omitted):
- `trajectories.csv`: Tabular log of sampled vehicle positions, orientations, and velocities over time.
- `trajectories.json`: JSON representation of the recorded trajectories (omit with `--no-save-json`).
- `trajectories.png`: Optional plot generated when `--plot-trajectories` is provided.

Each run creates or reuses the output directory; existing files with the same names are overwritten unless the script provides rotation options. Review the generated CSV/JSON files to process trajectory data programmatically, and use the plot to quickly inspect vehicle paths.

## Sending CSV data over UDP
To stream arbitrary CSV rows to another process over UDP, use `send_data/send_udp_from_csv.py`:

```bash
python send_data/send_udp_from_csv.py data.csv --host 192.168.0.20 --port 5005 --message-column payload
```

The script treats each CSV row as a message. By default it serialises the entire row as JSON before transmitting it, but you can
pick a specific column with `--message-column`. Use `--interval` for a fixed delay between packets or `--delay-column` to use a per-row delay value stored in the CSV.
