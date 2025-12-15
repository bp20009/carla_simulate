# CARLA Simulation Utilities

## Introduction
`scripts/autopilot_simulation.py` provides a convenient way to launch an autopilot scenario in CARLA. The script connects to a running CARLA server, spawns vehicles with autopilot enabled, and continuously records their trajectories. Each run produces structured trajectory logs that can be inspected or visualised for further analysis.

`scripts/vehicle_state_stream.py` exposes a lightweight CLI for watching the world state in an existing CARLA simulation. It assigns a stable identifier to every `vehicle.*` actor discovered in the world and writes a CSV row per vehicle for each frame received from the simulator.

The stream can also merge per-actor control mode hints produced by the UDP replay utility. When a control state file is provided, each CSV row will include whether the actor is currently being driven by the replay PID controller (`tracking`) or handed off to CARLA's autopilot (`autopilot`). Position and rotation values always come directly from the CARLA server, so the stream reflects authoritative simulator poses even while ingesting external control metadata.

`scripts/plot_vehicle_trajectories.py` reads one of those CSV logs and renders a static XY plot of the actors' motion. You can filter by actor category, annotate actor IDs, and optionally save the figure to disk instead of opening an interactive window.

`scripts/animate_vehicle_trajectories.py` builds on the same CSV data to generate a Matplotlib animation. The command-line tool lets you configure the playback FPS, the amount of history to retain in the position trail, and the output resolution before exporting to formats supported by your Matplotlib installation (e.g. MP4, GIF).

## Prerequisites
- A running CARLA 0.10 server accessible from the machine executing the script.
- Python dependencies:
- `carla`
- `matplotlib` (required for the plotting and animation tools)
- `pillow` (optional, enables GIF export in the animation script)
- A working `ffmpeg` binary on your `PATH` for MP4 export from Matplotlib (optional but recommended)

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

### Replaying UDP tracking data with control state sharing
`scripts/udp_replay/replay_from_udp_carla_pred.py` consumes tracking messages over UDP, mirrors them into the CARLA world, and after a configurable tracking period hands control to CARLA's autopilot. It can optionally mirror each actor's control mode to disk so other tools (like the vehicle state stream) can annotate rows with the current tracking/autopilot status while continuing to read transforms from the simulator.

```bash
python scripts/udp_replay/replay_from_udp_carla_pred.py \
  --carla-host 127.0.0.1 \
  --carla-port 2000 \
  --listen-port 5005 \
  --control-state-file /tmp/control_state.json
```

Key options:
- `--control-state-file`: Optional JSON file that is updated with a mapping from CARLA actor ID to control flags (`autopilot_enabled`, `control_mode`). Downstream consumers can read this file between frames to detect when the replay switches from PID tracking to autopilot.
- `--enable-completion`: Fill in missing yaw/heading values based on motion direction when incoming data omits yaw.
- `--measure-update-times`: Emit per-frame update timing CSV for profiling replay performance.
- `--max-runtime`: Optional maximum duration before the replay loop exits.

## Streaming vehicle states to CSV
Use the vehicle state streaming tool to monitor the traffic in a running simulation:

```bash
python scripts/vehicle_state_stream.py \
  --host 127.0.0.1 \
  --port 2000 \
  --interval 0.5 \
  --control-state-file /tmp/control_state.json \
  --output vehicle_states.csv
```

Key options:
- `--host` / `--port`: CARLA server address to connect to.
- `--timeout`: Maximum time to wait for the client connection to establish (seconds).
- `--interval`: Optional delay (seconds) between snapshots to throttle logging.
- `--mode`: Choose `wait` (default) to block on `wait_for_tick`, or `on-tick` to register a `World.on_tick` listener.
- `--output`: Destination for the CSV log (`-` for `stdout`, default).
- `--wall-clock`: Add a `wall_time` column with the local UNIX timestamp for each frame.
- `--frame-elapsed`: Prepend a `frame_elapsed` column with the delta time reported by CARLA between frames.
- `--control-state-file`: Optional path to a JSON file that overrides the control/autopilot columns per actor. This is intended to be fed by `replay_from_udp_carla_pred.py` so mode transitions are visible in the CSV while poses remain sourced from CARLA.

Each CSV row includes the frame index reported by CARLA, the stable ID assigned by the script, the original CARLA actor ID, the vehicle blueprint, as well as its world-space location and rotation. Because the header is emitted once at startup and the writer flushes after every frame, the command can be safely redirected to a file or piped into another process. When `--wall-clock` is used, the `wall_time` column is prepended to the CSV. The plotting and animation utilities ignore extra columns, so either layout can be used with the downstream tools.
When `--frame-elapsed` is set, a `frame_elapsed` column is added before any optional timestamp fields, and it contains the delta time reported in each `WorldSnapshot`. Using either or both timestamp flags preserves the same column order across `wait` and `on-tick` modes, so downstream tools can skip unknown columns as before.

## Visualising trajectories from CSV logs

### Static plots
Use the plotting utility to quickly inspect recorded trajectories:

```bash
python scripts/plot_vehicle_trajectories.py vehicle_states.csv --only vehicle --save trajectories.png
```

Useful flags:
- `--only vehicle` or `--only walker`: Filter by actor categories found in the CSV.
- `--hide-ids`: Suppress text labels for actor IDs on the plot.
- `--no-endpoints`: Do not mark the start and end points of each trajectory.
- `--save`: Store the figure on disk instead of showing it interactively.

### Animated videos
Generate an animation that shows vehicles moving over time:

```bash
python scripts/animate_vehicle_trajectories.py vehicle_states.csv trajectories.mp4 --fps 15 --history 60
```

Useful flags:
- `--fps`: Playback speed in frames per second.
- `--history`: Number of past samples to keep in the trail (default: full history).
- `--dpi`: Resolution passed to Matplotlib when rendering each frame.
- `--only`: Filter to a subset of actor categories (e.g. only vehicles).
- Any output format supported by your Matplotlib writers can be used by changing the file extension (e.g. `.gif`).

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

Use `--frame-stride` with the frame-specific sender to skip frames while streaming. For example, `--frame-stride 5` sends only every fifth frame.

When replaying world snapshots captured by `scripts/vehicle_state_stream.py`, first reduce the dataset and then transmit the per-frame payloads:

```bash
python scripts/convert_vehicle_state_csv.py vehicle_states.csv vehicle_states_reduced.csv
python send_data/send_udp_frames_from_csv.py vehicle_states_reduced.csv --host 192.168.0.20 --port 5005
```

`send_udp_frames_from_csv.py` expects the reduced CSV layout generated by the converter (columns: `frame`, `id`, `type`, `x`, `y`, `z`) and sends one UDP datagram per frame containing the actors present in that snapshot.

## Replaying tracking data in CARLA
`scripts/udp_replay/replay_from_udp.py` listens for UDP messages describing actor positions and replays them inside a CARLA world. Run `python scripts/udp_replay/replay_from_udp.py --help` for the full list of options, including:

- `--enable-completion`: Default off. When set, the script fills in missing yaw/heading values using the direction of motion (calculated from position deltas).
