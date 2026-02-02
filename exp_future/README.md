# exp_future

Experimental scripts used for CARLA studies.

## Time acceleration benchmark

`measure_time_acceleration.py` measures how much simulation time advances within
fixed wall-clock windows while sweeping actor counts and rendering modes.
It spawns autopilot vehicles (no external data required), runs CARLA in
synchronous mode, and reports per-run speedup.

Example:

```bash
python exp_future/measure_time_acceleration.py \
  --duration 15 \
  --actor-counts 0:50:10 \
  --output results/accel_benchmark.csv
```

Use `--render-only` or `--no-render-only` to restrict rendering modes, and
`--fixed-delta` to override the synchronous timestep if desired.

### Batch runner (Windows)

Run `exp_future/run_time_accel_benchmark.bat` after starting CARLA to execute
one sweep and save the CSV to `results/time_accel_benchmark.csv`.
