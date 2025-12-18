# CARLAシミュレーション用ユーティリティ

## 概要
`scripts/autopilot_simulation.py` は、CARLAでオートパイロットシナリオを手軽に起動するスクリプトです。実行中のCARLAサーバに接続し、オートパイロットを有効にした車両をスポーンして軌跡を継続的に記録します。各実行では構造化された軌跡ログが生成され、確認や可視化に利用できます。

`scripts/vehicle_state_stream.py` は、既存のCARLAシミュレーション内の世界状態を監視するための軽量CLIです。ワールド内のすべての `vehicle.*` アクターに安定した識別子を割り当て、受信した各フレームごとに車両単位のCSV行を出力します。

このストリームは、UDPリプレイスクリプトが生成するアクターごとの制御モード情報も取り込めます。制御状態ファイルが与えられた場合、各CSV行にはPID制御による追従中（`tracking`）かCARLAのオートパイロットへ切り替わったか（`autopilot`）が記録されます。位置と回転は常にCARLAサーバから直接取得するため、外部の制御メタデータを取り込んでも、ストリームにはシミュレータの正しい姿勢が反映されます。

`scripts/plot_vehicle_trajectories.py` は、上記CSVログを読み込み静的なXYプロットとして描画します。アクター種別によるフィルタ、アクターIDの注釈、インタラクティブ表示の代わりにファイルへ保存するオプションなどを備えています。

`scripts/animate_vehicle_trajectories.py` は同じCSVデータを用いてMatplotlibアニメーションを生成します。CLIでは再生FPS、軌跡に残す履歴長、出力解像度などを設定でき、環境で利用可能なフォーマット（例: MP4, GIF）にエクスポートできます。

## 事前準備
- 実行マシンからアクセス可能なCARLA 0.10サーバ
- Python依存関係:
  - `carla`
  - `matplotlib`（プロット・アニメーションツールに必要）
  - `pillow`（任意。アニメーションをGIF出力する場合に必要）
- MatplotlibでMP4を出力する場合は、`PATH`上にある`ffmpeg`バイナリ（任意だが推奨）

## 使い方
接続先やシナリオのパラメータを指定してオートパイロットシミュレーションスクリプトを実行します。

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

主なオプション:
- `--host` / `--port`: CARLAサーバのアドレス（デフォルト: `127.0.0.1:2000`）。
- `--vehicles`: スポーンするオートパイロット車両数（デフォルト: 10）。
- `--duration`: シミュレーション時間（秒、デフォルト: 60）。
- `--output-dir`: 生成物の保存先ディレクトリ（デフォルト: `outputs`）。
- `--log-level`: ログの詳細度（`DEBUG`、`INFO`、`WARNING` など）。
- `--plot-trajectories`: 軌跡プロットの生成を有効化（`matplotlib`が必要）。
- `--no-save-json`: CSVのみ必要な場合はJSON出力を無効化。

全オプションは `python scripts/autopilot_simulation.py --help` を参照してください。

### UDPトラッキングデータのリプレイと制御状態共有
`scripts/udp_replay/replay_from_udp_carla_pred.py` は、UDP経由で受信したトラッキングメッセージをCARLAワールドに反映し、設定した条件でCARLAのオートパイロットへ制御を引き継ぎます。受信ペイロードが持つ `frame` を基準に切替タイミングや終了フレームを指定でき、オフライン評価で「何フレーム前に切り替えれば事故を避けられるか」を検証できます。任意で各アクターの制御モードをファイルへ書き出し、車両状態ストリームなどが追従/オートパイロットの切り替えを検出できるようにします。

```bash
python scripts/udp_replay/replay_from_udp_carla_pred.py \
  --carla-host 127.0.0.1 \
  --carla-port 2000 \
  --listen-port 5005 \
  --control-state-file /tmp/control_state.json
```

主なオプション:
- `--control-state-file`: CARLAアクターIDをキーに、`autopilot_enabled` と `control_mode` を保持するJSONファイルを更新します。リプレイがPID追従からオートパイロットへ切り替わったことを、下流のツールがこのファイルを読むことで検出できます。
- `--enable-completion`: 受信データにyawが欠落している場合、移動方向からヘディングを補完します。
- `--measure-update-times`: パフォーマンスプロファイル用にフレームごとの更新時間をCSVで出力します。
- `--switch-payload-frame`: 受信ペイロードの `frame` が指定値に達した瞬間に、全車両をオートパイロットへ切替えます。`--lead-time-sec` と `--end-payload-frame` を組み合わせると、終了フレームから逆算して切替フレームを自動計算できます。
- `--end-payload-frame`: 指定フレームに到達したらリプレイを終了します。リードタイム実験のためのバッチ実行に便利です。
- `--max-runtime`: リプレイループの最大実行時間を指定（任意）。
- `--actor-log` / `--id-map-file`: パスを指定すると、受信ペイロードの `frame` をキーに各アクターの姿勢と制御状態を `actors.csv` へ記録し、外部IDとCARLAアクターIDの対応表を `id_map.csv` として出力します（未指定なら無効）。`actors.csv` にはデバッグ用の `carla_frame` 列と `frame_source` 列（`payload`/`carla`）が含まれ、未来フェーズでペイロードが途切れてもCARLAフレームで時系列を継続します。

衝突評価のため、車両・自転車アクターには衝突センサが自動付与され、`pred_collisions.csv` に衝突イベントを記録します（ペイロードフレーム、ペイロードフレームの由来 `payload_frame_source`、CARLAフレーム、相手ID/種別、接触座標、強度、事故判定を含む）。事故判定には `ACCIDENT_THRESHOLD`、`VEHICLE_ONLY`、`COOLDOWN_SEC` のフィルタが適用され、同一フレーム内の多重衝突は最も強いもののみ残ります。

## 車両状態をCSVへストリーミング
実行中のシミュレーションを監視するには、車両状態ストリームツールを使用します。

```bash
python scripts/vehicle_state_stream.py \
  --host 127.0.0.1 \
  --port 2000 \
  --interval 0.5 \
  --control-state-file /tmp/control_state.json \
  --output vehicle_states.csv
```

主なオプション:
- `--host` / `--port`: 接続先CARLAサーバ。
- `--timeout`: クライアント接続の最大待ち時間（秒）。
- `--interval`: ログを間引くためのスナップショット間隔（秒、任意）。
- `--mode`: `wait`（デフォルト。`wait_for_tick` を使用）または `on-tick`（`World.on_tick` を登録）。
- `--output`: CSVの出力先（デフォルトは`stdout`、`-`指定）。
- `--wall-clock`: 各フレームにローカルのUNIX時刻 `wall_time` 列を追加。
- `--frame-elapsed`: 各 `WorldSnapshot` が報告する経過時間を `frame_elapsed` 列として先頭に追加。
- `--control-state-file`: アクターごとの制御/オートパイロット状態を上書きするJSONファイルのパス。`replay_from_udp_carla_pred.py` から供給されることを想定しており、ポーズはCARLAから取得しつつ、CSVでモード遷移を可視化できます。

各CSV行にはCARLAが報告するフレーム番号、スクリプトが付与する安定ID、元のCARLAアクターID、ブループリント、ワールド座標系の位置と回転が含まれます。ヘッダーは起動時に1度だけ出力され、各フレーム後にフラッシュされるため、ファイルへのリダイレクトや別プロセスへのパイプも安全です。`--wall-clock` を使うと `wall_time` 列が先頭に追加されます。プロット/アニメーションツールは余分な列を無視するため、どちらのカラム順でも下流ツールと併用できます。`--frame-elapsed` を指定すると、任意のタイムスタンプ列よりも前に `frame_elapsed` が追加され、各フレーム間のデルタ時間を保持します。`wait` と `on-tick` のどちらでも同じ列順が維持され、従来通り未知の列はスキップできます。

## CSVログから軌跡を可視化

### 静的プロット
記録した軌跡を手早く確認するには、プロットユーティリティを使用します。

```bash
python scripts/plot_vehicle_trajectories.py vehicle_states.csv --only vehicle --save trajectories.png
```

便利なフラグ:
- `--only vehicle` または `--only walker`: CSVに含まれるアクター種別でフィルタ。
- `--hide-ids`: プロット上のアクターIDテキストを非表示。
- `--no-endpoints`: 各軌跡の始点・終点を描かない。
- `--save`: インタラクティブ表示の代わりにファイルへ保存。

### アニメーション動画
時間とともに車両が動く様子を表示するにはアニメーションを生成します。

```bash
python scripts/animate_vehicle_trajectories.py vehicle_states.csv trajectories.mp4 --fps 15 --history 60
```

便利なフラグ:
- `--fps`: 再生フレームレート。
- `--history`: 軌跡に残す過去サンプル数（デフォルト: 全履歴）。
- `--dpi`: 各フレームを描画する際の解像度（DPI）。
- `--only`: アクター種別で絞り込み（例: vehicleのみ）。
- Matplotlibがサポートする任意の出力形式を、拡張子を変えることで利用可能（例: `.gif`）。

## 出力
デフォルトでは、指定した `--output-dir`（未指定の場合は `outputs`）配下に結果を保存します。
- `trajectories.csv`: 時系列で記録した車両位置・姿勢・速度の表形式ログ。
- `trajectories.json`: 軌跡のJSON表現（`--no-save-json` 指定で省略）。
- `trajectories.png`: `--plot-trajectories` を指定した場合に生成される軌跡プロット。

各実行では出力ディレクトリを作成（または再利用）し、同名ファイルが存在する場合はスクリプトがローテーションを提供しない限り上書きされます。生成されたCSV/JSONはプログラム的に処理でき、プロットは軌跡をすばやく確認するのに役立ちます。

## CSVデータをUDP送信
任意のCSV行をUDPで他プロセスへ送信するには `send_data/send_udp_from_csv.py` を使用します。

```bash
python send_data/send_udp_from_csv.py data.csv --host 192.168.0.20 --port 5005 --message-column payload
```

スクリプトは各CSV行をメッセージとして扱います。デフォルトでは行全体をJSON化して送信しますが、`--message-column` で特定のカラムを選択できます。`--interval` で一定時間遅延を入れるか、`--delay-column` で行ごとの遅延値を使用できます。

フレーム専用の送信では `--frame-stride` を使ってフレームを間引けます。例: `--frame-stride 5` は5フレームごとに送信します。

`scripts/vehicle_state_stream.py` で取得したワールドスナップショットを送信する場合は、まずデータセットを縮小し、フレームごとのペイロードを送信します。

```bash
python scripts/convert_vehicle_state_csv.py vehicle_states.csv vehicle_states_reduced.csv
python send_data/send_udp_frames_from_csv.py vehicle_states_reduced.csv --host 192.168.0.20 --port 5005
```

`send_udp_frames_from_csv.py` は、コンバータが生成する簡易CSV（列: `frame`, `id`, `type`, `x`, `y`, `z`）を前提とし、各フレームのアクター情報を1データグラムにまとめて送信します。

## トラッキングデータをCARLAで再生
`scripts/udp_replay/replay_from_udp.py` は、UDPメッセージで送られるアクター位置情報を受信し、CARLAワールド上で再生します。`python scripts/udp_replay/replay_from_udp.py --help` で利用可能な全オプションを確認してください。

- `--enable-completion`: デフォルトでは無効。設定すると、位置差分から移動方向を計算し、欠落したyaw/ヘディングを補完します。
