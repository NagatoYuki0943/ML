import numpy as np
import pandas as pd
import json
from pathlib import Path


result_path = Path("./results/history.jsonl")
csv_path = Path("./results/history.csv")

raw_data = []
with open(result_path, "r") as f:
    for line in f:
        raw_data.append(json.loads(line))
print("Number of data points:", len(raw_data))


pixel_move_results = []
pixel_move_results_without_ref = []
real_move_results = []
real_move_results_without_ref = []
is_temp_stable_results = []
times = []
for data in raw_data:
    camera0: dict = data["camera0"]

    pixel_move_result: dict = camera0["pixel_move_result"]
    pixel_move_result = dict(sorted(pixel_move_result.items(), key=lambda x: x[0]))
    pixel_move_results.append(
        [i for v in pixel_move_result.values() for i in v]
    )

    pixel_move_result_without_ref: dict = camera0["pixel_move_result_without_ref"]
    pixel_move_result_without_ref = dict(sorted(pixel_move_result_without_ref.items(), key=lambda x: x[0]))
    pixel_move_results_without_ref.append(
        [i for v in pixel_move_result_without_ref.values() for i in v]
    )

    real_move_result: dict = camera0["real_move_result"]
    real_move_result = dict(sorted(real_move_result.items(), key=lambda x: x[0]))
    real_move_results.append(
        [i for v in real_move_result.values() for i in v]
    )

    real_move_result_without_ref: dict = camera0["real_move_result_without_ref"]
    real_move_result_without_ref = dict(sorted(real_move_result_without_ref.items(), key=lambda x: x[0]))
    real_move_results_without_ref.append(
        [i for v in real_move_result_without_ref.values() for i in v]
    )

    is_temp_stable: bool = data["is_temp_stable"]
    is_temp_stable_results.append([is_temp_stable])

    time = data["time"]
    times.append([time])


concat_results = np.concatenate(
    (
        pixel_move_results,
        pixel_move_results_without_ref,
        real_move_results,
        real_move_results_without_ref,
        is_temp_stable_results,
        times,
    ),
    axis=1,
)

df = pd.DataFrame(
    concat_results,
    columns=[
        "pixel_move_0_x",
        "pixel_move_0_y",
        "pixel_move_1_x",
        "pixel_move_1_y",
        "pixel_move_without_ref_0_x",
        "pixel_move_without_ref_0_y",
        "pixel_move_without_ref_1_x",
        "pixel_move_without_ref_1_y",
        "real_move_0_x",
        "real_move_0_y",
        "real_move_1_x",
        "real_move_1_y",
        "real_move_without_ref_0_x",
        "real_move_without_ref_0_y",
        "real_move_without_ref_1_x",
        "real_move_without_ref_1_y",
        "is_temp_stable",
        "time",
    ],
)
print(df.head())

df.to_csv(csv_path)
print("Saved to", csv_path)
