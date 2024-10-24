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


camera0_pixel_move_results = []
camera0_pixel_move_results_without_ref = []
camera0_real_move_results = []
camera0_real_move_results_without_ref = []
camera0_is_temp_stable_results = []
temperature_results = []
camera0_times = []
for data in raw_data:
    camera0: dict = data["camera0"]

    pixel_move_result: dict = camera0["pixel_move_result"]
    pixel_move_result = dict(sorted(pixel_move_result.items(), key=lambda x: x[0]))
    camera0_pixel_move_results.append(
        [i for v in pixel_move_result.values() for i in v]
    )

    pixel_move_result_without_ref: dict = camera0["pixel_move_result_without_ref"]
    pixel_move_result_without_ref = dict(
        sorted(pixel_move_result_without_ref.items(), key=lambda x: x[0])
    )
    camera0_pixel_move_results_without_ref.append(
        [i for v in pixel_move_result_without_ref.values() for i in v]
    )

    real_move_result: dict = camera0["real_move_result"]
    real_move_result = dict(sorted(real_move_result.items(), key=lambda x: x[0]))
    camera0_real_move_results.append([i for v in real_move_result.values() for i in v])

    real_move_result_without_ref: dict = camera0["real_move_result_without_ref"]
    real_move_result_without_ref = dict(
        sorted(real_move_result_without_ref.items(), key=lambda x: x[0])
    )
    camera0_real_move_results_without_ref.append(
        [i for v in real_move_result_without_ref.values() for i in v]
    )

    temperature_results.append(data["temperature"])

    is_temp_stable: bool = data["is_temp_stable"]
    camera0_is_temp_stable_results.append([is_temp_stable])

    time = data["time"]
    camera0_times.append([time])


concat_results = np.concatenate(
    (
        camera0_pixel_move_results,
        camera0_pixel_move_results_without_ref,
        camera0_real_move_results,
        camera0_real_move_results_without_ref,
        camera0_is_temp_stable_results,
    ),
    axis=1,
)


pixel_move_columns = [
    [f"pixel_move_{i}_x", f"pixel_move_{i}_y"]
    for i in range(len(camera0_pixel_move_results[0]) // 2)
]
pixel_move_columns = [item for sublist in pixel_move_columns for item in sublist]

pixel_move_without_ref_columns = [
    [f"pixel_move_without_ref_{i}_x", f"pixel_move_without_ref_{i}_y"]
    for i in range(len(camera0_pixel_move_results_without_ref[0]) // 2)
]
pixel_move_without_ref_columns = [item for sublist in pixel_move_without_ref_columns for item in sublist]

real_move_columns = [
    [f"real_move_{i}_x", f"real_move_{i}_y"]
    for i in range(len(camera0_real_move_results[0]) // 2)
]
real_move_columns = [item for sublist in real_move_columns for item in sublist]

real_move_without_ref_columns = [
    [f"real_move_without_ref_{i}_x", f"real_move_without_ref_{i}_y"]
    for i in range(len(camera0_real_move_results_without_ref[0]) // 2)
]
real_move_without_ref_columns = [item for sublist in real_move_without_ref_columns for item in sublist]

df1 = pd.DataFrame(
    concat_results,
    columns=pixel_move_columns
    + pixel_move_without_ref_columns
    + real_move_columns
    + real_move_without_ref_columns
    + [
        "is_temp_stable",
    ],
)

temp_columns = ["inside", "outside", "CPU", "holder", "lens", "undefined1", "undefined2", "undefined3"]
if len(temperature_results[0]) == 8:
    df2 = pd.DataFrame(temperature_results)
    df2.columns = temp_columns
else:
    df2 = pd.DataFrame()

df3 = pd.DataFrame(camera0_times, columns=["time"])

if len(temperature_results[0]) > 0:
    df = pd.concat([df1, df2, df3], axis=1)
else:
    df = pd.concat([df1, df3], axis=1)
print(df.head())

df.to_csv(csv_path)
print("Saved to", csv_path)
