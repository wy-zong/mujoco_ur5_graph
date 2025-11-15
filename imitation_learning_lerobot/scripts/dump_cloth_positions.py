# python dump_cloth_positions.py --xml /home/wuc120/benchmarking_cloth/bcm/assets/mujoco3_cloth.xml --steps 0 --csv cloth_positions.csv 
# dump_cloth_positions.py
import argparse
import re
import csv
from pathlib import Path

import mujoco


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True, help="Path to scene XML (or cloth XML)")
    ap.add_argument("--steps", type=int, default=0,
                    help="Extra mj_step to settle the cloth before reading (default: 0)")
    ap.add_argument("--csv", type=str, default="cloth_positions.csv",
                    help="Where to save the CSV (default: cloth_positions.csv)")
    return ap.parse_args()


def numeric_suffix(name: str):
    """
    取得像 'cloth_121' 的數字 121。若沒有數字，傳回 None。
    """
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else None


def main():
    args = parse_args()
    xml_path = Path(args.xml)

    if not xml_path.exists():
        raise FileNotFoundError(f"XML not found: {xml_path}")

    # Load model & data
    m = mujoco.MjModel.from_xml_path(str(xml_path))
    d = mujoco.MjData(m)

    # Forward once to compute world transforms
    mujoco.mj_forward(m, d)

    # Optional extra settling steps
    for _ in range(max(0, args.steps)):
        mujoco.mj_step(m, d)

    # Collect cloth_* bodies
    cloth_bodies = []
    for bid in range(m.nbody):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid)
        if not name:
            continue
        if name.startswith("cloth_"):
            idx = numeric_suffix(name)
            xpos = d.xpos[bid].copy()  # (3,): world position
            xmat = d.xmat[bid].copy()  # (9,): world rotation (row-major)
            cloth_bodies.append({
                "bid": bid,
                "name": name,
                "idx": idx,
                "x": xpos[0],
                "y": xpos[1],
                "z": xpos[2],
                # 也存旋轉，方便對齊檢查（可省略）
                "R00": xmat[0], "R01": xmat[1], "R02": xmat[2],
                "R10": xmat[3], "R11": xmat[4], "R12": xmat[5],
                "R20": xmat[6], "R21": xmat[7], "R22": xmat[8],
            })

    # 依數字索引排序（若沒有數字就放後面）
    cloth_bodies.sort(key=lambda r: (10**9 if r["idx"] is None else r["idx"], r["name"]))

    # Pretty print to console
    print(f"Found {len(cloth_bodies)} cloth bodies")
    print(f"{'ord':>4} {'bid':>4} {'name':>12} {'idx':>5} | {'x':>10} {'y':>10} {'z':>10}")
    for k, row in enumerate(cloth_bodies):
        print(f"{k:>4} {row['bid']:>4} {row['name']:>12} {str(row['idx']):>5} | "
              f"{row['x']:>10.6f} {row['y']:>10.6f} {row['z']:>10.6f}")

    # Save to CSV
    csv_path = Path(args.csv)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ord", "bid", "name", "idx", "x", "y", "z",
                "R00", "R01", "R02", "R10", "R11", "R12", "R20", "R21", "R22"
            ]
        )
        writer.writeheader()
        for k, row in enumerate(cloth_bodies):
            writer.writerow({
                "ord": k, "bid": row["bid"], "name": row["name"], "idx": row["idx"],
                "x": row["x"], "y": row["y"], "z": row["z"],
                "R00": row["R00"], "R01": row["R01"], "R02": row["R02"],
                "R10": row["R10"], "R11": row["R11"], "R12": row["R12"],
                "R20": row["R20"], "R21": row["R21"], "R22": row["R22"],
            })
    print(f"\nCSV saved to: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
