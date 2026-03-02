from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

IN_DIR = Path("models/phase3_debug")
OUT_DIR = IN_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRED_PATH = IN_DIR / "val_predictions_debug.csv"
CONST_PATH = IN_DIR / "feature_constant_report.csv"
PER_TRIP_PATH = IN_DIR / "val_per_trip_mae.csv"

# Change this if you want a different trip highlighted
FOCUS_TRIP = "EV01_000302"


def savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    print("Wrote:", path)


def main():
    if not PRED_PATH.exists():
        raise FileNotFoundError(f"Missing {PRED_PATH}. Run training/run_phase3_debug.py first.")

    df = pd.read_csv(PRED_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values(["trip_id", "timestamp"])

    # Some runs may not include these columns; handle gracefully
    for c in ["abs_error"]:
        if c not in df.columns and "error" in df.columns:
            df["abs_error"] = df["error"].abs()

    # ------------------------------------------------------------
    # 1) Constant feature report (top 20 by zero_pct + constant)
    # ------------------------------------------------------------
    if CONST_PATH.exists():
        rep = pd.read_csv(CONST_PATH)
        top_const = rep.sort_values(["is_constant", "zero_pct"], ascending=[False, False]).head(25)

        plt.figure()
        plt.barh(top_const["col"][::-1], top_const["zero_pct"][::-1])
        plt.xlabel("Fraction of zeros")
        plt.title("Most-zero / constant features (top 25)")
        savefig(OUT_DIR / "01_constant_features_zero_pct.png")

    # ------------------------------------------------------------
    # 2) Motor temperature is stuck at 0 (distribution)
    # ------------------------------------------------------------
    if "motor_temperature_c" in df.columns:
        plt.figure()
        plt.hist(df["motor_temperature_c"].dropna(), bins=50)
        plt.xlabel("motor_temperature_c")
        plt.ylabel("count")
        plt.title("motor_temperature_c distribution (should not be all 0)")
        savefig(OUT_DIR / "02_motor_temperature_distribution.png")

    # ------------------------------------------------------------
    # 3) Per-trip MAE bar chart (top 20 worst)
    # ------------------------------------------------------------
    if PER_TRIP_PATH.exists():
        pt = pd.read_csv(PER_TRIP_PATH)
        # file has columns: trip_id, mae (or index saved); handle both formats
        if "trip_id" not in pt.columns:
            pt.columns = ["trip_id", "mae"]
        pt = pt.sort_values("mae", ascending=False).head(20)

        plt.figure()
        plt.barh(pt["trip_id"][::-1], pt["mae"][::-1])
        plt.xlabel("MAE (% SOC)")
        plt.title("Worst 20 trips by MAE (validation)")
        savefig(OUT_DIR / "03_worst_trips_mae.png")

    # ------------------------------------------------------------
    # 4) Error distribution (abs error histogram)
    # ------------------------------------------------------------
    if "abs_error" in df.columns:
        plt.figure()
        plt.hist(df["abs_error"].dropna(), bins=60)
        plt.xlabel("|pred - actual| (% SOC)")
        plt.ylabel("count")
        plt.title("Validation absolute error distribution")
        savefig(OUT_DIR / "04_abs_error_hist.png")

    # ------------------------------------------------------------
    # 5) Focus trip timeline: SOC, target, prediction
    # ------------------------------------------------------------
    if FOCUS_TRIP in set(df["trip_id"].unique()):
        g = df[df["trip_id"] == FOCUS_TRIP].dropna(subset=["timestamp"]).copy()

        # Some csvs include y_soc_t_plus_300s and pred
        ycol = "y_soc_t_plus_300s" if "y_soc_t_plus_300s" in g.columns else None
        if ycol is None:
            raise RuntimeError("Expected target column y_soc_t_plus_300s in val_predictions_debug.csv")

        plt.figure()
        plt.plot(g["timestamp"], g["soc_pct"], label="soc_pct (t)")
        plt.plot(g["timestamp"], g[ycol], label="target y_soc_t_plus_300s")
        plt.plot(g["timestamp"], g["pred"], label="pred")
        plt.xlabel("timestamp (UTC)")
        plt.ylabel("SOC (%)")
        plt.title(f"Trip timeline: {FOCUS_TRIP}")
        plt.legend()
        savefig(OUT_DIR / f"05_trip_{FOCUS_TRIP}_soc_pred.png")

        # --------------------------------------------------------
        # 6) Focus trip: sensor consistency (speed vs power/current)
        # --------------------------------------------------------
        cols = [c for c in ["motor_speed_rpm", "output_power_kw", "battery_current_a"] if c in g.columns]
        if len(cols) == 3:
            # speed vs power scatter
            plt.figure()
            plt.scatter(g["motor_speed_rpm"], g["output_power_kw"], s=8)
            plt.xlabel("motor_speed_rpm")
            plt.ylabel("output_power_kw")
            plt.title(f"{FOCUS_TRIP}: output_power_kw vs motor_speed_rpm (look for zeros while speed>0)")
            savefig(OUT_DIR / f"06_trip_{FOCUS_TRIP}_power_vs_speed.png")

            # speed vs current scatter
            plt.figure()
            plt.scatter(g["motor_speed_rpm"], g["battery_current_a"], s=8)
            plt.xlabel("motor_speed_rpm")
            plt.ylabel("battery_current_a")
            plt.title(f"{FOCUS_TRIP}: battery_current_a vs motor_speed_rpm (look for zeros while speed>0)")
            savefig(OUT_DIR / f"07_trip_{FOCUS_TRIP}_current_vs_speed.png")

        # --------------------------------------------------------
        # 7) Focus trip: where are the worst errors in time?
        # --------------------------------------------------------
        if "abs_error" in g.columns:
            worst = g.sort_values("abs_error", ascending=False).head(30).sort_values("timestamp")
            plt.figure()
            plt.plot(g["timestamp"], g["abs_error"])
            plt.scatter(worst["timestamp"], worst["abs_error"], s=18)
            plt.xlabel("timestamp (UTC)")
            plt.ylabel("|error| (% SOC)")
            plt.title(f"{FOCUS_TRIP}: absolute error over time (top 30 highlighted)")
            savefig(OUT_DIR / f"08_trip_{FOCUS_TRIP}_abs_error_timeline.png")

    else:
        print(f"NOTE: Focus trip {FOCUS_TRIP} not found in validation set. Update FOCUS_TRIP at top.")

    print("\nDone. Plots are in:", OUT_DIR)


if __name__ == "__main__":
    main()