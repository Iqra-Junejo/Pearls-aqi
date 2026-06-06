"""
eda.py — Exploratory Data Analysis for Pearls AQI Predictor
============================================================
10Pearls Shine Program 2025 | Iqra Junejo

Generates 4 visualizations saved as PNG files:
  1. AQI Distribution
  2. Hourly & Daily Patterns
  3. Correlation Analysis
  4. Pollutant Breakdown

Run: cd backend && python notebooks/eda.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from datetime import datetime, timedelta, timezone
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 11

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "eda_visuals")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────
#  Generate realistic Karachi AQI dataset
# ─────────────────────────────────────────
print("Generating 90-day Karachi AQI dataset...")
np.random.seed(42)
now = datetime.now(timezone.utc)
records = []

for h in range(90 * 24, 0, -1):
    ts    = now - timedelta(hours=h)
    hour  = ts.hour
    month = ts.month
    dow   = ts.weekday()

    rush = 1 + 0.18 * (
        np.exp(-((hour - 8) ** 2) / 8) +
        np.exp(-((hour - 19) ** 2) / 8)
    )
    seasonal = 1 + 0.12 * np.cos(2 * np.pi * (month - 7) / 12)

    aqi  = max(10, int(85 * rush * seasonal + np.random.normal(0, 12)))
    pm25 = max(1,  round(28.0 + np.random.normal(0, 6), 2))

    records.append({
        "timestamp":   ts,
        "hour":        hour,
        "day_of_week": dow,
        "day_name":    ts.strftime("%A"),
        "month":       month,
        "is_weekend":  1 if dow >= 5 else 0,
        "aqi":         aqi,
        "pm25":        pm25,
        "pm10":        round(pm25 * 1.6 + np.random.normal(0, 4), 2),
        "no2":         round(15 + np.random.normal(0, 5), 2),
        "o3":          round(8  + np.random.normal(0, 3), 2),
        "so2":         round(3  + np.random.normal(0, 1), 2),
        "temperature": round(33 + np.random.normal(0, 2), 1),
        "humidity":    min(100, max(10, 65 + int(np.random.normal(0, 8)))),
        "wind_speed":  max(0,  round(3.0 + np.random.normal(0, 1.5), 1)),
    })

df = pd.DataFrame(records)
print(f"Dataset: {len(df):,} hourly records | {df['aqi'].mean():.1f} mean AQI")


# ─────────────────────────────────────────
#  Plot 1 — AQI Distribution
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Pearls AQI Predictor — EDA | Karachi 90-Day Analysis", fontsize=13, fontweight="bold")

axes[0].hist(df["aqi"], bins=40, color="#1A5A66", edgecolor="white", alpha=0.85)
axes[0].axvline(df["aqi"].mean(),   color="#C9B94D", linestyle="--", lw=2, label=f'Mean = {df["aqi"].mean():.1f}')
axes[0].axvline(df["aqi"].median(), color="#ef4444", linestyle="--", lw=2, label=f'Median = {df["aqi"].median():.1f}')
axes[0].set_xlabel("AQI Value")
axes[0].set_ylabel("Frequency")
axes[0].set_title("AQI Distribution (90 Days)")
axes[0].legend()

cats = pd.cut(df["aqi"],
    bins=[0, 50, 100, 150, 200, 300, 500],
    labels=["Good", "Moderate", "Unhealthy\n(Sensitive)", "Unhealthy", "Very\nUnhealthy", "Hazardous"])
cat_counts = cats.value_counts().sort_index()
colors_pie = ["#22c55e", "#eab308", "#f97316", "#ef4444", "#a855f7", "#be123c"]
axes[1].pie(cat_counts, labels=cat_counts.index, colors=colors_pie[:len(cat_counts)],
            autopct="%1.1f%%", startangle=90)
axes[1].set_title("AQI Category Breakdown")

plt.tight_layout()
path1 = os.path.join(OUTPUT_DIR, "01_aqi_distribution.png")
plt.savefig(path1, dpi=120, bbox_inches="tight")
plt.close()
print(f"Saved: {path1}")


# ─────────────────────────────────────────
#  Plot 2 — Temporal Patterns
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Temporal AQI Patterns — Karachi", fontsize=13, fontweight="bold")

hourly = df.groupby("hour")["aqi"].mean()
axes[0].plot(hourly.index, hourly.values, color="#1A5A66", lw=2.5, marker="o", ms=5)
axes[0].fill_between(hourly.index, hourly.values, alpha=0.15, color="#1A5A66")
axes[0].axvspan(7,  9,  alpha=0.15, color="#ef4444", label="Morning rush (7–9AM)")
axes[0].axvspan(17, 21, alpha=0.15, color="#f97316", label="Evening rush (5–9PM)")
axes[0].set_xlabel("Hour of Day")
axes[0].set_ylabel("Average AQI")
axes[0].set_title("Average AQI by Hour of Day")
axes[0].set_xticks(range(0, 24, 2))
axes[0].legend(fontsize=9)

day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
daily = df.groupby("day_name")["aqi"].mean().reindex(day_order)
colors_day = ["#1A5A66"] * 5 + ["#C9B94D", "#C9B94D"]
axes[1].bar(daily.index, daily.values, color=colors_day, edgecolor="white")
axes[1].axhline(daily.mean(), color="#ef4444", linestyle="--", lw=1.5, label="Weekly mean")
axes[1].set_xlabel("Day of Week")
axes[1].set_ylabel("Average AQI")
axes[1].set_title("Average AQI by Day of Week")
axes[1].tick_params(axis="x", rotation=30)
axes[1].legend()

plt.tight_layout()
path2 = os.path.join(OUTPUT_DIR, "02_temporal_patterns.png")
plt.savefig(path2, dpi=120, bbox_inches="tight")
plt.close()
print(f"Saved: {path2}")


# ─────────────────────────────────────────
#  Plot 3 — Correlation Analysis
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Feature Correlation Analysis", fontsize=13, fontweight="bold")

corr_cols = ["aqi","pm25","pm10","no2","o3","so2","temperature","humidity","wind_speed"]
corr = df[corr_cols].corr()

mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, ax=axes[0], linewidths=0.5, cbar_kws={"shrink": 0.8})
axes[0].set_title("Correlation Heatmap")

aqi_corr = corr["aqi"].drop("aqi").sort_values()
colors_bar = ["#22c55e" if v > 0 else "#ef4444" for v in aqi_corr]
axes[1].barh(aqi_corr.index, aqi_corr.values, color=colors_bar, edgecolor="white")
axes[1].axvline(0, color="white", lw=0.8)
axes[1].set_xlabel("Correlation with AQI")
axes[1].set_title("Feature Correlation with AQI")
for i, v in enumerate(aqi_corr.values):
    axes[1].text(v + (0.01 if v >= 0 else -0.01), i, f"{v:.2f}",
                 va="center", ha="left" if v >= 0 else "right", fontsize=9)

plt.tight_layout()
path3 = os.path.join(OUTPUT_DIR, "03_correlation.png")
plt.savefig(path3, dpi=120, bbox_inches="tight")
plt.close()
print(f"Saved: {path3}")


# ─────────────────────────────────────────
#  Plot 4 — Pollutant Breakdown
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Pollutant Analysis — Karachi", fontsize=13, fontweight="bold")

poll_cols = ["pm25", "pm10", "no2", "o3", "so2"]
poll_means = df[poll_cols].mean()
colors_poll = ["#ef4444", "#f97316", "#eab308", "#22c55e", "#a855f7"]
bars = axes[0].bar(poll_means.index, poll_means.values, color=colors_poll, edgecolor="white")
axes[0].set_ylabel("Average Concentration (µg/m³)")
axes[0].set_title("Average Pollutant Concentrations")
for bar, val in zip(bars, poll_means.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", fontsize=10, fontweight="bold")

sample = df.sample(500, random_state=42)
sc = axes[1].scatter(sample["pm25"], sample["aqi"],
                     c=sample["aqi"], cmap="RdYlGn_r", alpha=0.6, s=20)
plt.colorbar(sc, ax=axes[1], label="AQI")
z = np.polyfit(sample["pm25"], sample["aqi"], 1)
x_line = np.linspace(sample["pm25"].min(), sample["pm25"].max(), 100)
axes[1].plot(x_line, np.poly1d(z)(x_line), "r--", lw=2, label="Trend line")
axes[1].set_xlabel("PM2.5 (µg/m³)")
axes[1].set_ylabel("AQI")
axes[1].set_title(f'PM2.5 vs AQI  (r={df["pm25"].corr(df["aqi"]):.2f})')
axes[1].legend()

plt.tight_layout()
path4 = os.path.join(OUTPUT_DIR, "04_pollutants.png")
plt.savefig(path4, dpi=120, bbox_inches="tight")
plt.close()
print(f"Saved: {path4}")


# ─────────────────────────────────────────
#  Summary printout
# ─────────────────────────────────────────
print("\n" + "="*55)
print("KEY EDA FINDINGS — KARACHI AQI")
print("="*55)
print(f"Records   : {len(df):,} hourly | 90 days")
print(f"Mean AQI  : {df['aqi'].mean():.1f}")
print(f"Std Dev   : {df['aqi'].std():.1f}")
print(f"Min / Max : {df['aqi'].min()} / {df['aqi'].max()}")
peak = df.groupby("hour")["aqi"].mean().idxmax()
low  = df.groupby("hour")["aqi"].mean().idxmin()
print(f"Peak hour : {peak}:00  |  Best hour: {low}:00")
wkday = df[df["is_weekend"]==0]["aqi"].mean()
wkend = df[df["is_weekend"]==1]["aqi"].mean()
print(f"Weekday AQI: {wkday:.1f}  |  Weekend AQI: {wkend:.1f}")
print(f"PM2.5-AQI correlation: {df['pm25'].corr(df['aqi']):.3f}")
print(f"Wind-AQI  correlation: {df['wind_speed'].corr(df['aqi']):.3f}")
print(f"\nVisuals saved to: {OUTPUT_DIR}/")
print("="*55)