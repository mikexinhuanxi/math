# -*- coding: utf-8 -*-
"""Generate supplemental explanation figures for question 2.

The main model script already produces the core plots. This helper adds two
report-facing figures that make the impact-law section easier to read:
front/middle/rear windbox average effects and key-variable partial dependence.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "第2问" / "结果数据"
ALIGNED_PATH = ROOT / "data" / "第2问" / "中间数据" / "滞后配准建模数据.csv"
MODEL_PATH = ROOT / "result" / "第2问" / "CO预测模型_过程变量模型.json"
FIG_DIR = ROOT / "figure" / "第2问" / "异常修正后建模" / "基础图"

SENS_PATH = DATA_DIR / "过程变量敏感性分析_过程变量模型.csv"
SEGMENT_CSV = DATA_DIR / "风箱分区平均影响.csv"
PDP_CSV = DATA_DIR / "关键变量偏依赖曲线.csv"

VALUE_COL = "增加1个历史标准差后的平均CO变化"


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        (
            "<style>"
            "text{font-family:Arial,Microsoft YaHei,sans-serif;font-size:13px;fill:#263238}"
            ".title{font-size:21px;font-weight:700}"
            ".subtitle{font-size:12px;fill:#607D8B}"
            ".axis{stroke:#607D8B;stroke-width:1}"
            ".grid{stroke:#ECEFF1;stroke-width:1}"
            ".zero{stroke:#455A64;stroke-width:1.2}"
            "</style>"
        ),
    ]


def fmt(value: float, digits: int = 1) -> str:
    if not np.isfinite(value):
        return ""
    if abs(value) >= 100:
        return f"{value:.0f}"
    return f"{value:.{digits}f}"


def windbox_no(name: str) -> int | None:
    match = re.search(r"(\d+)#风箱", str(name))
    return int(match.group(1)) if match else None


def segment_name(no: int) -> str:
    if 1 <= no <= 6:
        return "前段(1#-6#)"
    if 7 <= no <= 12:
        return "中段(7#-12#)"
    return "后段(13#-18#)"


def y_map(value: float, ymin: float, ymax: float, top: float, height: float) -> float:
    if ymax == ymin:
        return top + height / 2
    return top + (ymax - value) / (ymax - ymin) * height


def draw_segment_plot(segment_df: pd.DataFrame, out_path: Path) -> None:
    width, height = 980, 520
    left, right, top, bottom = 94, 42, 92, 86
    plot_w, plot_h = width - left - right, height - top - bottom
    groups = ["前段(1#-6#)", "中段(7#-12#)", "后段(13#-18#)"]
    series = [("风箱负压强度", "#1565C0", "负压增强"), ("风箱平均温度", "#EF6C00", "温度升高")]
    max_abs = float(segment_df["平均CO变化"].abs().max())
    bound = max(10.0, math.ceil(max_abs / 10.0) * 10.0)
    ymin, ymax = -bound * 1.18, bound * 1.18

    svg = svg_header(width, height)
    svg.append(f'<text x="{width/2:.0f}" y="38" text-anchor="middle" class="title">前/中/后风箱分区平均影响</text>')
    svg.append(
        f'<text x="{width/2:.0f}" y="62" text-anchor="middle" class="subtitle">'
        "基于过程变量模型敏感性：单个风箱变量增加1个历史标准差后的平均 CO 响应"
        "</text>"
    )

    for tick in np.linspace(-bound, bound, 5):
        y = y_map(float(tick), ymin, ymax, top, plot_h)
        cls = "zero" if abs(tick) < 1e-9 else "grid"
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" class="{cls}"/>')
        svg.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end">{fmt(float(tick), 0)}</text>')
    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" class="axis"/>')
    svg.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" class="axis"/>')
    svg.append(
        f'<text x="22" y="{top+plot_h/2:.1f}" transform="rotate(-90,22,{top+plot_h/2:.1f})" '
        'text-anchor="middle">平均 CO 变化 (mg/m^3)</text>'
    )

    group_w = plot_w / len(groups)
    bar_w = 72
    for gi, group in enumerate(groups):
        cx = left + group_w * (gi + 0.5)
        svg.append(f'<text x="{cx:.1f}" y="{top+plot_h+34:.1f}" text-anchor="middle">{group}</text>')
        for si, (var_group, color, _) in enumerate(series):
            row = segment_df[(segment_df["分区"] == group) & (segment_df["变量组"] == var_group)]
            value = float(row.iloc[0]["平均CO变化"]) if not row.empty else 0.0
            x = cx + (si - 0.5) * (bar_w + 12)
            y0 = y_map(0, ymin, ymax, top, plot_h)
            y1 = y_map(value, ymin, ymax, top, plot_h)
            h = abs(y1 - y0)
            y = min(y0, y1)
            svg.append(f'<rect x="{x-bar_w/2:.1f}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" fill="{color}" opacity="0.86"/>')
            label_y = y - 8 if value >= 0 else y + h + 18
            svg.append(f'<text x="{x:.1f}" y="{label_y:.1f}" text-anchor="middle">{fmt(value)}</text>')

    legend_x = left + plot_w - 240
    for i, (_, color, label) in enumerate(series):
        x = legend_x + i * 126
        svg.append(f'<rect x="{x}" y="74" width="18" height="12" fill="{color}" opacity="0.86"/>')
        svg.append(f'<text x="{x+25}" y="85">{label}</text>')

    svg.append("</svg>")
    out_path.write_text("\n".join(svg), encoding="utf-8")


def prepare_segment_effects() -> pd.DataFrame:
    sensitivity = pd.read_csv(SENS_PATH)
    data = sensitivity[sensitivity["变量组"].isin(["风箱负压强度", "风箱平均温度"])].copy()
    data["风箱编号"] = data["原始字段"].map(windbox_no)
    data = data.dropna(subset=["风箱编号"])
    data["风箱编号"] = data["风箱编号"].astype(int)
    data["分区"] = data["风箱编号"].map(segment_name)
    out = (
        data.groupby(["分区", "变量组"], as_index=False)
        .agg(
            平均CO变化=(VALUE_COL, "mean"),
            平均绝对CO变化=(VALUE_COL, lambda x: float(np.mean(np.abs(x)))),
            风箱数=("风箱编号", "count"),
        )
    )
    out["分区排序"] = out["分区"].map({"前段(1#-6#)": 1, "中段(7#-12#)": 2, "后段(13#-18#)": 3})
    out = out.sort_values(["分区排序", "变量组"]).drop(columns=["分区排序"])
    out.to_csv(SEGMENT_CSV, index=False, encoding="utf-8-sig")
    return out


def predict_from_model(frame: pd.DataFrame, model: dict) -> np.ndarray:
    features = model["特征字段"]
    x = frame[features].to_numpy(dtype=float)
    mean = np.array(model["标准化均值"], dtype=float)
    std = np.array(model["标准化标准差"], dtype=float)
    z = (x - mean) / np.where(std == 0, 1.0, std)
    linear = np.array(model["线性系数"], dtype=float)
    square = np.array(model["平方项系数"], dtype=float)
    return float(model["截距"]) + z @ linear + (z**2) @ square


def feature_for_field(features: list[str], field: str) -> str:
    matches = [feature for feature in features if feature.startswith(f"{field}__")]
    if not matches:
        raise KeyError(f"Cannot find lagged feature for {field}")
    return matches[0]


def prepare_partial_dependence() -> pd.DataFrame:
    with MODEL_PATH.open("r", encoding="utf-8") as f:
        model = json.load(f)

    aligned = pd.read_csv(ALIGNED_PATH)
    test = aligned[aligned["数据集"] == "测试集"].copy()
    features = model["特征字段"]
    key_fields = ["17#风箱负压强度", "1#大烟道温度", "18#风箱平均温度", "1#大烟道负压强度"]
    rows: list[dict[str, float | str]] = []

    for field in key_fields:
        feature = feature_for_field(features, field)
        idx = features.index(feature)
        x_series = test[feature].astype(float)
        lo, hi = np.nanquantile(x_series, [0.05, 0.95])
        if not np.isfinite(lo) or not np.isfinite(hi) or math.isclose(lo, hi):
            center = float(model["标准化均值"][idx])
            spread = float(model["标准化标准差"][idx])
            lo, hi = center - 2 * spread, center + 2 * spread
        grid = np.linspace(float(lo), float(hi), 61)

        baseline_frame = test.copy()
        baseline_frame[feature] = float(model["标准化均值"][idx])
        baseline = float(np.mean(predict_from_model(baseline_frame, model)))
        for x_value in grid:
            varied = test.copy()
            varied[feature] = x_value
            pdp = float(np.mean(predict_from_model(varied, model)) - baseline)
            rows.append({"变量": field, "模型特征": feature, "取值": x_value, "相对训练均值CO变化": pdp})

    out = pd.DataFrame(rows)
    out.to_csv(PDP_CSV, index=False, encoding="utf-8-sig")
    return out


def draw_partial_dependence(pdp_df: pd.DataFrame, out_path: Path) -> None:
    width, height = 1120, 760
    margin_x, margin_y = 74, 88
    panel_w, panel_h = 460, 250
    gap_x, gap_y = 82, 92
    color = "#00695C"

    svg = svg_header(width, height)
    svg.append(f'<text x="{width/2:.0f}" y="38" text-anchor="middle" class="title">关键变量偏依赖曲线</text>')
    svg.append(
        f'<text x="{width/2:.0f}" y="62" text-anchor="middle" class="subtitle">'
        "基于不含历史 CO 的过程变量模型；纵轴为相对训练均值取值的平均 CO 变化"
        "</text>"
    )

    variables = list(dict.fromkeys(pdp_df["变量"].tolist()))
    for i, variable in enumerate(variables):
        row, col = divmod(i, 2)
        left = margin_x + col * (panel_w + gap_x)
        top = margin_y + row * (panel_h + gap_y)
        data = pdp_df[pdp_df["变量"] == variable].copy()
        xs = data["取值"].to_numpy(dtype=float)
        ys = data["相对训练均值CO变化"].to_numpy(dtype=float)
        xmin, xmax = float(np.min(xs)), float(np.max(xs))
        max_abs = max(1.0, float(np.max(np.abs(ys))) * 1.16)
        ymin, ymax = -max_abs, max_abs

        svg.append(f'<text x="{left+panel_w/2:.1f}" y="{top-18:.1f}" text-anchor="middle" font-weight="700">{variable}</text>')
        for tick in np.linspace(ymin, ymax, 5):
            y = y_map(float(tick), ymin, ymax, top, panel_h)
            cls = "zero" if abs(tick) < max_abs * 1e-6 else "grid"
            svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+panel_w}" y2="{y:.1f}" class="{cls}"/>')
            svg.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end">{fmt(float(tick))}</text>')
        svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+panel_h}" class="axis"/>')
        svg.append(f'<line x1="{left}" y1="{top+panel_h}" x2="{left+panel_w}" y2="{top+panel_h}" class="axis"/>')

        points = []
        for x_value, y_value in zip(xs, ys):
            px = left + (x_value - xmin) / (xmax - xmin) * panel_w if xmax != xmin else left + panel_w / 2
            py = y_map(float(y_value), ymin, ymax, top, panel_h)
            points.append(f"{px:.1f},{py:.1f}")
        svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.4"/>')
        svg.append(f'<text x="{left}" y="{top+panel_h+24}" text-anchor="start">{fmt(xmin)}</text>')
        svg.append(f'<text x="{left+panel_w}" y="{top+panel_h+24}" text-anchor="end">{fmt(xmax)}</text>')
        svg.append(f'<text x="{left+panel_w/2:.1f}" y="{top+panel_h+44}" text-anchor="middle">变量取值</text>')
        svg.append(
            f'<text x="{left-54}" y="{top+panel_h/2:.1f}" transform="rotate(-90,{left-54},{top+panel_h/2:.1f})" '
            'text-anchor="middle">ΔCO (mg/m^3)</text>'
        )

    svg.append("</svg>")
    out_path.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    segment_df = prepare_segment_effects()
    draw_segment_plot(segment_df, FIG_DIR / "图2-9_前中后风箱分区平均影响.svg")
    pdp_df = prepare_partial_dependence()
    draw_partial_dependence(pdp_df, FIG_DIR / "图2-10_关键变量偏依赖曲线.svg")


if __name__ == "__main__":
    main()
