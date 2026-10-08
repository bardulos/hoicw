#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone interactive GUI for the HOI4 combat-width frontage estimator."""

import json
from itertools import product
import re
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


DATA_FILENAME = "data.json"
SCHEMA_VERSION = 1
TERRAIN_CATEGORIES = [
    "Desert", "Forest", "Hills", "Jungle", "Marsh", "Mountain", "Plains", "Urban"
]
ATTACK_DIRECTIONS = [1, 2, 3, 4]
DEFAULT_ATTACK_WEIGHTS = [40, 40, 15, 5]
BATTLE_WIDTHS = {
    "Desert": [70, 105, 140, 175],
    "Forest": [60, 90, 120, 150],
    "Hills": [70, 105, 140, 175],
    "Jungle": [60, 90, 120, 150],
    "Marsh": [50, 75, 100, 125],
    "Mountain": [50, 75, 100, 125],
    "Plains": [70, 105, 140, 175],
    "Urban": [80, 120, 160, 200],
}
MAX_DIVISION_WIDTH = 40
DEFENSIVE_DIVISIONS = 3
OFFENSIVE_DIVISIONS = 2
WATER_TERRAIN_TOKENS = ("sea", "ocean", "lake", "water", "coast", "river")


def is_water_terrain(raw_terrain):
    terrain = str(raw_terrain or "").lower()
    return any(token in terrain for token in WATER_TERRAIN_TOKENS)


def calculate_profile_scores(terrain_counts, attack_weights, objective="average"):
    """Return frontage-fit scores using the selected outcome objective."""
    if objective not in ("average", "worst", "balanced"):
        raise ValueError(f"Unknown outcome objective: {objective}")
    weight_values = np.asarray(attack_weights, dtype=float)
    weight_total = weight_values.sum()
    if weight_total <= 0:
        raise ValueError("Set a non-zero weight for at least one attack direction.")
    weight_values /= weight_total

    scenario_widths = []
    scenario_weights = []
    for terrain in TERRAIN_CATEGORIES:
        province_count = max(0, int(terrain_counts.get(terrain, 0)))
        scenario_widths.extend(BATTLE_WIDTHS[terrain])
        scenario_weights.extend(province_count * weight_values)

    scenario_widths = np.asarray(scenario_widths, dtype=float)
    scenario_weights = np.asarray(scenario_weights, dtype=float)
    if scenario_weights.sum() <= 0:
        raise ValueError("Enter at least one recognized land province.")
    scenario_weights /= scenario_weights.sum()

    widths = np.arange(1, MAX_DIVISION_WIDTH + 1, dtype=int)

    def profile(base_divisions):
        divisions_per_scenario = np.full(scenario_widths.shape, base_divisions, dtype=float)
        while np.any(scenario_widths / divisions_per_scenario > MAX_DIVISION_WIDTH):
            too_wide = scenario_widths / divisions_per_scenario > MAX_DIVISION_WIDTH
            divisions_per_scenario[too_wide] *= 2

        max_total_width = scenario_widths * (1.0 + 0.33 / 1.5)
        max_active = np.floor(max_total_width[None, :] / widths[:, None])
        active_divisions = np.minimum(
            divisions_per_scenario[None, :], np.maximum(max_active, 1.0)
        )
        participation = active_divisions / divisions_per_scenario[None, :]
        total_width = active_divisions * widths[:, None]
        frontage_ratio = total_width / scenario_widths[None, :]
        filled_frontage = np.minimum(frontage_ratio, 1.0)
        excess_ratio = np.maximum(frontage_ratio - 1.0, 0.0)
        overwidth_penalty = np.minimum(1.5 * excess_ratio, 0.33)
        scores = participation * filled_frontage * (1.0 - overwidth_penalty)
        average_scores = scores @ scenario_weights
        if objective == "average":
            return average_scores
        active_scenarios = scenario_weights > 0
        worst_scores = np.min(scores[:, active_scenarios], axis=1)
        if objective == "worst":
            return worst_scores
        return np.sqrt(np.maximum(average_scores, 0.0) * np.maximum(worst_scores, 0.0))

    return widths, profile(DEFENSIVE_DIVISIONS), profile(OFFENSIVE_DIVISIONS)


def calculate_weight_sweep(terrain_counts, values=range(0, 101, 10), objective="average"):
    """Evaluate every 10%-step four-way distribution whose normalized weights sum to 100%."""
    if objective not in ("average", "worst", "balanced"):
        raise ValueError(f"Unknown outcome objective: {objective}")
    province_counts = np.asarray(
        [max(0, int(terrain_counts.get(terrain, 0))) for terrain in TERRAIN_CATEGORIES],
        dtype=float,
    )
    total_provinces = province_counts.sum()
    if total_provinces <= 0:
        raise ValueError("Enter at least one recognized land province before running the sweep.")

    candidates = np.asarray(list(product(values, repeat=4)), dtype=int)
    combinations = candidates[candidates.sum(axis=1) == 100].astype(float)
    normalized_directions = combinations / 100.0
    scenario_weights = np.concatenate(
        [normalized_directions * count for count in province_counts], axis=1
    ) / total_provinces

    scenario_widths = np.asarray(
        [width for terrain in TERRAIN_CATEGORIES for width in BATTLE_WIDTHS[terrain]],
        dtype=float,
    )
    widths = np.arange(1, MAX_DIVISION_WIDTH + 1, dtype=int)

    def scenario_score_matrix(base_divisions):
        divisions_per_scenario = np.full(scenario_widths.shape, base_divisions, dtype=float)
        while np.any(scenario_widths / divisions_per_scenario > MAX_DIVISION_WIDTH):
            too_wide = scenario_widths / divisions_per_scenario > MAX_DIVISION_WIDTH
            divisions_per_scenario[too_wide] *= 2

        max_total_width = scenario_widths[None, :] * (1.0 + 0.33 / 1.5)
        max_active = np.floor(max_total_width / widths[:, None])
        active_divisions = np.minimum(
            divisions_per_scenario[None, :], np.maximum(max_active, 1.0)
        )
        participation = active_divisions / divisions_per_scenario[None, :]
        total_width = active_divisions * widths[:, None]
        frontage_ratio = total_width / scenario_widths[None, :]
        filled_frontage = np.minimum(frontage_ratio, 1.0)
        excess_ratio = np.maximum(frontage_ratio - 1.0, 0.0)
        overwidth_penalty = np.minimum(1.5 * excess_ratio, 0.33)
        return (participation * filled_frontage * (1.0 - overwidth_penalty)).T

    defensive_scenario_scores = scenario_score_matrix(DEFENSIVE_DIVISIONS)
    offensive_scenario_scores = scenario_score_matrix(OFFENSIVE_DIVISIONS)
    active_scenarios = scenario_weights > 0

    def aggregate_outcomes(scenario_scores):
        average_scores = scenario_weights @ scenario_scores
        if objective == "average":
            return average_scores
        worst_scores = np.min(
            np.where(active_scenarios[:, :, None], scenario_scores[None, :, :], np.inf),
            axis=1,
        )
        if objective == "worst":
            return worst_scores
        return np.sqrt(np.maximum(average_scores, 0.0) * np.maximum(worst_scores, 0.0))

    defensive_scores = aggregate_outcomes(defensive_scenario_scores)
    offensive_scores = aggregate_outcomes(offensive_scenario_scores)
    defensive_best = widths[np.argmax(defensive_scores, axis=1)]
    offensive_best = widths[np.argmax(offensive_scores, axis=1)]

    def best_width_summary(best_widths):
        frequencies = np.bincount(best_widths, minlength=MAX_DIVISION_WIDTH + 1)
        return {
            "minimum": int(best_widths.min()),
            "maximum": int(best_widths.max()),
            "mode": int(np.argmax(frequencies)),
        }

    return {
        "widths": widths,
        "combination_count": len(combinations),
        "defensive_scores": defensive_scores,
        "offensive_scores": offensive_scores,
        "defensive_best": best_width_summary(defensive_best),
        "offensive_best": best_width_summary(offensive_best),
    }


def find_near_peak_interval(widths, scores, tolerance=0.05):
    """Find the maximum and its contiguous interval within tolerance of the maximum."""
    peak_index = int(np.argmax(scores))
    peak_score = float(scores[peak_index])
    threshold = peak_score * (1.0 - tolerance)
    left_index = peak_index
    right_index = peak_index
    while left_index > 0 and scores[left_index - 1] >= threshold:
        left_index -= 1
    while right_index < len(scores) - 1 and scores[right_index + 1] >= threshold:
        right_index += 1
    return {
        "best": int(widths[peak_index]),
        "low": int(widths[left_index]),
        "high": int(widths[right_index]),
        "peak": peak_score,
    }


def shade_interval_under_curve(axes, widths, scores, interval, color, alpha=0.18):
    """Shade a curve's area down to the x-axis only over its near-peak interval."""
    interval_mask = (widths >= interval["low"]) & (widths <= interval["high"])
    return axes.fill_between(
        widths[interval_mask],
        0,
        scores[interval_mask],
        color=color,
        alpha=alpha,
        linewidth=0,
    )


def find_second_peak(widths, scores):
    """Find the strongest distinct local peak separated from the global peak by a valley."""
    scores = np.asarray(scores, dtype=float)
    global_index = int(np.argmax(scores))
    peak_groups = []

    index = 0
    while index < len(scores):
        end = index
        while end + 1 < len(scores) and scores[end + 1] == scores[index]:
            end += 1
        if index > 0 and end < len(scores) - 1:
            if scores[index] > scores[index - 1] and scores[end] > scores[end + 1]:
                peak_groups.append((index, end, (index + end) // 2, float(scores[index])))
        index = end + 1

    primary_group = next(
        (group for group in peak_groups if group[0] <= global_index <= group[1]),
        None,
    )
    if primary_group is None:
        return None

    secondary_candidates = []
    for candidate in peak_groups:
        if candidate == primary_group:
            continue
        start = min(primary_group[2], candidate[2])
        end = max(primary_group[2], candidate[2])
        if end - start < 2:
            continue
        valley_offset = int(np.argmin(scores[start + 1:end]))
        valley_index = start + 1 + valley_offset
        valley_score = float(scores[valley_index])
        if valley_score < min(primary_group[3], candidate[3]):
            secondary_candidates.append((candidate, valley_index, valley_score))

    if not secondary_candidates:
        return None

    candidate, valley_index, valley_score = max(
        secondary_candidates, key=lambda item: item[0][3]
    )
    return {
        "width": int(widths[candidate[2]]),
        "score": candidate[3],
        "valley_width": int(widths[valley_index]),
        "valley_score": valley_score,
    }


def redistribute_attack_weights(current_weights, changed_index, new_value):
    """Pin the dragged slider and proportionally rebalance the others to total 100."""
    weights = list(current_weights)
    changed_value = max(0, min(100, int(round(float(new_value)))))
    other_indices = [index for index in range(len(weights)) if index != changed_index]
    remaining = 100 - changed_value
    previous_total = sum(weights[index] for index in other_indices)

    if previous_total > 0:
        exact_shares = [remaining * weights[index] / previous_total for index in other_indices]
        allocations = [int(share) for share in exact_shares]
        leftover = remaining - sum(allocations)
        remainders = [share - allocation for share, allocation in zip(exact_shares, allocations)]
        for offset in sorted(range(len(other_indices)), key=lambda item: remainders[item], reverse=True)[:leftover]:
            allocations[offset] += 1
    else:
        quotient, leftover = divmod(remaining, len(other_indices))
        allocations = [quotient + (1 if offset < leftover else 0) for offset in range(len(other_indices))]

    weights[changed_index] = changed_value
    for index, allocation in zip(other_indices, allocations):
        weights[index] = allocation
    return weights


def load_data_file():
    data_path = Path(__file__).resolve().with_name(DATA_FILENAME)
    with data_path.open("r", encoding="utf-8") as source:
        data = json.load(source)
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported data schema in {data_path}.")
    return data


class CombatWidthGUI:
    def __init__(self, root, data):
        self.root = root
        self.data = data
        self.counts = {category: 0 for category in TERRAIN_CATEGORIES}
        self.attack_weight_vars = []
        self.last_attack_weights = list(DEFAULT_ATTACK_WEIGHTS)
        self._syncing_attack_sliders = False
        self.objective_var = tk.StringVar(value="average")
        self.sweep_signature = None
        self.objective_var.trace_add("write", self._on_objective_changed)

        root.title("HOI4 Combat Width Estimator")
        root.geometry("1880x820")
        root.minsize(1180, 680)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        self.control_panel = ttk.Frame(root, padding=10)
        self.control_panel.grid(row=0, column=0, sticky="nsew")
        self.control_panel.columnconfigure(0, weight=1)

        self.chart_panel = ttk.Frame(root, padding=(0, 10, 10, 10))
        self.chart_panel.grid(row=0, column=1, sticky="nsew")
        self.chart_panel.rowconfigure(0, weight=1)
        self.chart_panel.columnconfigure(0, weight=1)
        self.chart_panel.columnconfigure(1, weight=1)

        self._build_area_controls()
        self._build_terrain_controls()
        self._build_attack_controls()
        self._build_objective_controls()
        self._build_chart()
        self._update_area_choices()
        self._refresh_chart()

    def _build_area_controls(self):
        area_frame = ttk.LabelFrame(self.control_panel, text="Terrain data source", padding=8)
        area_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        area_frame.columnconfigure(1, weight=1)

        ttk.Label(area_frame, text="Source").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=3)
        self.source_var = tk.StringVar(value="Continent")
        self.source_combo = ttk.Combobox(
            area_frame,
            textvariable=self.source_var,
            values=["Continent", "Strategic region", "Country", "Custom"],
            state="readonly",
            width=20,
        )
        self.source_combo.grid(row=0, column=1, sticky="ew", pady=3)
        self.source_combo.bind("<<ComboboxSelected>>", self._on_source_changed)

        ttk.Label(area_frame, text="Area").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=3)
        self.area_var = tk.StringVar()
        self.area_combo = ttk.Combobox(area_frame, textvariable=self.area_var, state="readonly", width=20)
        self.area_combo.grid(row=1, column=1, sticky="ew", pady=3)
        self.area_combo.bind("<<ComboboxSelected>>", self._on_area_changed)

    def _build_terrain_controls(self):
        terrain_frame = ttk.LabelFrame(self.control_panel, text="Terrain mix", padding=8)
        terrain_frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        terrain_frame.columnconfigure(0, weight=1)

        self.terrain_tree = ttk.Treeview(
            terrain_frame, columns=("count", "share"), show="tree headings", height=8, selectmode="browse"
        )
        self.terrain_tree.heading("#0", text="Terrain")
        self.terrain_tree.heading("count", text="Provinces")
        self.terrain_tree.heading("share", text="Share")
        self.terrain_tree.column("#0", width=130, stretch=True)
        self.terrain_tree.column("count", width=80, anchor="e", stretch=False)
        self.terrain_tree.column("share", width=70, anchor="e", stretch=False)
        self.terrain_tree.grid(row=0, column=0, columnspan=3, sticky="ew")
        self.terrain_tree.bind("<<TreeviewSelect>>", self._on_terrain_selected)

        ttk.Label(terrain_frame, text="Terrain").grid(row=1, column=0, sticky="w", pady=(8, 2))
        ttk.Label(terrain_frame, text="Count").grid(row=1, column=1, sticky="w", pady=(8, 2))
        self.terrain_var = tk.StringVar(value=TERRAIN_CATEGORIES[0])
        self.terrain_combo = ttk.Combobox(
            terrain_frame,
            textvariable=self.terrain_var,
            values=TERRAIN_CATEGORIES,
            state="readonly",
            width=14,
        )
        self.terrain_combo.grid(row=2, column=0, sticky="ew", padx=(0, 5))
        self.count_var = tk.StringVar(value="0")
        self.count_spin = ttk.Spinbox(terrain_frame, from_=0, to=1000000, textvariable=self.count_var, width=9)
        self.count_spin.grid(row=2, column=1, sticky="ew", padx=(0, 5))
        ttk.Button(terrain_frame, text="Set count", command=self._set_terrain_count).grid(
            row=2, column=2, sticky="ew"
        )

    def _build_attack_controls(self):
        attack_frame = ttk.LabelFrame(self.control_panel, text="Attack direction weights", padding=8)
        attack_frame.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        attack_frame.columnconfigure(1, weight=1)

        for row, (direction, default_weight) in enumerate(zip(ATTACK_DIRECTIONS, DEFAULT_ATTACK_WEIGHTS)):
            ttk.Label(attack_frame, text=f"{direction} direction{'s' if direction != 1 else ''}").grid(
                row=row, column=0, sticky="w", padx=(0, 6)
            )
            weight_var = tk.IntVar(value=default_weight)
            self.attack_weight_vars.append(weight_var)
            scale = tk.Scale(
                attack_frame,
                from_=0,
                to=100,
                orient="horizontal",
                resolution=1,
                showvalue=True,
                variable=weight_var,
                command=lambda value, index=row: self._on_attack_weight_changed(index, value),
                length=210,
                highlightthickness=0,
            )
            scale.grid(row=row, column=1, sticky="ew")

    def _build_objective_controls(self):
        objective_frame = ttk.LabelFrame(self.control_panel, text="Outcome objective", padding=8)
        objective_frame.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        ttk.Radiobutton(
            objective_frame,
            text="Maximize average outcome",
            value="average",
            variable=self.objective_var,
        ).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            objective_frame,
            text="Minimize worst-case loss",
            value="worst",
            variable=self.objective_var,
        ).grid(row=1, column=0, sticky="w")
        ttk.Radiobutton(
            objective_frame,
            text="Balanced",
            value="balanced",
            variable=self.objective_var,
        ).grid(row=2, column=0, sticky="w")

    def _on_objective_changed(self, *_trace_args):
        self.sweep_signature = None
        self._refresh_chart()
        self._refresh_sweep()

    def _build_chart(self):
        self.current_box = ttk.LabelFrame(self.chart_panel, text="Current slider mix", padding=4)
        self.current_box.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self.current_figure, self.axes = plt.subplots(figsize=(6.0, 6.0), dpi=90)
        self.canvas = FigureCanvasTkAgg(self.current_figure, master=self.current_box)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.current_box.rowconfigure(0, weight=1)
        self.current_box.columnconfigure(0, weight=1)

        self.sweep_box = ttk.LabelFrame(self.chart_panel, text="", padding=4)
        self.sweep_box.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        self.sweep_box.rowconfigure(0, weight=1)
        self.sweep_box.columnconfigure(0, weight=1)
        self.sweep_figure, self.sweep_axes = plt.subplots(figsize=(6.0, 5.8), dpi=90)
        self.sweep_canvas = FigureCanvasTkAgg(self.sweep_figure, master=self.sweep_box)
        self.sweep_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def _is_land_dominant_region(self, area):
        land_provinces = 0
        water_provinces = 0
        province_data = self.data.get("provinces", {})
        for province_id in area.get("province_ids", []):
            record = province_data.get(str(province_id), {})
            if is_water_terrain(record.get("raw_terrain")):
                water_provinces += 1
            elif record.get("category") in TERRAIN_CATEGORIES:
                land_provinces += 1
        return land_provinces > water_provinces

    def _area_mapping(self):
        source = self.source_var.get()
        if source == "Continent":
            return self.data.get("continents", {})
        if source == "Strategic region":
            return {
                name: area
                for name, area in self.data.get("regions", {}).items()
                if self._is_land_dominant_region(area)
            }
        if source == "Country":
            return {
                name: area
                for name, area in self.data.get("countries", {}).items()
                if not re.search(r"\$[^$]+\$", name)
            }
        return {}

    def _update_area_choices(self):
        areas = self._area_mapping()
        names = sorted(areas)
        self.area_combo.configure(values=names, state="readonly" if names else "disabled")
        if names:
            default_area = next(
                (name for name in names if self.source_var.get() == "Continent" and name.casefold() == "europe"),
                names[0],
            )
            self.area_var.set(default_area)
            self._load_selected_area()
        else:
            self.area_var.set("")
            self.counts = {category: 0 for category in TERRAIN_CATEGORIES}
            self._refresh_terrain_table()
            self._refresh_chart()

    def _on_source_changed(self, _event=None):
        self._update_area_choices()

    def _on_area_changed(self, _event=None):
        self._load_selected_area()

    def _load_selected_area(self):
        area = self._area_mapping().get(self.area_var.get())
        if not area:
            return
        stored_counts = area.get("terrain_counts", {})
        self.counts = {
            category: int(stored_counts.get(category, 0))
            for category in TERRAIN_CATEGORIES
        }
        self._refresh_terrain_table()
        self._refresh_chart()

    def _refresh_terrain_table(self):
        for item in self.terrain_tree.get_children():
            self.terrain_tree.delete(item)
        total = sum(max(0, count) for count in self.counts.values())
        for category in TERRAIN_CATEGORIES:
            count = max(0, int(self.counts.get(category, 0)))
            share = f"{count / total:.1%}" if total else "0.0%"
            self.terrain_tree.insert("", "end", iid=category, text=category, values=(count, share))
        self._refresh_sweep()

    def _refresh_sweep(self):
        signature = (
            self.objective_var.get(),
            tuple(int(self.counts.get(category, 0)) for category in TERRAIN_CATEGORIES),
        )
        if signature == self.sweep_signature:
            return

        objective_label = {
            "average": "weighted average",
            "worst": "worst-case maximin",
            "balanced": "balanced geometric mean",
        }[self.objective_var.get()]

        try:
            result = calculate_weight_sweep(self.counts, objective=self.objective_var.get())
        except ValueError as error:
            self.sweep_axes.clear()
            self.sweep_axes.set_title(str(error))
            self.sweep_canvas.draw_idle()
            return

        self.sweep_signature = signature
        defensive_best = result["defensive_best"]
        offensive_best = result["offensive_best"]

        widths = result["widths"]
        axes = self.sweep_axes
        axes.clear()
        axes.set_title(f"Attack-Weight Sweep — {objective_label.title()}")
        for scores, color, name in (
            (result["defensive_scores"], "C0", "Defense"),
            (result["offensive_scores"], "C1", "Offense"),
        ):
            low = np.min(scores, axis=0)
            p10 = np.quantile(scores, 0.10, axis=0)
            p25 = np.quantile(scores, 0.25, axis=0)
            median = np.median(scores, axis=0)
            p75 = np.quantile(scores, 0.75, axis=0)
            p90 = np.quantile(scores, 0.90, axis=0)
            high = np.max(scores, axis=0)
            peak_interval = find_near_peak_interval(widths, median)
            axes.fill_between(
                widths, low, high, color=color, alpha=0.05, linewidth=0, label="_nolegend_"
            )
            axes.fill_between(widths, p10, p90, color=color, alpha=0.09, linewidth=0, label="_nolegend_")
            axes.fill_between(widths, p25, p75, color=color, alpha=0.14, linewidth=0, label="_nolegend_")
            shade_interval_under_curve(axes, widths, median, peak_interval, color, alpha=0.12)
            best_widths = defensive_best if name == "Defense" else offensive_best
            aggregate_label = {
                "average": "average",
                "worst": "worst-case",
                "balanced": "balanced geometric",
            }[self.objective_var.get()]
            axes.plot(
                widths,
                median,
                color=color,
                linewidth=2,
                zorder=3,
                label=(f"{name} {aggregate_label} median; "
                       f"best width {best_widths['minimum']}-{best_widths['maximum']}w"),
            )

        axes.set_xlabel("Width of Each Division")
        axes.set_ylabel("Weighted Frontage-Fit Score")
        axes.set_xlim(0, MAX_DIVISION_WIDTH)
        axes.set_ylim(0, 1.05)
        axes.set_xticks(np.arange(0, MAX_DIVISION_WIDTH + 1, 2))
        axes.xaxis.set_minor_locator(plt.MultipleLocator(1))
        axes.tick_params(axis="x", which="both", top=False, labeltop=False)
        axes.spines["top"].set_visible(True)
        axes.grid(True, which="both", linestyle="--", linewidth=0.5)
        axes.legend(loc="upper left", fontsize=9, framealpha=0.92)
        self.sweep_canvas.draw()

    def _on_terrain_selected(self, _event=None):
        selection = self.terrain_tree.selection()
        if not selection:
            return
        category = selection[0]
        self.terrain_var.set(category)
        self.count_var.set(str(self.counts.get(category, 0)))

    def _set_terrain_count(self):
        category = self.terrain_var.get()
        try:
            count = int(self.count_var.get())
        except ValueError:
            messagebox.showerror("Invalid count", "Enter a whole number of provinces.", parent=self.root)
            return
        if count < 0:
            messagebox.showerror("Invalid count", "Province counts cannot be negative.", parent=self.root)
            return
        self.counts[category] = count
        self._refresh_terrain_table()
        self._refresh_chart()

    def _attack_weights(self):
        values = [max(0, int(variable.get())) for variable in self.attack_weight_vars]
        total = sum(values)
        if total == 0:
            return values, None
        proportions = [value / total for value in values]
        return values, proportions

    def _on_attack_weight_changed(self, changed_index, slider_value):
        if self._syncing_attack_sliders:
            return

        weights = redistribute_attack_weights(self.last_attack_weights, changed_index, slider_value)

        self._syncing_attack_sliders = True
        try:
            for variable, weight in zip(self.attack_weight_vars, weights):
                variable.set(weight)
        finally:
            self._syncing_attack_sliders = False

        self.last_attack_weights = weights
        self._refresh_chart()

    def _refresh_chart(self):
        if not hasattr(self, "axes"):
            return
        raw_weights, normalized_weights = self._attack_weights()
        if normalized_weights is None:
            self.axes.clear()
            self.axes.set_title("No attack-direction weight selected")
            self.canvas.draw_idle()
            return

        try:
            widths, defensive_scores, offensive_scores = calculate_profile_scores(
                self.counts, raw_weights, objective=self.objective_var.get()
            )
        except ValueError as error:
            self.axes.clear()
            self.axes.set_title("Set a terrain mix to calculate widths")
            self.canvas.draw_idle()
            return

        defensive = find_near_peak_interval(widths, defensive_scores)
        offensive = find_near_peak_interval(widths, offensive_scores)
        defensive_secondary = find_second_peak(widths, defensive_scores)
        offensive_secondary = find_second_peak(widths, offensive_scores)

        self.axes.clear()
        defensive_color = "C0"
        offensive_color = "C1"
        defensive_line, = self.axes.plot(
            widths,
            defensive_scores,
            color=defensive_color,
            linewidth=2,
            label=f"Defense: {defensive['low']}-{defensive['high']}w",
        )
        offensive_line, = self.axes.plot(
            widths,
            offensive_scores,
            color=offensive_color,
            linewidth=2,
            label=f"Offense: {offensive['low']}-{offensive['high']}w",
        )

        self.axes.set_ylim(0, 1.05)
        shade_interval_under_curve(self.axes, widths, defensive_scores, defensive, defensive_color)
        shade_interval_under_curve(self.axes, widths, offensive_scores, offensive, offensive_color)

        objective_label = {
            "average": "Weighted Average",
            "worst": "Worst-Case Maximin",
            "balanced": "Balanced Geometric Mean",
        }[self.objective_var.get()]
        self.axes.set_title(f"Estimated Combat Width Fit — {objective_label}")
        self.axes.set_xlabel("Width of Each Division")
        self.axes.set_ylabel("Weighted Frontage-Fit Score")
        self.axes.set_xlim(0, MAX_DIVISION_WIDTH)
        self.axes.set_ylim(0, 1.05)
        self.axes.set_xticks(np.arange(0, MAX_DIVISION_WIDTH + 1, 2))
        self.axes.xaxis.set_minor_locator(plt.MultipleLocator(1))
        self.axes.tick_params(axis="x", which="both", top=False, labeltop=False)
        self.axes.spines["top"].set_visible(True)
        self.axes.grid(True, which="both", linestyle="--", linewidth=0.5)
        self.axes.legend(
            handles=[defensive_line, offensive_line],
            loc="upper left",
            fontsize=8,
            framealpha=0.92,
        )
        self.canvas.draw_idle()

def main():
    try:
        data = load_data_file()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Could not load {DATA_FILENAME}: {error}")
        return 1

    root = tk.Tk()
    CombatWidthGUI(root, data)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
