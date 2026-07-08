#!/usr/bin/env python3
"""
Timeseries giornaliera di POD, FAR e conteggi per soglie cumulative di ampiezza.

Metodo coerente con l'esempio ufficiale py-eddy-tracker "Parameter Histogram":
- l'ampiezza viene letta direttamente dal catalogo con dataset["amplitude"];
- l'ampiezza, memorizzata in metri, viene convertita in centimetri con * 100;
- il matching viene eseguito PRIMA della classificazione per ampiezza;
- per ogni gruppo (matched, missed, false alarms) si contano cumulativamente
  le osservazioni con amplitude >= 5, 10, 20 e 30 cm.

Definizioni condizionate per ampiezza:

POD(thr) = H_ref(thr) / [H_ref(thr) + M_ref(thr)]

  H_ref: matched eddies selezionati usando l'ampiezza del riferimento;
  M_ref: missed eddies selezionati usando l'ampiezza del riferimento.

FAR(thr) = FA_model(thr) / [H_model(thr) + FA_model(thr)]

  H_model: matched eddies selezionati usando l'ampiezza del modello;
  FA_model: false alarms selezionati usando l'ampiezza del modello.

Sono quindi presenti due conteggi degli hits, perché POD e FAR sono condizionati
rispettivamente nello spazio del riferimento e nello spazio del modello. Questo evita
di mescolare ampiezze osservate e simulate nello stesso denominatore.
"""

from pathlib import Path
from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netCDF4 import date2num, num2date
from py_eddy_tracker.generic import reverse_index
from py_eddy_tracker.observations.tracking import TrackEddiesObservations


# ============================================================================
# CONFIGURAZIONE
# ============================================================================

REFERENCE_NAME = "DUACS 1/8°"
LIFETIME_LABEL = "4 days"

# Soglie cumulative richieste: amplitude >= threshold
THRESHOLDS_CM = np.array([5.0, 10.0, 20.0])

# Lasciare None per usare tutto il periodo comune ai cataloghi.
# Per il periodo principale SWOT:
# START_DATE = "2023-08-01"
# END_DATE = "2025-05-01"
START_DATE = None
END_DATE = None

OUTPUT_DIR = Path("POD_FAR_amplitude_thresholds_28days_pyeddytracker")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIME_UNITS = "days since 1950-01-01"
TIME_CALENDAR = "standard"

# Riferimento SWOT/MIOST
FILENAME_REFERENCE_AC = (
    "/ec/res4/scratch/ita6648/cmems/eddy_output_north_atlantic/"
    "eddy_tracking/Anticyclonic_4days.nc"
)
FILENAME_REFERENCE_CC = (
    "/ec/res4/scratch/ita6648/cmems/eddy_output_north_atlantic/"
    "eddy_tracking/Cyclonic_4days.nc"
)

# GLORYS12V1
FILENAME_GLORYS12V1_AC = (
    "/ec/res4/scratch/ita6648/GLORYS12V1/ALL_FILES/"
    "eddy_output_north_atlantic_remapped_swot_grid/eddy_tracking/"
    "Anticyclonic_cmems_4days.nc"
)
FILENAME_GLORYS12V1_CC = (
    "/ec/res4/scratch/ita6648/GLORYS12V1/ALL_FILES/"
    "eddy_output_north_atlantic_remapped_swot_grid/eddy_tracking/"
    "Cyclonic_cmems_4days.nc"
)

# GLORYS2V4
FILENAME_GLORYS2V4_AC = (
    "/ec/res4/scratch/ita6648/GLORYS2V4/ALL_FILES/"
    "eddy_output_north_atlantic_remapped_1_8/eddy_tracking/"
    "Anticyclonic_cmems_4days.nc"
)
FILENAME_GLORYS2V4_CC = (
    "/ec/res4/scratch/ita6648/GLORYS2V4/ALL_FILES/"
    "eddy_output_north_atlantic_remapped_1_8/eddy_tracking/"
    "Cyclonic_cmems_4days.nc"
)

MODEL_COLORS = {
    "GLORYS12V1": "tab:blue",
    "GLORYS2V4": "tab:green",
}


# ============================================================================
# CARICAMENTO E FUNZIONI DI BASE
# ============================================================================

def load_catalogs():
    """Carica il catalogo di riferimento e i due cataloghi modello."""
    reference = {
        "AC": TrackEddiesObservations.load_file(FILENAME_REFERENCE_AC),
        "CC": TrackEddiesObservations.load_file(FILENAME_REFERENCE_CC),
    }

    models = {
        "GLORYS12V1": {
            "AC": TrackEddiesObservations.load_file(FILENAME_GLORYS12V1_AC),
            "CC": TrackEddiesObservations.load_file(FILENAME_GLORYS12V1_CC),
        },
        "GLORYS2V4": {
            "AC": TrackEddiesObservations.load_file(FILENAME_GLORYS2V4_AC),
            "CC": TrackEddiesObservations.load_file(FILENAME_GLORYS2V4_CC),
        },
    }
    return reference, models


def safe_ratio(numerator, denominator):
    """Restituisce NaN quando la metrica non è definita."""
    return numerator / denominator if denominator > 0 else np.nan


def amplitude_cm(dataset, indices=None):
    """
    Legge l'ampiezza secondo la sintassi py-eddy-tracker e la converte in cm.

    L'esempio ufficiale usa:
        dataset["amplitude"] * 100
    """
    values = np.ma.asarray(dataset["amplitude"], dtype=float)
    if indices is not None:
        values = values[indices]

    values = np.ma.filled(values, np.nan) * 100.0
    return np.asarray(values, dtype=float)


def cumulative_counts_ge(values_cm, thresholds_cm=THRESHOLDS_CM):
    """
    Conteggi cumulativi amplitude >= soglia.

    È l'equivalente numerico della coda superiore rappresentata nell'esempio
    py-eddy-tracker con un istogramma cumulative=-1, ma restituisce direttamente
    i conteggi esatti alle soglie richieste.
    """
    values_cm = np.asarray(values_cm, dtype=float)
    values_cm = values_cm[np.isfinite(values_cm)]

    return {
        float(threshold): int(np.count_nonzero(values_cm >= threshold))
        for threshold in thresholds_cm
    }


def common_day_range(reference, models):
    """Determina i giorni interi comuni a tutti i cataloghi."""
    catalogs = [reference["AC"], reference["CC"]]
    for model_catalogs in models.values():
        catalogs.extend([model_catalogs["AC"], model_catalogs["CC"]])

    # La property period è fornita direttamente da py-eddy-tracker.
    periods = [catalog.period for catalog in catalogs]
    start_day = int(np.ceil(max(float(period[0]) for period in periods)))
    end_day = int(np.floor(min(float(period[1]) for period in periods)))

    if START_DATE is not None:
        requested_start = int(
            np.ceil(
                date2num(
                    datetime.strptime(START_DATE, "%Y-%m-%d"),
                    units=TIME_UNITS,
                    calendar=TIME_CALENDAR,
                )
            )
        )
        start_day = max(start_day, requested_start)

    if END_DATE is not None:
        requested_end = int(
            np.floor(
                date2num(
                    datetime.strptime(END_DATE, "%Y-%m-%d"),
                    units=TIME_UNITS,
                    calendar=TIME_CALENDAR,
                )
            )
        )
        end_day = min(end_day, requested_end)

    if end_day < start_day:
        raise ValueError(
            "Il periodo richiesto non si sovrappone al periodo comune dei cataloghi."
        )

    return np.arange(start_day, end_day + 1, dtype=int)


# ============================================================================
# MATCHING E CLASSIFICAZIONE PER AMPIEZZA
# ============================================================================

def match_and_get_amplitudes(reference_day, model_day):
    """
    Esegue il matching giornaliero completo e restituisce le ampiezze dei gruppi.

    Il matching NON viene ripetuto per ciascuna soglia. Le soglie sono applicate
    dopo il matching ai gruppi già classificati.
    """
    n_reference = len(reference_day)
    n_model = len(model_day)

    if n_reference == 0 and n_model == 0:
        empty = np.empty(0, dtype=float)
        return {
            "hit_reference_cm": empty,
            "hit_model_cm": empty,
            "miss_reference_cm": empty,
            "false_alarm_model_cm": empty,
            "matched_total": 0,
            "reference_total": 0,
            "model_total": 0,
        }

    if n_reference == 0:
        empty = np.empty(0, dtype=float)
        return {
            "hit_reference_cm": empty,
            "hit_model_cm": empty,
            "miss_reference_cm": empty,
            "false_alarm_model_cm": amplitude_cm(model_day),
            "matched_total": 0,
            "reference_total": 0,
            "model_total": n_model,
        }

    if n_model == 0:
        empty = np.empty(0, dtype=float)
        return {
            "hit_reference_cm": empty,
            "hit_model_cm": empty,
            "miss_reference_cm": amplitude_cm(reference_day),
            "false_alarm_model_cm": empty,
            "matched_total": 0,
            "reference_total": n_reference,
            "model_total": 0,
        }

    reference_id, model_id, _ = reference_day.tracking(model_day)

    miss_reference_id = reverse_index(reference_id, n_reference)
    false_alarm_model_id = reverse_index(model_id, n_model)

    return {
        # Stessa coppia matched, ma ampiezza letta nei due cataloghi distinti.
        "hit_reference_cm": amplitude_cm(reference_day, reference_id),
        "hit_model_cm": amplitude_cm(model_day, model_id),
        "miss_reference_cm": amplitude_cm(reference_day, miss_reference_id),
        "false_alarm_model_cm": amplitude_cm(model_day, false_alarm_model_id),
        "matched_total": int(len(reference_id)),
        "reference_total": int(n_reference),
        "model_total": int(n_model),
    }


def threshold_counts(groups):
    """Calcola tutti i conteggi cumulativi dei quattro gruppi di ampiezza."""
    return {
        "hits_reference": cumulative_counts_ge(groups["hit_reference_cm"]),
        "hits_model": cumulative_counts_ge(groups["hit_model_cm"]),
        "misses_reference": cumulative_counts_ge(groups["miss_reference_cm"]),
        "false_alarms_model": cumulative_counts_ge(
            groups["false_alarm_model_cm"]
        ),
    }


def calculate_daily_timeseries(reference, models, days):
    """Calcola conteggi, POD e FAR giorno per giorno e per soglia."""
    records = []

    for day_number, day in enumerate(days, start=1):
        date_str = num2date(
            day, units=TIME_UNITS, calendar=TIME_CALENDAR
        ).strftime("%Y-%m-%d")

        reference_day = {
            polarity: reference[polarity].extract_with_period((day, day))
            for polarity in ("AC", "CC")
        }

        models_day = {
            model_name: {
                polarity: catalog[polarity].extract_with_period((day, day))
                for polarity in ("AC", "CC")
            }
            for model_name, catalog in models.items()
        }

        for model_name, model_day in models_day.items():
            # Matching completo, separato per polarità.
            groups_ac = match_and_get_amplitudes(
                reference_day["AC"], model_day["AC"]
            )
            groups_cc = match_and_get_amplitudes(
                reference_day["CC"], model_day["CC"]
            )

            counts_ac = threshold_counts(groups_ac)
            counts_cc = threshold_counts(groups_cc)

            for threshold in THRESHOLDS_CM:
                threshold = float(threshold)

                h_ref_ac = counts_ac["hits_reference"][threshold]
                h_ref_cc = counts_cc["hits_reference"][threshold]
                h_model_ac = counts_ac["hits_model"][threshold]
                h_model_cc = counts_cc["hits_model"][threshold]
                m_ref_ac = counts_ac["misses_reference"][threshold]
                m_ref_cc = counts_cc["misses_reference"][threshold]
                fa_model_ac = counts_ac["false_alarms_model"][threshold]
                fa_model_cc = counts_cc["false_alarms_model"][threshold]

                hits_reference = h_ref_ac + h_ref_cc
                hits_model = h_model_ac + h_model_cc
                misses_reference = m_ref_ac + m_ref_cc
                false_alarms_model = fa_model_ac + fa_model_cc

                reference_count_threshold = hits_reference + misses_reference
                model_count_threshold = hits_model + false_alarms_model

                records.append(
                    {
                        "date": date_str,
                        "threshold_cm": threshold,
                        "model": model_name,
                        # Conteggi per polarità, condizionati sul riferimento.
                        "hits_reference_ac": h_ref_ac,
                        "hits_reference_cc": h_ref_cc,
                        "misses_reference_ac": m_ref_ac,
                        "misses_reference_cc": m_ref_cc,
                        # Conteggi per polarità, condizionati sul modello.
                        "hits_model_ac": h_model_ac,
                        "hits_model_cc": h_model_cc,
                        "false_alarms_model_ac": fa_model_ac,
                        "false_alarms_model_cc": fa_model_cc,
                        # Totali usati nelle metriche.
                        "hits_reference": hits_reference,
                        "misses_reference": misses_reference,
                        "hits_model": hits_model,
                        "false_alarms_model": false_alarms_model,
                        "reference_count_threshold": reference_count_threshold,
                        "model_count_threshold": model_count_threshold,
                        # Conteggi completi prima della soglia.
                        "matched_total_before_threshold": (
                            groups_ac["matched_total"] + groups_cc["matched_total"]
                        ),
                        "reference_total_before_threshold": (
                            groups_ac["reference_total"]
                            + groups_cc["reference_total"]
                        ),
                        "model_total_before_threshold": (
                            groups_ac["model_total"] + groups_cc["model_total"]
                        ),
                        "POD": safe_ratio(
                            hits_reference,
                            hits_reference + misses_reference,
                        ),
                        "FAR": safe_ratio(
                            false_alarms_model,
                            hits_model + false_alarms_model,
                        ),
                    }
                )

        if day_number == 1 or day_number % 50 == 0 or day_number == len(days):
            print(f"Processati {day_number:4d}/{len(days)} giorni - {date_str}")

    df = pd.DataFrame.from_records(records)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "threshold_cm", "model"]).reset_index(
        drop=True
    )


# ============================================================================
# RIEPILOGO E OUTPUT
# ============================================================================

def build_summary(df):
    """Riepilogo per modello e soglia basato sui conteggi aggregati."""
    rows = []

    for (threshold_cm, model_name), group in df.groupby(
        ["threshold_cm", "model"], sort=True
    ):
        h_ref_total = int(group["hits_reference"].sum())
        m_ref_total = int(group["misses_reference"].sum())
        h_model_total = int(group["hits_model"].sum())
        fa_model_total = int(group["false_alarms_model"].sum())

        rows.append(
            {
                "threshold_cm": threshold_cm,
                "model": model_name,
                "days": int(group["date"].nunique()),
                "hits_reference_total": h_ref_total,
                "misses_reference_total": m_ref_total,
                "hits_model_total": h_model_total,
                "false_alarms_model_total": fa_model_total,
                "reference_eddies_total": h_ref_total + m_ref_total,
                "model_eddies_total": h_model_total + fa_model_total,
                "POD_global": safe_ratio(
                    h_ref_total, h_ref_total + m_ref_total
                ),
                "FAR_global": safe_ratio(
                    fa_model_total, h_model_total + fa_model_total
                ),
                "POD_daily_mean": group["POD"].mean(skipna=True),
                "FAR_daily_mean": group["FAR"].mean(skipna=True),
                "valid_POD_days": int(group["POD"].notna().sum()),
                "valid_FAR_days": int(group["FAR"].notna().sum()),
            }
        )

    return pd.DataFrame(rows).sort_values(["threshold_cm", "model"])


def save_tables(df, summary):
    """Salva CSV lungo, largo e riassuntivo."""
    long_path = OUTPUT_DIR / "POD_FAR_daily_amplitude_thresholds_long.csv"
    df.to_csv(long_path, index=False, float_format="%.6f")

    value_columns = [
        "hits_reference",
        "misses_reference",
        "hits_model",
        "false_alarms_model",
        "reference_count_threshold",
        "model_count_threshold",
        "POD",
        "FAR",
    ]

    wide = df.pivot(
        index="date",
        columns=["threshold_cm", "model"],
        values=value_columns,
    )
    wide.columns = [
        f"{metric}_thr{threshold:g}cm_{model}"
        for metric, threshold, model in wide.columns
    ]
    wide = wide.reset_index()

    wide_path = OUTPUT_DIR / "POD_FAR_daily_amplitude_thresholds_wide.csv"
    wide.to_csv(wide_path, index=False, float_format="%.6f")

    summary_path = OUTPUT_DIR / "POD_FAR_amplitude_thresholds_summary.csv"
    summary.to_csv(summary_path, index=False, float_format="%.6f")

    print(f"[OK] CSV giornaliero lungo: {long_path}")
    print(f"[OK] CSV giornaliero largo: {wide_path}")
    print(f"[OK] CSV riassuntivo:       {summary_path}")


def configure_plot_style():
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 15,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "axes.linewidth": 1.3,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.dpi": 150,
            "savefig.dpi": 600,
        }
    )


def format_time_axis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="both", direction="out", length=5, width=1.1)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def models_order(df):
    preferred = ["GLORYS12V1", "GLORYS2V4"]
    available = list(df["model"].drop_duplicates())
    return [name for name in preferred if name in available] + [
        name for name in available if name not in preferred
    ]


def plot_scores(df, summary):
    """Figura POD/FAR: una riga per soglia."""
    configure_plot_style()
    nrows = len(THRESHOLDS_CM)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=2,
        figsize=(18, 4.0 * nrows),
        sharex=True,
        sharey="col",
        constrained_layout=True,
    )

    summary_indexed = summary.set_index(["threshold_cm", "model"])

    for row, threshold_cm in enumerate(THRESHOLDS_CM):
        threshold_cm = float(threshold_cm)
        ax_pod, ax_far = axes[row, 0], axes[row, 1]

        for model_name in models_order(df):
            group = df[
                (df["threshold_cm"] == threshold_cm)
                & (df["model"] == model_name)
            ]
            values = summary_indexed.loc[(threshold_cm, model_name)]
            color = MODEL_COLORS.get(model_name)

            ax_pod.plot(
                group["date"],
                group["POD"],
                linewidth=1.8,
                color=color,
                label=f"{model_name} (global POD = {values['POD_global']:.3f})",
            )
            ax_far.plot(
                group["date"],
                group["FAR"],
                linewidth=1.8,
                color=color,
                label=f"{model_name} (global FAR = {values['FAR_global']:.3f})",
            )

        ax_pod.set_title(f"POD - amplitude >= {threshold_cm:g} cm")
        ax_far.set_title(f"FAR - amplitude >= {threshold_cm:g} cm")
        ax_pod.set_ylabel("POD")
        ax_far.set_ylabel("FAR")
        ax_pod.set_ylim(0.0, 1.02)
        ax_far.set_ylim(0.0, 1.02)
        ax_pod.legend(loc="best", frameon=True, edgecolor="black")
        ax_far.legend(loc="best", frameon=True, edgecolor="black")
        format_time_axis(ax_pod)
        format_time_axis(ax_far)

    axes[-1, 0].set_xlabel("Date")
    axes[-1, 1].set_xlabel("Date")

    fig.suptitle(
        f"POD & FAR - {REFERENCE_NAME} comparison - "
        f"{LIFETIME_LABEL} lifetime threshold",
        fontsize=19,
        fontweight="bold",
    )

    png_path = OUTPUT_DIR / "POD_FAR_daily_by_amplitude_threshold.png"
    pdf_path = OUTPUT_DIR / "POD_FAR_daily_by_amplitude_threshold.pdf"
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"[OK] Figura POD/FAR PNG: {png_path}")
    print(f"[OK] Figura POD/FAR PDF: {pdf_path}")


def plot_counts(df):
    """
    Figura dei conteggi giornalieri.

    Colonne:
    1) hits e misses nello spazio del riferimento, usati per POD;
    2) hits e false alarms nello spazio del modello, usati per FAR.
    """
    configure_plot_style()
    nrows = len(THRESHOLDS_CM)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=2,
        figsize=(19, 4.0 * nrows),
        sharex=True,
        constrained_layout=True,
    )

    for row, threshold_cm in enumerate(THRESHOLDS_CM):
        threshold_cm = float(threshold_cm)
        ax_reference, ax_model = axes[row, 0], axes[row, 1]

        for model_name in models_order(df):
            group = df[
                (df["threshold_cm"] == threshold_cm)
                & (df["model"] == model_name)
            ]
            color = MODEL_COLORS.get(model_name)

            ax_reference.plot(
                group["date"],
                group["hits_reference"],
                color=color,
                linewidth=1.7,
                label=f"{model_name} hits",
            )
            ax_reference.plot(
                group["date"],
                group["misses_reference"],
                color=color,
                linewidth=1.3,
                linestyle="--",
                label=f"{model_name} misses",
            )

            ax_model.plot(
                group["date"],
                group["hits_model"],
                color=color,
                linewidth=1.7,
                label=f"{model_name} hits",
            )
            ax_model.plot(
                group["date"],
                group["false_alarms_model"],
                color=color,
                linewidth=1.3,
                linestyle="--",
                label=f"{model_name} false alarms",
            )

        ax_reference.set_title(
            f"Reference-space counts - amplitude >= {threshold_cm:g} cm"
        )
        ax_model.set_title(
            f"Model-space counts - amplitude >= {threshold_cm:g} cm"
        )
        ax_reference.set_ylabel("Daily count")
        ax_model.set_ylabel("Daily count")
        ax_reference.set_ylim(bottom=0)
        ax_model.set_ylim(bottom=0)
        ax_reference.legend(loc="best", frameon=True, edgecolor="black")
        ax_model.legend(loc="best", frameon=True, edgecolor="black")
        format_time_axis(ax_reference)
        format_time_axis(ax_model)

    axes[-1, 0].set_xlabel("Date")
    axes[-1, 1].set_xlabel("Date")

    fig.suptitle(
        f"Daily cumulative counts by amplitude threshold - {REFERENCE_NAME} comparison - "
        f"{LIFETIME_LABEL} lifetime threshold",
        fontsize=19,
        fontweight="bold",
    )

    png_path = OUTPUT_DIR / "matching_counts_daily_by_amplitude_threshold.png"
    pdf_path = OUTPUT_DIR / "matching_counts_daily_by_amplitude_threshold.pdf"
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"[OK] Figura conteggi PNG: {png_path}")
    print(f"[OK] Figura conteggi PDF: {pdf_path}")


def print_summary(summary):
    """Stampa una tabella compatta nel terminale."""
    columns = [
        "threshold_cm",
        "model",
        "hits_reference_total",
        "misses_reference_total",
        "hits_model_total",
        "false_alarms_model_total",
        "POD_global",
        "FAR_global",
    ]

    print("\nRiepilogo globale per soglia:\n")
    print(summary[columns].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("Caricamento cataloghi...")
    reference, models = load_catalogs()

    days = common_day_range(reference, models)
    first_date = num2date(
        days[0], units=TIME_UNITS, calendar=TIME_CALENDAR
    ).strftime("%Y-%m-%d")
    last_date = num2date(
        days[-1], units=TIME_UNITS, calendar=TIME_CALENDAR
    ).strftime("%Y-%m-%d")

    print(f"Periodo elaborato: {first_date} -> {last_date} ({len(days)} giorni)")
    print(f"Soglie cumulative: {THRESHOLDS_CM.tolist()} cm")

    df = calculate_daily_timeseries(reference, models, days)
    summary = build_summary(df)

    save_tables(df, summary)
    plot_scores(df, summary)
    plot_counts(df)
    print_summary(summary)

    print("\nElaborazione completata.")


if __name__ == "__main__":
    main()

