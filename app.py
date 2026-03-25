import re
from io import StringIO
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(page_title="Enhancement Ratio Analyzer", layout="wide")

# -------------------------------------------------
# Session state
# -------------------------------------------------
if "er_results_ready" not in st.session_state:
    st.session_state.er_results_ready = False
if "er_summary_df" not in st.session_state:
    st.session_state.er_summary_df = pd.DataFrame()
if "er_review_df" not in st.session_state:
    st.session_state.er_review_df = pd.DataFrame()
if "er_warnings_df" not in st.session_state:
    st.session_state.er_warnings_df = pd.DataFrame()
if "er_details" not in st.session_state:
    st.session_state.er_details = {}
if "er_thickness_editor_df" not in st.session_state:
    st.session_state.er_thickness_editor_df = pd.DataFrame()


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def load_spectrum(uploaded_file, skiprows=1, max_rows=1024):
    data = np.loadtxt(uploaded_file, skiprows=skiprows, max_rows=max_rows)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError("Spectrum file must contain at least 3 columns.")
    channels = data[:, 0]
    intensity = data[:, 2]
    return channels, intensity


def calculate_wavelengths(channels, center_wavelength=550, grating_number=1):
    g = 0.4196 if grating_number == 1 else 0.4192
    return np.array([center_wavelength - ((i - 513) * g) for i in channels], dtype=float)


def normalize_name(name: str) -> str:
    stem = Path(name).stem.upper().strip()
    stem = re.sub(r"\s+", " ", stem)
    return stem


def detect_is_reference(name: str) -> bool:
    n = normalize_name(name)
    return "REF" in n


def detect_material_family(name: str):
    n = normalize_name(name)
    ordered = ["PMMA", "EMA", "PET", "PE", "LAM"]
    for fam in ordered:
        if fam in n:
            return fam
    return None


def extract_sample_name(filename: str):
    stem = Path(filename).stem
    stem = stem.replace("_Excplasma_Cen550_NewM266Gr1_Slit100_Filter4_t500ms", "")
    stem = re.sub(r"_Exc.*$", "", stem, flags=re.IGNORECASE)
    return normalize_name(stem)


def extract_thickness_from_name(sample_name: str):
    # Supports names like SAMPLE 1-50-A, SAMPLE 2-100-B, etc.
    wet_to_dry = {50: 11.0, 100: 12.0, 150: 18.0, 200: 38.0, 400: 60.0}
    parts = re.split(r"[-_ ]+", normalize_name(sample_name))
    for part in parts:
        if part.isdigit():
            val = int(part)
            if val in wet_to_dry:
                return wet_to_dry[val], f"inferred_from_name({val})"
    return None, "missing"


def match_reference(sample_name: str, available_references: list[str]):
    sample_norm = normalize_name(sample_name)
    family = detect_material_family(sample_norm)

    if not available_references:
        return None, "no_references_uploaded"

    # 1. same family + REF
    if family is not None:
        family_matches = [r for r in available_references if family in normalize_name(r)]
        if len(family_matches) == 1:
            return family_matches[0], f"matched_family:{family}"
        if len(family_matches) > 1:
            # prefer A if present
            a_matches = [r for r in family_matches if normalize_name(r).endswith(" A")]
            if len(a_matches) == 1:
                return a_matches[0], f"matched_family_prefer_A:{family}"
            return family_matches[0], f"multiple_family_matches:{family}"

    # 2. fallback generic LAM / PET style refs
    generic_priority = ["LAM", "PET", "EMA", "PMMA", "PE"]
    for fam in generic_priority:
        fam_matches = [r for r in available_references if fam in normalize_name(r)]
        if len(fam_matches) == 1:
            return fam_matches[0], f"fallback_family:{fam}"
        if len(fam_matches) > 1:
            return fam_matches[0], f"fallback_multiple_family:{fam}"

    # 3. final fallback first ref
    return available_references[0], "fallback_first_reference"


def build_review_table(measurement_files, manual_thickness_map=None):
    manual_thickness_map = manual_thickness_map or {}
    rows = []

    uploaded_names = [extract_sample_name(f.name) for f in measurement_files]
    ref_names = [n for n in uploaded_names if detect_is_reference(n)]

    for f in measurement_files:
        parsed_name = extract_sample_name(f.name)
        is_ref = detect_is_reference(parsed_name)
        family = detect_material_family(parsed_name)
        matched_ref, match_reason = (parsed_name, "self_reference") if is_ref else match_reference(parsed_name, ref_names)

        if parsed_name in manual_thickness_map and manual_thickness_map[parsed_name] not in [None, "", np.nan]:
            thickness = float(manual_thickness_map[parsed_name])
            thickness_source = "manual_upload"
        else:
            thickness, thickness_source = extract_thickness_from_name(parsed_name)

        rows.append({
            "File": f.name,
            "Parsed name": parsed_name,
            "Type": "Reference" if is_ref else "Sample",
            "Family": family,
            "Matched reference": matched_ref,
            "Reference match reason": match_reason,
            "Thickness (µm)": thickness,
            "Thickness source": thickness_source,
        })

    return pd.DataFrame(rows)
def build_manual_review_table(measurement_files, manual_thickness_map=None):
    manual_thickness_map = manual_thickness_map or {}
    rows = []

    parsed_names = [extract_sample_name(f.name) for f in measurement_files]

    for f in measurement_files:
        parsed_name = extract_sample_name(f.name)

        if parsed_name in manual_thickness_map and manual_thickness_map[parsed_name] not in [None, "", np.nan]:
            thickness = float(manual_thickness_map[parsed_name])
            thickness_source = "manual_upload"
        else:
            thickness, thickness_source = extract_thickness_from_name(parsed_name)

        rows.append({
            "File": f.name,
            "Parsed name": parsed_name,
            "Type": "Sample",
            "Family": detect_material_family(parsed_name),
            "Matched reference": None,
            "Reference match reason": "manual",
            "Thickness (µm)": thickness,
            "Thickness source": thickness_source,
        })

    return pd.DataFrame(rows)

def build_thickness_map_from_editor(df_editor: pd.DataFrame):
    out = {}
    if df_editor is None or df_editor.empty:
        return out
    for _, row in df_editor.iterrows():
        name = row.get("Parsed name")
        thickness = row.get("Thickness (µm)")
        if pd.notna(name) and pd.notna(thickness):
            out[str(name)] = float(thickness)
    return out


def select_measurement_files(review_df: pd.DataFrame):
    if review_df.empty:
        return [], []
    refs = review_df[review_df["Type"] == "Reference"]["Parsed name"].tolist()
    samples = review_df[review_df["Type"] == "Sample"]["Parsed name"].tolist()
    return refs, samples


def plot_enhancement_ratio(ax, wl, ratio, sample_name, thickness=None):
    label = f"{sample_name} / {thickness:.1f} µm" if thickness is not None and pd.notna(thickness) else sample_name
    ax.plot(wl, ratio, label=label, lw=2)
    ax.hlines(y=1, xmin=350, xmax=950, colors="black", linestyles="-", lw=1.5)
    ax.set_xlim(360, 770)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Transmission normalized")
    ax.grid(True, alpha=0.2)


def plot_raw_data(ax, wl, sample_i, ref_i, ref_name, sample_name):
    ax.plot(wl, sample_i, label=sample_name, lw=1)
    ax.plot(wl, ref_i, label=ref_name, lw=1)
    ax.set_xlim(360, 770)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Intensity")
    ax.grid(True, alpha=0.2)


def make_downloadable_summary(results_long: pd.DataFrame):
    if results_long.empty:
        return pd.DataFrame()
    summary = results_long[[
        "Sample",
        "Reference",
        "Family",
        "Thickness (µm)",
        "Thickness source",
        "Mean ratio 400-700",
        "Min ratio 400-700",
        "Max ratio 400-700",
    ]].copy()
    return summary.sort_values(by=["Sample"]).reset_index(drop=True)


def band_stats(wl, ratio, lo=400, hi=700):
    mask = (wl >= lo) & (wl <= hi)
    if not np.any(mask):
        return np.nan, np.nan, np.nan
    vals = ratio[mask]
    return float(np.mean(vals)), float(np.min(vals)), float(np.max(vals))


def parse_thickness_csv(uploaded_file):
    if uploaded_file is None:
        return {}
    df = pd.read_csv(uploaded_file)
    required = {"Parsed name", "Thickness (µm)"}
    if not required.issubset(df.columns):
        raise ValueError("Thickness CSV must contain columns: 'Parsed name' and 'Thickness (µm)'.")
    return {
        normalize_name(str(row["Parsed name"])): float(row["Thickness (µm)"])
        for _, row in df.iterrows()
        if pd.notna(row["Parsed name"]) and pd.notna(row["Thickness (µm)"])
    }


# -------------------------------------------------
# UI
# -------------------------------------------------
def make_sample_label(sample_name, thickness=None, ref_name=None):
    label = sample_name
    if thickness is not None and pd.notna(thickness):
        label += f" / {float(thickness):.1f} µm"
    if ref_name:
        label += f" | ref: {ref_name}"
    return label


def build_plotly_figure(
    details_dict,
    selected_samples,
    mode="ratio",
    d_ref=None,
    x_range=None,
):
    fig = go.Figure()

    for sample_name in selected_samples:
        d = details_dict[sample_name]

        if mode == "ratio":
            y = d["ratio"]
            y_label = "Transmission normalized"
            title = "Enhancement ratio"
            trace_label = make_sample_label(
                d["sample_name"], d["thickness"], d["ref_name"]
            )

        elif mode == "raw_sample":
            y = d["sample_i"]
            y_label = "Intensity"
            title = "Raw sample spectra"
            trace_label = d["sample_name"]

        elif mode == "raw_reference":
            y = d["ref_i"]
            y_label = "Intensity"
            title = "Raw reference spectra"
            trace_label = f"{d['sample_name']} | {d['ref_name']}"

        elif mode == "thickness_norm":
            if d["norm_ratio"] is None:
                continue
            y = d["norm_ratio"]
            y_label = f"Transmission at d = {d_ref:.1f} µm" if d_ref is not None else "Transmission"
            title = "Thickness-normalized transmission"
            trace_label = f"{d['sample_name']} → {d_ref:.1f} µm"

        else:
            continue

        fig.add_trace(
            go.Scatter(
                x=d["wl"],
                y=y,
                mode="lines",
                name=trace_label,
            )
        )

    if mode in ["ratio", "thickness_norm"]:
        fig.add_hline(y=1, line_width=1.5, line_color="black")

    fig.update_layout(
        title=title,
        xaxis_title="Wavelength (nm)",
        yaxis_title=y_label,
        hovermode="x unified",
        legend_title="Samples",
    )

    if x_range is not None:
        fig.update_xaxes(range=x_range)
    else:
        fig.update_xaxes(range=[360, 770])

    return fig

st.title("Enhancement Ratio Analyzer")
st.caption(
    "Upload all measurement files. The app will detect references, match samples to references, suggest thickness values, and run enhancement-ratio analysis."
)

left, right = st.columns([1, 1.7], gap="large")

with left:
    st.subheader("Inputs")

    measurement_files = st.file_uploader(
        "1. Drop all measurement files",
        type=["txt", "dat", "csv"],
        accept_multiple_files=True,
        key="er_measurement_files",
    )

    matching_mode = st.radio(
        "2. Reference matching mode",
        options=["Smart mode", "Manual mode"],
        index=0,
    )

    center_wavelength = st.number_input(
        "3. Center wavelength (nm)",
        min_value=200,
        max_value=1200,
        value=550,
        step=1,
    )

    grating_number = st.selectbox("4. Grating", options=[1, 2], index=1)

    st.subheader("Optional switches")
    plot_raw = st.toggle("Show raw spectra", value=False)
    solve_thickness = st.toggle("Run thickness normalization", value=False)
    run_simulation = st.toggle("Run simulation", value=False)

    simulated_thickness = None
    if run_simulation:
        simulated_thickness = st.number_input(
            "Simulated thickness (µm)",
            min_value=1.0,
            max_value=5000.0,
            value=146.0,
            step=1.0,
        )

    thickness_csv = st.file_uploader(
        "Optional thickness CSV",
        type=["csv"],
        help="Optional CSV with columns: Parsed name, Thickness (µm)",
        key="er_thickness_csv",
    )

    preview = st.button("Preview parsing", type="secondary", width="stretch")
    run_analysis = st.button("Run enhancement analysis", type="primary", width="stretch")

with right:
    st.subheader("Review and results")

    manual_thickness_map = {}
    if thickness_csv is not None:
        try:
            manual_thickness_map = parse_thickness_csv(thickness_csv)
        except Exception as e:
            st.error(f"Thickness CSV error: {e}")

    if preview or (measurement_files and st.session_state.er_thickness_editor_df.empty):
        if not measurement_files:
            st.warning("Upload the measurement files first.")
        else:
            if matching_mode == "Smart mode":
                review_df = build_review_table(
                    measurement_files,
                    manual_thickness_map=manual_thickness_map,
                )
            else:
                review_df = build_manual_review_table(
                    measurement_files,
                    manual_thickness_map=manual_thickness_map,
                )

            st.session_state.er_review_df = review_df
            thickness_editor_df = review_df[
                ["Parsed name", "Type", "Family", "Matched reference", "Thickness (µm)", "Thickness source"]
            ].copy()
            st.session_state.er_thickness_editor_df = thickness_editor_df

    if not st.session_state.er_thickness_editor_df.empty:
        st.subheader("Editable thickness / reference review")

        if matching_mode == "Smart mode":
            disabled_cols = ["Parsed name", "Type", "Family", "Matched reference", "Thickness source"]
        else:
            disabled_cols = ["Parsed name", "Family", "Thickness source"]

        reference_options = st.session_state.er_thickness_editor_df["Parsed name"].tolist()

        edited_df = st.data_editor(
            st.session_state.er_thickness_editor_df,
            width="stretch",
            num_rows="fixed",
            key="er_data_editor",
            disabled=disabled_cols,
            column_config={
                "Type": st.column_config.SelectboxColumn(
                    "Type",
                    options=["Sample", "Reference"],
                    required=True,
                ),
                "Matched reference": st.column_config.SelectboxColumn(
                    "Matched reference",
                    options=reference_options,
                ),
                "Thickness (µm)": st.column_config.NumberColumn(
                    "Thickness (µm)",
                    min_value=0.0,
                    step=1.0,
                ),
            },
        )
        st.session_state.er_thickness_editor_df = edited_df.copy()

        review_csv = edited_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download thickness review CSV",
            data=review_csv,
            file_name="enhancement_ratio_thickness_review.csv",
            mime="text/csv",
            width="stretch",
        )

    if run_analysis:
        try:
            if not measurement_files:
                st.error("Please upload the measurement files.")
                st.stop()

            editor_df = st.session_state.er_thickness_editor_df.copy()

            if editor_df.empty:
                if matching_mode == "Smart mode":
                    editor_df = build_review_table(
                        measurement_files,
                        manual_thickness_map=manual_thickness_map,
                    )
                else:
                    editor_df = build_manual_review_table(
                        measurement_files,
                        manual_thickness_map=manual_thickness_map,
                    )

            manual_map = build_thickness_map_from_editor(editor_df)

            if matching_mode == "Smart mode":
                review_df = build_review_table(
                    measurement_files,
                    manual_thickness_map=manual_map,
                )
            else:
                review_df = editor_df.copy()

            file_lookup = {extract_sample_name(f.name): f for f in measurement_files}
            warnings = []
            details = {}
            results_long = []
            mu_list = []
            sim_wl = None
            d_ref = None

            if solve_thickness:
                for _, row in review_df.iterrows():
                    if row["Type"] == "Sample" and pd.notna(row["Thickness (µm)"]):
                        t = float(row["Thickness (µm)"])
                        if d_ref is None or t > d_ref:
                            d_ref = t

            for _, row in review_df.iterrows():
                if row["Type"] != "Sample":
                    continue

                sample_name = row["Parsed name"]
                ref_name = row["Matched reference"]
                family = row["Family"]
                thickness = row["Thickness (µm)"]
                thickness_source = row["Thickness source"]

                if pd.isna(ref_name) or ref_name is None or str(ref_name).strip() == "":
                    warnings.append({
                        "Type": "Missing reference assignment",
                        "Sample": sample_name,
                        "Message": "No reference assigned to this sample.",
                    })
                    continue

                if ref_name not in file_lookup:
                    warnings.append({
                        "Type": "Missing reference file",
                        "Sample": sample_name,
                        "Message": f"Matched reference '{ref_name}' was not found among uploaded files.",
                    })
                    continue

                sample_file = file_lookup[sample_name]
                ref_file = file_lookup[ref_name]

                try:
                    try:
                        sample_file.seek(0)
                        ref_file.seek(0)
                    except Exception:
                        pass

                    channels_s, sample_i = load_spectrum(sample_file)
                    channels_r, ref_i = load_spectrum(ref_file)

                    if len(channels_s) != len(channels_r):
                        raise ValueError("Sample and reference files do not have the same number of points.")

                    wl = calculate_wavelengths(
                        channels_s,
                        center_wavelength=center_wavelength,
                        grating_number=grating_number,
                    )
                    ratio = sample_i / np.clip(ref_i, 1e-12, None)
                    mean_ratio, min_ratio, max_ratio = band_stats(wl, ratio, 400, 700)

                    norm_ratio = None
                    mu_lambda = None
                    if solve_thickness and pd.notna(thickness) and thickness and d_ref is not None and thickness > 0:
                        ratio_clipped = np.clip(ratio, 1e-9, None)
                        norm_ratio = ratio_clipped ** (d_ref / float(thickness))
                        mu_lambda = (-np.log(ratio_clipped)) / float(thickness)
                        mu_list.append(mu_lambda)
                        if sim_wl is None:
                            sim_wl = wl

                    results_long.append({
                        "Sample": sample_name,
                        "Reference": ref_name,
                        "Family": family,
                        "Thickness (µm)": thickness,
                        "Thickness source": thickness_source,
                        "Mean ratio 400-700": mean_ratio,
                        "Min ratio 400-700": min_ratio,
                        "Max ratio 400-700": max_ratio,
                    })

                    details[sample_name] = {
                        "wl": wl,
                        "sample_i": sample_i,
                        "ref_i": ref_i,
                        "ratio": ratio,
                        "norm_ratio": norm_ratio,
                        "mu_lambda": mu_lambda,
                        "sample_name": sample_name,
                        "ref_name": ref_name,
                        "family": family,
                        "thickness": thickness,
                    }

                except Exception as e:
                    warnings.append({
                        "Type": "Processing error",
                        "Sample": sample_name,
                        "Message": str(e),
                    })

            summary_df = make_downloadable_summary(pd.DataFrame(results_long))
            warnings_df = pd.DataFrame(warnings)

            simulation = None
            if run_simulation and mu_list and sim_wl is not None and simulated_thickness is not None:
                mu_stack = np.vstack(mu_list)
                mu_mean = np.mean(mu_stack, axis=0)
                d_sim = float(simulated_thickness)
                t_sim = np.exp(-mu_mean * d_sim)
                t_all = np.exp(-mu_stack * d_sim)
                t_lo = np.min(t_all, axis=0)
                t_hi = np.max(t_all, axis=0)
                simulation = {
                    "wl": sim_wl,
                    "mean": t_sim,
                    "lower": t_lo,
                    "upper": t_hi,
                    "thickness": d_sim,
                }

            st.session_state.er_summary_df = summary_df
            st.session_state.er_review_df = review_df
            st.session_state.er_warnings_df = warnings_df
            st.session_state.er_details = {
                "samples": details,
                "simulation": simulation,
                "d_ref": d_ref,
                "plot_raw": plot_raw,
                "solve_thickness": solve_thickness,
                "run_simulation": run_simulation,
            }
            st.session_state.er_results_ready = True

        except Exception as e:
            st.error(f"Error while running enhancement analysis: {e}")

    if st.session_state.er_results_ready:
        summary_df = st.session_state.er_summary_df
        review_df = st.session_state.er_review_df
        warnings_df = st.session_state.er_warnings_df
        details_state = st.session_state.er_details
        details = details_state.get("samples", {})
        simulation = details_state.get("simulation")
        d_ref = details_state.get("d_ref")
        plot_raw_state = details_state.get("plot_raw", False)
        solve_thickness_state = details_state.get("solve_thickness", False)
        run_simulation_state = details_state.get("run_simulation", False)

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Summary",
            "Review table",
            "Graphs",
            "Simulation",
            "Warnings",
        ])

        with tab1:
            st.subheader("Enhancement ratio summary")
            if summary_df.empty:
                st.info("No results generated.")
            else:
                st.dataframe(summary_df, width="stretch")
                csv_bytes = summary_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download summary CSV",
                    data=csv_bytes,
                    file_name="enhancement_ratio_summary.csv",
                    mime="text/csv",
                    width="stretch",
                )

        with tab2:
            st.subheader("Parsed files and matching")
            st.dataframe(review_df, width="stretch")

        with tab3:
            st.subheader("Interactive graphs")

            if not details:
                st.info("No graphable results available.")
            else:
                sample_options = sorted(details.keys())

                selected_samples = st.multiselect(
                    "Select one or more samples to overlay",
                    options=sample_options,
                    default=sample_options[: min(3, len(sample_options))],
                    key="er_graph_multiselect",
                )

                if not selected_samples:
                    st.warning("Select at least one sample.")
                else:
                    graph_mode_options = ["Enhancement ratio"]
                    if plot_raw_state:
                        graph_mode_options.extend(["Raw sample spectra", "Raw reference spectra"])
                    if solve_thickness_state:
                        has_norm = any(details[s]["norm_ratio"] is not None for s in selected_samples)
                        if has_norm:
                            graph_mode_options.append("Thickness-normalized transmission")

                    graph_mode = st.radio(
                        "Graph type",
                        options=graph_mode_options,
                        horizontal=True,
                        key="er_graph_mode",
                    )

                    x_min, x_max = st.slider(
                        "Displayed wavelength range (nm)",
                        min_value=300,
                        max_value=900,
                        value=(360, 770),
                        step=1,
                        key="er_graph_range",
                    )

                    if graph_mode == "Enhancement ratio":
                        fig = build_plotly_figure(
                            details_dict=details,
                            selected_samples=selected_samples,
                            mode="ratio",
                            d_ref=d_ref,
                            x_range=[x_min, x_max],
                        )
                    elif graph_mode == "Raw sample spectra":
                        fig = build_plotly_figure(
                            details_dict=details,
                            selected_samples=selected_samples,
                            mode="raw_sample",
                            d_ref=d_ref,
                            x_range=[x_min, x_max],
                        )
                    elif graph_mode == "Raw reference spectra":
                        fig = build_plotly_figure(
                            details_dict=details,
                            selected_samples=selected_samples,
                            mode="raw_reference",
                            d_ref=d_ref,
                            x_range=[x_min, x_max],
                        )
                    else:
                        fig = build_plotly_figure(
                            details_dict=details,
                            selected_samples=selected_samples,
                            mode="thickness_norm",
                            d_ref=d_ref,
                            x_range=[x_min, x_max],
                        )

                    st.plotly_chart(fig, use_container_width=True)

                    st.subheader("Selected sample details")
                    detail_rows = []
                    for s in selected_samples:
                        d = details[s]
                        detail_rows.append({
                            "Sample": d["sample_name"],
                            "Reference": d["ref_name"],
                            "Family": d["family"],
                            "Thickness (µm)": d["thickness"],
                        })
                    st.dataframe(pd.DataFrame(detail_rows), width="stretch")

        with tab4:
            st.subheader("Simulation")
            if not run_simulation_state:
                st.info("Simulation was turned off.")
            elif simulation is None:
                st.info("No simulation available. Enable thickness normalization and ensure at least one valid thickness is present.")
            else:
                fig_sim, ax_sim = plt.subplots(figsize=(8, 4.8))
                ax_sim.plot(
                    simulation["wl"],
                    simulation["mean"],
                    lw=2,
                    label=f"Simulated, d = {simulation['thickness']:.1f} µm (mean μ)",
                )
                ax_sim.fill_between(
                    simulation["wl"],
                    simulation["lower"],
                    simulation["upper"],
                    alpha=0.3,
                    label="Envelope from all μ(λ) samples",
                )
                ax_sim.set_xlim(360, 770)
                ax_sim.set_xlabel("Wavelength (nm)")
                ax_sim.set_ylabel(f"Predicted transmission at d = {simulation['thickness']:.1f} µm")
                ax_sim.grid(True, alpha=0.2)
                ax_sim.legend(loc="best", fontsize=10)
                ax_sim.set_title("Predicted transmission by thickness")
                st.pyplot(fig_sim)

                sim_df = pd.DataFrame({
                    "Wavelength_nm": simulation["wl"],
                    "T_sim": simulation["mean"],
                    "T_sim_lower": simulation["lower"],
                    "T_sim_upper": simulation["upper"],
                })
                st.dataframe(sim_df, width="stretch")
                st.download_button(
                    "Download simulation CSV",
                    data=sim_df.to_csv(index=False).encode("utf-8"),
                    file_name="enhancement_ratio_simulation.csv",
                    mime="text/csv",
                    width="stretch",
                )

        with tab5:
            st.subheader("Warnings")
            if warnings_df.empty:
                st.success("No warnings.")
            else:
                st.dataframe(warnings_df, width="stretch")

    else:
        st.info("Upload files, preview parsing, optionally edit thickness, then run the analysis.")
