"""Streamlit dashboard for Fata Cognita.

Three-page layout:
1. Individual Explorer — predict trajectory for a person
2. Archetype Gallery — browse discovered archetypes
3. What-If Analysis — counterfactual sensitivity analysis
"""

from __future__ import annotations

import os

import httpx
import plotly.graph_objects as go
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000/api/v1")


def main() -> None:
    """Entry point for the Streamlit dashboard."""
    st.set_page_config(page_title="Fata Cognita", layout="wide")
    st.title("Fata Cognita — Neural Actuarial Trajectory Model")

    page = st.sidebar.radio(
        "Navigate",
        ["Individual Explorer", "Archetype Gallery", "What-If Analysis"],
    )

    if page == "Individual Explorer":
        _individual_explorer()
    elif page == "Archetype Gallery":
        _archetype_gallery()
    else:
        _what_if_analysis()


def _get_feature_inputs() -> dict[str, float]:
    """Render sidebar inputs for static features and return values."""
    st.sidebar.header("Person Features")
    features: dict[str, float] = {}

    def _sex_label(x: float) -> str:
        return "Female" if x == 0 else "Male"

    features["sex"] = st.sidebar.selectbox(
        "Sex",
        [0.0, 1.0],
        format_func=_sex_label,
    )

    race_sel = st.sidebar.selectbox(
        "Race/Ethnicity",
        ["Non-Hispanic/Non-Black", "Hispanic", "Black"],
    )
    features["race_hispanic"] = 1.0 if race_sel == "Hispanic" else 0.0
    features["race_black"] = 1.0 if race_sel == "Black" else 0.0
    features["race_other"] = 1.0 if race_sel == "Non-Hispanic/Non-Black" else 0.0

    features["birth_year"] = st.sidebar.slider(
        "Birth Year (normalized)",
        -2.0,
        2.0,
        0.0,
        0.1,
    )
    features["parent_education"] = st.sidebar.slider(
        "Parent Education (normalized)",
        -2.0,
        2.0,
        0.0,
        0.1,
    )
    features["family_income_14"] = st.sidebar.slider(
        "Family Income at 14 (normalized)",
        -2.0,
        2.0,
        0.0,
        0.1,
    )

    region = st.sidebar.selectbox(
        "Region at 14",
        ["Northeast", "North Central", "South", "West"],
    )
    features["region_northeast"] = 1.0 if region == "Northeast" else 0.0
    features["region_north_central"] = 1.0 if region == "North Central" else 0.0
    features["region_south"] = 1.0 if region == "South" else 0.0
    features["region_west"] = 1.0 if region == "West" else 0.0

    features["afqt_score"] = st.sidebar.slider(
        "AFQT Score (normalized)",
        -2.0,
        2.0,
        0.0,
        0.1,
    )
    features["afqt_available"] = 1.0

    def _cohort_label(x: float) -> str:
        return "NLSY79" if x == 0 else "NLSY97"

    features["cohort"] = st.sidebar.selectbox(
        "Cohort",
        [0.0, 1.0],
        format_func=_cohort_label,
    )

    return features


def _individual_explorer() -> None:
    """Page 1: Individual trajectory prediction."""
    features = _get_feature_inputs()
    mode = st.sidebar.radio("Mode", ["Deterministic", "Monte Carlo"])

    if st.sidebar.button("Predict"):
        if mode == "Deterministic":
            _show_deterministic(features)
        else:
            _show_monte_carlo(features)


def _show_deterministic(features: dict[str, float]) -> None:
    """Show deterministic prediction."""
    with st.spinner("Predicting..."):
        try:
            resp = httpx.post(
                f"{API_URL}/predict",
                json={"static_features": features, "deterministic": True},
                timeout=30.0,
            )
            resp.raise_for_status()
        except httpx.HTTPError as e:
            st.error(f"API error: {e}")
            return

    data = resp.json()
    traj = data["trajectory"]
    ages = [t["age"] for t in traj]
    incomes = [t["income"] for t in traj]
    satisfaction = [t["satisfaction"] for t in traj]
    [t["life_state"] for t in traj]

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=ages,
                y=incomes,
                name="Income",
                line=dict(color="steelblue"),
            )
        )
        fig.update_layout(
            title="Income Trajectory",
            xaxis_title="Age",
            yaxis_title="Income ($)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=ages,
                y=satisfaction,
                name="Satisfaction",
                line=dict(color="coral"),
            )
        )
        fig.update_layout(
            title="Life Satisfaction",
            xaxis_title="Age",
            yaxis_title="Satisfaction",
            yaxis_range=[0, 1],
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"Archetype: {data['archetype_id']}")
    st.json(data["archetype_membership"])


def _show_monte_carlo(features: dict[str, float]) -> None:
    """Show Monte Carlo simulation with percentile bands."""
    n_sim = st.sidebar.slider("Simulations", 100, 5000, 1000, 100)

    with st.spinner(f"Simulating {n_sim} trajectories..."):
        try:
            resp = httpx.post(
                f"{API_URL}/simulate",
                json={"static_features": features, "n_simulations": n_sim},
                timeout=60.0,
            )
            resp.raise_for_status()
        except httpx.HTTPError as e:
            st.error(f"API error: {e}")
            return

    data = resp.json()
    bands = data["percentile_bands"]
    ages = bands["age"]
    inc = bands["income"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ages,
            y=inc["p90"],
            fill=None,
            line=dict(color="lightblue"),
            name="P90",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ages,
            y=inc["p10"],
            fill="tonexty",
            line=dict(color="lightblue"),
            name="P10-P90",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ages,
            y=inc["p75"],
            fill=None,
            line=dict(color="steelblue", dash="dash"),
            name="P75",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ages,
            y=inc["p25"],
            fill="tonexty",
            line=dict(color="steelblue", dash="dash"),
            name="P25-P75",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ages,
            y=inc["p50"],
            line=dict(color="darkblue", width=2),
            name="Median",
        )
    )
    fig.update_layout(
        title="Income Forecast (Monte Carlo)",
        xaxis_title="Age",
        yaxis_title="Income ($)",
    )
    st.plotly_chart(fig, use_container_width=True)


def _archetype_gallery() -> None:
    """Page 2: Browse discovered archetypes."""
    with st.spinner("Loading archetypes..."):
        try:
            resp = httpx.get(f"{API_URL}/archetypes", timeout=10.0)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            st.error(f"API error: {e}")
            return

    data = resp.json()
    k = data["k_selected"]
    n = data["total_individuals"]
    st.subheader(f"Discovered {k} Archetypes ({n} individuals)")

    for arch in data["archetypes"]:
        dominant = arch["dominant_life_state"]
        prevalence = arch["prevalence"]
        label = f"Archetype {arch['id']} — {dominant} ({prevalence:.1%})"
        with st.expander(label):
            st.write(f"**Members:** {arch['member_count']}")
            st.write(
                f"**Peak Income:** ${arch['median_peak_income']:,.0f}",
            )

            try:
                traj_resp = httpx.get(
                    f"{API_URL}/archetypes/{arch['id']}/trajectory",
                    timeout=10.0,
                )
                traj_resp.raise_for_status()
                traj_data = traj_resp.json()
                traj = traj_data["canonical_trajectory"]

                ages = [t["age"] for t in traj]
                incomes = [t["income"] for t in traj]

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=ages,
                        y=incomes,
                        name="Income",
                    )
                )
                fig.update_layout(
                    title=f"Archetype {arch['id']} Canonical Trajectory",
                    xaxis_title="Age",
                    yaxis_title="Log Income",
                )
                st.plotly_chart(fig, use_container_width=True)
            except httpx.HTTPError:
                st.warning("Could not load trajectory")


def _what_if_analysis() -> None:
    """Page 3: Counterfactual sensitivity analysis."""
    features = _get_feature_inputs()

    st.sidebar.header("Perturbation")
    perturb_var = st.sidebar.selectbox(
        "Variable to Perturb",
        [
            "birth_year",
            "parent_education",
            "family_income_14",
            "afqt_score",
            "sex",
            "cohort",
        ],
    )
    perturb_val = st.sidebar.slider(
        "New Value",
        -3.0,
        3.0,
        1.0,
        0.1,
    )
    n_sim = st.sidebar.slider(
        "Simulations",
        100,
        10000,
        1000,
        100,
        key="whatif_nsim",
    )

    if st.sidebar.button("Analyze"):
        with st.spinner("Running counterfactual analysis..."):
            try:
                resp = httpx.post(
                    f"{API_URL}/inflection-points",
                    json={
                        "static_features": features,
                        "perturb_variable": perturb_var,
                        "perturb_value": perturb_val,
                        "n_simulations": n_sim,
                    },
                    timeout=120.0,
                )
                resp.raise_for_status()
            except httpx.HTTPError as e:
                st.error(f"API error: {e}")
                return

        data = resp.json()
        deltas = data["deltas_by_age"]
        ages = [d["age"] for d in deltas]
        income_deltas = [d["delta_income"] for d in deltas]
        satis_deltas = [d["delta_satisfaction"] for d in deltas]

        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=ages,
                    y=income_deltas,
                    name="Income Delta",
                )
            )
            fig.update_layout(
                title=f"Income Impact of {perturb_var}={perturb_val}",
                xaxis_title="Age",
                yaxis_title="Delta ($)",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=ages,
                    y=satis_deltas,
                    name="Satisfaction Delta",
                    marker_color="coral",
                )
            )
            fig.update_layout(
                title="Satisfaction Impact",
                xaxis_title="Age",
                yaxis_title="Delta",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Inflection Points")
        for ip in data["inflection_points"]:
            st.write(
                f"**Age {ip['age']}**: "
                f"Income delta ${ip['delta_income']:,.0f}, "
                f"Satisfaction delta {ip['delta_satisfaction']:.3f} "
                f"(significance: {ip['significance']:.2f})",
            )

        st.metric(
            "Overall Effect Size",
            f"${data['overall_effect_size']:,.0f}",
        )
        st.write(
            f"Archetype shift: {data['base_archetype']} → {data['perturbed_archetype']}",
        )


if __name__ == "__main__":
    main()
