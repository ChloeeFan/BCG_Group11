import streamlit as st
import pandas as pd
import geopandas as gpd
import json
import folium
import plotly.graph_objects as go
from streamlit_folium import st_folium
import unicodedata
import re

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BCG · Agri Supply-Chain Strategy",
    layout="wide",
    page_icon="�",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS — Premium Dark Theme
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main container */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
        max-width: 1200px;
    }
    /* KPI Cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.78rem !important;
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #f1f5f9 !important;
    }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #e2e8f0 !important;
    }
    /* Title */
    h1 {
        background: linear-gradient(90deg, #10b981, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# HELPER: Normalize department names for fuzzy matching
# ─────────────────────────────────────────────────────────────
def normalize_name(name: str) -> str:
    """Strip accents, lowercase, replace hyphens/underscores/spaces with _"""
    s = unicodedata.normalize("NFD", name)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")  # strip accents
    s = s.lower().strip()
    s = re.sub(r"[\s\-']+", "_", s)  # unify separators
    return s


# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("production_forecast_optimized.csv")
    except Exception:
        st.error("❌ `production_forecast_optimized.csv` not found. Run the Optimization Layer first.")
        return None, None, None

    try:
        gdf = gpd.read_file("france_departments.geojson")
    except Exception:
        st.warning("⚠️ GeoJSON not found — map will be disabled.")
        return df, None, None

    # Build fuzzy name lookup: normalized → original GeoJSON name
    gdf["_norm"] = gdf["nom"].apply(normalize_name)
    df["_norm"]  = df["department"].apply(normalize_name)

    # Merge on normalized key
    merged = gdf.merge(df, on="_norm", how="inner")

    # Build GeoJSON dict for Folium (needs JSON, not GeoDataFrame)
    geojson_dict = json.loads(gdf.to_json())
    # Inject normalized name into each feature for Folium lookup
    for feat, (_, row) in zip(geojson_dict["features"], gdf.iterrows()):
        feat["properties"]["_norm"] = row["_norm"]

    return df, gdf, geojson_dict


df, gdf, geojson_dict = load_data()

if df is None:
    st.stop()

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.title("🌾 Agricultural Supply-Chain Optimization")
st.caption("Interactive strategy dashboard · France departments · 2015 → 2050")

# ─────────────────────────────────────────────────────────────
# 2. SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls")

    min_year = int(df["year"].min())
    max_year = int(df["year"].max())
    selected_year = st.slider("📅 Year", min_year, max_year, 2030)

    st.markdown("---")
    st.markdown("### 🗺️ Map Metric")
    metric_options = {
        "Production Volume (t)": "Recommended Allocation (Vol)",
        "Allocation Share (%)":  "Recommended %",
        "Reallocation Adjustment (t)": "Adjustment",
        "Shift vs Climate Only (pp)":  "Delta pp",
    }
    selected_metric_label = st.selectbox(
        "Colour departments by", list(metric_options.keys()), label_visibility="collapsed"
    )
    selected_metric_col = metric_options[selected_metric_label]

    st.markdown("---")
    st.caption("Built for BCG Financial Model")

# ─────────────────────────────────────────────────────────────
# 3. FILTER DATA FOR SELECTED YEAR
# ─────────────────────────────────────────────────────────────
df_year = df[df["year"] == selected_year].copy()

# ─────────────────────────────────────────────────────────────
# 4. KPI ROW
# ─────────────────────────────────────────────────────────────
st.markdown("")
c1, c2, c3 = st.columns(3)
total_vol   = df_year["Recommended Allocation (Vol)"].sum()
total_adj   = df_year["Adjustment"].abs().sum() / 2  # avoid double-count
unalloc     = df_year["unallocated_volume_tonnes"].sum() if "unallocated_volume_tonnes" in df_year.columns else 0

c1.metric("🏭 National Production Target", f"{total_vol:,.0f} t")
c2.metric("🔄 Volume Reallocated",        f"{total_adj:,.0f} t")
c3.metric("🚫 Stranded Capacity",         f"{unalloc:,.0f} t")

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# 5. TABS
# ─────────────────────────────────────────────────────────────
tab_map, tab_ts, tab_shift = st.tabs([
    "🗺️  France Map",
    "📈  Department Time Series",
    "🏆  Gainers & Losers",
])

# ═══════════════════ TAB 1: FOLIUM MAP ═══════════════════
with tab_map:
    st.subheader(f"{selected_metric_label} — {selected_year}")

    if gdf is not None and geojson_dict is not None:
        # Build a lookup: normalised_name → metric value
        value_map = dict(zip(df_year["_norm"], df_year[selected_metric_col]))

        # Determine colour scale
        vals = df_year[selected_metric_col].dropna()
        is_diverging = "Shift" in selected_metric_label or "Adjustment" in selected_metric_label

        # Create Folium map centered on France
        m = folium.Map(
            location=[46.8, 2.3],
            zoom_start=6,
            tiles="CartoDB positron",
            control_scale=True,
        )

        # Choropleth layer
        choropleth = folium.Choropleth(
            geo_data=geojson_dict,
            data=df_year,
            columns=["_norm", selected_metric_col],
            key_on="feature.properties._norm",
            fill_color="RdYlGn" if is_diverging else "YlGnBu",
            fill_opacity=0.75,
            line_opacity=0.4,
            line_weight=1,
            legend_name=selected_metric_label,
            nan_fill_color="#2d3748",
            highlight=True,
        )
        choropleth.add_to(m)

        # Tooltip
        style_func = lambda x: {
            "fillOpacity": 0.75,
            "weight": 1,
            "color": "#475569",
        }
        highlight_func = lambda x: {
            "fillOpacity": 0.95,
            "weight": 3,
            "color": "#10b981",
        }

        # Build tooltip data
        tooltip_data = {}
        for _, row in df_year.iterrows():
            tooltip_data[row["_norm"]] = {
                "dept": row["department"],
                "vol": f'{row["Recommended Allocation (Vol)"]:,.0f} t',
                "adj": f'{row["Adjustment"]:+,.0f} t',
                "pct": f'{row["Recommended %"]:.2f}%',
            }

        # Add GeoJSON overlay with tooltips
        folium.GeoJson(
            geojson_dict,
            style_function=lambda x: {"fillOpacity": 0, "weight": 0},
            highlight_function=highlight_func,
            tooltip=folium.GeoJsonTooltip(
                fields=["nom"],
                aliases=["Department"],
                style="background-color:#1e293b;color:white;font-size:13px;padding:8px;border-radius:6px;",
            ),
        ).add_to(m)

        st_folium(m, width=None, height=560, returned_objects=[])
    else:
        st.info("Map disabled — GeoJSON file not found.")
        st.dataframe(
            df_year.sort_values("Recommended Allocation (Vol)", ascending=False)
            .head(20)[["department", selected_metric_col, "Recommended Allocation (Vol)"]]
        )

# ═══════════════════ TAB 2: TIME SERIES ═══════════════════
with tab_ts:
    st.subheader("Department Reallocation Trajectory")
    st.markdown(
        "Compare the **passive climate baseline** _(grey dashed)_ vs. the **optimised allocation** _(blue)_. "
        "Green/red bars show the net volume added or removed by the optimiser each year."
    )

    departments = sorted(df["department"].unique())
    default_ix  = departments.index("Gironde") if "Gironde" in departments else 0
    selected_dept = st.selectbox("Select department", departments, index=default_ix)

    df_dept = df[df["department"] == selected_dept].sort_values("year")

    if not df_dept.empty:
        fig_ts = go.Figure()

        # Adjustment bars (secondary y-axis)
        bar_colors = ["#10b981" if v >= 0 else "#ef4444" for v in df_dept["Adjustment"]]
        fig_ts.add_trace(go.Bar(
            x=df_dept["year"], y=df_dept["Adjustment"],
            name="Net Reallocation",
            marker_color=bar_colors, opacity=0.45,
            yaxis="y2",
        ))

        # Passive baseline
        fig_ts.add_trace(go.Scatter(
            x=df_dept["year"],
            y=df_dept["Passive Climate Allocation (Vol)"],
            name="Passive Climate Baseline",
            mode="lines",
            line=dict(color="#94a3b8", width=2, dash="dash"),
        ))

        # Optimised line
        fig_ts.add_trace(go.Scatter(
            x=df_dept["year"],
            y=df_dept["Recommended Allocation (Vol)"],
            name="Optimised Allocation",
            mode="lines+markers",
            line=dict(color="#3b82f6", width=3),
            marker=dict(size=5),
        ))

        fig_ts.update_layout(
            title=dict(text=f"<b>{selected_dept}</b> — Volume & Reallocation", font=dict(size=16)),
            xaxis=dict(title="Year", dtick=5),
            yaxis=dict(
                title=dict(text="Volume (t)", font=dict(color="#3b82f6")),
                tickfont=dict(color="#3b82f6"),
            ),
            yaxis2=dict(
                title=dict(text="Reallocation (t)", font=dict(color="#94a3b8")),
                tickfont=dict(color="#94a3b8"),
                anchor="x", overlaying="y", side="right",
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
            hovermode="x unified",
            margin=dict(l=0, r=0, t=60, b=0),
            template="plotly_dark",
            height=480,
        )

        st.plotly_chart(fig_ts, use_container_width=True)

# ═══════════════════ TAB 3: WINNERS & LOSERS ═══════════════════
with tab_shift:
    st.subheader(f"Strategic Reallocation — {selected_year}")
    st.markdown("Departments **gaining** volume from vulnerable areas vs. departments **shedding** volume.")

    col_l, col_r = st.columns(2)

    display_cols = ["department", "Delta pp", "Adjustment", "Recommended Allocation (Vol)"]

    with col_l:
        st.markdown("#### 🟢 Top 10 Gaining Regions")
        winners = df_year.nlargest(10, "Adjustment")[display_cols].reset_index(drop=True)
        st.dataframe(
            winners.style.format({
                "Delta pp": "{:+.4f}",
                "Adjustment": "{:+,.0f}",
                "Recommended Allocation (Vol)": "{:,.0f}",
            }).background_gradient(subset=["Adjustment"], cmap="Greens"),
            hide_index=True,
            height=420,
        )

    with col_r:
        st.markdown("#### 🔴 Top 10 Losing Regions")
        losers = df_year.nsmallest(10, "Adjustment")[display_cols].reset_index(drop=True)
        st.dataframe(
            losers.style.format({
                "Delta pp": "{:+.4f}",
                "Adjustment": "{:+,.0f}",
                "Recommended Allocation (Vol)": "{:,.0f}",
            }).background_gradient(subset=["Adjustment"], cmap="Reds_r"),
            hide_index=True,
            height=420,
        )