"""
🌊 Analyse de l'Érosion Côtière — Nouakchott, Mauritanie
Sentinel-2 (NDWI) | Google Earth Engine | Streamlit
"""

import streamlit as st
import ee
import folium
from folium import plugins
from streamlit_folium import st_folium
import plotly.graph_objects as go
import pandas as pd

# ═══════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════
st.set_page_config(
    page_title="🌊 Érosion Côtière - Nouakchott",
    page_icon="🌊",
    layout="wide",
)

st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    .hero {
        background: linear-gradient(135deg, #0f172a, #1e40af, #0ea5e9);
        color: white;
        padding: 24px 32px;
        border-radius: 14px;
        margin-bottom: 20px;
    }
    .hero h1 { margin: 0; font-size: 2rem; }
    .hero p  { margin: 4px 0 0 0; opacity: 0.85; font-size: 1.05rem; }
    .stat-box {
        background: white;
        border-radius: 10px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-top: 4px solid;
    }
    .stat-box h2 { margin: 6px 0 2px 0; font-size: 1.6rem; }
    .stat-box p  { margin: 0; color: #64748b; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
#  GEE INIT
# ═══════════════════════════════════════════════════════
GEE_PROJECT = "concise-metrics-471614-f2"

@st.cache_resource
def init_ee():
    # 1) Essayer avec les credentials locales (earthengine authenticate)
    try:
        ee.Initialize(project=GEE_PROJECT)
        return True
    except Exception:
        pass

    # 2) Essayer avec un Service Account (Streamlit Cloud)
    try:
        sa = st.secrets["GEE_SERVICE_ACCOUNT"]
        ck = st.secrets["GEE_CREDENTIALS"]
        creds = ee.ServiceAccountCredentials(sa, key_data=ck)
        ee.Initialize(creds, project=GEE_PROJECT)
        return True
    except (KeyError, FileNotFoundError):
        pass
    except Exception as e:
        st.error(f"❌ Erreur GEE (Service Account) : {e}")
        st.stop()

    # 3) Dernier recours : authentification interactive
    try:
        ee.Authenticate()
        ee.Initialize(project=GEE_PROJECT)
        return True
    except Exception as e:
        st.error(f"❌ Impossible d'initialiser GEE : {e}")
        st.info("💡 Exécutez `earthengine authenticate` dans le terminal.")
        st.stop()

init_ee()

# ═══════════════════════════════════════════════════════
#  ZONES PRÉDÉFINIES
# ═══════════════════════════════════════════════════════
ZONES = {
    "Nouakchott — Côte complète": {
        "bounds": [-16.04, 18.00, -15.88, 18.20],
        "center": [18.10, -15.96],
        "zoom": 12,
    },
    "Nouakchott — Plage Sud": {
        "bounds": [-16.03, 18.00, -15.95, 18.06],
        "center": [18.03, -15.99],
        "zoom": 14,
    },
    "Nouakchott — Centre-ville": {
        "bounds": [-16.03, 18.06, -15.92, 18.12],
        "center": [18.09, -15.97],
        "zoom": 14,
    },
    "Nouakchott — Nord (Toujounine)": {
        "bounds": [-16.03, 18.12, -15.90, 18.20],
        "center": [18.16, -15.96],
        "zoom": 14,
    },
    "Zone personnalisée": {
        "bounds": None,
        "center": [18.10, -15.96],
        "zoom": 12,
    },
}

TRANSECTS_ALL = [
    {"lon": -16.00, "lat": 18.01, "nom": "T01 Plage Sud"},
    {"lon": -16.00, "lat": 18.03, "nom": "T02 Plage Centre S"},
    {"lon": -16.00, "lat": 18.05, "nom": "T03 Port de Pêche"},
    {"lon": -16.00, "lat": 18.07, "nom": "T04 Centre-Ville"},
    {"lon": -16.00, "lat": 18.09, "nom": "T05 Tevragh Zeina"},
    {"lon": -16.00, "lat": 18.11, "nom": "T06 Ksar Nord"},
    {"lon": -16.00, "lat": 18.13, "nom": "T07 Sebkha"},
    {"lon": -16.00, "lat": 18.15, "nom": "T08 Toujounine S"},
    {"lon": -16.00, "lat": 18.17, "nom": "T09 Toujounine N"},
    {"lon": -16.00, "lat": 18.19, "nom": "T10 Nord Limite"},
]

# ═══════════════════════════════════════════════════════
#  GEE PROCESSING
# ═══════════════════════════════════════════════════════
def get_ndwi_and_water(start, end, aoi):
    """Charger Sentinel-2, calculer NDWI, retourner masque eau."""
    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 15))
        .select(["B3", "B8"])
    )
    count = col.size().getInfo()
    ndwi = col.median().clip(aoi).normalizedDifference(["B3", "B8"]).rename("ndwi")
    water = (
        ndwi.gt(0.1)
        .focal_max(1, "circle", "pixels")
        .focal_min(1, "circle", "pixels")
        .rename("eau")
    )
    return ndwi, water, count


def get_tile_url(image, vis):
    """Obtenir l'URL des tuiles pour affichage dans Folium."""
    map_id = image.getMapId(vis)
    return map_id["tile_fetcher"].url_format


@st.cache_data(ttl=3600, show_spinner=False)
def run_analysis(bounds, date1_start, date1_end, date2_start, date2_end):
    """Analyse complète : érosion, accrétion, transects."""

    aoi = ee.Geometry.Rectangle(bounds)

    # ── Charger les deux périodes ──
    ndwi1, water1, count1 = get_ndwi_and_water(date1_start, date1_end, aoi)
    ndwi2, water2, count2 = get_ndwi_and_water(date2_start, date2_end, aoi)

    # ── Changements ──
    erosion = water2.And(water1.Not()).selfMask().rename("eau")
    accretion = water1.And(water2.Not()).selfMask().rename("eau")

    pixel_area = ee.Image.pixelArea()

    tot_e = (
        erosion.multiply(pixel_area)
        .reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=20,
            maxPixels=1e10,
            bestEffort=True,
        )
        .getNumber("eau")
        .getInfo()
        or 0
    )
    tot_a = (
        accretion.multiply(pixel_area)
        .reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=20,
            maxPixels=1e10,
            bestEffort=True,
        )
        .getNumber("eau")
        .getInfo()
        or 0
    )

    # ── Tuiles pour la carte ──
    ndwi_vis = {
        "min": -0.5,
        "max": 0.5,
        "palette": ["8B4513", "F5DEB3", "FFFFFF", "87CEEB", "0000CD"],
    }
    tile1 = get_tile_url(ndwi1, ndwi_vis)
    tile2 = get_tile_url(ndwi2, ndwi_vis)
    tile_ero = get_tile_url(erosion, {"palette": ["FF0000"]})
    tile_acc = get_tile_url(accretion, {"palette": ["00CC00"]})

    # ── Transects dans la zone ──
    lat_min, lat_max = bounds[1], bounds[3]
    active_transects = [
        t for t in TRANSECTS_ALL if lat_min <= t["lat"] <= lat_max
    ]

    transect_rows = []
    for t in active_transects:
        zone = ee.Geometry.Point([t["lon"], t["lat"]]).buffer(800).bounds()

        se = (
            erosion.multiply(pixel_area)
            .reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=zone,
                scale=20,
                maxPixels=1e8,
                bestEffort=True,
            )
            .getNumber("eau")
            .getInfo()
            or 0
        )
        sa = (
            accretion.multiply(pixel_area)
            .reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=zone,
                scale=20,
                maxPixels=1e8,
                bestEffort=True,
            )
            .getNumber("eau")
            .getInfo()
            or 0
        )
        recul = round(se / 1600)
        avancee = round(sa / 1600)
        transect_rows.append(
            {
                "Transect": t["nom"],
                "Lat": t["lat"],
                "Recul (m)": recul,
                "Avancée (m)": avancee,
                "Bilan (m)": avancee - recul,
            }
        )

    return {
        "count1": count1,
        "count2": count2,
        "erosion_m2": round(tot_e),
        "accretion_m2": round(tot_a),
        "tile1": tile1,
        "tile2": tile2,
        "tile_ero": tile_ero,
        "tile_acc": tile_acc,
        "transects": transect_rows,
    }


# ═══════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Paramètres")

    # ── Zone ──
    st.markdown("### 📍 Zone d'étude")
    zone_name = st.selectbox("Choisir une zone", list(ZONES.keys()))
    zone = ZONES[zone_name]

    if zone_name == "Zone personnalisée":
        st.markdown("Entrez les coordonnées :")
        c1, c2 = st.columns(2)
        lon_min = c1.number_input("Lon min", value=-16.04, format="%.4f")
        lat_min = c2.number_input("Lat min", value=18.00, format="%.4f")
        lon_max = c1.number_input("Lon max", value=-15.88, format="%.4f")
        lat_max = c2.number_input("Lat max", value=18.20, format="%.4f")
        bounds = [lon_min, lat_min, lon_max, lat_max]
        center = [(lat_min + lat_max) / 2, (lon_min + lon_max) / 2]
        zoom = 12
    else:
        bounds = zone["bounds"]
        center = zone["center"]
        zoom = zone["zoom"]

    st.markdown("---")

    # ── Années ──
    st.markdown("### 📅 Périodes")

    YEARS = list(range(2015, 2027))

    st.markdown("**Période de référence**")
    cr1, cr2 = st.columns(2)
    y1_start = cr1.selectbox("De", YEARS, index=0, key="y1s")
    y1_end = cr2.selectbox("À", YEARS, index=1, key="y1e")

    st.markdown("**Période récente**")
    cr3, cr4 = st.columns(2)
    y2_start = cr3.selectbox("De", YEARS, index=9, key="y2s")
    y2_end = cr4.selectbox("À", YEARS, index=11, key="y2e")

    date1_start = f"{y1_start}-01-01"
    date1_end = f"{y1_end}-12-31"
    date2_start = f"{y2_start}-01-01"
    date2_end = f"{y2_end}-12-31"

    ecart = y2_start - y1_start

    st.markdown("---")
    st.markdown("### 🛰️ Données")
    st.caption("Sentinel-2 SR Harmonized")
    st.caption("NDWI = (B3 − B8) / (B3 + B8)")
    st.caption("Seuil eau : NDWI > 0.1")
    st.caption("Nuages < 15 %")

    st.markdown("---")
    run_btn = st.button("🚀 Lancer l'analyse", type="primary", use_container_width=True)

# ═══════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════
st.markdown(
    f"""
<div class='hero'>
    <h1>🌊 Analyse de l'Érosion Côtière</h1>
    <p>{zone_name} — Sentinel-2 NDWI — {y1_start}-{y1_end} vs {y2_start}-{y2_end} ({ecart} ans)</p>
</div>
""",
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════
#  STATE
# ═══════════════════════════════════════════════════════
if "results" not in st.session_state:
    st.session_state.results = None

if run_btn:
    with st.spinner("🔄 Analyse en cours… (1-3 minutes selon la zone)"):
        st.session_state.results = run_analysis(
            bounds, date1_start, date1_end, date2_start, date2_end
        )

results = st.session_state.results

if results is None:
    st.info("👈 Configurez les paramètres dans la barre latérale puis cliquez sur **Lancer l'analyse**.")
    st.stop()

# ═══════════════════════════════════════════════════════
#  MÉTRIQUES
# ═══════════════════════════════════════════════════════
st.markdown("### 📊 Résultats globaux")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f"""<div class='stat-box' style='border-top-color:#ef4444;'>
        <p>🔴 Érosion totale</p>
        <h2>{results['erosion_m2']:,} m²</h2>
        <p>{results['erosion_m2']/10000:,.2f} ha</p>
        </div>""",
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""<div class='stat-box' style='border-top-color:#22c55e;'>
        <p>🟢 Accrétion totale</p>
        <h2>{results['accretion_m2']:,} m²</h2>
        <p>{results['accretion_m2']/10000:,.2f} ha</p>
        </div>""",
        unsafe_allow_html=True,
    )

with c3:
    net = results["accretion_m2"] - results["erosion_m2"]
    color = "#22c55e" if net >= 0 else "#ef4444"
    label = "Gain net" if net >= 0 else "Perte nette"
    st.markdown(
        f"""<div class='stat-box' style='border-top-color:{color};'>
        <p>📐 {label}</p>
        <h2 style='color:{color};'>{abs(net):,} m²</h2>
        <p>{abs(net)/10000:,.2f} ha</p>
        </div>""",
        unsafe_allow_html=True,
    )

with c4:
    rate = round(results["erosion_m2"] / 20000 / ecart, 2) if ecart > 0 else 0
    st.markdown(
        f"""<div class='stat-box' style='border-top-color:#3b82f6;'>
        <p>📏 Taux de recul</p>
        <h2>{rate} m/an</h2>
        <p>{results['count1']} + {results['count2']} images S-2</p>
        </div>""",
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════
#  CARTE
# ═══════════════════════════════════════════════════════
st.markdown("### 🗺️ Carte interactive")

tab_compare, tab_erosion = st.tabs(
    ["🔄 Comparaison NDWI", "🔴🟢 Érosion / Accrétion"]
)

with tab_compare:
    st.caption("Glissez le curseur au centre pour comparer les deux périodes")

    m1 = plugins.DualMap(location=center, zoom_start=zoom, layout="horizontal")

    folium.TileLayer(
        tiles=results["tile1"],
        attr="GEE Sentinel-2",
        name=f"NDWI {y1_start}-{y1_end}",
        overlay=True,
    ).add_to(m1.m1)

    folium.TileLayer(
        tiles=results["tile2"],
        attr="GEE Sentinel-2",
        name=f"NDWI {y2_start}-{y2_end}",
        overlay=True,
    ).add_to(m1.m2)

    # Ajouter zone d'étude
    folium.Rectangle(
        bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
        color="red",
        weight=2,
        fill=False,
        popup="Zone d'étude",
    ).add_to(m1.m1)

    folium.Rectangle(
        bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
        color="red",
        weight=2,
        fill=False,
        popup="Zone d'étude",
    ).add_to(m1.m2)

    st_folium(m1, height=500, use_container_width=True, returned_objects=[])

with tab_erosion:
    m2 = folium.Map(location=center, zoom_start=zoom, tiles="Esri.WorldImagery")

    folium.TileLayer(
        tiles=results["tile_ero"],
        attr="GEE",
        name="🔴 Érosion",
        overlay=True,
    ).add_to(m2)

    folium.TileLayer(
        tiles=results["tile_acc"],
        attr="GEE",
        name="🟢 Accrétion",
        overlay=True,
    ).add_to(m2)

    # Transects sur la carte
    for t in results["transects"]:
        color = "#ef4444" if t["Bilan (m)"] < 0 else "#22c55e"
        folium.CircleMarker(
            location=[t["Lat"], -16.00],
            radius=6,
            color=color,
            fill=True,
            fill_opacity=0.8,
            popup=f"{t['Transect']}<br>Bilan: {t['Bilan (m)']} m",
        ).add_to(m2)

    folium.LayerControl().add_to(m2)
    st_folium(m2, height=500, use_container_width=True, returned_objects=[])

# ═══════════════════════════════════════════════════════
#  TRANSECTS
# ═══════════════════════════════════════════════════════
if results["transects"]:
    st.markdown("### 📐 Analyse par transects")

    df = pd.DataFrame(results["transects"])

    # ── Graphique ──
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        fig1 = go.Figure()
        fig1.add_trace(
            go.Bar(
                x=df["Transect"],
                y=df["Recul (m)"],
                name="Recul",
                marker_color="#ef4444",
            )
        )
        fig1.add_trace(
            go.Bar(
                x=df["Transect"],
                y=df["Avancée (m)"],
                name="Avancée",
                marker_color="#22c55e",
            )
        )
        fig1.update_layout(
            title="Recul vs Avancée par transect",
            barmode="group",
            template="plotly_white",
            height=380,
            xaxis_tickangle=-45,
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col_g2:
        colors = ["#22c55e" if v >= 0 else "#ef4444" for v in df["Bilan (m)"]]
        fig2 = go.Figure(
            go.Bar(
                x=df["Transect"],
                y=df["Bilan (m)"],
                marker_color=colors,
                text=[f"{v:+d}" for v in df["Bilan (m)"]],
                textposition="outside",
            )
        )
        fig2.update_layout(
            title="Bilan net par transect",
            template="plotly_white",
            height=380,
            xaxis_tickangle=-45,
            shapes=[
                dict(
                    type="line",
                    y0=0,
                    y1=0,
                    x0=-0.5,
                    x1=len(df) - 0.5,
                    line=dict(color="gray", dash="dash"),
                )
            ],
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Tableau ──
    def color_bilan(val):
        if isinstance(val, (int, float)):
            if val < 0:
                return "color: #ef4444; font-weight: bold"
            elif val > 0:
                return "color: #22c55e; font-weight: bold"
        return ""

    styled = df.drop(columns=["Lat"]).style.applymap(
        color_bilan, subset=["Bilan (m)"]
    )
    st.dataframe(styled, use_container_width=True, height=380)

    # ── CSV ──
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "📥 Télécharger les résultats (CSV)",
        data=csv,
        file_name=f"transects_{y1_start}_{y2_end}.csv",
        mime="text/csv",
    )

# ═══════════════════════════════════════════════════════
#  MÉTHODOLOGIE
# ═══════════════════════════════════════════════════════
with st.expander("📖 Méthodologie"):
    st.markdown(
        f"""
**Données** : Sentinel-2 SR Harmonized (ESA/Copernicus) — B3 (Vert, 10m) + B8 (NIR, 10m)

**Indice** : NDWI = (B3 − B8) / (B3 + B8) — seuil > 0.1 → eau

**Nettoyage** : focal_max + focal_min (morphologie) pour supprimer le bruit

**Changements** :
- Érosion = terre en {y1_start}-{y1_end} devenue eau en {y2_start}-{y2_end}
- Accrétion = eau en {y1_start}-{y1_end} devenue terre en {y2_start}-{y2_end}

**Transects** : 10 zones perpendiculaires au trait de côte (buffer 800m = 1600m de large).
Recul (m) = surface érodée dans le transect / 1600 m

**Pourquoi Sentinel-2 et pas Sentinel-1 (radar) ?**
Le sable du Sahara mauritanien produit une rétrodiffusion radar très faible en bande C,
similaire à celle de l'eau. Cela rend la classification mer/terre par SAR peu fiable.
Le NDWI optique sépare nettement eau, sable et terre.

**Limites** :
- Effets de marée (influence la position du trait de côte)
- Résolution de 10m (ne détecte pas les changements < 10m)
- Composition médiane lisse les variations saisonnières
"""
    )

# ═══════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#94a3b8; font-size:13px;'>"
    "🌊 Érosion Côtière — Nouakchott | Sentinel-2 NDWI | GEE + Streamlit<br>"
    "Observation de la Terre — TP 2026"
    "</div>",
    unsafe_allow_html=True,
)