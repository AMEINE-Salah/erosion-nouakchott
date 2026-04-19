"""
🌊 Analyse de l'Érosion Côtière — Mauritanie
Sentinel-2 (NDWI) | Google Earth Engine | Streamlit
"""

import streamlit as st
import ee
import folium
from folium import plugins
from folium.plugins import Draw
from streamlit_folium import st_folium
import plotly.graph_objects as go
import pandas as pd
import json

# ═══════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════
st.set_page_config(
    page_title="🌊 Érosion Côtière - Mauritanie",
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
    .step-box {
        background: #f0f9ff;
        border: 1px solid #bae6fd;
        border-radius: 10px;
        padding: 16px;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
#  GEE INIT
# ═══════════════════════════════════════════════════════
GEE_PROJECT = "concise-metrics-471614-f2"

@st.cache_resource
def init_ee():
    try:
        ee.Initialize(project=GEE_PROJECT)
        return True
    except Exception:
        pass
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
#  GEE FUNCTIONS
# ═══════════════════════════════════════════════════════
def get_ndwi_and_water(start, end, aoi):
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
    map_id = image.getMapId(vis)
    return map_id["tile_fetcher"].url_format


def run_analysis(bounds, date1_start, date1_end, date2_start, date2_end):
    aoi = ee.Geometry.Rectangle(bounds)

    ndwi1, water1, count1 = get_ndwi_and_water(date1_start, date1_end, aoi)
    ndwi2, water2, count2 = get_ndwi_and_water(date2_start, date2_end, aoi)

    erosion = water2.And(water1.Not()).selfMask().rename("eau")
    accretion = water1.And(water2.Not()).selfMask().rename("eau")

    pixel_area = ee.Image.pixelArea()

    tot_e = (
        erosion.multiply(pixel_area)
        .reduceRegion(reducer=ee.Reducer.sum(), geometry=aoi, scale=20, maxPixels=1e10, bestEffort=True)
        .getNumber("eau").getInfo() or 0
    )
    tot_a = (
        accretion.multiply(pixel_area)
        .reduceRegion(reducer=ee.Reducer.sum(), geometry=aoi, scale=20, maxPixels=1e10, bestEffort=True)
        .getNumber("eau").getInfo() or 0
    )

    ndwi_vis = {"min": -0.5, "max": 0.5, "palette": ["8B4513", "F5DEB3", "FFFFFF", "87CEEB", "0000CD"]}
    tile1 = get_tile_url(ndwi1, ndwi_vis)
    tile2 = get_tile_url(ndwi2, ndwi_vis)
    tile_ero = get_tile_url(erosion, {"palette": ["FF0000"]})
    tile_acc = get_tile_url(accretion, {"palette": ["00CC00"]})

    # Transects automatiques le long de la côte sélectionnée
    lon_min, lat_min, lon_max, lat_max = bounds
    coast_lon = lon_min + (lon_max - lon_min) * 0.3  # approximation de la ligne de côte
    lat_range = lat_max - lat_min
    n_transects = min(10, max(3, int(lat_range / 0.02)))

    transect_rows = []
    for i in range(n_transects):
        lat = lat_min + (i + 0.5) * lat_range / n_transects
        zone = ee.Geometry.Point([coast_lon, lat]).buffer(800).bounds()

        se = (
            erosion.multiply(pixel_area)
            .reduceRegion(reducer=ee.Reducer.sum(), geometry=zone, scale=20, maxPixels=1e8, bestEffort=True)
            .getNumber("eau").getInfo() or 0
        )
        sa = (
            accretion.multiply(pixel_area)
            .reduceRegion(reducer=ee.Reducer.sum(), geometry=zone, scale=20, maxPixels=1e8, bestEffort=True)
            .getNumber("eau").getInfo() or 0
        )
        recul = round(se / 1600)
        avancee = round(sa / 1600)
        transect_rows.append({
            "Transect": f"T{i+1:02d}",
            "Lat": round(lat, 4),
            "Lon": round(coast_lon, 4),
            "Recul (m)": recul,
            "Avancée (m)": avancee,
            "Bilan (m)": avancee - recul,
        })

    return {
        "count1": count1, "count2": count2,
        "erosion_m2": round(tot_e), "accretion_m2": round(tot_a),
        "tile1": tile1, "tile2": tile2,
        "tile_ero": tile_ero, "tile_acc": tile_acc,
        "transects": transect_rows,
        "bounds": bounds,
    }


# ═══════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════
st.markdown("""
<div class='hero'>
    <h1>🌊 Analyse de l'Érosion Côtière</h1>
    <p>Côte Atlantique — Mauritanie | Sentinel-2 NDWI | Google Earth Engine</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
#  SIDEBAR — PARAMÈTRES
# ═══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Paramètres")

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

# ═══════════════════════════════════════════════════════
#  ÉTAPE 1 — SÉLECTION DE LA ZONE
# ═══════════════════════════════════════════════════════
st.markdown("### 📍 Étape 1 : Sélectionnez votre zone d'étude")

st.markdown("""
<div class='step-box'>
    <b>🖊️ Instructions :</b> Utilisez l'outil rectangle <b>▭</b> (à gauche de la carte) 
    pour dessiner un rectangle sur la zone côtière que vous souhaitez analyser.<br>
    <b>⚠️ Important :</b> La zone doit contenir du littoral (mer + terre).
</div>
""", unsafe_allow_html=True)

# Carte centrée sur toute la côte mauritanienne
# Côte mauritanienne : du Cap Blanc (20.85°N) au fleuve Sénégal (16.05°N)
m_select = folium.Map(
    location=[18.5, -16.2],
    zoom_start=7,
    tiles="Esri.WorldImagery",
)

# Ajouter les noms des villes côtières
villes = [
    {"nom": "Nouadhibou", "lat": 20.93, "lon": -17.03},
    {"nom": "Cap Blanc", "lat": 20.85, "lon": -17.05},
    {"nom": "Nouakchott", "lat": 18.09, "lon": -15.98},
    {"nom": "Parc Banc d'Arguin", "lat": 19.70, "lon": -16.45},
    {"nom": "Rosso", "lat": 16.51, "lon": -15.81},
    {"nom": "Saint-Louis (SN)", "lat": 16.02, "lon": -16.50},
]

for v in villes:
    folium.Marker(
        location=[v["lat"], v["lon"]],
        popup=v["nom"],
        icon=folium.Icon(color="blue", icon="info-sign"),
    ).add_to(m_select)

# Trait de côte approximatif pour guider l'utilisateur
coast_line = [
    [-17.08, 20.90], [-17.05, 20.50], [-16.90, 20.20],
    [-16.60, 19.80], [-16.50, 19.50], [-16.40, 19.20],
    [-16.30, 18.80], [-16.15, 18.40], [-16.05, 18.10],
    [-16.00, 17.80], [-16.05, 17.50], [-16.10, 17.20],
    [-16.20, 16.80], [-16.30, 16.50], [-16.45, 16.10],
]
folium.PolyLine(
    locations=[[p[1], p[0]] for p in coast_line],
    color="#00bcd4",
    weight=3,
    opacity=0.6,
    dash_array="10",
    popup="Trait de côte approximatif",
).add_to(m_select)

# Outil de dessin (rectangle uniquement)
Draw(
    export=False,
    draw_options={
        "polyline": False,
        "polygon": False,
        "circle": False,
        "circlemarker": False,
        "marker": False,
        "rectangle": {
            "shapeOptions": {
                "color": "#ff4444",
                "weight": 3,
                "fillOpacity": 0.1,
            }
        },
    },
    edit_options={"edit": False},
).add_to(m_select)

map_data = st_folium(m_select, height=500, use_container_width=True, returned_objects=["all_drawings"])

# ═══════════════════════════════════════════════════════
#  EXTRAIRE LA ZONE DESSINÉE
# ═══════════════════════════════════════════════════════
drawn_bounds = None

if map_data and map_data.get("all_drawings"):
    drawings = map_data["all_drawings"]
    if len(drawings) > 0:
        last_drawing = drawings[-1]
        geom = last_drawing.get("geometry", {})
        if geom.get("type") == "Polygon":
            coords = geom["coordinates"][0]
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            drawn_bounds = [min(lons), min(lats), max(lons), max(lats)]

# Afficher la zone sélectionnée ou message
if drawn_bounds:
    st.success(
        f"✅ Zone sélectionnée : "
        f"Lon [{drawn_bounds[0]:.3f}, {drawn_bounds[2]:.3f}] — "
        f"Lat [{drawn_bounds[1]:.3f}, {drawn_bounds[3]:.3f}]"
    )

    # Bouton pour lancer l'analyse
    if st.button("🚀 Lancer l'analyse", type="primary", use_container_width=True):
        with st.spinner("🔄 Analyse en cours… (1-3 minutes selon la taille de la zone)"):
            st.session_state.results = run_analysis(
                drawn_bounds, date1_start, date1_end, date2_start, date2_end
            )
            st.session_state.ecart = ecart
            st.session_state.y1 = f"{y1_start}-{y1_end}"
            st.session_state.y2 = f"{y2_start}-{y2_end}"
        st.rerun()
else:
    st.info("👆 Dessinez un rectangle sur la carte pour sélectionner votre zone d'étude, puis cliquez sur **Lancer l'analyse**.")

# ═══════════════════════════════════════════════════════
#  RÉSULTATS
# ═══════════════════════════════════════════════════════
if "results" not in st.session_state or st.session_state.results is None:
    st.stop()

results = st.session_state.results
ecart = st.session_state.get("ecart", 10)
y1_label = st.session_state.get("y1", "2015-2016")
y2_label = st.session_state.get("y2", "2024-2026")

st.markdown("---")

# ── MÉTRIQUES ──
st.markdown("### 📊 Résultats globaux")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f"""<div class='stat-box' style='border-top-color:#ef4444;'>
        <p>🔴 Érosion totale</p>
        <h2>{results['erosion_m2']:,} m²</h2>
        <p>{results['erosion_m2']/10000:,.2f} ha</p>
        </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(
        f"""<div class='stat-box' style='border-top-color:#22c55e;'>
        <p>🟢 Accrétion totale</p>
        <h2>{results['accretion_m2']:,} m²</h2>
        <p>{results['accretion_m2']/10000:,.2f} ha</p>
        </div>""", unsafe_allow_html=True)

with c3:
    net = results["accretion_m2"] - results["erosion_m2"]
    color = "#22c55e" if net >= 0 else "#ef4444"
    label = "Gain net" if net >= 0 else "Perte nette"
    st.markdown(
        f"""<div class='stat-box' style='border-top-color:{color};'>
        <p>📐 {label}</p>
        <h2 style='color:{color};'>{abs(net):,} m²</h2>
        <p>{abs(net)/10000:,.2f} ha</p>
        </div>""", unsafe_allow_html=True)

with c4:
    coast_len = (results["bounds"][3] - results["bounds"][1]) * 111000 * 0.3
    coast_len = max(coast_len, 1000)
    rate = round(results["erosion_m2"] / coast_len / max(ecart, 1), 2)
    st.markdown(
        f"""<div class='stat-box' style='border-top-color:#3b82f6;'>
        <p>📏 Taux de recul</p>
        <h2>{rate} m/an</h2>
        <p>{results['count1']} + {results['count2']} images S-2</p>
        </div>""", unsafe_allow_html=True)

# ── CARTES RÉSULTATS ──
st.markdown("### 🗺️ Carte des résultats")

tab_compare, tab_erosion = st.tabs(["🔄 Comparaison NDWI", "🔴🟢 Érosion / Accrétion"])

bounds = results["bounds"]
center_lat = (bounds[1] + bounds[3]) / 2
center_lon = (bounds[0] + bounds[2]) / 2

with tab_compare:
    st.caption("Glissez le curseur au centre pour comparer les deux périodes")

    m1 = plugins.DualMap(location=[center_lat, center_lon], zoom_start=13, layout="horizontal")

    folium.TileLayer(tiles=results["tile1"], attr="GEE", name=f"NDWI {y1_label}", overlay=True).add_to(m1.m1)
    folium.TileLayer(tiles=results["tile2"], attr="GEE", name=f"NDWI {y2_label}", overlay=True).add_to(m1.m2)

    folium.Rectangle(bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
                     color="red", weight=2, fill=False).add_to(m1.m1)
    folium.Rectangle(bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
                     color="red", weight=2, fill=False).add_to(m1.m2)

    st_folium(m1, height=500, use_container_width=True, returned_objects=[])

with tab_erosion:
    m2 = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="Esri.WorldImagery")

    folium.TileLayer(tiles=results["tile_ero"], attr="GEE", name="🔴 Érosion", overlay=True).add_to(m2)
    folium.TileLayer(tiles=results["tile_acc"], attr="GEE", name="🟢 Accrétion", overlay=True).add_to(m2)

    for t in results["transects"]:
        clr = "#ef4444" if t["Bilan (m)"] < 0 else "#22c55e"
        folium.CircleMarker(
            location=[t["Lat"], t["Lon"]],
            radius=7, color=clr, fill=True, fill_opacity=0.8,
            popup=f"<b>{t['Transect']}</b><br>Recul: {t['Recul (m)']}m<br>Avancée: {t['Avancée (m)']}m<br>Bilan: {t['Bilan (m)']}m",
        ).add_to(m2)

    folium.LayerControl().add_to(m2)
    st_folium(m2, height=500, use_container_width=True, returned_objects=[])

# ── TRANSECTS ──
if results["transects"]:
    st.markdown("### 📐 Analyse par transects")

    df = pd.DataFrame(results["transects"])

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(x=df["Transect"], y=df["Recul (m)"], name="Recul", marker_color="#ef4444"))
        fig1.add_trace(go.Bar(x=df["Transect"], y=df["Avancée (m)"], name="Avancée", marker_color="#22c55e"))
        fig1.update_layout(
            title="Recul vs Avancée par transect", barmode="group",
            template="plotly_white", height=380, xaxis_tickangle=-45,
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col_g2:
        colors = ["#22c55e" if v >= 0 else "#ef4444" for v in df["Bilan (m)"]]
        fig2 = go.Figure(go.Bar(
            x=df["Transect"], y=df["Bilan (m)"],
            marker_color=colors,
            text=[f"{v:+d}" for v in df["Bilan (m)"]],
            textposition="outside",
        ))
        fig2.update_layout(
            title="Bilan net par transect", template="plotly_white", height=380,
            xaxis_tickangle=-45,
            shapes=[dict(type="line", y0=0, y1=0, x0=-0.5, x1=len(df)-0.5,
                         line=dict(color="gray", dash="dash"))],
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Tableau
    def color_bilan(val):
        if isinstance(val, (int, float)):
            if val < 0:
                return "color: #ef4444; font-weight: bold"
            elif val > 0:
                return "color: #22c55e; font-weight: bold"
        return ""

    styled = df.drop(columns=["Lat", "Lon"]).style.map(color_bilan, subset=["Bilan (m)"])
    st.dataframe(styled, use_container_width=True, height=380)

    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("📥 Télécharger les résultats (CSV)", data=csv,
                       file_name=f"transects_{y1_label}_{y2_label}.csv", mime="text/csv")

# ── MÉTHODOLOGIE ──
with st.expander("📖 Méthodologie"):
    st.markdown(f"""
**Données** : Sentinel-2 SR Harmonized (ESA/Copernicus) — B3 (Vert, 10m) + B8 (NIR, 10m)

**Indice** : NDWI = (B3 − B8) / (B3 + B8) — seuil > 0.1 → eau

**Nettoyage** : focal_max + focal_min (morphologie) pour supprimer le bruit

**Changements** :
- Érosion = terre en {y1_label} devenue eau en {y2_label}
- Accrétion = eau en {y1_label} devenue terre en {y2_label}

**Transects** : zones perpendiculaires au trait de côte (buffer 800m = 1600m de large),
positionnées automatiquement selon la zone sélectionnée.
Recul (m) = surface érodée dans le transect / 1600 m

**Pourquoi Sentinel-2 et pas Sentinel-1 (radar) ?**
Le sable du Sahara mauritanien produit une rétrodiffusion radar très faible en bande C,
similaire à celle de l'eau. Cela rend la classification mer/terre par SAR peu fiable.
Le NDWI optique sépare nettement eau, sable et terre.

**Limites** :
- Effets de marée (influence la position du trait de côte)
- Résolution de 10m (ne détecte pas les changements < 10m)
- Composition médiane lisse les variations saisonnières
""")

# ── FOOTER ──
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#94a3b8; font-size:13px;'>"
    "🌊 Érosion Côtière — Mauritanie | Sentinel-2 NDWI | GEE + Streamlit<br>"
    "Observation de la Terre — TP 2026"
    "</div>", unsafe_allow_html=True)
