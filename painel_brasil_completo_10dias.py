import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State, callback, no_update
import plotly.express as px
import math
from datetime import date

# ----- Limites do Brasil (aprox.) + limites de zoom -----
BRAZIL_BBOX = {"west": -74.5, "east": -32.0, "south": -34.5, "north": 6.0}
CLAMP_ZOOM_MIN = 3.0
CLAMP_ZOOM_MAX = 7.0

# ================== CAMINHOS (Render-friendly) ==================
# Em Render/GitHub: use caminhos relativos ao arquivo .py
BASE_DIR = Path(__file__).resolve().parent

# Se estiver rodando local e quiser manter seu padrão, você pode setar ENV LOCAL_DATA_DIR
# Ex.: set LOCAL_DATA_DIR=C:\Users\Pedro\Downloads
LOCAL_DATA_DIR = os.environ.get("LOCAL_DATA_DIR", "").strip()
PASTA = Path(LOCAL_DATA_DIR) if LOCAL_DATA_DIR else BASE_DIR

ARQ_PREV = PASTA / "previsao_brasil_10dias.xlsx"
ARQ_ATTR = PASTA / "arquivo_completo_brasil.xlsx"

# Usar apenas o JSON (GeoJSON com extensão .json)
GEOJSON_MUN = BASE_DIR / "municipios_br.json"  # coloque esse arquivo no mesmo diretório do .py (no repo)

# ================== PALETAS & ORDENS ==================
CLASS_ORDER = ["Normal", "Baixo", "Severo", "Extremo"]
COLOR_MAP = {
    "Normal": "#C5E0B4",
    "Baixo":  "#F1C40F",
    "Severo": "#E67E22",
    "Extremo":"#C0392B"
}

RISK_ORDER = ["Normal", "Baixo", "Severo", "Extremo"]
RISK_COLORS = {
    "Normal": COLOR_MAP["Normal"],
    "Baixo":  COLOR_MAP["Baixo"],
    "Severo": COLOR_MAP["Severo"],
    "Extremo":COLOR_MAP["Extremo"]
}

BARS_COLORS = {"Tmín":"#BFDBFE","Tméd":"#60A5FA","Tmáx":"#1E3A8A"}

# ================== MAPAS AUXILIARES ==================
PREFIXO_UF = {"11":"RO","12":"AC","13":"AM","14":"RR","15":"PA","16":"AP","17":"TO",
              "21":"MA","22":"PI","23":"CE","24":"RN","25":"PB","26":"PE","27":"AL","28":"SE","29":"BA",
              "31":"MG","32":"ES","33":"RJ","35":"SP",
              "41":"PR","42":"SC","43":"RS",
              "50":"MS","51":"MT","52":"GO","53":"DF"}

UF_REGIAO = {"AC":"Norte","AP":"Norte","AM":"Norte","PA":"Norte","RO":"Norte","RR":"Norte","TO":"Norte",
             "AL":"Nordeste","BA":"Nordeste","CE":"Nordeste","MA":"Nordeste","PB":"Nordeste","PE":"Nordeste",
             "PI":"Nordeste","RN":"Nordeste","SE":"Nordeste",
             "DF":"Centro-Oeste","GO":"Centro-Oeste","MT":"Centro-Oeste","MS":"Centro-Oeste",
             "ES":"Sudeste","MG":"Sudeste","RJ":"Sudeste","SP":"Sudeste",
             "PR":"Sul","RS":"Sul","SC":"Sul"}

# ================== HELPERS ==================
def z7(s: pd.Series) -> pd.Series:
    return pd.Series(s, dtype=str).str.extract(r"(\d+)")[0].str.zfill(7)

def norm_key(x: pd.Series) -> pd.Series:
    s = pd.Series(x, dtype=str).str.normalize("NFKD").str.encode("ascii","ignore").str.decode("ascii")
    s = s.str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
    return s

def carregar_geojson_cdmun(path: Path):
    # Lê GeoJSON (mesmo sendo .json) e:
    # - garante properties.CD_MUN com 7 dígitos
    # - remove features sem geometry (isso pode “sumir” o mapa no Plotly/Mapbox)
    with open(path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    keys = ["CD_MUN","CD_GEOCMU","CD_GEOCODI","CD_MUNIC","CD_IBGE","GEOCODIGO","IBGE","id"]
    feats_ok = []

    for ft in gj.get("features", []):
        if not ft.get("geometry"):
            continue

        p = ft.get("properties", {}) or {}

        cd = None
        for k in keys:
            if k in p and str(p[k]).strip():
                cd = p[k]
                break
        if cd is None:
            cd = ft.get("id")

        if cd is None:
            continue

        p["CD_MUN"] = "".join(ch for ch in str(cd) if ch.isdigit()).zfill(7)
        ft["properties"] = p
        feats_ok.append(ft)

    gj["features"] = feats_ok
    return gj

def lookup_nomes_from_geojson(gj):
    name_keys = ["NM_MUN","NM_MUNICIP","NM_MUNICIPIO","NM_MUNIC","NM_NOME",
                 "NOME_MUN","NOME","name","Name","municipio","MUNICIPIO"]
    d = {}
    for ft in gj.get("features", []):
        p = ft.get("properties", {}) or {}
        cd = str(p.get("CD_MUN","")).zfill(7)
        nm = None
        for k in name_keys:
            if k in p and str(p[k]).strip():
                nm = str(p[k]).strip()
                break
        if cd and nm:
            d[cd] = nm.title()
    d.setdefault("5300108", "Brasília")
    return d

def calc_ehf(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["CD_MUN","data"]).reset_index(drop=True)

    if "Tmean" not in df or df["Tmean"].dropna().empty:
        if "Tmed" in df and not df["Tmed"].dropna().empty:
            df["Tmean"] = df["Tmed"]
        elif {"Tmin","Tmax"}.issubset(df.columns):
            df["Tmean"] = (df["Tmin"] + df["Tmax"]) / 2
        else:
            df["Tmean"] = np.nan

    def _t3_centered(s: pd.Series):
        m = s.rolling(3, center=True, min_periods=3).mean()
        if len(s) >= 3:
            m.iloc[0]  = s.iloc[:3].mean()
            m.iloc[-1] = s.iloc[-3:].mean()
        return m

    df["T3d_prev"] = (df.groupby("CD_MUN")["Tmean"].apply(_t3_centered)
                        .reset_index(level=0, drop=True))
    df["EHI_sig"]  = df["T3d_prev"] - df.get("Tmean_p95")
    df["EHI_accl"] = df["T3d_prev"] - df.get("Tmean_30d")
    df["EHF"]      = df["EHI_sig"].clip(lower=0) * df["EHI_accl"].apply(lambda x: x if pd.notna(x) and x > 1 else 1)
    return df

def classify_by_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df["ratio"] = np.where(df.get("EHF").gt(0) & df.get("EHF99").gt(0), df["EHF"]/df["EHF99"], np.nan)

    def _cls(r):
        if pd.isna(r["ratio"]) or r["EHF"] <= 0: return "Normal"
        if r["ratio"] >= 3:     return "Extremo"
        if r["ratio"] >= 1:     return "Severo"
        if r["ratio"] >= 0.85:  return "Baixo"
        return "Normal"

    df["classification"] = df.apply(_cls, axis=1)
    return df

def build_combined_risk(df: pd.DataFrame):
    if "GeoSES" not in df.columns:
        df["H_norm"]=np.nan; df["V"]=np.nan; df["risk_index"]=np.nan; df["risk_class"]="Normal"
        return df, False

    d = df.copy()
    geoses_num = pd.to_numeric(d["GeoSES"], errors="coerce")
    d["V"] = ((1 - geoses_num) / 2).clip(0, 1)  # vulnerabilidade social (0=baixa,1=alta)

    H = (d["EHF"].clip(lower=0)) / d.get("EHF99").replace(0, np.nan)
    d["H_norm"] = H.clip(0, 1).fillna(0)

    d["risk_index"] = 0.5*d["H_norm"] + 0.5*d["V"]

    d["risk_class"] = pd.cut(
        d["risk_index"].fillna(0),
        [-0.001, 0.33, 0.66, 1.0],
        labels=["Baixo", "Severo", "Extremo"],
        include_lowest=True
    ).astype(object)

    d.loc[d["classification"]=="Normal", "risk_class"] = "Normal"
    return d, d["V"].notna().any()

# ======== AUTO-ZOOM: BBOX helpers ========
def _geom_bbox(geom):
    west = south = float("inf")
    east = north = float("-inf")
    t = (geom or {}).get("type")
    coords = (geom or {}).get("coordinates", [])
    if t == "Polygon":
        rings = coords
    elif t == "MultiPolygon":
        rings = [r for poly in coords for r in poly]
    else:
        rings = []
    for ring in rings:
        for lon, lat in ring:
            if lon < west:  west = lon
            if lon > east:  east = lon
            if lat < south: south = lat
            if lat > north: north = lat
    if west == float("inf"):
        return None
    return (west, south, east, north)

def _union_bbox(bboxes):
    bboxes = [b for b in bboxes if b]
    if not bboxes: return None
    w = min(b[0] for b in bboxes); s = min(b[1] for b in bboxes)
    e = max(b[2] for b in bboxes); n = max(b[3] for b in bboxes)
    return (w, s, e, n)

def _merc_y(lat):
    lat = max(min(lat, 85.0511), -85.0511)
    return math.log(math.tan(math.pi/4 + math.radians(lat)/2))

def bbox_to_center_zoom(bbox, width=1100, height=650, padding=0.06):
    w, s, e, n = bbox
    lon_center = (w + e) / 2
    lat_center = (s + n) / 2
    lon_frac = max((e - w) / 360.0, 1e-9)
    lat_frac = max((_merc_y(n) - _merc_y(s)) / math.pi, 1e-9)
    lon_zoom = math.log2(width  / 256.0 / lon_frac)
    lat_zoom = math.log2(height / 256.0 / lat_frac)
    zoom = min(lon_zoom, lat_zoom) - math.log2(1 + 2*padding)
    return {"lat": lat_center, "lon": lon_center}, float(max(min(zoom, 10), 2.5))

# ================== DADOS ==================
if not ARQ_PREV.exists():
    raise FileNotFoundError(f"Arquivo de previsão não encontrado: {ARQ_PREV}")
if not ARQ_ATTR.exists():
    raise FileNotFoundError(f"Arquivo de atributos não encontrado: {ARQ_ATTR}")

prev = pd.read_excel(ARQ_PREV, engine="openpyxl")
attr = pd.read_excel(ARQ_ATTR, engine="openpyxl")

prev["CD_MUN"] = z7(prev["CD_MUN"])
attr["CD_MUN"] = z7(attr["CD_MUN"])
prev["data"]   = pd.to_datetime(prev["data"], errors="coerce")

keep_attr = ["CD_MUN","NM_MUN","SIGLA_UF","NM_REGIAO","GeoSES",
             "EHF85","EHF90","EHF95","EHF99",
             "Tmean_p90","Tmean_p95","Tmean_p99","Tmean_30d"]
attr = attr[[c for c in keep_attr if c in attr.columns]].drop_duplicates("CD_MUN")

base = prev.merge(attr, on="CD_MUN", how="left")

# GeoJSON (json), nomes e derivação UF/Região
if not GEOJSON_MUN.exists():
    raise FileNotFoundError(f"JSON/GeoJSON de municípios não encontrado: {GEOJSON_MUN}")

GJ = carregar_geojson_cdmun(GEOJSON_MUN)
NOME_MUN_LOOKUP = lookup_nomes_from_geojson(GJ)

# BBOX por município
BBOX_BY_MUN = {}
for ft in GJ.get("features", []):
    props = ft.get("properties", {}) or {}
    cd = str(props.get("CD_MUN","")).zfill(7)
    bb = _geom_bbox(ft.get("geometry"))
    if cd and bb:
        BBOX_BY_MUN[cd] = bb

base["CD_MUN"]   = z7(base["CD_MUN"])
uf_by_cd         = base["CD_MUN"].str[:2].map(PREFIXO_UF)
base["SIGLA_UF"] = base.get("SIGLA_UF", uf_by_cd).fillna(uf_by_cd).astype(str).str.strip().str.upper()
base["NM_REGIAO"] = base.get("NM_REGIAO", base["SIGLA_UF"].map(UF_REGIAO)).fillna(base["SIGLA_UF"].map(UF_REGIAO))
base["NM_MUN"] = base.get("NM_MUN", base["CD_MUN"].map(NOME_MUN_LOOKUP))
base["NM_MUN"] = base["NM_MUN"].fillna(base["CD_MUN"].map(NOME_MUN_LOOKUP))
base["NM_MUN"] = base["NM_MUN"].fillna("Município " + base["CD_MUN"])

base["UF_KEY"]  = base["SIGLA_UF"]
base["REG_KEY"] = norm_key(base["NM_REGIAO"])

# EHF + classificação + risco combinado
base = calc_ehf(base)
base = classify_by_ratio(base)
base, HAS_RISK = build_combined_risk(base)

# Default Brasília
def busca_brasilia(df):
    c = df[(df["UF_KEY"]=="DF") & (df["NM_MUN"].str.upper().str.contains("BRASILIA|BRASÍLIA", na=False))]
    if not c.empty: return c.iloc[0]["CD_MUN"]
    if (df["CD_MUN"]=="5300108").any(): return "5300108"
    return df["CD_MUN"].iloc[0]
DEFAULT_MUN = busca_brasilia(base)

# Opções dos filtros
REG_OPTS = (base[["REG_KEY","NM_REGIAO"]].dropna().drop_duplicates()
            .sort_values("NM_REGIAO")
            .rename(columns={"REG_KEY":"value","NM_REGIAO":"label"})
            .loc[:, ["label","value"]]
            .to_dict("records"))

UF_OPTS_ALL = (base[["UF_KEY","SIGLA_UF"]].dropna().drop_duplicates()
               .sort_values("SIGLA_UF")
               .rename(columns={"UF_KEY":"value","SIGLA_UF":"label"})
               .loc[:, ["label","value"]]
               .to_dict("records"))

DATES = sorted(base["data"].dropna().dt.date.unique().tolist())

def initial_date_index():
    if not DATES:
        return 0
    hoje = date.today()
    return DATES.index(hoje) if hoje in DATES else 0

# ================== APP ==================
app = Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1.0"}])
server = app.server  # <- necessário pro gunicorn no Render
app.title = "Fator de Excesso de Calor (EHF) – Brasil"

layer_opts = [{"label":"EHF", "value":"ehf"}]
if HAS_RISK:
    layer_opts.append({"label":"Risco combinado (EHF + GeoSES)", "value":"risk"})

# ================== LAYOUT (Responsivo, sem mudar lógica) ==================
app.layout = html.Div(
    style={
        "fontFamily":"Inter, system-ui, Arial",
        "padding":"12px",
        "maxWidth":"1400px",
        "margin":"0 auto"
    },
    children=[
        html.H3("Fator de Excesso de Calor (EHF) – Brasil", style={"marginBottom":"8px"}),

        # Barra de filtros (wrap)
        html.Div(
            [
                html.Div([
                    html.Label("Data"),
                    dcc.Slider(
                        id="date-slider",
                        min=0, max=max(len(DATES)-1,0), step=1,
                        value=initial_date_index(),
                        marks={i: d.strftime("%d/%m") for i, d in enumerate(DATES)} if DATES else {}
                    )
                ], style={"minWidth":"260px","flex":"2"}),

                html.Div([
                    html.Label("Região"),
                    dcc.Dropdown(id="regiao-filter", options=REG_OPTS, value=None, placeholder="Todas", clearable=True)
                ], style={"minWidth":"200px","flex":"1"}),

                html.Div([
                    html.Label("UF"),
                    dcc.Dropdown(id="uf-filter", options=UF_OPTS_ALL, value=[], multi=True, placeholder="SIGLA_UF")
                ], style={"minWidth":"220px","flex":"1"}),

                html.Div([
                    html.Label("Município"),
                    dcc.Dropdown(id="muni-filter", options=[], value=None, multi=False, placeholder="Município")
                ], style={"minWidth":"280px","flex":"2"}),

                html.Div([
                    html.Label("Camada"),
                    dcc.RadioItems(id="layer", options=layer_opts, value=layer_opts[0]["value"], inline=True)
                ], style={"minWidth":"260px","flex":"1"})
            ],
            style={
                "display":"flex",
                "gap":"10px",
                "alignItems":"center",
                "marginBottom":"10px",
                "flexWrap":"wrap"
            }
        ),

        # Grid responsiva (2 colunas no desktop, 1 no mobile)
        html.Div(
            [
                # ====== COL ESQUERDA ======
                html.Div(
                    [
                        dcc.Graph(
                            id="mapa",
                            style={"height":"58vh","marginBottom":"8px"},
                            config={
                                "scrollZoom": True,
                                "displaylogo": False,
                                "modeBarButtonsToRemove": ["pan2d","lasso2d","select2d","autoScale2d","toggleSpikelines"]
                            }
                        ),
                        html.Div(
                            id="cards-ehf",
                            style={
                                "display":"grid",
                                "gridTemplateColumns":"repeat(auto-fit, minmax(170px, 1fr))",
                                "gap":"8px",
                                "alignItems":"stretch",
                                "marginTop":"2px"
                            }
                        ),

                        html.Hr(),
                        html.Div("Consulta por classificação (dia atual)", style={"fontWeight":"700","margin":"6px 0"}),

                        html.Div(
                            [
                                # EHF
                                html.Div(
                                    [
                                        html.Div([
                                            html.Label("Classificação (EHF)"),
                                            dcc.Dropdown(
                                                id="ehf-cls-dd",
                                                options=[{"label":c,"value":c} for c in CLASS_ORDER],
                                                value=CLASS_ORDER[0],
                                                clearable=False
                                            )
                                        ], style={"marginBottom":"6px"}),

                                        html.Div(id="ehf-cls-count", style={"fontSize":"13px","marginBottom":"4px"}),

                                        html.Div(
                                            id="ehf-cls-list",
                                            style={
                                                "border":"1px solid #e5e7eb",
                                                "borderRadius":"8px",
                                                "padding":"8px",
                                                "minHeight":"48px",
                                                "maxHeight":"24vh",
                                                "overflowY":"auto",
                                                "backgroundColor":"#fff"
                                            }
                                        ),

                                        html.Div(
                                            [
                                                html.Button("Exportar XLSX (todos os dias)", id="btn-export-ehf", n_clicks=0),
                                                dcc.Download(id="dl-ehf"),
                                            ],
                                            style={"marginTop":"6px","display":"flex","justifyContent":"flex-end"}
                                        ),
                                    ],
                                    style={"flex":"1","minWidth":"280px"}
                                ),

                                # RISCO
                                html.Div(
                                    [
                                        html.Div([
                                            html.Label("Risco combinado"),
                                            dcc.Dropdown(
                                                id="risk-cls-dd",
                                                options=[{"label":c,"value":c} for c in RISK_ORDER],
                                                value=RISK_ORDER[0],
                                                clearable=False,
                                                disabled=(not HAS_RISK)
                                            )
                                        ], style={"marginBottom":"6px"}),

                                        html.Div(id="risk-cls-count", style={"fontSize":"13px","marginBottom":"4px"}),

                                        html.Div(
                                            id="risk-cls-list",
                                            style={
                                                "border":"1px solid #e5e7eb",
                                                "borderRadius":"8px",
                                                "padding":"8px",
                                                "minHeight":"48px",
                                                "maxHeight":"24vh",
                                                "overflowY":"auto",
                                                "backgroundColor":"#fff"
                                            }
                                        ),

                                        html.Div(
                                            [
                                                html.Button("Exportar XLSX (todos os dias)", id="btn-export-risk", n_clicks=0, disabled=(not HAS_RISK)),
                                                dcc.Download(id="dl-risk"),
                                            ],
                                            style={"marginTop":"6px","display":"flex","justifyContent":"flex-end"}
                                        ),
                                    ],
                                    style={"flex":"1","minWidth":"280px"}
                                )
                            ],
                            style={"display":"flex","gap":"8px","flexWrap":"wrap"}
                        )
                    ],
                    style={"minWidth":"320px"}
                ),

                # ====== COL DIREITA ======
                html.Div(
                    [
                        dcc.Graph(id="serie-municipio", style={"height":"50vh","marginBottom":"10px"}),
                        html.Div(
                            id="ehf-dia",
                            style={
                                "display":"grid",
                                "gridTemplateColumns":"repeat(auto-fit, minmax(140px, 1fr))",
                                "gap":"6px"
                            }
                        )
                    ],
                    style={"minWidth":"320px"}
                )
            ],
            style={
                "display":"grid",
                "gridTemplateColumns":"repeat(auto-fit, minmax(420px, 1fr))",
                "gap":"12px"
            }
        )
    ]
)

# ================== CALLBACKS ==================
@callback(
    Output("uf-filter","options"),
    Output("uf-filter","value"),
    Input("regiao-filter","value"),
    State("uf-filter","value")
)
def cb_ufs(reg_key, ufs_val):
    df = base[["UF_KEY","SIGLA_UF","REG_KEY"]].dropna().drop_duplicates()
    if reg_key:
        df = df[df["REG_KEY"] == reg_key]
    ops = (df.sort_values("SIGLA_UF")
             .rename(columns={"UF_KEY":"value","SIGLA_UF":"label"})
             .loc[:, ["label","value"]]
             .to_dict("records"))
    return ops, []

@callback(
    Output("muni-filter","options"),
    Output("muni-filter","value"),
    Input("regiao-filter","value"),
    Input("uf-filter","value"),
    Input("mapa","clickData"),
    State("muni-filter","value")
)
def cb_munis(reg_key, uf_keys, clickData, mval):
    uf_keys = uf_keys or []
    df = base[["CD_MUN","NM_MUN","UF_KEY","REG_KEY"]].copy()
    if reg_key:
        df = df[df["REG_KEY"] == reg_key]
    if uf_keys:
        df = df[df["UF_KEY"].isin(uf_keys)]

    df = df.dropna(subset=["CD_MUN"]).drop_duplicates("CD_MUN").sort_values(["UF_KEY","NM_MUN"])
    ops = [{"label": f"{r.NM_MUN} / {r.UF_KEY}", "value": r.CD_MUN} for r in df.itertuples()]
    valid = {o["value"] for o in ops}

    clicked = None
    if clickData and clickData.get("points"):
        clicked = str(clickData["points"][0].get("location") or "")
        if clicked not in valid:
            clicked = None

    val = clicked if clicked else (mval if (mval in valid) else None)
    return ops, val

@callback(
    Output("mapa","figure"),
    Output("serie-municipio","figure"),
    Output("cards-ehf","children"),
    Output("ehf-dia","children"),
    Input("date-slider","value"),
    Input("regiao-filter","value"),
    Input("uf-filter","value"),
    Input("muni-filter","value"),
    Input("layer","value"),
    Input("mapa","relayoutData"),
)
def cb_viz(idx_date, reg_key, uf_keys, muni_key, layer, relayout):
    if not DATES:
        return px.scatter(), px.bar(), [], []

    dia = DATES[idx_date] if 0 <= idx_date < len(DATES) else DATES[-1]
    uf_keys = uf_keys or []

    # ====== MAPA ======
    df = base[base["data"].dt.date == dia].copy()
    if reg_key:
        df = df[df["REG_KEY"] == reg_key]
    if uf_keys:
        df = df[df["UF_KEY"].isin(uf_keys)]

    filter_muni_active = bool(muni_key)
    if filter_muni_active:
        df = df[df["CD_MUN"] == muni_key]

    if layer == "risk" and HAS_RISK:
        color_col = "risk_class"; cmap = RISK_COLORS; ordem = RISK_ORDER
        legend_title = "Risco combinado (EHF + GeoSES)"
    else:
        color_col = "classification"; cmap = COLOR_MAP; ordem = CLASS_ORDER
        legend_title = "EHF"

    for c in ["NM_MUN","SIGLA_UF","EHF","GeoSES","risk_index",color_col]:
        if c not in df.columns:
            df[c] = np.nan
    df[color_col] = pd.Categorical(df[color_col], categories=ordem, ordered=True)

    vis = df[["CD_MUN","NM_MUN","SIGLA_UF","NM_REGIAO", color_col, "EHF", "GeoSES", "risk_index"]].copy()
    vis["muni_label"] = vis["NM_MUN"].astype(str) + " / " + vis["SIGLA_UF"].astype(str)

    fig_map = px.choropleth_mapbox(
        vis,
        geojson=GJ,
        locations="CD_MUN",
        featureidkey="properties.CD_MUN",
        color=color_col,
        color_discrete_map=cmap,
        hover_name="muni_label",
        custom_data=["EHF","GeoSES","risk_index"],
        category_orders={color_col: ordem},
        mapbox_style="carto-positron",
        center={"lat": -14.2, "lon": -51.9},
        zoom=3.4,
        opacity=0.85
    )

    fig_map.update_traces(
        marker_line_width=0.2, marker_line_color="#000000",
        hovertemplate="<b>%{hovertext}</b><br>" +
                      "EHF: %{customdata[0]:.1f}<br>" +
                      "GeoSES: %{customdata[1]:.2f}<br>" +
                      "Risco combinado: %{customdata[2]:.2f}<extra></extra>"
    )
    fig_map.update_layout(clickmode="event+select")

    # Auto-zoom quando aplicar filtro geográfico (clamp ao Brasil)
    apply_zoom = bool(reg_key) or bool(uf_keys) or bool(muni_key)
    if apply_zoom and not vis.empty:
        target_cds = [muni_key] if filter_muni_active else vis["CD_MUN"].astype(str).unique().tolist()
        bbox = _union_bbox([BBOX_BY_MUN.get(cd) for cd in target_cds])
        if bbox:
            center, z = bbox_to_center_zoom(bbox, width=1100, height=650, padding=0.06)
            z = max(min(z, CLAMP_ZOOM_MAX), CLAMP_ZOOM_MIN)
            lat = min(max(center["lat"], BRAZIL_BBOX["south"]), BRAZIL_BBOX["north"])
            lon = min(max(center["lon"], BRAZIL_BBOX["west"]),  BRAZIL_BBOX["east"])
            fig_map.update_layout(mapbox_center={"lat": lat, "lon": lon}, mapbox_zoom=z)

    # Clamp após interações do usuário (relayoutData)
    if isinstance(relayout, dict):
        c = relayout.get("mapbox.center") or relayout.get("mapbox._center")
        z = relayout.get("mapbox.zoom")
        if c or (z is not None):
            if c:
                lat = float(c.get("lat", -14.2))
                lon = float(c.get("lon", -51.9))
                lat = min(max(lat, BRAZIL_BBOX["south"]), BRAZIL_BBOX["north"])
                lon = min(max(lon, BRAZIL_BBOX["west"]),  BRAZIL_BBOX["east"])
            else:
                lat = fig_map.layout.mapbox.center.lat
                lon = fig_map.layout.mapbox.center.lon
            if z is None:
                z = fig_map.layout.mapbox.zoom
            z = max(min(float(z), CLAMP_ZOOM_MAX), CLAMP_ZOOM_MIN)
            fig_map.update_layout(mapbox_center={"lat": lat, "lon": lon}, mapbox_zoom=z)

    fig_map.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0},
        legend_title_text=legend_title,
        uirevision=f"reg:{reg_key}|ufs:{','.join(uf_keys)}|mun:{muni_key or ''}"
    )

    # ====== BARRAS MUNICÍPIO ======
    muni_sel = muni_key if muni_key else DEFAULT_MUN
    dmun = base[base["CD_MUN"] == muni_sel].copy()

    if "Tmean" not in dmun or dmun["Tmean"].dropna().empty:
        if "Tmed" in dmun and not dmun["Tmed"].dropna().empty:
            dmun["Tmean"] = dmun["Tmed"]
        elif {"Tmin","Tmax"}.issubset(dmun.columns):
            dmun["Tmean"] = (dmun["Tmin"] + dmun["Tmax"]) / 2
        else:
            dmun["Tmean"] = np.nan

    serie = (dmun[["data","Tmin","Tmean","Tmax","NM_MUN","SIGLA_UF","classification"]]
             .dropna(subset=["data"]).sort_values("data"))

    MES = {1:"JAN",2:"FEV",3:"MAR",4:"ABR",5:"MAI",6:"JUN",7:"JUL",8:"AGO",9:"SET",10:"OUT",11:"NOV",12:"DEZ"}
    if not serie.empty:
        serie["data_lbl"] = serie["data"].dt.day.astype(str).str.zfill(2) + " " + serie["data"].dt.month.map(MES)
        bt = serie.melt(id_vars=["data","data_lbl","NM_MUN","SIGLA_UF"], value_vars=["Tmin","Tmean","Tmax"],
                        var_name="Série", value_name="Valor")
        bt["Série"] = bt["Série"].map({"Tmin":"Tmín","Tmean":"Tméd","Tmax":"Tmáx"})
        titulo = f"Previsão – {serie.iloc[0]['NM_MUN']} / {serie.iloc[0]['SIGLA_UF']}"
        cat_array = serie["data_lbl"].tolist()
    else:
        bt = pd.DataFrame({"data_lbl":[], "Valor":[], "Série":[]})
        titulo = "Previsão"
        cat_array = []

    fig_bar = px.bar(bt, x="data_lbl", y="Valor", color="Série",
                     color_discrete_map=BARS_COLORS, barmode="group", title=titulo)
    fig_bar.update_traces(texttemplate="%{y:.1f}", textposition="outside")

    fig_bar.update_xaxes(
        categoryorder="array",
        categoryarray=cat_array,
        showgrid=False,
        fixedrange=True
    )
    fig_bar.update_yaxes(
        range=[0, 45], dtick=5,
        showgrid=True, gridcolor="#e5e7eb", gridwidth=1,
        fixedrange=True
    )
    fig_bar.update_layout(
        xaxis_title="", yaxis_title="°C", legend_title="",
        margin=dict(l=10,r=10,t=40,b=10),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    # ====== CARDS ======
    cont = df[color_col].value_counts().reindex(ordem, fill_value=0)
    cards = []
    for lbl in ordem:
        val = int(cont.get(lbl, 0))
        pal = (RISK_COLORS if layer=="risk" and HAS_RISK else COLOR_MAP).get(lbl, "#6b7280")
        cards.append(
            html.Div(
                [
                    html.Div(lbl, style={"fontWeight":"600","fontSize":"13px","marginBottom":"4px","textAlign":"center"}),
                    html.Div(f"{val:,}".replace(",","."), style={"fontSize":"20px","fontWeight":"800","textAlign":"center"})
                ],
                style={
                    "backgroundColor":"#FFF","border":"1px solid #e5e7eb","borderLeft":"8px solid "+pal,
                    "borderRadius":"10px","padding":"10px","height":"78px","display":"flex",
                    "flexDirection":"column","justifyContent":"center"
                }
            )
        )

    # ====== EHF POR DIA (município) ======
    ehf_boxes = []
    if not serie.empty:
        for dstr, cl in zip(serie["data_lbl"], serie["classification"]):
            cor = COLOR_MAP.get(cl, "#6b7280")
            ehf_boxes.append(
                html.Div(
                    [
                        html.Div(dstr, style={"fontWeight":"600","marginBottom":"2px"}),
                        html.Div(cl,   style={"fontSize":"14px","fontWeight":"700"})
                    ],
                    style={
                        "backgroundColor":"#FFF","border":"1px solid #e5e7eb",
                        "borderLeft":"10px solid "+cor,"borderRadius":"10px","padding":"8px"
                    }
                )
            )

    return fig_map, fig_bar, cards, ehf_boxes

# ===== CONSULTA POR CLASSE (listas) =====
@callback(
    Output("ehf-cls-count","children"),
    Output("ehf-cls-list","children"),
    Output("risk-cls-count","children"),
    Output("risk-cls-list","children"),
    Input("date-slider","value"),
    Input("regiao-filter","value"),
    Input("uf-filter","value"),
    Input("ehf-cls-dd","value"),
    Input("risk-cls-dd","value"),
)
def cb_listas(idx_date, reg_key, uf_keys, ehf_cls, risk_cls):
    if not DATES:
        msg = html.Div("Nenhum município com os filtros atuais.", style={"color":"#6b7280"})
        return "", msg, "", msg

    dia = DATES[idx_date] if 0 <= idx_date < len(DATES) else DATES[-1]
    uf_keys = uf_keys or []

    df = base[base["data"].dt.date == dia].copy()
    if reg_key:
        df = df[df["REG_KEY"] == reg_key]
    if uf_keys:
        df = df[df["UF_KEY"].isin(uf_keys)]

    d_ehf = df[df["classification"] == ehf_cls].sort_values(["SIGLA_UF","NM_MUN"])
    c_ehf = len(d_ehf["CD_MUN"].unique())
    list_ehf = ([html.Div(f"{r.NM_MUN} / {r.SIGLA_UF}") for r in d_ehf.itertuples()]
                or [html.Div("Nenhum município com os filtros atuais.", style={"color":"#6b7280"})])
    txt_ehf = f"{c_ehf} município(s) na categoria selecionada."

    if "risk_class" in df.columns:
        d_risk = df[df["risk_class"] == risk_cls].sort_values(["SIGLA_UF","NM_MUN"])
        c_risk = len(d_risk["CD_MUN"].unique())
        list_risk = ([html.Div(f"{r.NM_MUN} / {r.SIGLA_UF}") for r in d_risk.itertuples()]
                     or [html.Div("Nenhum município com os filtros atuais.", style={"color":"#6b7280"})])
        txt_risk = f"{c_risk} município(s) na categoria selecionada."
    else:
        list_risk = [html.Div("Risco combinado indisponível.", style={"color":"#6b7280"})]
        txt_risk = ""

    return txt_ehf, list_ehf, txt_risk, list_risk

# ===== EXPORTAR XLSX (TODOS OS DIAS/TODOS MUNICÍPIOS) =====
def _df_export_full():
    df = base.copy()
    if "Tmean" not in df or df["Tmean"].dropna().empty:
        if "Tmed" in df and not df["Tmed"].dropna().empty:
            df["Tmean"] = df["Tmed"]
        elif {"Tmin","Tmax"}.issubset(df.columns):
            df["Tmean"] = (df["Tmin"] + df["Tmax"]) / 2
        else:
            df["Tmean"] = np.nan

    cols = {
        "data": "Data",
        "CD_MUN": "CD_MUN",
        "NM_MUN": "NM_MUN",
        "SIGLA_UF": "UF",
        "NM_REGIAO": "Região",
        "classification": "Classificação EHF",
        "risk_class": "Classificação Risco Combinado",
        "risk_index": "Índice Risco Combinado",
        "GeoSES": "GeoSES",
        "EHF": "EHF",
        "Tmax": "Tmáxima",
        "Tmin": "Tmínima",
        "Tmean": "Tmédia",
    }

    out = (df[list(cols.keys())]
           .rename(columns=cols)
           .sort_values(["Data","UF","NM_MUN"]))
    out["Data"] = pd.to_datetime(out["Data"]).dt.date
    return out

@callback(
    Output("dl-ehf", "data"),
    Input("btn-export-ehf", "n_clicks"),
    prevent_initial_call=True
)
def exportar_ehf_full(n_clicks):
    if not n_clicks:
        return no_update
    out = _df_export_full()
    fname = "ehf_todas_datas.xlsx" if not DATES else f"ehf_{DATES[0].isoformat()}_a_{DATES[-1].isoformat()}.xlsx"
    return dcc.send_data_frame(out.to_excel, fname, index=False)

@callback(
    Output("dl-risk", "data"),
    Input("btn-export-risk", "n_clicks"),
    prevent_initial_call=True
)
def exportar_risco_full(n_clicks):
    if not n_clicks:
        return no_update
    out = _df_export_full()
    fname = "risco_todas_datas.xlsx" if not DATES else f"risco_{DATES[0].isoformat()}_a_{DATES[-1].isoformat()}.xlsx"
    return dcc.send_data_frame(out.to_excel, fname, index=False)

# ================== RUN (Render-friendly) ==================
if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", "8069"))
    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=False,
        dev_tools_ui=False,
        dev_tools_props_check=False,
    )











