# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════╗
║        NAFTAL — Système de Signalement des Pannes                       ║
║        Login · 4 Pages · KPI comparaison · Format heure corrigé        ║
║        Lancement : streamlit run app.py                                 ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
import torch
import torch.nn as nn
import json
import hashlib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from transformers import CamembertModel, CamembertTokenizer
from datetime import datetime, date, timedelta
import os
import glob
import re
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
BASE_PATH       = '.'
MODEL_PATH      = os.path.join(BASE_PATH, 'best_model.pt')
LABEL_MAPS_PATH = os.path.join(BASE_PATH, 'label_maps .json')
TOKENIZER_DIR   = BASE_PATH
MAX_LENGTH      = 128
CSV_DIR         = '.'
HISTORIQUE_PATH = os.path.join(CSV_DIR, 'historique_pannes.csv')
LOGO_PATH       = "logo_naftal.png"
COMPTES_PATH    = "comptes.json"

# ── Téléchargement automatique du modèle depuis Hugging Face ─────────────────
HF_REPO_ID = "hammadi-sofia2004/Camembert-naftal"

@st.cache_resource(show_spinner="⏳ Chargement du modèle IA...")
def telecharger_modele_si_absent():
    if not os.path.exists(MODEL_PATH):
        try:
            from huggingface_hub import hf_hub_download
            hf_token = st.secrets.get("HF_TOKEN", None)
            chemin = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename="best_model.pt",
                token=hf_token,
                local_dir="."
            )
            st.success("✅ Modèle chargé avec succès !")
        except Exception as e:
            st.error(f"❌ Impossible de télécharger le modèle : {e}")
            st.stop()
    return True

telecharger_modele_si_absent()
# ─────────────────────────────────────────────────────────────────────────────

COULEURS_GRAVITE  = {'Critique':'#C0392B','Modérée':'#E8A838','Mineure':'#27AE60'}
COULEURS_PRIORITE = {'Urgente':'#C0392B','Moyenne':'#E8A838','Basse':'#27AE60'}

STATIONS = [
    "GDR1504","GDR1506","GDR1507","GDR1509","GDR1513",
    "GDR3522","GDR3523","MGX","PVA AGOUDJIL","PVA CHABANE",
    "Autre / Non précisé",
]

EXEMPLES = [
    "Fuite d'huile importante sous la pompe hydraulique n°2, bruit anormal depuis ce matin",
    "Disjoncteur principal déclenché, plus d'alimentation électrique depuis 2 heures",
    "Vanne de sécurité bloquée en position fermée, pression anormale sur le circuit",
    "Filtre à gasoil colmaté, débit insuffisant sur la pompe de refoulement",
    "Corrosion avancée sur la tuyauterie principale, petite fuite visible",
]

# ─────────────────────────────────────────────────────────────────────────────
# 2. CONFIG PAGE
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NAFTAL — Signalement des Pannes",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# 3. CSS
# ─────────────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600;700&family=Barlow+Condensed:wght@700&display=swap');

    #MainMenu,footer,header,[data-testid="stToolbar"],.stDeployButton
        {visibility:hidden !important;display:none !important;}

    .stApp{background:#EEF2F7 !important;font-family:'Barlow',sans-serif !important;}
    .main .block-container{padding:1.5rem 2rem !important;max-width:1350px !important;}

    [data-testid="stSidebar"]{
        background:linear-gradient(180deg,#0d1b2a 0%,#1F3864 60%,#0d1b2a 100%) !important;
        border-right:2.5px solid #E8A838 !important;
    }
    [data-testid="stSidebar"] *{color:#fff !important;font-family:'Barlow',sans-serif !important;}
    [data-testid="stSidebar"] hr{border-color:rgba(232,168,56,0.3) !important;}

    h1,h2,h3{font-family:'Barlow Condensed',sans-serif !important;color:#1F3864 !important;}

    /* ── Page header ── */
    .page-header{display:flex;align-items:center;gap:1rem;padding-bottom:1rem;
                 margin-bottom:1.4rem;border-bottom:3px solid #E8A838;}
    .ph-icon{width:48px;height:48px;border-radius:10px;background:#1F3864;
             display:flex;align-items:center;justify-content:center;font-size:1.5rem;flex-shrink:0;}
    .ph-title{font-family:'Barlow Condensed',sans-serif;font-size:1.4rem;font-weight:700;color:#1F3864;margin:0;}
    .ph-sub{font-size:0.82rem;color:#666;margin:0.1rem 0 0;}

    /* ── Login page ── */
    .login-wrap{max-width:480px;margin:3rem auto;}
    .login-card{background:white;border-radius:16px;padding:2.5rem 2rem;
                border:0.5px solid #dde4ee;box-shadow:0 8px 30px rgba(31,56,100,0.12);}
    .login-logo{text-align:center;margin-bottom:1.5rem;}
    .login-title{font-family:'Barlow Condensed',sans-serif;font-size:1.4rem;
                 font-weight:700;color:#1F3864;text-align:center;margin-bottom:0.3rem;}
    .login-sub{font-size:0.82rem;color:#888;text-align:center;margin-bottom:1.5rem;}

    /* ── User badge sidebar ── */
    .user-badge{background:rgba(232,168,56,0.15);border:1px solid rgba(232,168,56,0.35);
                border-radius:9px;padding:0.6rem 0.8rem;margin-top:0.5rem;}
    .user-nom{font-size:0.82rem;font-weight:600;color:white;}
    .user-station{font-size:0.7rem;color:rgba(255,255,255,0.6);}

    /* ── Section headers ── */
    .section-header{display:flex;align-items:center;gap:0.7rem;padding:0.7rem 1rem;
                    border-radius:10px;background:linear-gradient(135deg,#1F3864,#2E75B6);
                    margin:1.4rem 0 1rem;color:white;}
    .section-header span{font-family:'Barlow Condensed',sans-serif;font-size:1.05rem;
                         font-weight:700;color:white;}

    /* ── KPI cards ── */
    .kpi{background:white;border-radius:12px;padding:1.1rem 0.8rem;text-align:center;
         border-top:3px solid #2E75B6;box-shadow:0 2px 10px rgba(31,56,100,0.07);}
    .kpi.rouge{border-top-color:#C0392B;}
    .kpi.orange{border-top-color:#E8A838;}
    .kpi.vert{border-top-color:#27AE60;}
    .kpi.bleu{border-top-color:#2E75B6;}
    .kpi-icon{font-size:1.4rem;margin-bottom:0.3rem;}
    .kpi-val{font-family:'Barlow Condensed',sans-serif;font-size:2.1rem;font-weight:700;
             color:#1F3864;line-height:1;}
    .kpi-lbl{font-size:0.68rem;color:#888;text-transform:uppercase;
             letter-spacing:0.05em;margin-top:0.28rem;}
    /* ── Delta KPI ── */
    .kpi-delta{font-size:0.72rem;margin-top:0.3rem;font-weight:600;}
    .kpi-delta.up{color:#C0392B;}      /* hausse = mauvais pour les pannes */
    .kpi-delta.down{color:#27AE60;}    /* baisse = bon */
    .kpi-delta.neutral{color:#888;}

    /* ── Res cards ── */
    .res-card{border-radius:12px;padding:1.2rem;text-align:center;border:1.5px solid;}
    .res-card.rouge{background:#FDE8E8;border-color:#C0392B;}
    .res-card.orange{background:#FEF6E4;border-color:#E8A838;}
    .res-card.bleu{background:#EBF5FF;border-color:#2E75B6;}
    .res-card.vert{background:#E8F8F0;border-color:#27AE60;}
    .res-lbl{font-size:0.68rem;text-transform:uppercase;letter-spacing:0.07em;
             color:#666;margin-bottom:0.45rem;}
    .res-val{font-family:'Barlow Condensed',sans-serif;font-size:1.5rem;font-weight:700;}
    .res-val.rouge{color:#C0392B;}
    .res-val.orange{color:#B7800A;}
    .res-val.bleu{color:#1F3864;}
    .res-val.vert{color:#1D7A44;}
    .res-action{font-size:0.72rem;margin-top:0.5rem;color:#555;}

    .card{background:white;border-radius:13px;padding:1.4rem;border:0.5px solid #dde4ee;
          margin-bottom:1rem;box-shadow:0 2px 12px rgba(31,56,100,0.06);}

    .stButton>button{
        background:#1F3864 !important;color:white !important;
        border:none !important;border-radius:9px !important;
        font-family:'Barlow',sans-serif !important;font-weight:600 !important;
        font-size:1rem !important;padding:0.65rem 1.5rem !important;
        box-shadow:0 3px 12px rgba(31,56,100,0.22) !important;transition:all 0.2s !important;
    }
    .stButton>button:hover{background:#2E75B6 !important;transform:translateY(-1px) !important;}

    .banner-success{background:#E8F8F0;border:1.5px solid #27AE60;border-radius:10px;
                    padding:0.85rem 1.1rem;color:#1D7A44;font-size:0.85rem;margin-top:0.9rem;}
    .banner-info{background:#EBF5FF;border:1.5px solid #2E75B6;border-radius:10px;
                 padding:0.85rem 1.1rem;color:#1F3864;font-size:0.85rem;}
    .banner-error{background:#FDE8E8;border:1.5px solid #C0392B;border-radius:10px;
                  padding:0.85rem 1.1rem;color:#922B21;font-size:0.85rem;}

    .badge{display:inline-block;padding:0.18em 0.6em;border-radius:20px;
           font-size:0.72rem;font-weight:700;}
    .b-rouge{background:#FDE8E8;color:#C0392B;}
    .b-orange{background:#FEF6E4;color:#B7800A;}
    .b-vert{background:#E8F8F0;color:#1D7A44;}
    .b-bleu{background:#EBF5FF;color:#1F3864;}

    .divider{height:3px;background:linear-gradient(90deg,#E8A838,#2E75B6,#1F3864);
             border-radius:2px;margin:1.2rem 0;}

    .stSelectbox label,.stTextArea label,.stTextInput label
        {font-size:0.88rem !important;font-weight:600 !important;color:#1F3864 !important;}

    .step{display:flex;gap:0.8rem;align-items:flex-start;margin-bottom:0.85rem;}
    .step-num{width:30px;height:30px;border-radius:50%;background:#1F3864;color:white;
              font-size:0.82rem;font-weight:700;display:flex;align-items:center;
              justify-content:center;flex-shrink:0;}
    .step-txt{font-size:0.83rem;color:#444;line-height:1.65;padding-top:0.2rem;}
    .step-txt b{color:#1F3864;}
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4. ARCHITECTURE MODÈLE
# ─────────────────────────────────────────────────────────────────────────────
class CamemBERTMultiTask(nn.Module):
    def __init__(self, model_name, num_gravite, num_priorite,
                 num_equipement, num_type_panne, dropout_rate=0.3):
        super().__init__()
        self.bert            = CamembertModel.from_pretrained(model_name)
        hidden               = self.bert.config.hidden_size
        self.dropout         = nn.Dropout(dropout_rate)
        self.head_gravite    = nn.Linear(hidden, num_gravite)
        self.head_priorite   = nn.Linear(hidden, num_priorite)
        self.head_equipement = nn.Linear(hidden, num_equipement)
        self.head_type_panne = nn.Linear(hidden, num_type_panne)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0, :])
        return (self.head_gravite(cls), self.head_priorite(cls),
                self.head_equipement(cls), self.head_type_panne(cls))


# ─────────────────────────────────────────────────────────────────────────────
# 5. CHARGEMENT MODÈLE
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def charger_modele():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open(LABEL_MAPS_PATH, 'r', encoding='utf-8') as f:
        lm = json.load(f)
    g_cls  = lm['gravite']['classes']
    p_cls  = lm['priorite']['classes']
    eq_cls = lm['equipement']['classes']
    tp_cls = lm['type_panne']['classes']
    tokenizer = CamembertTokenizer.from_pretrained(TOKENIZER_DIR)
    model = CamemBERTMultiTask('camembert-base', len(g_cls), len(p_cls),
                               len(eq_cls), len(tp_cls))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()
    return model, tokenizer, device, g_cls, p_cls, eq_cls, tp_cls


# ─────────────────────────────────────────────────────────────────────────────
# 6. FONCTIONS UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────

def normaliser_heure(h: str) -> str:
    """
    Normalise tous les formats d'heure vers HH:MM.
    Accepte : '13H02min', '13:02', '13h02', '1302', etc.
    """
    if pd.isna(h) or str(h).strip() == '':
        return ''
    h = str(h).strip()
    # Format HH:MM déjà correct
    if re.match(r'^\d{2}:\d{2}$', h):
        return h
    # Format 13H02min ou 13h02min
    m = re.match(r'(\d{1,2})[Hh](\d{2})', h)
    if m:
        return f"{int(m.group(1)):02d}:{m.group(2)}"
    # Format HHMM sans séparateur
    m2 = re.match(r'^(\d{2})(\d{2})$', h)
    if m2:
        return f"{m2.group(1)}:{m2.group(2)}"
    return h   # retourner tel quel si format inconnu


def predire(description: str, lieu: str) -> dict:
    model, tokenizer, device, g_cls, p_cls, eq_cls, tp_cls = charger_modele()
    now   = datetime.now()
    heure = now.strftime('%H:%M')   # format normalisé HH:MM
    texte = (f"[DATE:{now.strftime('%Y-%m-%d')}]"
             f"[HEURE:{heure}]"
             f"[LIEU:{lieu}] {description.strip()}")
    enc  = tokenizer(texte, max_length=MAX_LENGTH, padding='max_length',
                     truncation=True, return_tensors='pt')
    ids  = enc['input_ids'].to(device)
    mask = enc['attention_mask'].to(device)
    with torch.no_grad():
        lg, lp, leq, ltp = model(ids, mask)
    gravite    = g_cls[lg.argmax(1).item()]
    priorite   = p_cls[lp.argmax(1).item()]
    equipement = eq_cls[leq.argmax(1).item()]
    type_panne = tp_cls[ltp.argmax(1).item()]
    sauvegarder(description, lieu, gravite, priorite, equipement, type_panne, now, heure)
    return dict(gravite=gravite, priorite=priorite, equipement=equipement,
                type_panne=type_panne, date=now.strftime('%Y-%m-%d'),
                heure=heure, lieu=lieu, mois=now.strftime('%Y-%m'))


def sauvegarder(desc, lieu, gravite, priorite, equipement, type_panne, now, heure):
    """Sauvegarde avec heure au format normalisé HH:MM."""
    employe = st.session_state.get('employe_nom', 'Inconnu')
    ligne = {
        'date':       now.strftime('%Y-%m-%d'),
        'heure':      heure,                       # ← toujours HH:MM
        'lieu':       lieu,
        'employe':    employe,
        'description': desc[:250],
        'gravite':    gravite,
        'priorite':   priorite,
        'equipement': equipement,
        'type_panne': type_panne,
        'mois':       now.strftime('%Y-%m'),
        'annee':      now.year,
    }
    if os.path.exists(HISTORIQUE_PATH):
        df = pd.read_csv(HISTORIQUE_PATH, encoding='utf-8-sig')
        df = pd.concat([df, pd.DataFrame([ligne])], ignore_index=True)
    else:
        df = pd.DataFrame([ligne])
    df.to_csv(HISTORIQUE_PATH, index=False, encoding='utf-8-sig')


def charger_tous_csv() -> pd.DataFrame | None:
    """Charge et fusionne tous les CSV. Normalise la colonne heure."""
    fichiers = glob.glob(os.path.join(CSV_DIR, '*.csv'))
    if not fichiers:
        return None
    frames = []
    for f in fichiers:
        try:
            tmp = pd.read_csv(f, encoding='utf-8-sig')
            tmp['_source'] = os.path.basename(f)
            frames.append(tmp)
        except Exception:
            pass
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip().lower() for c in df.columns]
    # ── Normaliser heure ──
    if 'heure' in df.columns:
        df['heure'] = df['heure'].apply(normaliser_heure)
    return df if len(df) > 0 else None


def charger_historique() -> pd.DataFrame | None:
    if not os.path.exists(HISTORIQUE_PATH):
        return None
    try:
        df = pd.read_csv(HISTORIQUE_PATH, encoding='utf-8-sig')
        df.columns = [c.strip().lower() for c in df.columns]
        if 'heure' in df.columns:
            df['heure'] = df['heure'].apply(normaliser_heure)
        return df if len(df) > 0 else None
    except Exception:
        return None


def get_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def ref_signalement() -> str:
    n = 0
    if os.path.exists(HISTORIQUE_PATH):
        try:
            n = len(pd.read_csv(HISTORIQUE_PATH, encoding='utf-8-sig'))
        except Exception:
            pass
    return f"NAFT-{datetime.now().strftime('%Y%m%d')}-{n+1:04d}"


def coul_grav(g):
    return {'Critique':'rouge','Modérée':'orange','Mineure':'vert'}.get(g,'bleu')

def coul_prio(p):
    return {'Urgente':'rouge','Moyenne':'orange','Basse':'vert'}.get(p,'bleu')

def action_grav(g):
    return {'Critique':'🔴 Intervention immédiate',
            'Modérée': '🟡 À traiter rapidement',
            'Mineure': '🟢 Planifiable'}.get(g,'')

def plotly_style(fig, height=300):
    fig.update_layout(
        plot_bgcolor='white', paper_bgcolor='white',
        title_font=dict(size=13, color='#1F3864', family='Barlow Condensed'),
        font=dict(family='Barlow', size=11),
        height=height, margin=dict(t=40, b=30, l=20, r=20),
    )
    return fig


def filtrer_periode(df, col_date, granularite, debut, fin):
    """Filtre par période et ajoute colonne _groupe."""
    if not col_date or col_date not in df.columns:
        return df, None
    df = df.copy()
    df['_dt'] = pd.to_datetime(df[col_date], errors='coerce')
    df = df.dropna(subset=['_dt'])
    debut_dt = pd.Timestamp(debut)
    fin_dt   = pd.Timestamp(fin).replace(hour=23, minute=59, second=59)
    df = df[(df['_dt'] >= debut_dt) & (df['_dt'] <= fin_dt)]
    if granularite == 'Jour':
        df['_groupe'] = df['_dt'].dt.strftime('%Y-%m-%d')
    elif granularite == 'Mois':
        df['_groupe'] = df['_dt'].dt.strftime('%Y-%m')
    else:
        df['_groupe'] = df['_dt'].dt.strftime('%Y')
    return df, '_groupe'


def periode_precedente(debut: date, fin: date):
    """Calcule la période précédente de même durée."""
    duree = (fin - debut).days + 1
    fin_prec   = debut - timedelta(days=1)
    debut_prec = fin_prec - timedelta(days=duree - 1)
    return debut_prec, fin_prec


def kpi_avec_delta(col, icon, valeur_cur, valeur_prec, label, cls='', inverse=False):
    """
    Affiche un KPI avec flèche ▲▼ de comparaison période précédente.
    inverse=True : hausse = bon (ex: taux sans critique)
    inverse=False : hausse = mauvais (ex: nb pannes critiques)
    """
    if valeur_prec is None or valeur_prec == 0:
        delta_html = '<div class="kpi-delta neutral">— pas de données précédentes</div>'
    else:
        diff = valeur_cur - valeur_prec
        pct  = round((diff / valeur_prec) * 100, 1)
        if diff == 0:
            delta_html = '<div class="kpi-delta neutral">➡ stable</div>'
        elif diff > 0:
            couleur = 'down' if inverse else 'up'
            delta_html = f'<div class="kpi-delta {couleur}">▲ +{pct}% vs période préc.</div>'
        else:
            couleur = 'up' if inverse else 'down'
            delta_html = f'<div class="kpi-delta {couleur}">▼ {pct}% vs période préc.</div>'

    with col:
        st.markdown(f"""
        <div class="kpi {cls}">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-val">{valeur_cur}</div>
            <div class="kpi-lbl">{label}</div>
            {delta_html}
        </div>""", unsafe_allow_html=True)


def section_header(icon, titre):
    st.markdown(f"""
    <div class="section-header">
        <span style="font-size:1.3rem;">{icon}</span>
        <span>{titre}</span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 7. COMPOSANTS UI
# ─────────────────────────────────────────────────────────────────────────────

def logo_sidebar():
    if os.path.exists(LOGO_PATH):
        st.sidebar.image(LOGO_PATH, use_container_width=True)
    else:
        st.sidebar.markdown("""
        <div style="text-align:center;padding:0.8rem 0;">
            <span style="font-family:'Barlow Condensed',sans-serif;font-size:2rem;
                         font-weight:700;color:#E8A838;letter-spacing:0.1em;">NAFTAL</span>
        </div>
        """, unsafe_allow_html=True)
    st.sidebar.markdown("""
    <p style="font-size:0.58rem;color:rgba(255,255,255,0.38);text-align:center;
              margin:0.2rem 0 0;letter-spacing:0.1em;text-transform:uppercase;">
        Signalement des Pannes
    </p>
    <hr style="border-color:rgba(232,168,56,0.28);margin:0.5rem 0 0.8rem;">
    """, unsafe_allow_html=True)


def page_header(icon, titre, sous_titre):
    st.markdown(f"""
    <div class="page-header">
        <div class="ph-icon">{icon}</div>
        <div>
            <p class="ph-title">{titre}</p>
            <p class="ph-sub">{sous_titre}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 8. SYSTÈME DE COMPTES — FONCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def hacher_mdp(mdp: str) -> str:
    """Hash SHA-256 — mot de passe jamais stocké en clair."""
    return hashlib.sha256(mdp.strip().encode()).hexdigest()


def charger_comptes() -> dict:
    if os.path.exists(COMPTES_PATH):
        try:
            with open(COMPTES_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def sauvegarder_comptes(comptes: dict):
    with open(COMPTES_PATH, 'w', encoding='utf-8') as f:
        json.dump(comptes, f, ensure_ascii=False, indent=2)


def compte_existe(nom: str) -> bool:
    return nom.strip().lower() in charger_comptes()


def creer_compte(nom: str, mdp: str, role: str = 'employe') -> bool:
    comptes = charger_comptes()
    cle = nom.strip().lower()
    if cle in comptes:
        return False
    comptes[cle] = {
        'nom':      nom.strip(),
        'mdp_hash': hacher_mdp(mdp),
        'role':     role,
        'cree_le':  datetime.now().strftime('%Y-%m-%d %H:%M'),
    }
    sauvegarder_comptes(comptes)
    return True


def supprimer_compte(nom: str) -> bool:
    comptes = charger_comptes()
    cle = nom.strip().lower()
    if cle not in comptes or comptes[cle]['role'] == 'admin':
        return False
    del comptes[cle]
    sauvegarder_comptes(comptes)
    return True


def verifier_login(nom: str, mdp: str):
    comptes = charger_comptes()
    cle = nom.strip().lower()
    if cle not in comptes:
        return None
    if comptes[cle]['mdp_hash'] == hacher_mdp(mdp):
        return comptes[cle]
    return None


def admin_existe() -> bool:
    return any(c['role'] == 'admin' for c in charger_comptes().values())


def lister_employes() -> list:
    return [c for c in charger_comptes().values() if c['role'] == 'employe']


def sidebar_user_badge():
    nom  = st.session_state.get('employe_nom', '')
    role = st.session_state.get('employe_role', 'employe')
    badge_role = "👑 Administrateur" if role == 'admin' else "👤 Employé"
    couleur    = "rgba(232,168,56,0.2)"  if role == 'admin' else "rgba(46,117,182,0.2)"
    bordure    = "rgba(232,168,56,0.45)" if role == 'admin' else "rgba(46,117,182,0.45)"
    st.sidebar.markdown(f"""
    <div style="background:{couleur};border:1px solid {bordure};
                border-radius:9px;padding:0.6rem 0.8rem;margin-bottom:0.5rem;">
        <div style="font-size:0.82rem;font-weight:600;color:white;">{badge_role}</div>
        <div style="font-size:0.78rem;color:rgba(255,255,255,0.75);margin-top:0.1rem;">{nom}</div>
    </div>
    """, unsafe_allow_html=True)
    if st.sidebar.button("🚪 Se déconnecter"):
        for key in ['connecte','employe_nom','employe_role']:
            st.session_state.pop(key, None)
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# 9. PAGE — SETUP ADMIN (premier lancement uniquement)
# ─────────────────────────────────────────────────────────────────────────────

def page_setup_admin():
    inject_css()
    _, col_c, _ = st.columns([1, 2, 1])
    with col_c:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, use_container_width=True)
        else:
            st.markdown("""
            <div style="text-align:center;font-family:'Barlow Condensed',sans-serif;
                        font-size:2.5rem;font-weight:700;color:#1F3864;">NAFTAL</div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background:linear-gradient(135deg,#1F3864,#2E75B6);border-radius:14px;
                    padding:1.5rem 2rem;margin:1rem 0;text-align:center;">
            <div style="font-size:1.5rem;margin-bottom:0.3rem;">🔧</div>
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.3rem;
                        font-weight:700;color:white;">Première utilisation</div>
            <div style="font-size:0.82rem;color:rgba(255,255,255,0.78);margin-top:0.3rem;">
                Créez votre compte administrateur pour commencer
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("form_setup_admin"):
            nom_admin = st.text_input(
                "👤 Nom d'utilisateur admin",
                placeholder="Ex : admin_naftal"
            )
            mdp1 = st.text_input(
                "🔒 Mot de passe",
                type="password",
                placeholder="Choisissez un mot de passe..."
            )
            mdp2 = st.text_input(
                "🔒 Confirmer le mot de passe",
                type="password",
                placeholder="Répétez le mot de passe..."
            )
            creer = st.form_submit_button(
                "✅ Créer le compte administrateur",
                use_container_width=True
            )

        if creer:
            if not nom_admin or len(nom_admin.strip()) < 3:
                st.error("❌ Le nom doit contenir au moins 3 caractères.")
            elif not mdp1 or len(mdp1) < 4:
                st.error("❌ Le mot de passe doit contenir au moins 4 caractères.")
            elif mdp1 != mdp2:
                st.error("❌ Les mots de passe ne correspondent pas.")
            else:
                creer_compte(nom_admin, mdp1, role='admin')
                st.success("✅ Compte administrateur créé ! Connectez-vous maintenant.")
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# 10. PAGE — LOGIN
# ─────────────────────────────────────────────────────────────────────────────

def page_login():
    if not admin_existe():
        page_setup_admin()
        return

    inject_css()
    _, col_c, _ = st.columns([1, 2, 1])
    with col_c:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, use_container_width=True)
        else:
            st.markdown("""
            <div style="text-align:center;font-family:'Barlow Condensed',sans-serif;
                        font-size:2.5rem;font-weight:700;color:#1F3864;">NAFTAL</div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background:white;border-radius:14px;padding:2rem;
                    border:0.5px solid #dde4ee;
                    box-shadow:0 6px 25px rgba(31,56,100,0.10);margin-top:1rem;">
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.3rem;
                        font-weight:700;color:#1F3864;text-align:center;margin-bottom:0.2rem;">
                Bienvenue
            </div>
            <div style="font-size:0.82rem;color:#888;text-align:center;margin-bottom:1.2rem;">
                Connectez-vous pour accéder au système
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("form_login"):
            nom = st.text_input(
                "👤 Nom d'utilisateur",
                placeholder="Entrez votre identifiant..."
            )
            mdp = st.text_input(
                "🔒 Mot de passe",
                type="password",
                placeholder="Entrez votre mot de passe..."
            )
            connecter = st.form_submit_button(
                "🔐 Se connecter",
                use_container_width=True
            )

        if connecter:
            if not nom or not mdp:
                st.warning("⚠️ Remplissez tous les champs.")
            else:
                compte = verifier_login(nom, mdp)
                if compte is None:
                    st.error("❌ Identifiant ou mot de passe incorrect.")
                else:
                    st.session_state['connecte']     = True
                    st.session_state['employe_nom']  = compte['nom']
                    st.session_state['employe_role'] = compte['role']
                    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# 9. PAGE 1 — SIGNALER UNE PANNE
# ─────────────────────────────────────────────────────────────────────────────
def page_signaler():
    page_header("⚠️", "Signaler une Panne",
                "Décrivez ce que vous observez — le système s'occupe du reste")

    modele_ok = os.path.exists(MODEL_PATH) and os.path.exists(LABEL_MAPS_PATH)
    if not modele_ok:
        st.markdown("""<div class="banner-error">
            ⚠️ <b>Fichier modèle introuvable.</b>
            Vérifiez que <code>best_model.pt</code> et <code>label_maps.json</code>
            sont dans le même dossier que <code>app.py</code>.
        </div>""", unsafe_allow_html=True)

    with st.form("form_signalement"):
        st.markdown("#### 📍 Station")
        station = st.selectbox("Station", STATIONS, label_visibility="collapsed")
        st.markdown("#### ✍️ Décrivez la panne")
        st.caption("En français, sans termes techniques — écrivez simplement ce que vous voyez.")
        description = st.text_area(
            "Description", height=145,
            placeholder="Exemple : fuite d'huile sous la pompe, bruit anormal depuis ce matin...",
            label_visibility="collapsed"
        )
        envoyer = st.form_submit_button(
            "📤 Envoyer le signalement",
            use_container_width=True,
            disabled=not modele_ok
        )

    with st.expander("💡 Voir des exemples de description"):
        for i, ex in enumerate(EXEMPLES):
            st.markdown(f"**{i+1}.** {ex}")

    if envoyer:
        if not description or len(description.strip()) < 5:
            st.warning("⚠️ Veuillez écrire une description d'au moins quelques mots.")
        else:
            with st.spinner("⏳ Analyse en cours..."):
                try:
                    res = predire(description, station)
                    ref = ref_signalement()

                    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                    st.markdown("#### 📋 Résultat de l'analyse")

                    c1, c2, c3, c4 = st.columns(4)
                    cg = coul_grav(res['gravite'])
                    cp = coul_prio(res['priorite'])

                    with c1:
                        st.markdown(f"""
                        <div class="res-card {cg}">
                            <div class="res-lbl">⚠️ Gravité</div>
                            <div class="res-val {cg}">{res['gravite']}</div>
                            <div class="res-action">{action_grav(res['gravite'])}</div>
                        </div>""", unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"""
                        <div class="res-card {cp}">
                            <div class="res-lbl">🚨 Priorité</div>
                            <div class="res-val {cp}">{res['priorite']}</div>
                            <div class="res-action">Équipe notifiée</div>
                        </div>""", unsafe_allow_html=True)
                    with c3:
                        st.markdown(f"""
                        <div class="res-card bleu">
                            <div class="res-lbl">🔧 Équipement</div>
                            <div class="res-val bleu" style="font-size:1.1rem;">{res['equipement']}</div>
                        </div>""", unsafe_allow_html=True)
                    with c4:
                        st.markdown(f"""
                        <div class="res-card bleu">
                            <div class="res-lbl">⚡ Type de panne</div>
                            <div class="res-val bleu" style="font-size:1.1rem;">{res['type_panne']}</div>
                        </div>""", unsafe_allow_html=True)

                    st.markdown(f"""
                    <div class="banner-success">
                        ✅ <b>Signalement enregistré</b> — Réf. : <b>{ref}</b><br>
                        <span style="font-size:0.78rem;">
                            {res['date']} à {res['heure']} &nbsp;·&nbsp; {station}
                            &nbsp;·&nbsp; {st.session_state.get('employe_nom','')}
                        </span>
                    </div>""", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"❌ Erreur lors de l'analyse : {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 10. PAGE 2 — MES SIGNALEMENTS
# ─────────────────────────────────────────────────────────────────────────────
def page_mes_signalements():
    page_header("📋", "Mes Signalements",
                "Toutes les pannes enregistrées dans le système")

    df = charger_historique()

    if df is None:
        st.markdown("""<div class="banner-info">
            ℹ️ Aucun signalement trouvé. Déclarez votre première panne dans
            <b>⚠️ Signaler une Panne</b>.
        </div>""", unsafe_allow_html=True)
        return

    col_grav  = get_col(df, ['gravite'])
    col_prio  = get_col(df, ['priorite'])
    col_equip = get_col(df, ['equipement'])
    col_date  = get_col(df, ['date', 'date_intervention'])
    col_lieu  = get_col(df, ['lieu', 'station', 'station_type'])
    col_desc  = get_col(df, ['description'])

    cf1, cf2, cf3 = st.columns(3)
    with cf1:
        opts_g = ['Toutes'] + (sorted(df[col_grav].dropna().unique().tolist()) if col_grav else [])
        f_grav = st.selectbox("🔍 Gravité", opts_g)
    with cf2:
        opts_p = ['Toutes'] + (sorted(df[col_prio].dropna().unique().tolist()) if col_prio else [])
        f_prio = st.selectbox("🔍 Priorité", opts_p)
    with cf3:
        opts_m = ['Tous']
        if col_date and col_date in df.columns:
            mois_list = sorted(df[col_date].str[:7].dropna().unique().tolist(), reverse=True)
            opts_m += mois_list
        f_mois = st.selectbox("🔍 Mois", opts_m)

    dff = df.copy()
    if f_grav != 'Toutes' and col_grav:   dff = dff[dff[col_grav] == f_grav]
    if f_prio != 'Toutes' and col_prio:   dff = dff[dff[col_prio] == f_prio]
    if f_mois != 'Tous' and col_date:     dff = dff[dff[col_date].str[:7] == f_mois]

    st.caption(f"**{len(dff)}** signalement(s) sur **{len(df)}** au total")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    afficher = [c for c in [col_date, 'heure', col_lieu, 'employe', col_desc,
                             col_grav, col_prio, col_equip]
                if c and c in dff.columns]
    rename = {col_date:'Date', 'heure':'Heure', col_lieu:'Lieu',
              'employe':'Employé', col_desc:'Description',
              col_grav:'Gravité', col_prio:'Priorité', col_equip:'Équipement'}
    df_aff = dff[afficher].rename(columns={k:v for k,v in rename.items() if k in afficher})
    if 'Date' in df_aff.columns:
        df_aff = df_aff.sort_values('Date', ascending=False)

    st.dataframe(df_aff, use_container_width=True, height=420)

    csv_bytes = dff.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
    st.download_button(
        "⬇️ Télécharger (CSV)", data=csv_bytes,
        file_name=f"naftal_signalements_{datetime.now().strftime('%Y%m%d')}.csv",
        mime='text/csv'
    )


# ─────────────────────────────────────────────────────────────────────────────
# 11. PAGE 3 — TABLEAU DE BORD & STATISTIQUES (fusionnée + filtre + delta KPI)
# ─────────────────────────────────────────────────────────────────────────────
def page_dashboard_stats():
    page_header("📊", "Tableau de Bord & Statistiques",
                "Analyse complète · Filtrez par période · Comparaison automatique")

    df_brut = charger_tous_csv()

    if df_brut is None:
        st.markdown("""<div class="banner-info">
            ℹ️ Aucun fichier CSV trouvé. Placez vos fichiers <code>.csv</code>
            dans le même dossier que <code>app.py</code>.
        </div>""", unsafe_allow_html=True)
        return

    col_grav  = get_col(df_brut, ['gravite'])
    col_prio  = get_col(df_brut, ['priorite'])
    col_equip = get_col(df_brut, ['equipement'])
    col_type  = get_col(df_brut, ['type_panne'])
    col_date  = get_col(df_brut, ['date', 'date_intervention'])
    col_lieu  = get_col(df_brut, ['lieu', 'station', 'station_type'])

    # ── Plage dates disponibles ──────────────────────────────────────────────
    date_min = date(2020, 1, 1)
    date_max = date.today()
    if col_date:
        d_val = pd.to_datetime(df_brut[col_date], errors='coerce').dropna()
        if len(d_val) > 0:
            date_min = d_val.min().date()
            date_max = d_val.max().date()

    # ── Barre de filtres ─────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:white;border-radius:12px;padding:0.9rem 1.4rem;
                border:0.5px solid #dde4ee;margin-bottom:1.1rem;
                box-shadow:0 2px 10px rgba(31,56,100,0.06);">
        <span style="font-size:0.88rem;font-weight:600;color:#1F3864;">
            🗓️ Filtres de période
        </span>
    </div>
    """, unsafe_allow_html=True)

    fc1, fc2, fc3, fc4 = st.columns([1, 1.5, 1.5, 2])
    with fc1:
        granularite = st.radio("Afficher par", ['Jour','Mois','Année'],
                               index=1, horizontal=False)
    with fc2:
        debut = st.date_input("📅 Début", value=date_min,
                              min_value=date_min, max_value=date_max)
    with fc3:
        fin = st.date_input("📅 Fin", value=date_max,
                            min_value=date_min, max_value=date_max)
    with fc4:
        nb_sources = df_brut['_source'].nunique() if '_source' in df_brut.columns else 1
        debut_prec, fin_prec = periode_precedente(debut, fin)
        st.markdown(f"""
        <div style="background:#EEF2F7;border-radius:9px;padding:0.65rem 1rem;margin-top:0.3rem;">
            <div style="font-size:0.68rem;color:#888;text-transform:uppercase;">Données chargées</div>
            <div style="font-size:1.05rem;font-weight:700;color:#1F3864;
                        font-family:'Barlow Condensed',sans-serif;">
                {len(df_brut)} enreg. · {nb_sources} fichier(s) CSV
            </div>
            <div style="font-size:0.68rem;color:#999;margin-top:0.2rem;">
                Période préc. : {debut_prec.strftime('%d/%m/%Y')} → {fin_prec.strftime('%d/%m/%Y')}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Filtrer période courante ──────────────────────────────────────────────
    df, col_groupe = filtrer_periode(df_brut, col_date, granularite, debut, fin)

    # ── Filtrer période précédente ────────────────────────────────────────────
    df_prec, _ = filtrer_periode(df_brut, col_date, granularite, debut_prec, fin_prec)

    if len(df) == 0:
        st.warning("⚠️ Aucune donnée sur cette période. Élargissez la plage de dates.")
        return

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 1 — VUE GÉNÉRALE + KPIs avec delta ▲▼
    # ═══════════════════════════════════════════════════════════════════════
    section_header("📋", "Vue Générale")

    total_cur  = len(df)
    total_prec = len(df_prec) if df_prec is not None else 0

    crit_cur  = int((df[col_grav] == 'Critique').sum())       if col_grav else 0
    crit_prec = int((df_prec[col_grav] == 'Critique').sum())  if (df_prec is not None and col_grav) else 0

    urg_cur  = int((df[col_prio] == 'Urgente').sum())         if col_prio else 0
    urg_prec = int((df_prec[col_prio] == 'Urgente').sum())    if (df_prec is not None and col_prio) else 0

    taux_cur  = round((1 - crit_cur  / total_cur)  * 100) if total_cur  > 0 else 0
    taux_prec = round((1 - crit_prec / total_prec) * 100) if total_prec > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    kpi_avec_delta(c1, "📋", total_cur,  total_prec,  "Total signalements", "bleu",   inverse=False)
    kpi_avec_delta(c2, "🔴", crit_cur,   crit_prec,   "Pannes critiques",   "rouge",  inverse=False)
    kpi_avec_delta(c3, "🚨", urg_cur,    urg_prec,    "Urgences",           "orange", inverse=False)
    kpi_avec_delta(c4, "✅", f"{taux_cur}%", taux_prec, "Taux sans critique","vert",  inverse=True)

    g1, g2 = st.columns(2)

    with g1:
        if col_groupe and col_groupe in df.columns:
            pm = df.groupby(col_groupe).size().reset_index(name='Signalements')
            pm = pm.sort_values(col_groupe)
            fig = px.bar(pm, x=col_groupe, y='Signalements',
                         title=f'📅 Signalements par {granularite.lower()}',
                         color_discrete_sequence=['#2E75B6'],
                         labels={col_groupe: granularite})
            fig.update_traces(marker_line_width=0)
            if granularite == 'Jour' and len(pm) > 20:
                fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(plotly_style(fig, 300), use_container_width=True)

    with g2:
        if col_grav:
            gc = df[col_grav].value_counts().reset_index()
            gc.columns = ['gravite','count']
            fig2 = px.pie(gc, names='gravite', values='count',
                          title='🎯 Répartition par gravité',
                          color='gravite',
                          color_discrete_map=COULEURS_GRAVITE, hole=0.42)
            fig2.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(plotly_style(fig2, 300), use_container_width=True)

    g3, g4 = st.columns(2)

    with g3:
        if col_equip:
            te = df[col_equip].value_counts().head(10).reset_index()
            te.columns = ['equipement','count']
            fig3 = px.bar(te, x='count', y='equipement', orientation='h',
                          title='🔧 Top 10 équipements défaillants',
                          color='count',
                          color_continuous_scale=['#2E75B6','#1F3864'],
                          labels={'count':'Pannes','equipement':''})
            fig3.update_layout(yaxis={'categoryorder':'total ascending'},
                               coloraxis_showscale=False, showlegend=False)
            st.plotly_chart(plotly_style(fig3, 320), use_container_width=True)

    with g4:
        if col_prio and col_groupe and col_groupe in df.columns:
            prm = df.groupby([col_groupe, col_prio]).size().reset_index(name='count')
            prm = prm.sort_values(col_groupe)
            fig4 = px.bar(prm, x=col_groupe, y='count', color=col_prio,
                          title=f'🚨 Priorités par {granularite.lower()}',
                          color_discrete_map=COULEURS_PRIORITE, barmode='stack',
                          labels={col_groupe:granularite,'count':'Pannes',col_prio:'Priorité'})
            st.plotly_chart(plotly_style(fig4, 320), use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 2 — ANALYSE DÉTAILLÉE
    # ═══════════════════════════════════════════════════════════════════════
    section_header("🔍", "Analyse Détaillée")

    ga, gb = st.columns(2)
    with ga:
        if col_type:
            tc = df[col_type].value_counts().reset_index()
            tc.columns = ['type','count']
            fig5 = px.bar(tc, x='count', y='type', orientation='h',
                          title='⚡ Types de pannes les plus fréquents',
                          color_discrete_sequence=['#E8A838'],
                          labels={'count':'Occurrences','type':''})
            fig5.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)
            st.plotly_chart(plotly_style(fig5, 360), use_container_width=True)

    with gb:
        if col_lieu:
            lc = df[col_lieu].value_counts().head(10).reset_index()
            lc.columns = ['lieu','count']
            fig6 = px.bar(lc, x='count', y='lieu', orientation='h',
                          title='📍 Pannes par station / lieu',
                          color_discrete_sequence=['#2E75B6'],
                          labels={'count':'Pannes','lieu':''})
            fig6.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)
            st.plotly_chart(plotly_style(fig6, 360), use_container_width=True)

    gc2, gd2 = st.columns(2)
    with gc2:
        if col_lieu and col_grav:
            lg2 = df.groupby([col_lieu, col_grav]).size().reset_index(name='count')
            top_lieux = df[col_lieu].value_counts().head(6).index.tolist()
            lg2 = lg2[lg2[col_lieu].isin(top_lieux)]
            fig7 = px.bar(lg2, x=col_lieu, y='count', color=col_grav,
                          title='🎯 Gravité par station (top 6)',
                          color_discrete_map=COULEURS_GRAVITE, barmode='stack',
                          labels={col_lieu:'Station','count':'Pannes',col_grav:'Gravité'})
            fig7.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(plotly_style(fig7, 360), use_container_width=True)

    with gd2:
        if col_grav and col_prio:
            cross = pd.crosstab(df[col_grav], df[col_prio])
            fig8 = px.imshow(cross, title='🔲 Matrice Gravité × Priorité',
                             color_continuous_scale='Blues', text_auto=True,
                             labels={'x':'Priorité','y':'Gravité','color':'Pannes'})
            fig8.update_layout(paper_bgcolor='white',
                               title_font=dict(size=13,color='#1F3864',family='Barlow Condensed'),
                               font=dict(family='Barlow'), height=360,
                               margin=dict(t=40,b=30,l=20,r=20))
            st.plotly_chart(fig8, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 3 — COURBES TEMPORELLES
    # ═══════════════════════════════════════════════════════════════════════
    section_header("📈", "Courbes Temporelles")

    if col_groupe and col_groupe in df.columns:
        jj = df.groupby(col_groupe).size().reset_index(name='count')
        jj = jj.sort_values(col_groupe)
        if len(jj) > 1:
            fig9 = px.line(jj, x=col_groupe, y='count',
                           title=f'Évolution des signalements par {granularite.lower()}',
                           labels={col_groupe:granularite,'count':'Signalements'},
                           line_shape='spline', color_discrete_sequence=['#2E75B6'])
            fig9.update_traces(fill='tozeroy', fillcolor='rgba(46,117,182,0.08)', line_width=2)
            st.plotly_chart(plotly_style(fig9, 270), use_container_width=True)

        if col_grav:
            mg = df.groupby([col_groupe, col_grav]).size().reset_index(name='count')
            mg = mg.sort_values(col_groupe)
            fig10 = px.line(mg, x=col_groupe, y='count', color=col_grav,
                            title=f'Évolution par gravité',
                            color_discrete_map=COULEURS_GRAVITE, markers=True,
                            labels={col_groupe:granularite,'count':'Pannes',col_grav:'Gravité'})
            fig10.update_traces(line_width=2)
            st.plotly_chart(plotly_style(fig10, 270), use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 4 — COMPARAISON CSV
    # ═══════════════════════════════════════════════════════════════════════
    if '_source' in df.columns and df['_source'].nunique() > 1 and col_grav:
        section_header("📁", "Comparaison entre Fichiers CSV")
        sg = df.groupby(['_source', col_grav]).size().reset_index(name='count')
        fig11 = px.bar(sg, x='_source', y='count', color=col_grav,
                       title='Gravité par fichier source',
                       color_discrete_map=COULEURS_GRAVITE, barmode='group',
                       labels={'_source':'Fichier','count':'Pannes',col_grav:'Gravité'})
        st.plotly_chart(plotly_style(fig11, 300), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# 12. PAGE 4 — COMMENT UTILISER ?
# ─────────────────────────────────────────────────────────────────────────────
def page_aide():
    page_header("❓", "Comment utiliser ?", "Guide rapide — tout ce dont vous avez besoin")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🔐 Connexion")
        st.markdown("""
        <div class="step">
            <div class="step-num">1</div>
            <div class="step-txt"><b>Entrez votre nom</b> et sélectionnez votre station au démarrage.</div>
        </div>
        <div class="step">
            <div class="step-num">2</div>
            <div class="step-txt">Votre nom est <b>enregistré automatiquement</b> avec chaque signalement.</div>
        </div>
        <div class="step">
            <div class="step-num">3</div>
            <div class="step-txt">Pour changer d'utilisateur, cliquez sur <b>Se déconnecter</b> dans la sidebar.</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### ⚠️ Signaler une panne — 3 étapes")
        st.markdown("""
        <div class="step">
            <div class="step-num">1</div>
            <div class="step-txt"><b>Votre station</b> est pré-remplie depuis la connexion.</div>
        </div>
        <div class="step">
            <div class="step-num">2</div>
            <div class="step-txt">
                <b>Décrivez ce que vous voyez</b> en français simple.<br>
                <span style="color:#888;font-size:0.78rem;">"Fuite d'huile sous la pompe, bruit bizarre depuis ce matin"</span>
            </div>
        </div>
        <div class="step">
            <div class="step-num">3</div>
            <div class="step-txt"><b>Cliquez sur Envoyer</b> — vous recevez une référence.</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🎨 Que signifient les couleurs ?")
        st.markdown("""
        <div style="display:flex;flex-direction:column;gap:0.65rem;margin-top:0.4rem;">
            <div style="display:flex;align-items:center;gap:0.8rem;">
                <span class="badge b-rouge" style="min-width:75px;text-align:center;">Critique</span>
                <span style="font-size:0.82rem;color:#444;">Panne grave — intervention immédiate</span>
            </div>
            <div style="display:flex;align-items:center;gap:0.8rem;">
                <span class="badge b-orange" style="min-width:75px;text-align:center;">Modérée</span>
                <span style="font-size:0.82rem;color:#444;">Panne importante — à traiter rapidement</span>
            </div>
            <div style="display:flex;align-items:center;gap:0.8rem;">
                <span class="badge b-vert" style="min-width:75px;text-align:center;">Mineure</span>
                <span style="font-size:0.82rem;color:#444;">Panne légère — planifiable sans urgence</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 📊 Tableau de Bord & Statistiques")
        st.markdown("""
        <div class="step">
            <div class="step-num">1</div>
            <div class="step-txt">
                <b>Choisissez la granularité</b> : Jour, Mois ou Année.
            </div>
        </div>
        <div class="step">
            <div class="step-num">2</div>
            <div class="step-txt">
                <b>Sélectionnez la période</b> — les KPIs et graphiques se recalculent automatiquement.
            </div>
        </div>
        <div class="step">
            <div class="step-num">3</div>
            <div class="step-txt">
                Les <b>flèches ▲▼</b> sur les KPIs comparent avec la période précédente de même durée.
                <span style="color:#C0392B;">▲ Rouge</span> = hausse des pannes (mauvais).
                <span style="color:#27AE60;">▼ Vert</span> = baisse (bon).
            </div>
        </div>
        <div class="step">
            <div class="step-num">4</div>
            <div class="step-txt">
                Placez un fichier <code>.csv</code> dans le dossier — il est <b>intégré automatiquement</b>.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### ❓ Questions fréquentes")
        with st.expander("Que se passe-t-il après mon signalement ?"):
            st.write("Il est enregistré automatiquement avec votre nom et transmis au responsable maintenance.")
        with st.expander("Puis-je modifier un signalement ?"):
            st.write("Non. En cas d'erreur, faites un nouveau signalement avec la correction.")
        with st.expander("Que signifient les flèches ▲▼ sur les KPIs ?"):
            st.write("Elles comparent la période sélectionnée avec la période précédente de même durée. Ex : si vous regardez janvier 2026, la comparaison se fait avec décembre 2025.")
        with st.expander("Quels fichiers CSV sont analysés ?"):
            st.write("Tous les fichiers .csv dans le dossier de l'application sont fusionnés et analysés ensemble.")
        with st.expander("Comment exporter mes données ?"):
            st.write("Dans 'Mes Signalements', cliquez sur ⬇️ Télécharger (CSV).")
        st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 13. POINT D'ENTRÉE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
# PAGE — GESTION DES COMPTES (admin uniquement)
# ─────────────────────────────────────────────────────────────────────────────

def page_gestion_comptes():
    if st.session_state.get('employe_role') != 'admin':
        st.error("❌ Accès refusé.")
        return

    page_header("⚙️", "Gestion des Comptes",
                "Créez et gérez les comptes des employés")

    col1, col2 = st.columns(2, gap="large")

    # ── Créer un compte employé ──────────────────────────────────────────────
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### ➕ Créer un compte employé")
        with st.form("form_creer_employe"):
            new_nom  = st.text_input("👤 Nom d'utilisateur",
                                     placeholder="Ex : ahmed.bensalem")
            new_mdp  = st.text_input("🔒 Mot de passe", type="password",
                                     placeholder="Choisissez un mot de passe...")
            new_mdp2 = st.text_input("🔒 Confirmer le mot de passe", type="password",
                                     placeholder="Répétez le mot de passe...")
            btn_creer = st.form_submit_button("✅ Créer le compte",
                                              use_container_width=True)
        if btn_creer:
            if not new_nom or len(new_nom.strip()) < 3:
                st.error("❌ Le nom doit contenir au moins 3 caractères.")
            elif not new_mdp or len(new_mdp) < 4:
                st.error("❌ Le mot de passe doit contenir au moins 4 caractères.")
            elif new_mdp != new_mdp2:
                st.error("❌ Les mots de passe ne correspondent pas.")
            elif compte_existe(new_nom):
                st.error(f"❌ Le nom **{new_nom}** existe déjà.")
            else:
                creer_compte(new_nom, new_mdp, role='employe')
                st.success(f"✅ Compte **{new_nom}** créé avec succès !")
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Liste des comptes employés ───────────────────────────────────────────
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 👥 Comptes employés existants")
        employes = lister_employes()
        if not employes:
            st.markdown("""<div class="banner-info">
                ℹ️ Aucun compte employé créé pour l'instant.
            </div>""", unsafe_allow_html=True)
        else:
            for emp in employes:
                cn, cd, cs = st.columns([2, 1.5, 1])
                with cn:
                    st.markdown(f"""
                    <div style="padding:0.4rem 0;font-size:0.85rem;
                                font-weight:500;color:#1F3864;">👤 {emp['nom']}</div>
                    """, unsafe_allow_html=True)
                with cd:
                    st.markdown(f"""
                    <div style="padding:0.4rem 0;font-size:0.72rem;color:#888;">
                        {emp.get('cree_le','—')[:10]}
                    </div>
                    """, unsafe_allow_html=True)
                with cs:
                    if st.button("🗑️", key=f"del_{emp['nom']}",
                                 help=f"Supprimer {emp['nom']}"):
                        supprimer_compte(emp['nom'])
                        st.success(f"✅ Compte {emp['nom']} supprimé.")
                        st.rerun()
                st.markdown("<hr style='margin:0.2rem 0;border-color:#EEF2F7;'>",
                            unsafe_allow_html=True)
        st.markdown(f"""
        <div style="margin-top:0.8rem;font-size:0.75rem;color:#888;">
            Total : <b>{len(employes)}</b> compte(s) employé(s)
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Changer son propre mot de passe ──────────────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    with st.expander("🔑 Changer mon mot de passe admin"):
        with st.form("form_changer_mdp"):
            ancien = st.text_input("Ancien mot de passe", type="password")
            nv1    = st.text_input("Nouveau mot de passe", type="password")
            nv2    = st.text_input("Confirmer le nouveau",  type="password")
            btn_chg = st.form_submit_button("💾 Enregistrer",
                                            use_container_width=True)
        if btn_chg:
            nom_admin = st.session_state.get('employe_nom', '')
            if verifier_login(nom_admin, ancien) is None:
                st.error("❌ Ancien mot de passe incorrect.")
            elif not nv1 or len(nv1) < 4:
                st.error("❌ Le nouveau mot de passe doit contenir au moins 4 caractères.")
            elif nv1 != nv2:
                st.error("❌ Les mots de passe ne correspondent pas.")
            else:
                comptes = charger_comptes()
                comptes[nom_admin.strip().lower()]['mdp_hash'] = hacher_mdp(nv1)
                sauvegarder_comptes(comptes)
                st.success("✅ Mot de passe mis à jour avec succès !")


# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
def main():
    inject_css()

    # ── 1. Vérifier connexion ─────────────────────────────────────────────────
    if not st.session_state.get('connecte', False):
        page_login()
        return

    # ── 2. Sidebar ────────────────────────────────────────────────────────────
    logo_sidebar()
    sidebar_user_badge()
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Menu")

    # ── 3. Menu selon le rôle ─────────────────────────────────────────────────
    role = st.session_state.get('employe_role', 'employe')

    pages_employe = [
        "⚠️  Signaler une Panne",
        "📋  Mes Signalements",
        "📊  Tableau de Bord & Statistiques",
        "❓  Comment utiliser ?",
    ]
    pages_admin = pages_employe + ["⚙️  Gestion des Comptes"]

    pages = pages_admin if role == 'admin' else pages_employe
    page  = st.sidebar.radio("", pages, label_visibility="collapsed")

    # ── 4. Routing ────────────────────────────────────────────────────────────
    if   "Signaler"    in page: page_signaler()
    elif "Signalement" in page: page_mes_signalements()
    elif "Bord"        in page: page_dashboard_stats()
    elif "utiliser"    in page: page_aide()
    elif "Gestion"     in page: page_gestion_comptes()


if __name__ == "__main__":
    main()
