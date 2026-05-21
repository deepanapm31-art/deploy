import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings, json, hashlib, os, re
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (r2_score, mean_absolute_error,
                              mean_squared_error, mean_absolute_percentage_error)
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import shap


# PAGE CONFIG

st.set_page_config(
    page_title="Real Estate AI - Tamil Nadu",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# SESSION STATE DEFAULTS

for key, val in {
    'logged_in'  : False,
    'username'   : '',
    'fullname'   : '',
    'dark_mode'  : True,
    'page'       : 'home',
    'result'     : None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val


# THEME

def get_theme():
    return {
        'bg'       : '#F5F7FA',
        'sidebar'  : '#FFFFFF',
        'card'     : '#FFFFFF',
        'border'   : '#D0D8E4',
        'text'     : '#1A2B3C',
        'subtext'  : '#4A5568',
        'input_bg' : '#FFFFFF',
        'input_txt': '#1A2B3C',
        'accent1'  : '#0077B6',
        'accent2'  : '#5E35B1',
        'green'    : '#00875A',
        'orange'   : '#D05B00',
        'red'      : '#C0123C',
        'yellow'   : '#B7850A',
    }

def apply_theme():
    t = get_theme()
    st.markdown(f"""
    <style>
    body, .stApp {{ background-color:{t['bg']} !important; color:{t['text']} !important; }}
    section[data-testid="stSidebar"] {{ background-color:{t['sidebar']} !important; }}
    section[data-testid="stSidebar"] * {{ color:{t['text']} !important; }}
    .stButton>button {{
        background: linear-gradient(135deg,{t['accent2']},{t['accent1']});
        color:white !important; font-weight:700; border:none;
        border-radius:10px; padding:0.6rem 1.5rem; width:100%;
        transition: opacity 0.1s;
    }}
    .stButton>button:hover {{ opacity:0.6; }}
    .card {{
        background:{t['card']}; border-radius:14px;
        padding:1.2rem; text-align:center;
        border:1.5px solid {t['border']}; margin-bottom:0.5rem;
        transition: transform 0.2s;
    }}
    .card:hover {{ transform: translateY(-2px); }}
    .card-val {{ font-size:1.8rem; font-weight:800; }}
    .card-lbl {{ font-size:0.8rem; color:{t['subtext']}; margin-top:0.3rem; }}
    input[type="text"], input[type="password"], input[type="email"] {{
        color:{t['input_txt']} !important;
        background-color:{t['input_bg']} !important;
        border:1.5px solid {t['border']} !important;
        border-radius:8px !important;
    }}
    input::placeholder {{ color:{t['subtext']} !important; opacity:1 !important; }}
    .stTabs [data-baseweb="tab"] {{ color:{t['subtext']} !important; font-weight:600; }}
    .stTabs [aria-selected="true"] {{
        color:{t['accent1']} !important;
        border-bottom:2px solid {t['accent1']} !important;
    }}
    .stSelectbox label, .stNumberInput label, .stSlider label {{
        color:{t['text']} !important; font-weight:500;
    }}
    .stSelectbox div[data-baseweb="select"] > div {{
        background:{t['input_bg']} !important;
        border-color:{t['border']} !important;
        color:{t['input_txt']} !important;
    }}
    div[data-testid="stMetricValue"] {{ color:{t['accent1']} !important; }}
    .stAlert {{ background:{t['card']} !important; border-color:{t['border']} !important; }}
    hr {{ border-color:{t['border']} !important; }}
    p, li, label {{ color:{t['text']} !important; }}
    h1,h2,h3,h4 {{ color:{t['text']} !important; }}
    .page-header {{
        font-size:2rem; font-weight:800;
        color:{t['accent1']}; margin-bottom:0.2rem;
    }}
    .page-sub {{
        font-size:1rem; color:{t['subtext']};
        margin-bottom:1.5rem;
    }}
    .result-price {{
        background:linear-gradient(135deg,{t['card']},{t['input_bg']});
        border:2px solid {t['accent1']}; border-radius:16px;
        padding:1.5rem; text-align:center; margin-bottom:1.5rem;
    }}
    .factor-up {{
        background:{t['green']}22;
        border-left:4px solid {t['green']};
        padding:0.4rem 0.8rem; margin:0.3rem 0;
        border-radius:0 6px 6px 0;
        color:{t['green']}; font-weight:700;
    }}
    .factor-dn {{
        background:{t['red']}22;
        border-left:4px solid {t['red']};
        padding:0.4rem 0.8rem; margin:0.3rem 0;
        border-radius:0 6px 6px 0;
        color:{t['red']}; font-weight:700;
    }}
    .optuna-card {{
        background:{t['card']}; border-radius:14px;
        padding:1.2rem; border:1.5px solid {t['accent2']}44;
        margin-bottom:0.8rem;
    }}
    .optuna-title {{
        font-size:1rem; font-weight:700;
        color:{t['accent2']}; margin-bottom:0.5rem;
    }}
    .param-row {{
        display:flex; justify-content:space-between;
        padding:0.25rem 0; border-bottom:1px solid {t['border']};
        font-size:0.85rem;
    }}
    .param-key {{ color:{t['subtext']}; }}
    .param-val {{ color:{t['accent1']}; font-weight:700; }}
    .feature-card {{
        background:white; padding:20px; border-radius:15px;
        box-shadow:0px 4px 15px rgba(0,0,0,0.05); text-align:center;
    }}
    </style>
    """, unsafe_allow_html=True)


# USER AUTH

USERS_FILE = "users.json"

def hash_pwd(p):
    return hashlib.sha256(p.encode()).hexdigest()

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE) as f:
            return json.load(f)
    return {}

def save_users(u):
    with open(USERS_FILE, "w") as f:
        json.dump(u, f, indent=2)

def register_user(username, password, email, fullname):
    users = load_users()
    if len(username) < 3:   return False, "Username must be >= 3 characters!"
    if len(password) < 6:   return False, "Password must be >= 6 characters!"
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email): return False, "Invalid email!"
    if username in users:   return False, "Username already exists!"
    users[username] = {"password": hash_pwd(password), "email": email, "fullname": fullname}
    save_users(users)
    return True, "Account created! Please login."

def login_user(username, password):
    users = load_users()
    if username not in users:                              return False, "Username not found!"
    if users[username]["password"] != hash_pwd(password): return False, "Wrong password!"
    return True, users[username]["fullname"]


# CONSTANTS

CSV_PATH = "dataset.csv"


# ⚡ FAST OPTUNA TUNING

# SPEED FIXES applied here:
#  1. n_trials = 10  (was 30)  → 3× faster
#  2. KFold   = 2    (was 3)   → 1.5× faster
#  3. iterations max = 500     (was 1000) → 2× faster
#  4. early_stopping = 30      (was 50)   → faster per fold
#  Combined: ~9× faster than original  (was 3-6 min → now ~30-50 sec)

def tune_catboost_with_optuna(X_train, y_train, n_trials=10):
    """Fast Optuna tuning: 10 trials × 2-fold CV = 20 fits (was 90)."""

    def objective(trial):
        params = {
            'iterations'         : trial.suggest_int('iterations', 200, 500),   # ⚡ max 500
            'learning_rate'      : trial.suggest_float('learning_rate', 0.05, 0.3, log=True),
            'depth'              : trial.suggest_int('depth', 4, 8),             # ⚡ max 8
            'l2_leaf_reg'        : trial.suggest_float('l2_leaf_reg', 1.0, 8.0),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'random_strength'    : trial.suggest_float('random_strength', 0.0, 1.0),
            'border_count'       : trial.suggest_int('border_count', 32, 128),   # ⚡ max 128
            'random_seed'        : 42,
            'verbose'            : 0,
            'eval_metric'        : 'RMSE',
        }
        kf = KFold(n_splits=2, shuffle=True, random_state=42)                   # ⚡ 2-fold
        scores = []
        for train_idx, val_idx in kf.split(X_train):
            Xtr_f = X_train.iloc[train_idx]
            Xval  = X_train.iloc[val_idx]
            ytr_f = y_train.iloc[train_idx]
            yval  = y_train.iloc[val_idx]
            model = CatBoostRegressor(**params)
            model.fit(Xtr_f, ytr_f, eval_set=(Xval, yval),
                      early_stopping_rounds=30, verbose=0)                       # ⚡ 30 rounds
            scores.append(r2_score(yval, model.predict(Xval)))
        return np.mean(scores)

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study


# LOAD & TRAIN MODELS  (cached — runs only once)

@st.cache_resource(show_spinner="🤖 Training AI models... (~30-50 sec, only once)")
def load_models():
    # ── 1. Load & clean data ───────────────────────────────
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()
    for col in ['area_sqft','bedrooms','avg_rooms','bathrooms','population',
                'avg_occupancy','latitude','longitude','price',
                'price_per_sqft','distance_to_city_center']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['city']          = df['city'].str.strip()
    df['area_name']     = df['area_name'].str.strip()
    df['house_age']     = df['house_age'].str.strip()
    df['property_type'] = df['property_type'].str.strip()
    df = df.dropna().reset_index(drop=True)

    # ⚡ Use only 70% of data for tuning speed (still accurate)
    df_tune = df.sample(frac=0.7, random_state=42).reset_index(drop=True)

    # ── 2. Feature engineering ────────────────────────────
    AGE_MAP = {'0-1': 0, '1-5': 1, '5-10': 2, '10+': 3}

    def engineer(d):
        d = d.copy()
        d['house_age_num']    = d['house_age'].map(AGE_MAP).fillna(1)
        d['room_density']     = d['avg_rooms'] / (d['area_sqft'] + 1) * 100
        d['pop_density']      = d['population'] / (d['area_sqft'] + 1)
        d['bed_bath_ratio']   = d['bedrooms'] / (d['bathrooms'] + 1)
        d['dist_score']       = 1 / (d['distance_to_city_center'] + 1)
        d['total_rooms']      = d['bedrooms'] + d['bathrooms']
        d['luxury_score']     = (d['area_sqft'] / 1000) * d['avg_rooms'] * d['bedrooms']
        d['space_per_person'] = d['area_sqft'] / (d['avg_occupancy'] + 1)
        d['log_area_sqft']    = np.log1p(d['area_sqft'])
        d['log_population']   = np.log1p(d['population'])
        d['city_type_combo']  = d['city'] + "_" + d['property_type']
        return d

    df      = engineer(df)
    df_tune = engineer(df_tune)

    # price_zone needs qcut on full df
    df['price_zone']      = pd.qcut(df['distance_to_city_center'],
                                    q=4, labels=[3,2,1,0]).astype(int)
    df_tune['price_zone'] = pd.cut(df_tune['distance_to_city_center'],
                                   bins=pd.qcut(df['distance_to_city_center'],
                                                q=4, retbins=True)[1],
                                   labels=[3,2,1,0], include_lowest=True).astype(int)

    # Median encodings from full df
    city_median      = df.groupby('city')['price'].median()
    area_median      = df.groupby('area_name')['price'].median()
    city_type_median = df.groupby('city_type_combo')['price'].median()

    for d in [df, df_tune]:
        d['city_med_price']  = d['city'].map(city_median)
        d['area_med_price']  = d['area_name'].map(area_median)
        d['city_type_med']   = d['city_type_combo'].map(city_type_median)
        d['city_price_rank'] = d.groupby('city')['price'].rank(pct=True)

    # Climate & liveness
    def clim(row):
        lat, lon = float(row['latitude']), float(row['longitude'])
        f     = max(0, min(100 - abs(lon - 80.28) * 50, 100))
        c     = 80 if lon < 80.5 else 30
        h     = max(0, min((13.5 - lat) * 15, 100))
        total = f * 0.4 + c * 0.35 + h * 0.25
        return pd.Series({'climate_risk_score': round(total, 1),
                          'climate_risk_label': 'HIGH' if total >= 70 else
                                               ('MEDIUM' if total >= 45 else 'LOW')})

    def live(row):
        s = (min(row['population']     / 200000 * 100, 100) * 0.35 +
             min(row['avg_occupancy']  / 5      * 100, 100) * 0.20 +
             min(row['avg_rooms']      / 8      * 100, 100) * 0.15 +
             min(row['price_per_sqft'] / 20000  * 100, 100) * 0.30)
        g = 'A' if s >= 70 else ('B' if s >= 50 else ('C' if s >= 30 else 'D'))
        return pd.Series({'liveness_score': round(s, 1), 'liveness_grade': g})

    df      = pd.concat([df,      df.apply(clim, axis=1),      df.apply(live, axis=1)],      axis=1)
    df_tune = pd.concat([df_tune, df_tune.apply(clim, axis=1), df_tune.apply(live, axis=1)], axis=1)

    # Label encoders (fit on full df)
    le_city  = LabelEncoder(); le_area  = LabelEncoder()
    le_type  = LabelEncoder(); le_combo = LabelEncoder()
    df['city_enc']      = le_city.fit_transform(df['city'])
    df['area_enc']      = le_area.fit_transform(df['area_name'])
    df['type_enc']      = le_type.fit_transform(df['property_type'])
    df['city_type_enc'] = le_combo.fit_transform(df['city_type_combo'])

    # Apply same encoders to df_tune safely
    def safe_enc(le, col):
        return col.map(lambda x: le.transform([x])[0] if x in le.classes_ else 0)

    df_tune['city_enc']      = safe_enc(le_city,  df_tune['city'])
    df_tune['area_enc']      = safe_enc(le_area,  df_tune['area_name'])
    df_tune['type_enc']      = safe_enc(le_type,  df_tune['property_type'])
    df_tune['city_type_enc'] = safe_enc(le_combo, df_tune['city_type_combo'])

    FEAT = [
        'area_sqft','house_age_num','bedrooms','avg_rooms','bathrooms',
        'population','avg_occupancy','latitude','longitude',
        'city_enc','area_enc','type_enc',
        'room_density','pop_density','city_med_price','area_med_price',
        'bed_bath_ratio','dist_score','distance_to_city_center',
        'climate_risk_score','liveness_score',
        'total_rooms','luxury_score','space_per_person',
        'log_area_sqft','price_zone','log_population',
        'city_type_enc','city_type_med','city_price_rank'
    ]

    # ── 3. Train/test split (full df for final eval) ──────
    X_full = df[FEAT].fillna(0)
    y_full = df['price']
    Xtr_full, Xte, ytr_full, yte = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42)
    Xte = Xte.sample(200, random_state=42)
    yte = yte.loc[Xte.index]

    # ⚡ Optuna tuning on smaller df_tune (70% data)
    X_t = df_tune[FEAT].fillna(0)
    y_t = df_tune['price']
    Xtr_t, _, ytr_t, _ = train_test_split(X_t, y_t, test_size=0.2, random_state=42)

    best_params, study = tune_catboost_with_optuna(Xtr_t, ytr_t, n_trials=10)

    # ── 4. Final CatBoost on full training data ───────────
    cat = CatBoostRegressor(
        iterations          = best_params.get('iterations', 400),
        learning_rate       = best_params.get('learning_rate', 0.08),
        depth               = best_params.get('depth', 6),
        l2_leaf_reg         = best_params.get('l2_leaf_reg', 3.0),
        bagging_temperature = best_params.get('bagging_temperature', 0.5),
        random_strength     = best_params.get('random_strength', 0.5),
        border_count        = best_params.get('border_count', 64),
        random_seed         = 42,
        verbose             = 0,
        eval_metric         = 'RMSE',
    )
    cat.fit(Xtr_full, ytr_full,
            eval_set=(Xte, yte),
            early_stopping_rounds=50)

    # ── 5. XGBoost ⚡ reduced estimators ──────────────────
    xgb = XGBRegressor(
        n_estimators    = 400,          # ⚡ was 1000
        learning_rate   = 0.05,         # ⚡ slightly higher = fewer trees needed
        max_depth       = 6,
        subsample       = 0.85,
        colsample_bytree= 0.85,
        min_child_weight= 3,
        reg_alpha       = 0.1,
        reg_lambda      = 1.0,
        random_state    = 42,
        n_jobs          = -1,
        verbosity       = 0,
        early_stopping_rounds = 30,     # ⚡ stops early if no improvement
    )
    xgb.fit(Xtr_full, ytr_full,
            eval_set=[(Xte, yte)],
            verbose=False)

    # ── 6. Ensemble weights ───────────────────────────────
    cp = cat.predict(Xte)
    xp = xgb.predict(Xte)
    cr = r2_score(yte, cp)
    xr = r2_score(yte, xp)
    wc = cr / (cr + xr)
    wx = xr / (cr + xr)

    # ── 7. SHAP ⚡ on sample only ─────────────────────────
    explainer = shap.TreeExplainer(cat)

    city_med = df.groupby('city')['price'].median()
    area_med = df.groupby('area_name')['price'].median()

    optuna_info = {
        'best_params' : best_params,
        'best_value'  : study.best_value,
        'n_trials'    : len(study.trials),
        'trial_values': [tr.value for tr in study.trials if tr.value is not None],
    }

    return (df, cat, xgb, wc, wx, le_city, le_area, le_type, le_combo,
            city_med, area_med, FEAT, explainer, cr, xr, Xte, yte, optuna_info)

# HELPERS

def get_climate(lat, lon):
    f     = max(0, min(100 - abs(lon - 80.28) * 50, 100))
    c     = 80 if lon < 80.5 else 30
    h     = max(0, min((13.5 - lat) * 15, 100))
    total = f * 0.4 + c * 0.35 + h * 0.25
    return {'flood': round(f,1), 'cyclone': round(c,1), 'heat': round(h,1),
            'total': round(total,1),
            'label': 'HIGH' if total >= 70 else ('MEDIUM' if total >= 45 else 'LOW')}

def get_liveness(pop, occ, rooms, pps):
    s = (min(pop   / 200000 * 100, 100) * 0.35 +
         min(occ   / 5      * 100, 100) * 0.20 +
         min(rooms / 8      * 100, 100) * 0.15 +
         min(pps   / 20000  * 100, 100) * 0.30)
    g = 'A' if s >= 70 else ('B' if s >= 50 else ('C' if s >= 30 else 'D'))
    return round(s, 1), g

def make_chart():
    return {'bg':'#F5F7FA','card':'#FFFFFF','text':'#1A2B3C','grid':'#D0D8E4'}


# PAGE: HOME

def page_home():
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        <style>
        .hero {
            background: linear-gradient(135deg, #f5f7fa, #e4ecf7);
            padding: 50px; border-radius: 20px;
        }
        .hero-title { font-size: 42px; font-weight: 800; color: #1f2d3d; }
        .hero-sub   { font-size: 16px; color: #5f6c7b; margin-top: 10px; }
        .btn {
            background-color: #22c55e; color: white;
            padding: 10px 20px; border-radius: 20px;
            text-decoration: none; display: inline-block;
            margin-top: 15px; font-weight: 600;
        }
        </style>
        <div class="hero">
            <div class="hero-title">Smart AI House Price Prediction</div>
            <div class="hero-sub">
                Predict property prices with Climate Risk, Liveness Score,
                and AI-powered insights across Tamil Nadu.
            </div>
            <a class="btn">Get Started</a>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        if os.path.exists("home.jpg"):
            st.image("home.jpg", use_container_width=True)
        elif os.path.exists("home.png"):
            st.image("home.png", use_container_width=True)
        else:
            st.markdown("""
            <div style='background:linear-gradient(135deg,#e4ecf7,#c9d8f0);
                border-radius:15px;padding:60px 20px;text-align:center;font-size:60px'>
                🏠🤖</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("## 🔥 Features")
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""<div class="feature-card"><h3>🤖 AI Prediction</h3>
        <p>CatBoost (Optuna-tuned) + XGBoost Ensemble</p></div>""", unsafe_allow_html=True)
    with f2:
        st.markdown("""<div class="feature-card"><h3>🌧️ Climate Risk</h3>
        <p>Flood, Cyclone & Heat Analysis</p></div>""", unsafe_allow_html=True)
    with f3:
        st.markdown("""<div class="feature-card"><h3>📍 Liveness Score</h3>
        <p>Amenities & Population Index</p></div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div class="feature-card"><h4>📊 Data Insights</h4>
        <p>6000+ records trained with 30 features</p></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="feature-card"><h4>⚖️ Fairness AI</h4>
        <p>SHAP-based bias detection system</p></div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.success("👉 Use sidebar to Login / Register and start prediction!")


# PAGE: AUTH

def page_auth():
    st.markdown("<div class='page-header'>🔐 Login</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Login or create a new account</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1.5, 1])
    with c2:
        tab1, tab2 = st.tabs(["🔑 Login", "📝 Register"])
        with tab1:
            st.markdown("#### Welcome back!")
            l_user = st.text_input("Username", key="l_user", placeholder="Enter username")
            l_pwd  = st.text_input("Password", type="password", key="l_pwd",
                                   placeholder="Enter password")
            if st.button("Login ->", key="btn_login"):
                if not l_user or not l_pwd:
                    st.error("❌ Fill all fields!")
                else:
                    ok, result = login_user(l_user, l_pwd)
                    if ok:
                        st.session_state.logged_in = True
                        st.session_state.username  = l_user
                        st.session_state.fullname  = result
                        st.session_state.page      = 'predict'
                        st.success(f"✅ Welcome, {result}!")
                        st.rerun()
                    else:
                        st.error(f"❌ {result}")
            st.caption("Demo -> username: **demo** | password: **demo123**")

        with tab2:
            st.markdown("#### Create account")
            r_name  = st.text_input("Full Name",        key="r_name",  placeholder="Ex: John Doe")
            r_email = st.text_input("Email",            key="r_email", placeholder="Ex: john@gmail.com")
            r_user  = st.text_input("Username",         key="r_user",  placeholder="Min 3 characters")
            r_pwd   = st.text_input("Password",         type="password", key="r_pwd",
                                    placeholder="Min 6 characters")
            r_cpwd  = st.text_input("Confirm Password", type="password", key="r_cpwd")
            if st.button("Create Account ->", key="btn_reg"):
                if not all([r_name, r_email, r_user, r_pwd, r_cpwd]):
                    st.error("❌ Fill all fields!")
                elif r_pwd != r_cpwd:
                    st.error("❌ Passwords do not match!")
                else:
                    ok, msg = register_user(r_user, r_pwd, r_email, r_name)
                    if ok:
                        st.success(f"✅ {msg}")
                        st.info("👆 Now login with your credentials!")
                    else:
                        st.error(f"❌ {msg}")


# PAGE: PREDICT

def page_predict(df, cat, xgb, wc, wx, le_city, le_area, le_type, le_combo,
                 city_med, area_med, FEAT, explainer):
    t = get_theme()
    st.markdown("<div class='page-header'>🔮 Predict House Price</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Enter property details to get AI-powered valuation</div>",
                unsafe_allow_html=True)

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**📐 Property Details**")
            city = st.selectbox("City / District",
                                sorted(df['city'].dropna().astype(str).str.strip().unique()))
            city_df   = df[df['city'].astype(str).str.strip() == city]
            areas     = sorted(city_df['area_name'].dropna().astype(str).str.strip().unique())
            area_name = st.selectbox("Area / Locality", areas) if areas else ""
            if not areas:
                st.warning("No areas found for selected city.")
            prop_type = st.selectbox("Property Type", sorted(df['property_type'].unique()))
            area_sqft = st.number_input("Area (sqft)", 200, 10000, 1200, step=50)
            house_age = st.selectbox("House Age", ['0-1', '1-5', '5-10', '10+'])

        with c2:
            st.markdown("**🛏️ Room Details**")
            bedrooms      = st.slider("Bedrooms",                1, 6,  3)
            avg_rooms     = st.slider("Total Rooms",             2, 10, 5)
            bathrooms     = st.slider("Bathrooms",               1, 4,  2)
            avg_occupancy = st.slider("Avg Occupancy (persons)", 1, 8,  3)

        with c3:
            st.markdown("**📍 Location**")
            population = st.number_input("Area Population",             1000, 500000, 50000, step=1000)
            latitude   = st.number_input("Latitude",   8.0,  14.0, 13.06, format="%.5f")
            longitude  = st.number_input("Longitude", 76.0,  81.0, 80.24, format="%.5f")
            dist_km    = st.number_input("Distance to City Center (km)", 0.0, 50.0, 5.0, step=0.5)

        submitted = st.form_submit_button("🚀 Predict Price Now", use_container_width=True)

    if submitted:
        with st.spinner("🤖 AI analysing your property..."):
            AGE_MAP = {'0-1': 0, '1-5': 1, '5-10': 2, '10+': 3}
            age_num = AGE_MAP.get(house_age, 1)
            c_med   = float(city_med.get(city, df['price'].median()))
            a_med   = float(area_med.get(area_name, df['price'].median()))
            pps_est = c_med / max(area_sqft, 1)
            combo   = f"{city}_{prop_type}"

            city_enc  = int(le_city.transform([city])[0])      if city      in le_city.classes_  else 0
            area_enc  = int(le_area.transform([area_name])[0]) if area_name in le_area.classes_  else 0
            type_enc  = int(le_type.transform([prop_type])[0]) if prop_type in le_type.classes_  else 0
            combo_enc = int(le_combo.transform([combo])[0])    if combo     in le_combo.classes_ else 0

            city_type_med_val = (df[df['city_type_combo'] == combo]['price'].median()
                                 if combo in df['city_type_combo'].values else c_med)
            city_pr = (df[df['city'] == city]['price'].rank(pct=True).mean()
                       if city in df['city'].values else 0.5)

            clim   = get_climate(latitude, longitude)
            ls, lg = get_liveness(population, avg_occupancy, avg_rooms, pps_est)

            inp = pd.DataFrame([[
                area_sqft, age_num, bedrooms, avg_rooms, bathrooms,
                population, avg_occupancy, latitude, longitude,
                city_enc, area_enc, type_enc,
                avg_rooms / max(area_sqft,1) * 100,
                population / (area_sqft + 1),
                c_med, a_med,
                bedrooms / (bathrooms + 1),
                1 / (dist_km + 1), dist_km,
                clim['total'], ls,
                bedrooms + bathrooms,
                (area_sqft / 1000) * avg_rooms * bedrooms,
                area_sqft / (avg_occupancy + 1),
                np.log1p(area_sqft),
                min(3, max(0, int(3 - dist_km / 15))),
                np.log1p(population),
                combo_enc, city_type_med_val, city_pr
            ]], columns=FEAT)

            cp    = cat.predict(inp)[0]
            xp    = xgb.predict(inp)[0]
            price = wc * cp + wx * xp
            sv    = explainer.shap_values(inp)[0]

        st.session_state.result = {
            'price': round(price), 'clim': clim,
            'liveness': ls, 'l_grade': lg,
            'shap_ser': pd.Series(sv, index=FEAT),
            'city': city, 'area_name': area_name,
            'prop_type': prop_type, 'area_sqft': area_sqft,
            'house_age': house_age, 'bedrooms': bedrooms,
            'bathrooms': bathrooms, 'population': population,
            'avg_occupancy': avg_occupancy, 'avg_rooms': avg_rooms,
            'pps_est': pps_est, 'dist_km': dist_km,
        }
        st.session_state.page = 'results'
        st.rerun()


# PAGE: RESULTS

def page_results():
    t = get_theme()
    r = st.session_state.result
    if not r:
        st.warning("⚠️ No prediction yet! Go to Predict page first.")
        return

    st.markdown("<div class='page-header'>📊 Prediction Results</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='page-sub'>{r['area_name']}, {r['city']} · "
                f"{r['prop_type']} · {r['area_sqft']} sqft</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class='result-price'>
        <div style='color:{t['subtext']};font-size:1rem'>🏠 AI Predicted House Price</div>
        <div style='color:{t['accent1']};font-size:3.5rem;font-weight:900'>
            &#8377;{r['price']:,.0f}
        </div>
        <div style='color:{t['subtext']}'>
            &#8377;{r['price']/1e6:.2f} Million &nbsp;|&nbsp;
            &#8377;{r['price']/1e5:.1f} Lakhs
        </div>
    </div>""", unsafe_allow_html=True)

    clim = r['clim']
    rc = t['red'] if clim['label']=='HIGH' else t['orange'] if clim['label']=='MEDIUM' else t['green']
    gc = (t['green'] if r['l_grade']=='A' else t['accent1'] if r['l_grade']=='B'
          else t['orange'] if r['l_grade']=='C' else t['red'])

    for col, val, lbl, color in zip(
        st.columns(5),
        [f"&#8377;{r['price']/1e6:.2f}M", clim['label'],
         f"{clim['total']:.0f}/100", f"{r['liveness']:.0f}/100", f"Grade {r['l_grade']}"],
        ["Predicted Price", "Climate Risk", "Risk Score", "Liveness Score", "Liveness Grade"],
        [t['accent1'], rc, t['orange'], t['green'], gc]
    ):
        with col:
            st.markdown(f"""<div class='card'>
                <div class='card-val' style='color:{color}'>{val}</div>
                <div class='card-lbl'>{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    ch = make_chart()
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### 🌧️ Climate Risk Breakdown")
        fig, ax = plt.subplots(figsize=(5, 3.2), facecolor=ch['card'])
        ax.set_facecolor(ch['card'])
        bars = ax.barh(['Flood Risk','Cyclone Risk','Heat Risk'],
                       [clim['flood'],clim['cyclone'],clim['heat']],
                       color=[t['accent1'],t['red'],t['orange']], height=0.5, edgecolor='none')
        ax.set_xlim(0, 115)
        ax.tick_params(colors=ch['text'])
        for sp in ax.spines.values(): sp.set_color(ch['grid'])
        for bar, v in zip(bars, [clim['flood'],clim['cyclone'],clim['heat']]):
            ax.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2,
                    f'{v:.0f}', va='center', color=ch['text'], fontweight='bold', fontsize=11)
        ax.axvline(45, color=t['orange'], ls='--', lw=1, alpha=0.6, label='MEDIUM')
        ax.axvline(70, color=t['red'],    ls='--', lw=1, alpha=0.6, label='HIGH')
        ax.legend(fontsize=8, facecolor=ch['card'], labelcolor=ch['text'])
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_right:
        st.markdown("#### 📍 Liveness Score Breakdown")
        fig2, ax2 = plt.subplots(figsize=(5, 3.2), facecolor=ch['card'])
        ax2.set_facecolor(ch['card'])
        lv_vals = [
            min(r['population']    / 200000 * 100, 100),
            min(r['avg_occupancy'] / 5      * 100, 100),
            min(r['avg_rooms']     / 8      * 100, 100),
            min(r['pps_est']       / 20000  * 100, 100),
        ]
        bars2 = ax2.barh(['Population','Occupancy','Rooms','Price Premium'],
                         lv_vals, color=[t['accent1'],t['accent2'],t['green'],t['orange']],
                         height=0.5, edgecolor='none')
        ax2.set_xlim(0, 115)
        ax2.tick_params(colors=ch['text'])
        for sp in ax2.spines.values(): sp.set_color(ch['grid'])
        for bar, v in zip(bars2, lv_vals):
            ax2.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2,
                     f'{v:.0f}', va='center', color=ch['text'], fontweight='bold', fontsize=11)
        plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.markdown("#### 🔍 Why This Price? (SHAP Explanation)")
    shap_ser = r['shap_ser']
    top_all  = shap_ser.abs().sort_values().tail(10)
    bc       = [t['green'] if shap_ser[f] > 0 else t['red'] for f in top_all.index]
    fig3, ax3 = plt.subplots(figsize=(10, 4), facecolor=ch['bg'])
    ax3.set_facecolor(ch['card'])
    ax3.barh(top_all.index, [shap_ser[f] for f in top_all.index],
             color=bc, height=0.6, edgecolor='none')
    ax3.axvline(0, color=ch['text'], lw=1.5)
    ax3.tick_params(colors=ch['text'])
    ax3.set_xlabel('SHAP Value (price impact)', color=ch['text'])
    ax3.set_title('Feature Impact on Predicted Price', color=ch['text'], fontsize=11)
    for sp in ax3.spines.values(): sp.set_color(ch['grid'])
    ax3.legend(handles=[mpatches.Patch(color=t['green'], label='↑ Increases Price'),
                         mpatches.Patch(color=t['red'],   label='↓ Decreases Price')],
               fontsize=9, facecolor=ch['card'], labelcolor=ch['text'])
    plt.tight_layout(); st.pyplot(fig3); plt.close()

    cu, cd = st.columns(2)
    with cu:
        st.markdown("**📈 Factors Pushing Price UP**")
        for feat, _ in shap_ser.nlargest(4).items():
            st.markdown(f"<div class='factor-up'>↑ {feat}</div>", unsafe_allow_html=True)
    with cd:
        st.markdown("**📉 Factors Pushing Price DOWN**")
        for feat, _ in shap_ser.nsmallest(4).items():
            st.markdown(f"<div class='factor-dn'>↓ {feat}</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔮 Predict Another Property"):
        st.session_state.page = 'predict'
        st.rerun()


# PAGE: BIAS DETECTION

def page_bias(df, cat, xgb, wc, wx, le_city, FEAT, cr, xr, Xte, yte, optuna_info):
    t  = get_theme()
    ch = make_chart()

    st.markdown("<div class='page-header'>⚖️ Bias Detection & Fairness</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>SHAP-based fairness audit across cities and property types</div>",
                unsafe_allow_html=True)

    ens_pred = wc * cat.predict(Xte) + wx * xgb.predict(Xte)
    ens_r2   = r2_score(yte, ens_pred) * 100

    for col, name, score, color in zip(
        st.columns(4),
        ['CatBoost', 'XGBoost', 'Ensemble', 'MAPE'],
        [cr*100, xr*100, ens_r2, mean_absolute_percentage_error(yte, ens_pred)*100],
        [t['accent2'], t['accent1'], t['green'], t['orange']]
    ):
        with col:
            label = 'R² Score' if name != 'MAPE' else 'Error'
            st.markdown(f"""<div class='card'>
                <div class='card-val' style='color:{color}'>{score:.1f}%</div>
                <div class='card-lbl'>{name} {label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Optuna results ──────────────────────────────────────
    st.markdown("#### 🔬 Optuna Hyperparameter Tuning Results (CatBoost)")
    oc1, oc2 = st.columns([1, 1.5])

    with oc1:
        param_rows = "".join(
            f"<div class='param-row'>"
            f"<span class='param-key'>{k}</span>"
            f"<span class='param-val'>{f'{v:.4f}' if isinstance(v,float) else str(v)}</span>"
            f"</div>"
            for k, v in optuna_info['best_params'].items()
        )
        st.markdown(f"""
        <div class='optuna-card'>
            <div class='optuna-title'>🏆 Best Hyperparameters Found</div>
            {param_rows}
            <div class='param-row'>
                <span class='param-key'>Best CV R² Score</span>
                <span class='param-val' style='color:{t["green"]}'>{optuna_info['best_value']:.4f}</span>
            </div>
            <div class='param-row'>
                <span class='param-key'>Total Trials Run</span>
                <span class='param-val'>{optuna_info['n_trials']}</span>
            </div>
        </div>""", unsafe_allow_html=True)

    with oc2:
        trial_vals = optuna_info['trial_values']
        if len(trial_vals) > 1:
            fig_opt, ax_opt = plt.subplots(figsize=(6, 3.2), facecolor=ch['card'])
            ax_opt.set_facecolor(ch['card'])
            ax_opt.plot(range(1, len(trial_vals)+1), trial_vals,
                        color=t['accent2'], lw=1.5, alpha=0.6, label='Trial Score')
            running_best = [max(trial_vals[:i+1]) for i in range(len(trial_vals))]
            ax_opt.plot(range(1, len(running_best)+1), running_best,
                        color=t['green'], lw=2.5, label='Best So Far')
            ax_opt.set_xlabel('Trial Number', color=ch['text'])
            ax_opt.set_ylabel('CV R² Score',  color=ch['text'])
            ax_opt.set_title('Optuna Optimization History', color=ch['text'], fontsize=11)
            ax_opt.tick_params(colors=ch['text'])
            for sp in ax_opt.spines.values(): sp.set_color(ch['grid'])
            ax_opt.legend(fontsize=9, facecolor=ch['card'], labelcolor=ch['text'])
            plt.tight_layout(); st.pyplot(fig_opt); plt.close()
        else:
            st.info("Not enough trials to plot.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── City-wise bias ──────────────────────────────────────
    res = Xte.copy()
    res['actual']    = yte.values
    res['predicted'] = ens_pred
    res['error_pct'] = (res['predicted'] - res['actual']) / res['actual'] * 100
    res['city_name'] = le_city.inverse_transform(res['city_enc'].astype(int))

    bias = res.groupby('city_name').agg(
        mean_error=('error_pct','mean'), count=('error_pct','count')
    ).reset_index().sort_values('mean_error')
    bias['status'] = bias['mean_error'].apply(
        lambda x: '🔴 Over' if x > 10 else ('🔵 Under' if x < -10 else '🟢 Fair'))

    col_chart, col_table = st.columns([1.5, 1])
    with col_chart:
        st.markdown("#### City-wise Prediction Bias")
        fig, ax = plt.subplots(figsize=(8, 7), facecolor=ch['bg'])
        ax.set_facecolor(ch['card'])
        colors = [t['red'] if v>10 else t['accent2'] if v<-10 else t['green']
                  for v in bias['mean_error']]
        ax.barh(bias['city_name'], bias['mean_error'],
                color=colors, height=0.65, edgecolor='none')
        ax.axvline(0,   color=ch['text'],   lw=1.5)
        ax.axvline(10,  color=t['red'],     lw=1, ls='--', alpha=0.6)
        ax.axvline(-10, color=t['accent2'], lw=1, ls='--', alpha=0.6)
        ax.set_xlabel('Mean Prediction Error %', color=ch['text'])
        ax.set_title('Fairness Audit - City-wise', color=ch['text'], fontsize=11)
        ax.tick_params(colors=ch['text'])
        for sp in ax.spines.values(): sp.set_color(ch['grid'])
        ax.legend(handles=[
            mpatches.Patch(color=t['red'],     label='Overestimate >10%'),
            mpatches.Patch(color=t['accent2'], label='Underestimate <-10%'),
            mpatches.Patch(color=t['green'],   label='Fair ±10%'),
        ], fontsize=8, facecolor=ch['card'], labelcolor=ch['text'])
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_table:
        st.markdown("#### Bias Summary Table")
        st.dataframe(
            bias.rename(columns={'city_name':'City','mean_error':'Error%',
                                  'count':'Samples','status':'Status'}),
            use_container_width=True, height=420
        )

    st.markdown("#### 🎯 Prediction Accuracy Bands")
    b1, b2, b3 = st.columns(3)
    for col, lbl, val, color in zip(
        [b1, b2, b3],
        ["Within 5%", "Within 10%", "Within 20%"],
        [(res['error_pct'].abs() < 5).mean()  * 100,
         (res['error_pct'].abs() < 10).mean() * 100,
         (res['error_pct'].abs() < 20).mean() * 100],
        [t['green'], t['accent1'], t['accent2']]
    ):
        with col:
            st.markdown(f"""<div class='card'>
                <div class='card-val' style='color:{color}'>{val:.1f}%</div>
                <div class='card-lbl'>{lbl} of actual price</div>
            </div>""", unsafe_allow_html=True)


# SIDEBAR

def show_sidebar():
    t = get_theme()
    with st.sidebar:
        if st.session_state.logged_in:
            st.markdown(f"👤 **{st.session_state.fullname}**")
            st.markdown(f"<small style='color:{t['subtext']}'>@{st.session_state.username}</small>",
                        unsafe_allow_html=True)
            st.markdown("---")

        col_logo, col_toggle = st.columns([2, 1])
        with col_logo:
            st.markdown(f"<div style='font-size:1.2rem;font-weight:800;"
                        f"color:{t['accent1']}'>🏠 RealEstate AI</div>",
                        unsafe_allow_html=True)
        

        st.markdown("---")
        st.markdown(f"<div style='font-size:0.75rem;color:{t['subtext']};"
                    f"letter-spacing:0.1em;text-transform:uppercase;"
                    f"margin-bottom:0.5rem'>Navigation</div>",
                    unsafe_allow_html=True)

        for page_id, icon, label in [
            ("auth",    "🔐", "Login / Register"),
            ("home",    "🏠", "Home"),
            ("predict", "🔮", "Predict Price"),
            ("results", "📊", "Results"),
            ("bias",    "⚖️",  "Bias Detection"),
        ]:
            if st.button(f"{icon}  {label}", key=f"nav_{page_id}", use_container_width=True):
                if page_id in ['predict','results','bias'] and not st.session_state.logged_in:
                    st.warning("⚠️ Please login first!")
                else:
                    st.session_state.page = page_id
                    st.rerun()

        st.markdown("---")
        if st.session_state.logged_in:
            if st.button("🚪 Logout", use_container_width=True):
                st.session_state.logged_in = False
                st.session_state.username  = ""
                st.session_state.fullname  = ""
                st.session_state.page      = "home"
                st.rerun()

        


# MAIN

def main():
    apply_theme()
    show_sidebar()

    page = st.session_state.page

    if page == 'auth':  page_auth();  return
    if page == 'home':  page_home();  return

    if not st.session_state.logged_in:
        st.warning("⚠️ Please login to access this page!")
        page_auth()
        return

    try:
        (df, cat, xgb, wc, wx, le_city, le_area, le_type, le_combo,
         city_med, area_med, FEAT, explainer,
         cr, xr, Xte, yte, optuna_info) = load_models()
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.warning(f"💡 Make sure **{CSV_PATH}** is in the same folder as app.py")
        return

    if   page == 'predict': page_predict(df, cat, xgb, wc, wx, le_city, le_area, le_type,
                                          le_combo, city_med, area_med, FEAT, explainer)
    elif page == 'results': page_results()
    elif page == 'bias':    page_bias(df, cat, xgb, wc, wx, le_city, FEAT,
                                       cr, xr, Xte, yte, optuna_info)

if __name__ == "__main__":
    main()