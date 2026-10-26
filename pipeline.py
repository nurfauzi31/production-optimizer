"""Rucika ML Production Optimizer: data prep -> train -> optimizer -> dashboard.html
Pakai: python pipeline.py <folder_data> (berisi xlsx, 3 CSV Insight, logo png)"""
import itertools, pandas as pd, numpy as np, json, base64, io, re, sys, glob, warnings, datetime as dt
from pathlib import Path
from PIL import Image
from sklearn.ensemble import HistGradientBoostingRegressor as HGB, RandomForestRegressor as RF
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error as mae, r2_score
warnings.filterwarnings("ignore")
D = Path(sys.argv[1] if len(sys.argv) > 1 else "."); OUT = Path(__file__).parent
SHIFT_START = {1: 7, 2: 15, 3: 23}          # ASUMSI jam mulai shift (ubah jika beda)
RANGES = {"melt": (120, 280), "load": (0, 110), "dos": (0.5, 400)}
HOLD_H, SPLIT = 48, "2026-08-01"

# ---------- 1. Excel (produksi) ----------
def oee(v):
    if isinstance(v, str): v = v.strip(); return float(v[:-1]) / 100 if v.endswith("%") else pd.to_numeric(v, errors="coerce")
    return v
def out_fix(v):                               # Excel salah baca "2229.5" jadi tanggal 2229-05-01
    if isinstance(v, dt.datetime): return v.year + 0.5 if v.month == 5 else np.nan
    return pd.to_numeric(v, errors="coerce")
xl = pd.ExcelFile(next(D.glob("Tabel_Data_Proses*.xlsx"))); fr = []; q = {"output_rusak": 0, "output_pulih": 0}
for sh in [s for s in xl.sheet_names if s.startswith("Line")]:
    d = xl.parse(sh, header=2); d.columns = [str(c).strip() for c in d.columns]
    raw = d["Production Output (kg))"]; bad = raw.map(lambda v: isinstance(v, dt.datetime))
    d["out"] = raw.map(out_fix); q["output_rusak"] += int(bad.sum()); q["output_pulih"] += int((bad & d["out"].notna()).sum())
    d["oee"] = d["OEE"].map(oee).astype(float); d["rej"] = pd.to_numeric(d["Reject Rate"], errors="coerce")
    d["line"] = int(re.findall(r"\d+", sh)[0]); d["product"] = d["Product (OEE)"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    d["recipe"] = d["Recipe (MES)"].astype(str).str.strip(); d["silo"] = d["Silo Dryblend (MES)"].astype(str).str.strip(); d["shift"] = pd.to_numeric(d["Shift"], errors="coerce")
    fr.append(d[["Tanggal", "shift", "line", "product", "recipe", "silo", "oee", "out", "rej"]])
E = pd.concat(fr, ignore_index=True); E["Tanggal"] = pd.to_datetime(E["Tanggal"], errors="coerce")
E["rej_pct"] = E.rej / (E.out + E.rej) * 100                                  # ASUMSI: reject dalam kg
E = E[(E.out > 0) & (E.oee > 0) & E.Tanggal.notna()].copy()

# ---------- 2. CSV sensor -> shift ----------
files = {"melt": "Melt_Temp", "load": "Extruder_Load", "dos": "Speed_Actual_Dosing"}; series = {}; q["sensor"] = {}
for f in glob.glob(str(D / "Insight_Data_*.csv")):
    c = pd.read_csv(f); var = next((k for k, v in files.items() if v in c.TagName.iloc[0]), None)
    if var is None: continue
    c["t"] = pd.to_datetime(c.DateTime, format="%m/%d/%Y %I:%M:%S.%f %p"); c["v"] = pd.to_numeric(c.Value, errors="coerce")
    c["line"] = c.TagName.str.extract(r"EXT_(\d+)")[0].astype(int); good = c[(c.Quality == 192) & c.v.notna()]
    ok = good[(good.v >= RANGES[var][0]) & (good.v <= RANGES[var][1])].sort_values("t")
    q["sensor"][var] = {"total": len(c), "good": len(good), "valid": len(ok)}
    series[var] = {l: (g.t.values.astype("datetime64[ns]").astype("int64"), g.v.values) for l, g in ok.groupby("line")}
def agg(var, line, st, en):
    t, v = series[var].get(line, (np.array([], dtype="int64"), np.array([])))
    a, b = np.searchsorted(t, st.value), np.searchsorted(t, en.value)
    if b > a: return v[a:b].mean()
    if a > 0 and (st.value - t[a - 1]) <= HOLD_H * 3.6e12: return v[a - 1]
    return np.nan
for var in files:
    E[var] = [agg(var, int(r.line), r.Tanggal + pd.Timedelta(hours=SHIFT_START[int(r.shift)]), r.Tanggal + pd.Timedelta(hours=SHIFT_START[int(r.shift)] + 8)) if r.shift in SHIFT_START else np.nan for r in E.itertuples()]
E["month"] = E.Tanggal.dt.month; ALL = E.copy()
E = E.dropna(subset=["melt", "load", "dos", "rej_pct"]); q["baris_latih"] = len(E)
cnt = E.groupby("line").size(); E = E[E.line.isin(cnt[cnt >= 100].index)]; q["line_dipakai"] = sorted(map(int, E.line.unique()))
E.to_csv(OUT / "dataset_bersih.csv", index=False)

# ---------- 3. Training ----------
NUM, CAT, TG = ["melt", "load", "dos", "shift"], ["line", "product", "recipe", "silo"], ["oee", "out", "rej_pct"]
X = pd.get_dummies(E[NUM + CAT].astype({c: str for c in CAT}), columns=CAT).astype(float); cols = list(X.columns)
tr, te = (E.Tanggal < SPLIT).values, (E.Tanggal >= SPLIT).values
mk = {"Ridge": lambda: Ridge(alpha=1.0), "RandomForest": lambda: RF(300, min_samples_leaf=3, n_jobs=-1, random_state=0),
      "GradientBoosting": lambda: HGB(max_iter=400, learning_rate=0.05, max_leaf_nodes=24, l2_regularization=1.0, random_state=0)}
metrics, best, final, pred_te = [], {}, {}, {}
for tg in TG:
    y = E[tg].values; base = E[tr].groupby(["line", "product"])[tg].median(); bp = E[te].set_index(["line", "product"]).index.map(base).values
    bp = np.where(pd.isna(bp), y[tr].mean(), bp); metrics.append(dict(target=tg, model="Baseline", mae=mae(y[te], bp), r2=r2_score(y[te], bp)))
    res = {}
    for n, f in mk.items():
        m = f().fit(X[tr], y[tr]); p = m.predict(X[te]); res[n] = (mae(y[te], p), p); metrics.append(dict(target=tg, model=n, mae=mae(y[te], p), r2=r2_score(y[te], p)))
    best[tg] = min(res, key=lambda k: res[k][0]); pred_te[tg] = res[best[tg]][1]; final[tg] = mk[best[tg]]().fit(X, y)
# importance (permutation per kelompok fitur, uji pada data test)
imp = {}; rng = np.random.default_rng(0); groups = {"Melt Temp": ["melt"], "Motor Load": ["load"], "Speed Dosing": ["dos"], "Shift": ["shift"],
      "Line": [c for c in cols if c.startswith("line_")], "Produk": [c for c in cols if c.startswith("product_")], "Recipe": [c for c in cols if c.startswith("recipe_")]}
for tg in ["oee", "out"]:
    m, Xt, base_m = final[tg], X[te], mae(E[tg].values[te], final[tg].predict(X[te])); imp[tg] = {}
    for g, cs in groups.items():
        Xp = Xt.copy(); Xp[cs] = Xp[cs].values[rng.permutation(len(Xp))]; imp[tg][g] = round(max(mae(E[tg].values[te], m.predict(Xp)) - base_m, 0), 4)

# ---------- 4. Rekomendasi set card (recipe + silo + melt + load + dosing) vs setting sebelumnya ----------
def setpred(df):
    Z = pd.get_dummies(df[NUM + CAT].astype({c: str for c in CAT}), columns=CAT).astype(float).reindex(columns=cols, fill_value=0)
    return np.column_stack([final[t].predict(Z) for t in TG])
combos = []
for (l, p), g in E.groupby(["line", "product"]):
    if len(g) < 25: continue
    g = g.sort_values("Tanggal"); pv = g.tail(10)                     # "setting sebelumnya" = 10 shift produktif terakhir
    ps = dict(recipe=pv.recipe.mode()[0], silo=pv.silo.mode()[0], melt=pv["melt"].median(), load=pv["load"].median(), dos=pv["dos"].median())
    rows = []
    for (r, si), h in g.groupby(["recipe", "silo"]):
        if len(h) < 10: continue
        ax = [np.quantile(h[v], [.1, .37, .63, .9]) for v in ["melt", "load", "dos"]]
        rows += [(a, b_, c_, r, si) for a, b_, c_ in itertools.product(*ax)]
    if not rows: continue
    C = pd.DataFrame(rows, columns=["melt", "load", "dos", "recipe", "silo"]); C["shift"] = 2; C["line"] = l; C["product"] = p
    Pc = setpred(C); Pp = setpred(pd.DataFrame([{**ps, "shift": 2, "line": l, "product": p}]))[0]
    rs = sorted({f"{r}|{si}" for r, si in zip(C.recipe, C.silo)}); ix = (C.recipe + "|" + C.silo).map({k: i for i, k in enumerate(rs)}).values
    combos.append(dict(line=int(l), product=p, n=len(g), rs=rs, prev=dict(recipe=ps["recipe"], silo=ps["silo"], s=[round(float(ps[k]), 1) for k in ["melt", "load", "dos"]],
        act=[round(float(pv[t].mean()), 3) for t in TG], pred=[round(float(x), 3) for x in Pp]),
        cand=np.round(np.column_stack([C[["melt", "load", "dos"]].values, ix, Pc]), 2).tolist()))
# pemilihan line per produk (historical performance) + insight parameter + pola energi
A = ALL.copy(); lp = [[p, int(l), len(g), round(g.oee.mean(), 3), round(g.out.mean(), 0), round(g.rej_pct.mean(), 2)] for (p, l), g in A.groupby(["product", "line"]) if len(g) >= 15]
bins = {}
for l, g in E.groupby("line"):
    bins[int(l)] = {}
    for v in ["melt", "load", "dos"]:
        b_ = pd.qcut(g[v], 5, duplicates="drop"); t = g.groupby(b_).agg(x=(v, "mean"), oee=("oee", "mean"), rej=("rej_pct", "mean"), out=("out", "mean"), n=(v, "size"))
        bins[int(l)][v] = t.round(3).values.tolist()
en, prod, ln = {}, None, None
for _, r in xl.parse("JAN-26", header=None).iloc[3:].iterrows():
    if isinstance(r[2], str): prod, ln = r[2].strip(), r[3]
    if r[4] == "Energy Consumption": en.setdefault(prod, []).append(dict(line=str(ln), v=[None if pd.isna(x) else round(float(x), 4) for x in r[5:14]]))

# ---------- 5. Dashboard ----------
lg = Image.open(next(D.glob("*Logo_Rucika*.png"))).convert("RGBA"); a = np.array(lg); m = (a[..., 3] > 0) & (a[..., :3].min(-1) < 245)
ys, xs = np.where(m); lg = lg.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)); lg.thumbnail((420, 160)); b = io.BytesIO(); lg.save(b, "PNG")
mth = ALL.groupby(["line", "month"]).oee.mean().unstack().round(3); tr_ = ALL.groupby("month").agg(oee=("oee", "mean"), out=("out", "sum"), rej=("rej_pct", "mean")).round(3)
data = dict(q=q, split=SPLIT, best=best, metrics=[{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in metrics],
            kpi=[dict(line=int(l), n=len(g), oee=round(g.oee.mean(), 3), out=round(g.out.mean(), 0), rej=round(g.rej_pct.mean(), 2)) for l, g in ALL.groupby("line")],
            heat={int(l): [None if pd.isna(v) else v for v in r.reindex(range(1, 10)).values] for l, r in mth.iterrows()},
            trend=dict(m=[int(i) for i in tr_.index], oee=tr_.oee.tolist(), out=(tr_.out / 1000).round(1).tolist(), rej=tr_.rej.tolist()), lp=lp, combos=combos, bins=bins, energy=en)
html = (OUT / "dashboard_template.html").read_text(encoding="utf-8").replace("__LOGO__", "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()).replace("__DATA__", json.dumps(data, separators=(",", ":"), default=float))
(OUT / "dashboard.html").write_text(html, encoding="utf-8")
print(pd.DataFrame(metrics).round(3).to_string(index=False)); print("best:", best, "| baris latih:", q["baris_latih"], "| combos:", len(combos), "| dashboard.html OK")
