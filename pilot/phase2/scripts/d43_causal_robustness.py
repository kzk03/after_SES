"""D43: RQ1(AI レビュー導入で人間のレビューは減るか)の因果推定の頑健性パッケージ。

査読会議(2026-09-27)で次の指摘を受けた。
  M1: 「人間のレビューが有意に増える(k=+3)」は頑健でない
      (長いリードで事前トレンドあり / wild cluster bootstrap で有意でなくなる / リポジトリ別トレンドで消える)
  M2: 導入の定義(AI レビュー比率の段差)が 1 - H_E そのもので、H_E の結果が循環している。
      SigNoz・vault・flipperzero は一時的な揺らぎで「導入」と判定されている
  M3: 実効的な処置リポジトリが少ない(home-assistant・grafana・sentry に処置行が集中)
  M7: 処置が AI 作成 PR の増加と束になっている

このスクリプトは d4 の主推定を再現したうえで、以下をまとめて出す。
  1. d4 の再現(k=-3..+3、端は束ねる、module + month 固定効果、リポジトリクラスタ)
  2. wild cluster restricted bootstrap(Webb 重み、帰無仮説を課す)の p 値と t(G-1) の臨界値
  3. 長いリード(k=-12..+6)と事前トレンドの結合検定
  4. リポジトリ別の線形トレンド
  5. 導入の再定義: 比率ではなく「AI ボットのレビューが月 N 件以上出て、その後 6ヶ月中 5ヶ月以上
     N 件以上続いた最初の月」。人間のレビュー量を一切使わないので、H_E と循環しない
  6. 処置リポジトリを1つずつ抜いた推定(leave-one-treated-out)
  7. リポジトリ等重みの推定(repo x month に集約した TWFE)と、never-treated を対照にした
     Callaway–Sant'Anna 風の 2x2 比較の集計(パッケージが無いので手で実装)
  8. AI が関与していない PR(人間が作成し AI trailer も無い PR)へのレビューだけに限定したアウトカム
  9. 同値性: 「5% / 10% 以上減った」を片側で棄却できるか(最も保守的な仕様で)
 10. HonestDiD の relative magnitudes を手計算した近似の区間

wild bootstrap は各リポジトリ(クラスタ)単位の集計量だけで回せるように式を変形しているので、
9,999 回でも数秒で終わる(クラスタごとの X'u と z'X を前計算し、重みベクトルとの行列積で t* を作る)。

実行: PILOT_RAW=raw_full PILOT_INTERIM=interim_full PHASE2_OUT=outputs_full \
      uv run python pilot/phase2/scripts/d43_causal_robustness.py
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from scipy import stats

from common import INTERIM_DIR, OUT_DIR, build_count_panel, load_repos, repo_dir_name
from d4_repo_event_study import build_balanced
from d4_repo_event_study import detect_adoption_share as detect_adoption  # 旧定義との比較用

SEED = 20260927
B_WCR = 9999          # wild bootstrap の反復回数(クラスタ集計で回すので多くても軽い)
B_CS = 999            # CS 風推定のリポジトリ・ブートストラップ回数
N_BOT = 10            # 新定義: 「AI レビューが出ている」とみなす月あたり件数(主仕様)
N_BOT_SENS = [1, 5, 25]  # 感度分析
N_CTRL = 5            # 対照群は「N_CTRL 件の基準でも定着しなかった」リポジトリに限る
PERSIST_WIN = 6       # 定着の判定窓(後続月数)
PERSIST_NEED = 5      # 窓のうち何ヶ月で N 件以上必要か
DELTAS = [0.05, 0.10]  # 同値性の限界(割合)。log アウトカムでは log(1-δ) を帰無値にする
M_RM = [0.5, 1.0, 2.0]  # HonestDiD relative magnitudes の M

OUTCOMES = ["log_n_E_human_nonauthor", "log_n_E_human_nonauthor_actors",
            "log_n_E_human", "log_n_E_human_actors", "e_human_per_pr", "H_E", "log_n_E_agent"]
OUTCOMES_NONAI = ["log_n_E_human_nonauthor_nonai", "log_n_E_human_nonai", "log_n_E_human_actors_nonai", "H_E_nonai", "log_n_E_agent_nonai"]
HUMAN_LOG = ["log_n_E_human_nonauthor", "log_n_E_human_nonauthor_actors",
             "log_n_E_human", "log_n_E_human_actors"]

WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])
rng = np.random.default_rng(SEED)
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# データ
# ---------------------------------------------------------------------------

def nonai_counts() -> pd.DataFrame:
    """AI が関与していない PR へのレビューだけを module x month で数える。

    「AI が関与していない PR」= 生成工程(G)のイベントが人間で、commit trailer にも
    AI エージェントが残っていない PR。作成者が観測できない PR(収集窓の前に作られた等)は
    どちらとも言えないので除く。AI ボットによるレビュー自体は処置なので、ここでは除外しない。
    """
    frames = []
    for repo in load_repos():
        p = INTERIM_DIR / repo_dir_name(repo) / "events.parquet"
        if not p.exists():
            continue
        ev = pd.read_parquet(p, columns=["stage", "actor_login", "actor_category", "module_id",
                                         "month", "pr_number", "trailer_agent_detected"])
        ev = ev[ev["stage"].isin(["G", "E"]) & (ev["actor_category"] != "legacy_bot")]
        g = ev[ev["stage"] == "G"]
        trailer = g["trailer_agent_detected"].fillna(False).astype(bool)
        human_prs = set(g.loc[(g["actor_category"] == "human") & ~trailer, "pr_number"])
        e = ev[(ev["stage"] == "E") & ev["pr_number"].isin(human_prs)].copy()
        if e.empty:
            continue
        e["is_human"] = e["actor_category"] == "human"
        # 作成者本人を除いた人間の評価(主指標と揃える)
        author_of = (g.sort_values("month").drop_duplicates("pr_number")
                      .set_index("pr_number")["actor_login"].to_dict())
        e["is_human_nonauthor"] = e["is_human"] & (e["actor_login"] != e["pr_number"].map(author_of))
        key = ["module_id", "month"]
        agg = e.groupby(key).agg(n_E_nonai=("is_human", "size"), n_E_human_nonai=("is_human", "sum"),
                                 n_E_human_nonauthor_nonai=("is_human_nonauthor", "sum"))
        act = e[e["is_human"]].groupby(key)["actor_login"].nunique().rename("n_E_human_actors_nonai")
        agg = agg.join(act).reset_index()
        agg["module_id_global"] = repo + "::" + agg["module_id"]
        frames.append(agg.drop(columns="module_id"))
    return pd.concat(frames, ignore_index=True)


def bot_adoption(df: pd.DataFrame, n: int) -> dict[str, str | None]:
    """AI ボットの出現と定着だけで導入月を決める(人間のレビュー量は使わない)。

    月 t に AI レビューが n 件以上あり、続く PERSIST_WIN ヶ月のうち PERSIST_NEED ヶ月以上で
    n 件以上ある最初の t。観測の終わりに近く後続が6ヶ月無い場合は、後続月の 5/6 以上
    (最低2ヶ月の後続が必要)で判定する。
    """
    by = df.groupby(["repo", "month"])["n_E_agent"].sum()
    out: dict[str, str | None] = {}
    for repo, s in by.groupby(level=0):
        s = s.droplevel(0).sort_index()
        # 観測期間内で抜けた月は 0 件として扱う(build_count_panel と同じ考え方)
        full = pd.period_range(s.index.min(), s.index.max(), freq="M").astype(str)
        s = s.reindex(full, fill_value=0.0)
        v = s.to_numpy()
        found = None
        for i in range(len(v)):
            if v[i] < n:
                continue
            f = v[i + 1:i + 1 + PERSIST_WIN]
            if len(f) < 2:
                continue
            req = int(np.ceil(len(f) * PERSIST_NEED / PERSIST_WIN))
            if (f >= n).sum() >= req:
                found = s.index[i]
                break
        out[repo] = found
    return out


def define_groups(df: pd.DataFrame, n: int) -> tuple[dict[str, str], list[str], list[str]]:
    """新定義での処置群・対照群・除外を返す。

    対照群は「N_CTRL 件の基準でも定着が一度も無い」リポジトリに限る(散発的な利用だけ)。
    n 件基準では定着しないが N_CTRL 件基準では定着するリポジトリは、どちらにも入れない。
    観測開始月に既に定着しているリポジトリは段差前が見えないので除外する。
    """
    a = bot_adoption(df, n)
    ctrl_rule = bot_adoption(df, min(n, N_CTRL))
    first = df.groupby("repo")["month"].min()
    treated = {r: m for r, m in a.items() if m is not None and m > first[r]}
    never = sorted(r for r, m in ctrl_rule.items() if m is None)
    excluded = sorted(set(a) - set(treated) - set(never))
    return treated, never, excluded


def add_outcomes(bal: pd.DataFrame) -> pd.DataFrame:
    bal["log_n_E_human"] = np.log1p(bal["n_E_human"])
    bal["log_n_E_agent"] = np.log1p(bal["n_E_agent"])
    bal["log_n_E_human_actors"] = np.log1p(bal["n_E_human_actors"])
    # ★作成者本人を除いた人間の評価(独立な審査)。主指標(2026-09-27 B2)
    bal["log_n_E_human_nonauthor"] = np.log1p(bal["n_E_human_nonauthor"])
    bal["log_n_E_human_nonauthor_actors"] = np.log1p(bal["n_E_human_nonauthor_actors"])
    bal["H_E"] = np.where(bal["n_E"] > 0, bal["n_E_human"] / bal["n_E"], np.nan)
    bal["e_human_per_pr"] = np.where(bal["n_E_prs"] > 0, bal["n_E_human"] / bal["n_E_prs"], np.nan)
    bal["log_n_E_human_nonai"] = np.log1p(bal["n_E_human_nonai"])
    bal["log_n_E_human_nonauthor_nonai"] = np.log1p(bal["n_E_human_nonauthor_nonai"])
    bal["log_n_E_human_actors_nonai"] = np.log1p(bal["n_E_human_actors_nonai"])
    ag = bal["n_E_nonai"] - bal["n_E_human_nonai"]
    bal["log_n_E_agent_nonai"] = np.log1p(ag)
    bal["H_E_nonai"] = np.where(bal["n_E_nonai"] > 0, bal["n_E_human_nonai"] / bal["n_E_nonai"], np.nan)
    return bal


def sample(bal: pd.DataFrame, adoption: dict[str, str], never: list[str], months: list[str]) -> pd.DataFrame:
    d = bal[bal["repo"].isin(set(adoption) | set(never))].copy()
    midx = {m: i for i, m in enumerate(months)}
    d["g"] = d["repo"].map({r: midx[m] for r, m in adoption.items()})
    d["event_time"] = d["month_idx"] - d["g"]
    return d


# ---------------------------------------------------------------------------
# 推定(二元固定効果を交互射影で除去 → OLS → リポジトリクラスタ CR1)
# ---------------------------------------------------------------------------

def es_cols(d: pd.DataFrame, leads: int, lags: int, ref: int = -1) -> list[str]:
    cols = []
    k = d["event_time"]
    for j in range(-leads, lags + 1):
        if j == ref:
            continue
        nm = f"k_m{abs(j)}" if j < 0 else f"k_p{j}"
        if j == -leads:
            v = k <= j
        elif j == lags:
            v = k >= j
        else:
            v = k == j
        d[nm] = v.fillna(False).astype(float)
        cols.append(nm)
    return cols


def add_post(d: pd.DataFrame) -> list[str]:
    d["post"] = ((d["event_time"] >= 0) & d["g"].notna()).astype(float)
    return ["post"]


def add_trends(d: pd.DataFrame) -> list[str]:
    cols = []
    for i, r in enumerate(sorted(d["repo"].unique())[1:]):
        c = f"trend_{i}"
        d[c] = (d["repo"] == r).astype(float) * d["month_idx"].astype(float)
        cols.append(c)
    return cols


def demean(M: np.ndarray, a: np.ndarray, b: np.ndarray, it: int = 500, tol: float = 1e-9) -> np.ndarray:
    M = M.copy()
    na, nb = a.max() + 1, b.max() + 1
    ca = np.maximum(np.bincount(a, minlength=na), 1).astype(float)
    cb = np.maximum(np.bincount(b, minlength=nb), 1).astype(float)
    for _ in range(it):
        delta = 0.0
        for j in range(M.shape[1]):
            ma = (np.bincount(a, M[:, j], na) / ca)[a]
            M[:, j] -= ma
            mb = (np.bincount(b, M[:, j], nb) / cb)[b]
            M[:, j] -= mb
            delta = max(delta, float(np.max(np.abs(mb))))
        if delta < tol:
            break
    return M


class Fit:
    """FE を除去した y, X に対する OLS とリポジトリクラスタ CR1、および高速な WCR。"""

    def __init__(self, d: pd.DataFrame, y: str, cols: list[str], unit: str = "module_id_global"):
        d = d.dropna(subset=[y])
        a = pd.factorize(d[unit])[0]
        b = pd.factorize(d["month_idx"])[0]
        self.cl, self.repos = pd.factorize(d["repo"])
        self.G = len(self.repos)
        M = demean(np.column_stack([d[y].to_numpy(float), d[cols].to_numpy(float)]), a, b)
        self.y, self.X = M[:, 0], M[:, 1:]
        # 固定効果に吸収されて消えた列(例: 対照群が居ない時期のダミー)は落とす
        keep = np.where(np.abs(self.X).max(axis=0) > 1e-10)[0]
        self.cols = [cols[i] for i in keep]
        self.X = self.X[:, keep]
        self.n, self.k = self.X.shape
        self.XtXi = np.linalg.pinv(self.X.T @ self.X)
        self.beta = self.XtXi @ self.X.T @ self.y
        self.e = self.y - self.X @ self.beta
        self.c = self.G / (self.G - 1) * (self.n - 1) / (self.n - self.k)
        S = np.column_stack([np.bincount(self.cl, self.X[:, j] * self.e, self.G) for j in range(self.k)])
        self.V = self.c * self.XtXi @ (S.T @ S) @ self.XtXi
        self.n_obs = int(self.n)

    def idx(self, name: str) -> int:
        return self.cols.index(name)

    def coef(self, name: str) -> dict:
        j = self.idx(name)
        b, se = float(self.beta[j]), float(np.sqrt(self.V[j, j]))
        t = b / se
        return {"coef": b, "se": se, "p_t": float(2 * stats.t.sf(abs(t), self.G - 1)),
                "p_z": float(2 * stats.norm.sf(abs(t)))}

    def wcr(self, name: str, null: float = 0.0, B: int = B_WCR) -> dict:
        """Wild cluster restricted bootstrap(Webb 重み)。

        帰無仮説 β_j = null を課した残差 u を使う。y* = Xγ + u∘w なので
        β*_j - null = a_j'(A'w)、スコアのクラスタ和 = w_g (z_g'u_g) - (Z X)_g XtXi A'w
        (A_g = Σ_{i∈g} X_i u_i, z = X a_j)と書け、観測数によらず G x k の行列演算で済む。
        """
        j = self.idx(name)
        X, y, cl, G = self.X, self.y, self.cl, self.G
        keep = [i for i in range(self.k) if i != j]
        Xr = X[:, keep]
        yn = y - null * X[:, j]
        br = np.linalg.lstsq(Xr, yn, rcond=None)[0]
        u = yn - Xr @ br
        aj = self.XtXi[j]
        z = X @ aj
        A = np.column_stack([np.bincount(cl, X[:, i] * u, G) for i in range(self.k)])  # G x k
        zu = np.bincount(cl, z * u, G)
        ZX = np.column_stack([np.bincount(cl, z * X[:, i], G) for i in range(self.k)])  # G x k
        t_obs = (self.beta[j] - null) / np.sqrt(self.V[j, j])
        ts = []
        for s in range(0, B, 2000):
            W = rng.choice(WEBB, size=(min(2000, B - s), G))
            AW = W @ A                   # B x k   (= (A'w)')
            num = AW @ aj                # β*_j - null
            shift = AW @ self.XtXi       # B x k   (= (XtXi A'w)')
            score = W * zu - shift @ ZX.T  # B x G
            V = self.c * np.sum(score ** 2, axis=1) * 1.0
            # V_jj = c Σ_g (Σ_{i∈g} z_i e*_i)^2(z に a_j を含めてあるので XtXi を挟む必要は無い)
            ts.append(num / np.sqrt(V))
        ts = np.concatenate(ts)
        return {"null": null, "t": float(t_obs), "p_wcr_two": float(np.mean(np.abs(ts) >= abs(t_obs))),
                "p_wcr_upper": float(np.mean(ts >= t_obs)), "B": int(B)}


def summarize(f: Fit, cols: list[str]) -> dict:
    return {c: {k: round(v, 5) for k, v in f.coef(c).items()} for c in cols if c in f.cols}


def joint_F(f: Fit, names: list[str]) -> dict:
    idx = [f.idx(c) for c in names if c in f.cols]
    b = f.beta[idx]
    V = f.V[np.ix_(idx, idx)]
    W = float(b @ np.linalg.pinv(V) @ b)
    q = len(idx)
    # クラスタ数が少なく q が G-1 に近いと CR1 の分散行列が退化しやすいので、Wald の χ² も併記
    return {"q": q, "F": W / q, "p_F": float(stats.f.sf(W / q, q, f.G - 1)), "p_chi2": float(stats.chi2.sf(W, q))}


# ---------------------------------------------------------------------------
# 各分析
# ---------------------------------------------------------------------------

def run_definition(label: str, bal: pd.DataFrame, adoption: dict, never: list, months: list,
                   full: bool = True) -> dict:
    d0 = sample(bal, adoption, never, months)
    res: dict = {"n_treated": len(adoption), "n_never": len(never), "n_modules": int(d0["module_id_global"].nunique())}

    # 処置群の偏り(実効クラスタ数)
    tr = d0[d0["g"].notna()]
    shares = tr.groupby("repo").size() / len(tr)
    res["treated_row_share"] = {r: round(float(v), 4) for r, v in shares.sort_values(ascending=False).items()}
    res["effective_treated_clusters_invHHI"] = round(float(1 / np.sum(shares ** 2)), 2)

    # 1-2. d4 再現 + WCR(k=+3)
    d = d0.copy()
    cols = es_cols(d, 3, 3)
    post_cols = add_post(d)
    trend_cols = add_trends(d) if full else []
    base = {}
    outs = OUTCOMES + (OUTCOMES_NONAI if full else [])
    for y in outs:
        f = Fit(d, y, cols)
        r = {"n_obs": f.n_obs, "G": f.G, "t_crit_975_G-1": float(stats.t.ppf(0.975, f.G - 1)),
             "coefs": summarize(f, cols), "pretrend_k-3_k-2": joint_F(f, ["k_m3", "k_m2"]),
             "k_p3_wcr": f.wcr("k_p3")}
        fp = Fit(d, y, post_cols)
        r["post_did"] = fp.coef("post") | {"wcr": fp.wcr("post")}
        if full:
            ft = Fit(d, y, cols + trend_cols)
            r["trend_coefs"] = summarize(ft, cols)
            r["trend_k_p3_wcr"] = ft.wcr("k_p3")
            ftp = Fit(d, y, post_cols + trend_cols)
            r["trend_post_did"] = ftp.coef("post") | {"wcr": ftp.wcr("post")}
        base[y] = r
        log(f"{label} {y}: k+3={r['coefs']['k_p3']['coef']:+.4f} p_t={r['coefs']['k_p3']['p_t']:.3f} "
            f"wcr={r['k_p3_wcr']['p_wcr_two']:.3f} | post={r['post_did']['coef']:+.4f} "
            f"wcr={r['post_did']['wcr']['p_wcr_two']:.3f}"
            + (f" | trend post={r['trend_post_did']['coef']:+.4f} wcr={r['trend_post_did']['wcr']['p_wcr_two']:.3f}" if full else ""))
    res["baseline"] = base
    if not full:
        return res

    # 3. 長いリード k=-12..+6 と 10. HonestDiD(RM)
    d = d0.copy()
    lcols = es_cols(d, 12, 6)
    lead_names = [c for c in lcols if c.startswith("k_m")]
    longres = {}
    for y in OUTCOMES + ["log_n_E_human_nonai"]:
        f = Fit(d, y, lcols)
        r = {"coefs": summarize(f, lcols),
             "joint_pre_all_leads_-12..-2": joint_F(f, lead_names),
             "joint_pre_unbinned_-11..-2": joint_F(f, [c for c in lead_names if c != "k_m12"])}
        if y in HUMAN_LOG + ["log_n_E_human_nonai"]:
            r["honestdid_rm"] = honest_rm(f)
        longres[y] = r
        log(f"{label} long {y}: joint pre p_F={r['joint_pre_all_leads_-12..-2']['p_F']:.4f}")
    res["long_event_study"] = longres

    # 6. leave-one-treated-out
    for yl in ["log_n_E_human_nonauthor", "log_n_E_human"]:
        loo = {}
        for r_out in sorted(adoption):
            dd = d0[d0["repo"] != r_out].copy()
            c = es_cols(dd, 3, 3)
            pc = add_post(dd)
            f = Fit(dd, yl, c)
            fp = Fit(dd, yl, pc)
            loo[r_out] = {"k_p3": f.coef("k_p3"), "post": fp.coef("post")}
        k3 = [v["k_p3"]["coef"] for v in loo.values()]
        res[f"leave_one_treated_out_{yl}"] = {
            "by_repo": loo,
            "k_p3_range": [min(k3), max(k3)],
            "k_p3_p_t_range": [min(v["k_p3"]["p_t"] for v in loo.values()),
                               max(v["k_p3"]["p_t"] for v in loo.values())],
            "post_range": [min(v["post"]["coef"] for v in loo.values()),
                           max(v["post"]["coef"] for v in loo.values())],
        }
        log(f"{label} LOO {yl} k+3 range {min(k3):+.4f}..{max(k3):+.4f}")

    # 7a. リポジトリ等重み: repo x month 平均の TWFE
    rl = {}
    R = repo_level(d0)
    rcols = es_cols(R, 3, 3)
    rpost = add_post(R)
    rtr = add_trends(R)
    for y in OUTCOMES + ["log_n_E_human_nonai", "log_n_E_human_nonauthor_nonai"]:
        f = Fit(R, y, rcols, unit="repo")
        fp = Fit(R, y, rpost, unit="repo")
        ftp = Fit(R, y, rpost + rtr, unit="repo")
        rl[y] = {"coefs": summarize(f, rcols), "k_p3_wcr": f.wcr("k_p3"),
                 "post_did": fp.coef("post") | {"wcr": fp.wcr("post")},
                 "trend_post_did": ftp.coef("post") | {"wcr": ftp.wcr("post")}}
    res["repo_weighted_twfe"] = rl
    log(f"{label} repo-level post log_h={rl['log_n_E_human']['post_did']['coef']:+.4f} "
        f"wcr={rl['log_n_E_human']['post_did']['wcr']['p_wcr_two']:.3f}")

    # 7b. Callaway–Sant'Anna 風
    res["cs_style"] = {y: cs_att(d0, y) for y in ["log_n_E_human_nonauthor", "log_n_E_human_nonauthor_nonai",
                                                  "log_n_E_human", "log_n_E_human_actors", "log_n_E_agent",
                                                  "log_n_E_human_nonai", "H_E"]}
    log(f"{label} CS done")

    # 9. 同値性(片側): H0 β <= log(1-δ) を WCR で
    eq = {}
    dq = d0.copy()
    cq = es_cols(dq, 3, 3)
    pq = add_post(dq)
    tq = add_trends(dq)
    for y in HUMAN_LOG + ["log_n_E_human_nonai"]:
        specs = {
            "es_k_p3": (dq, cq, "k_p3", "module_id_global"),
            "post_did": (dq, pq, "post", "module_id_global"),
            "trend_es_k_p3": (dq, cq + tq, "k_p3", "module_id_global"),
            "trend_post_did": (dq, pq + tq, "post", "module_id_global"),
            "repo_level_post_did": (R, rpost, "post", "repo"),
            "repo_level_trend_post_did": (R, rpost + rtr, "post", "repo"),
        }
        er = {}
        for sname, (dd, cc, name, unit) in specs.items():
            f = Fit(dd, y, cc, unit=unit)
            er[sname] = {"coef": f.coef(name)["coef"], "se": f.coef(name)["se"]}
            for dl in DELTAS:
                nul = float(np.log(1 - dl))
                w = f.wcr(name, null=nul)
                tval = (f.coef(name)["coef"] - nul) / f.coef(name)["se"]
                er[sname][f"decline_{int(dl * 100)}pct"] = {
                    "null": nul, "p_wcr_onesided": w["p_wcr_upper"],
                    "p_t_onesided": float(stats.t.sf(tval, f.G - 1))}
        for dl in DELTAS:
            key = f"decline_{int(dl * 100)}pct"
            er[f"max_p_wcr_{key}"] = max(v[key]["p_wcr_onesided"] for k, v in er.items() if isinstance(v, dict) and key in v)
        eq[y] = er
        log(f"{label} equivalence {y}: max p (10%)={er['max_p_wcr_decline_10pct']:.4f} (5%)={er['max_p_wcr_decline_5pct']:.4f}")
    res["equivalence"] = eq
    return res


def honest_rm(f: Fit) -> dict:
    """Rambachan–Roth の relative magnitudes を手で近似する。

    事前期間の隣接差の最大値 Δ̄ = max_s |β_s - β_{s-1}|(s = -10..-1、β_{-1}=0、束ねた k≤-12 は使わない)。
    事後の各月で |δ_{t} - δ_{t-1}| <= M Δ̄ を許すので、k=e のバイアスは最大 (e+1) M Δ̄。
    区間 = [β_e - (e+1)MΔ̄ - t_c se, β_e + (e+1)MΔ̄ + t_c se](Δ̄ の推定誤差は無視した近似)。
    k=0..3 の平均についてはバイアス上限 = mean(e+1) M Δ̄ = 2.5 M Δ̄。
    """
    pre = {-1: 0.0}
    for s in range(-11, -1):
        c = f"k_m{abs(s)}"
        if c in f.cols:
            pre[s] = float(f.beta[f.idx(c)])
    ss = sorted(pre)
    dbar = max(abs(pre[ss[i]] - pre[ss[i - 1]]) for i in range(1, len(ss)))
    tc = float(stats.t.ppf(0.975, f.G - 1))
    out = {"delta_bar": dbar, "t_crit": tc, "by_M": {}}
    names = [f"k_p{e}" for e in range(0, 4)]
    idx = [f.idx(c) for c in names]
    wv = np.zeros(f.k)
    wv[idx] = 0.25
    avg = float(wv @ f.beta)
    avg_se = float(np.sqrt(wv @ f.V @ wv))
    for M in M_RM:
        r = {}
        for e in [0, 1, 2, 3, 6]:
            c = f"k_p{e}"
            b, se = float(f.beta[f.idx(c)]), float(np.sqrt(f.V[f.idx(c), f.idx(c)]))
            bias = (e + 1) * M * dbar
            r[c] = {"coef": b, "lo": b - bias - tc * se, "hi": b + bias + tc * se}
        bias = 2.5 * M * dbar
        r["avg_k0_k3"] = {"coef": avg, "se": avg_se, "lo": avg - bias - tc * avg_se, "hi": avg + bias + tc * avg_se}
        out["by_M"][str(M)] = r
    return out


def repo_level(d0: pd.DataFrame) -> pd.DataFrame:
    """各リポジトリを等重みにするため repo x month にモジュール平均で集約する。"""
    ys = OUTCOMES + ["log_n_E_human_nonai", "log_n_E_human_nonauthor_nonai"]
    R = d0.groupby(["repo", "month_idx"])[ys].mean().reset_index()
    g = d0.groupby("repo")["g"].first()
    R["g"] = R["repo"].map(g)
    R["event_time"] = R["month_idx"] - R["g"]
    return R


def cs_att(d0: pd.DataFrame, y: str, E: range = range(-6, 7)) -> dict:
    """never-treated を対照にした Callaway–Sant'Anna 風の ATT(e)。

    処置リポジトリ r(導入月 g)ごとに、基準月 g-1 から g+e への変化を、同じ2時点の
    never-treated の変化(モジュール数で重み付けした平均)と比べる 2x2 DiD を作り、
    (a) モジュール数で重み付け、(b) リポジトリ等重み、で集計する。
    後者が「どのリポジトリも1票」の推定。標準誤差は処置群・対照群それぞれからリポジトリを
    復元抽出するブートストラップ。post 平均は e=0..3(各リポジトリで観測できる e のみ)。
    """
    rm = d0.dropna(subset=[y]).groupby(["repo", "month_idx"])[y].agg(["mean", "size"]).reset_index()
    piv = rm.pivot(index="repo", columns="month_idx", values="mean")
    npv = rm.pivot(index="repo", columns="month_idx", values="size").fillna(0)
    g = d0.groupby("repo")["g"].first()
    tre = [r for r in piv.index if pd.notna(g.get(r))]
    nev = [r for r in piv.index if pd.isna(g.get(r))]
    P = piv.to_numpy()
    N = npv.to_numpy()
    ri = {r: i for i, r in enumerate(piv.index)}
    ci = {c: i for i, c in enumerate(piv.columns)}
    Es = list(E)

    # 事前に各(処置 r, e)の処置側変化と対照側の各リポジトリ変化・重みを作っておく
    def cell(r, e):
        gi = int(g[r])
        if gi + e not in ci or gi - 1 not in ci:
            return None
        t1, t0 = ci[gi + e], ci[gi - 1]
        dt = P[ri[r], t1] - P[ri[r], t0]
        if np.isnan(dt):
            return None
        wt = N[ri[r], t0]
        return t1, t0, dt, wt

    cells = {(r, e): cell(r, e) for r in tre for e in Es}

    def att(ts, ns, weight):
        nidx = np.array([ri[c] for c in ns])
        out = {}
        for e in Es:
            num = den = 0.0
            for r in ts:
                c = cells[(r, e)]
                if c is None:
                    continue
                t1, t0, dt, wt = c
                dc_all = P[nidx, t1] - P[nidx, t0]
                w = N[nidx, t0]
                ok = ~np.isnan(dc_all) & (w > 0)
                if not ok.any():
                    continue
                dc = np.sum(w[ok] * dc_all[ok]) / w[ok].sum()
                ww = wt if weight == "module" else 1.0
                num += ww * (dt - dc)
                den += ww
            out[e] = num / den if den > 0 else np.nan
        out["post_avg_0_3"] = float(np.nanmean([out[e] for e in range(0, 4)]))
        out["pre_avg_-6_-2"] = float(np.nanmean([out[e] for e in range(-6, -1)]))
        return out

    res = {}
    for weight in ["module", "repo"]:
        pt = att(tre, nev, weight)
        bs = [att(list(rng.choice(tre, len(tre))), list(rng.choice(nev, len(nev))), weight) for _ in range(B_CS)]
        bs = pd.DataFrame(bs)
        r = {}
        for k in list(pt.keys()):
            se = float(bs[k].std())
            r[str(k)] = {"att": round(float(pt[k]), 5), "se_boot": round(se, 5),
                         "ci95_pct": [round(float(bs[k].quantile(0.025)), 5), round(float(bs[k].quantile(0.975)), 5)]}
        res[weight] = r
    res["B"] = B_CS
    return res


# ---------------------------------------------------------------------------

def main() -> None:
    df = build_count_panel()
    months = sorted(df["month"].unique())
    old_adopt, old_never, _ = detect_adoption(df)
    log(f"old adoption: {len(old_adopt)} treated, {len(old_never)} never")

    bal = build_balanced(df)
    na = nonai_counts()
    bal = bal.merge(na, on=["module_id_global", "month"], how="left")
    for c in ["n_E_nonai", "n_E_human_nonai", "n_E_human_actors_nonai", "n_E_human_nonauthor_nonai"]:
        bal[c] = bal[c].fillna(0.0)
    bal = add_outcomes(bal)
    log(f"balanced panel {len(bal)} rows")

    new_adopt, new_never, new_excl = define_groups(df, N_BOT)

    # 導入月の新旧比較と「揺らぎ」判定
    by = df.groupby(["repo", "month"])[["n_E_agent", "n_E_human"]].sum()
    comp = {}
    for r in sorted(set(old_adopt) | set(new_adopt) | set(old_never) | set(new_never) | set(new_excl)):
        o, nw = old_adopt.get(r), new_adopt.get(r)
        s = by.loc[r]
        ag = s["n_E_agent"]
        info = {
            "old": o or ("never" if r in old_never else "excluded"),
            "new": nw or ("never" if r in new_never else "excluded"),
            "gap_months": (pd.Period(nw, "M") - pd.Period(o, "M")).n if (o and nw) else None,
            "ai_reviews_total": int(ag.sum()),
            "ai_share_total": round(float(ag.sum() / (ag.sum() + s["n_E_human"].sum())), 4),
        }
        if o:
            post = ag[ag.index >= o]
            info["old_post_months_with_ai>=10"] = f"{int((post >= 10).sum())}/{len(post)}"
            info["old_def_blip"] = bool(nw is None or abs(info["gap_months"] or 0) > 6)
        comp[r] = info
    log("adoption comparison built")

    out: dict = {
        "spec": {
            "new_rule": f"first month with >= {N_BOT} AI(agent) reviews, and >= {N_BOT} in >= {PERSIST_NEED} of the next "
                        f"{PERSIST_WIN} months (pro-rata if fewer remain, min 2); controls = never persistent even at "
                        f">= {N_CTRL}/month; repos in between excluded",
            "B_wcr": B_WCR, "B_cs": B_CS, "weights": "Webb 6-point, null imposed", "seed": SEED,
            "nonai_pr_definition": "PR whose G event is human-authored with no AI trailer; reviews on other/unknown PRs dropped",
        },
        "adoption_old": old_adopt, "never_old": old_never,
        "adoption_new": new_adopt, "never_new": new_never, "excluded_new": new_excl,
        "adoption_comparison": comp,
        "results": {},
    }
    out["results"]["old_definition"] = run_definition("OLD", bal, old_adopt, old_never, months)
    out["results"][f"new_bot_N{N_BOT}"] = run_definition("NEW", bal, new_adopt, new_never, months)

    # 閾値の感度(基本推定と WCR のみ)
    sens = {}
    for n in N_BOT_SENS:
        a, nv, ex = define_groups(df, n)
        sens[f"N{n}"] = {"adoption": a, "never": nv, "excluded": ex} | run_definition(f"N{n}", bal, a, nv, months, full=False)
    # 旧定義の処置群のまま、導入月だけ新定義で付け替え(揺らぎで入ったリポジトリは落ちる)
    redated = {r: new_adopt[r] for r in old_adopt if r in new_adopt}
    sens["old_set_redated"] = {"adoption": redated, "never": old_never} | run_definition(
        "REDATED", bal, redated, old_never, months, full=False)
    out["results"]["sensitivity"] = sens

    path = OUT_DIR / "d43_causal_robustness.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    log(f"-> {path}")


if __name__ == "__main__":
    main()
