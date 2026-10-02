"""초기 5사이클 피처 추출 및 누수 없는 배터리 분류 파이프라인."""
from pathlib import Path
import re
import h5py
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42
FILES = {
    "batch1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "batch2": "2018-02-20_batchdata_updated_struct_errorcorrect.mat",
    "batch3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}
BASIC = ["mean_QD_1_5", "std_QD_1_5", "mean_IR_1_5", "mean_Tavg_1_5",
         "mean_Tmax_1_5", "mean_chargetime_1_5"]
DELTA = ["delta_logvar_5_4", "delta_min_5_4"]
POLICY = ["C_rate_1", "switch_SOC", "C_rate_2"]
FEATURE_SETS = {
    "basic": BASIC,
    "basic_delta": BASIC + DELTA,
    "basic_logvar": BASIC + DELTA[:1],
    "basic_delta_policy": BASIC + DELTA + POLICY,
}
# 검증 불가능 상황에서 Test를 보고 고르지 않기 위한 사전 지정 기준 모델.
DEFAULT_MODEL, DEFAULT_FEATURES = "Logistic", "basic_delta"


def read_values(f, dataset):
    """MATLAB 숫자 배열 또는 reference 배열을 1차원으로 읽습니다."""
    if h5py.check_dtype(ref=dataset.dtype) is not None:
        return np.concatenate([np.asarray(f[r], dtype=float).ravel()
                               for r in dataset[()].ravel()])
    return np.asarray(dataset, dtype=float).ravel()


def extract_features(data_dir, batch_names=("batch1", "batch2")):
    """summary에서 최초 5사이클을 선택하고 cycles에서는 Q4/Q5만 읽습니다."""
    rows = []
    for name in batch_names:
        path = Path(data_dir) / FILES[name]
        if not path.is_file():
            raise FileNotFoundError(f"원본 데이터가 없습니다: {path}. README의 데이터 준비를 확인하세요.")
        with h5py.File(path, "r") as f:
            batch = f["batch"]
            for i, ref in enumerate(batch["summary"][:, 0]):
                summary = f[ref]
                arrays = {k: read_values(f, summary[k]) for k in
                          ["cycle", "QDischarge", "QCharge", "IR", "Tavg", "Tmax", "chargetime"]}
                s = pd.DataFrame(arrays)
                if s.cycle.duplicated().any():
                    raise ValueError(f"{name}_cell{i}: 중복 cycle 번호")
                first = s.loc[s.cycle.between(1, 5)].copy()
                measures = first.columns.drop("cycle")
                empty = first[measures].fillna(0).eq(0).all(axis=1)
                first = first.loc[~empty].replace([np.inf, -np.inf], np.nan)
                for k in ["QDischarge", "QCharge"]:
                    first[k] = first[k].where(first[k].between(0, 1.43, inclusive="neither"))
                for k in ["IR", "chargetime"]:
                    first[k] = first[k].where(first[k] > 0)
                policy = "".join(chr(int(v)) for v in f[batch["policy_readable"][i, 0]][()].ravel())
                match = re.fullmatch(r"([\d.]+)C\(([\d.]+)%\)-([\d.]+)C", policy)
                settings = [float(v) for v in match.groups()] if match else [np.nan] * 3
                life = float(read_values(f, f[batch["cycle_life"][i, 0]])[0])
                row = dict(batch=name, cell_id=f"{name}_cell{i}", policy=policy,
                           cycle_life=life, empty_rows=int(empty.sum()),
                           observed_cycles=int(first.cycle.nunique()),
                           mean_QD_1_5=first.QDischarge.mean(), std_QD_1_5=first.QDischarge.std(ddof=1),
                           mean_IR_1_5=first.IR.mean(), mean_Tavg_1_5=first.Tavg.mean(),
                           mean_Tmax_1_5=first.Tmax.mean(), mean_chargetime_1_5=first.chargetime.mean(),
                           delta_logvar_5_4=np.nan, delta_min_5_4=np.nan,
                           C_rate_1=settings[0], switch_SOC=settings[1], C_rate_2=settings[2])
                cycles = f[batch["cycles"][i, 0]]
                if cycles["Qdlin"].size != len(s):
                    raise ValueError(f"{name}_cell{i}: summary/cycles 길이 불일치")
                positions = {int(c): j for j, c in enumerate(s.cycle) if c in (4, 5)}
                if len(positions) == 2:
                    q4, q5 = [read_values(f, f[cycles["Qdlin"][positions[c], 0]]) for c in (4, 5)]
                    voltage = read_values(f, f[batch["Vdlin"][i, 0]])
                    if len(q4) == len(q5) == len(voltage) and len(q4) > 2:
                        delta = q5 - q4
                        delta = delta[np.isfinite(delta) & np.isfinite(voltage)]
                        if len(delta) > 2:
                            row["delta_logvar_5_4"] = np.log10(max(np.var(delta), 1e-12))
                            row["delta_min_5_4"] = delta.min()
                rows.append(row)
    df = pd.DataFrame(rows)
    df["target"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    known = np.isfinite(df.cycle_life) & (df.cycle_life > 0)
    df.loc[known, "target"] = (df.loc[known, "cycle_life"] >= 550).astype(int)
    return df


def data_counts(df):
    return df.groupby("batch").agg(total=("cell_id", "size"), known=("target", "count"),
        unknown=("target", lambda s: s.isna().sum()),
        short=("target", lambda s: s.eq(0).sum()), long=("target", lambda s: s.eq(1).sum()))


def make_models():
    """결측 보완·표준화도 각 학습 분할 안에서 수행합니다."""
    estimators = {
        "Dummy": DummyClassifier(strategy="most_frequent"),
        "Logistic": LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=SEED),
        "SVM": SVC(C=1.0, kernel="rbf", class_weight="balanced"),
        # 공유 공분산 축소로 소표본·중복 피처의 불안정성을 완화합니다.
        "ShrinkageLDA": LinearDiscriminantAnalysis(solver="lsqr", shrinkage=0.5, priors=[0.5, 0.5]),
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=3, min_samples_leaf=2,
                                                class_weight="balanced", random_state=SEED, n_jobs=1),
    }
    return {name: make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                StandardScaler(), estimator) for name, estimator in estimators.items()}


def check_training_data(train):
    """학습 데이터와 입력 피처 범위를 검사합니다."""
    if train.empty or not train.batch.eq("batch1").all():
        raise ValueError("학습에는 Batch 1만 사용할 수 있습니다.")
    if train.target.isna().any() or train.cell_id.duplicated().any():
        raise ValueError("학습 라벨 결측 또는 중복 셀을 확인하세요.")
    allowed = set(BASIC + DELTA + POLICY)
    if any(not set(features) <= allowed for features in FEATURE_SETS.values()):
        raise ValueError("초기 5사이클 외 피처가 입력에 포함되었습니다.")


def split_batch1(train):
    """충전 정책 그룹을 분리하고, 학습에는 두 클래스가 남는 Hold-out을 찾습니다."""
    check_training_data(train)
    splitter = GroupShuffleSplit(n_splits=100, test_size=0.2, random_state=SEED)
    for fit_idx, valid_idx in splitter.split(train, groups=train.policy):
        if train.iloc[fit_idx].target.nunique() == 2:
            return fit_idx, valid_idx
    raise ValueError("학습에 두 클래스를 남길 수 있는 정책 그룹 Hold-out이 없습니다.")


def metrics(y, pred):
    y, pred = np.asarray(y, dtype=int), np.asarray(pred, dtype=int)
    both = len(np.unique(y)) == 2
    return {"F1": f1_score(y, pred, pos_label=1, zero_division=0),
            "Accuracy": accuracy_score(y, pred),
            "Macro_F1": f1_score(y, pred, average="macro", zero_division=0) if both else np.nan,
            "Short_Recall": recall_score(y, pred, pos_label=0, zero_division=0) if (y == 0).any() else np.nan,
            "Balanced_Accuracy": np.mean([recall_score(y, pred, pos_label=c, zero_division=0)
                                          for c in (0, 1)]) if both else np.nan}


def compare_validation(train, fit_idx, valid_idx):
    """Batch 1에서 후보를 비교하며, 소수 클래스 부족 시 CV는 결측으로 남깁니다."""
    check_training_data(train)
    if set(fit_idx) & set(valid_idx):
        raise ValueError("학습/검증 인덱스가 중복됩니다.")
    fit, valid = train.iloc[fit_idx], train.iloc[valid_idx]
    if not set(fit.policy).isdisjoint(set(valid.policy)):
        raise ValueError("학습/검증 정책 그룹이 중복됩니다.")
    rows = []
    minority = int(fit.target.value_counts().min())
    cv_splits = None
    if minority >= 2:
        cv = StratifiedGroupKFold(n_splits=min(3, minority), shuffle=True, random_state=SEED)
        splits = list(cv.split(fit, fit.target.astype(int), fit.policy))
        if all(fit.iloc[a].target.nunique() == fit.iloc[b].target.nunique() == 2 for a, b in splits):
            cv_splits = splits
    for feature_name, features in FEATURE_SETS.items():
        for name, model in make_models().items():
            fitted = clone(model).fit(fit[features], fit.target.astype(int))
            score = metrics(valid.target, fitted.predict(valid[features]))
            row = {"model": name, "features": feature_name, **score, "CV_F1": np.nan,
                   "CV_Accuracy": np.nan, "CV_Macro_F1": np.nan,
                   "CV_status": "단수명 표본/정책 그룹 부족으로 계산 불가"}
            if cv_splits:
                scores = cross_validate(model, fit[features], fit.target.astype(int), cv=cv_splits,
                    scoring={"F1": "f1", "Accuracy": "accuracy", "Macro_F1": "f1_macro"}, error_score="raise")
                row.update({"CV_" + k: scores["test_" + k].mean() for k in ["F1", "Accuracy", "Macro_F1"]})
                row["CV_status"] = "학습 부분 내 정책 그룹 CV"
            rows.append(row)
    return pd.DataFrame(rows)


def choose_candidate(comparison, valid):
    # 단일 클래스 검증으로는 단수명 분류기를 고를 수 없습니다.
    if valid.target.nunique() < 2 or comparison.CV_Macro_F1.isna().all():
        return DEFAULT_MODEL, DEFAULT_FEATURES, "검증 불충분: 사전 지정 기준 모델(잠정), 최적 모델 확정 불가"
    candidates = comparison[comparison.model != "Dummy"]
    best = candidates.sort_values(["CV_Macro_F1", "Macro_F1"], ascending=False).iloc[0]
    return best.model, best.features, "Batch 1 내부 CV 및 Hold-out 기준 선택"


def evaluate_test(train, test, feature_name):
    """모델 선택이 끝난 뒤 Batch 1 전체로 재학습해 Batch 2를 한 번 평가합니다."""
    check_training_data(train)
    if test.empty or not test.batch.eq("batch2").all() or test.target.isna().any():
        raise ValueError("최종 평가는 수명이 있는 Batch 2만 사용합니다.")
    if not set(train.cell_id).isdisjoint(set(test.cell_id)):
        raise ValueError("학습/평가 셀이 중복됩니다.")
    features, rows, predictions, fitted_models = FEATURE_SETS[feature_name], [], [], {}
    for name, model in make_models().items():
        fitted = clone(model).fit(train[features], train.target.astype(int))
        pred = fitted.predict(test[features])
        rows.append({"model": name, "features": feature_name, **metrics(test.target, pred)})
        p = test[["cell_id", "policy", "cycle_life", "target"]].copy()
        p["prediction"], p["model"] = pred, name
        p["error"] = p.target != p.prediction
        predictions.append(p)
        fitted_models[name] = fitted
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True), fitted_models


def reporting_table(comparison, test_scores, name, feature_name):
    v = comparison.query("model == @name and features == @feature_name").iloc[0]
    t = test_scores.query("model == @name").iloc[0]
    rows = []
    for label, f1, acc, note in [
        ("Train (Batch 1 CV)", v.CV_F1, v.CV_Accuracy, v.CV_status),
        ("Valid (Batch 1 Hold-out)", v.F1, v.Accuracy, "장수명만 포함: 단수명 검증 불가"),
        ("Test (Batch 2)", t.F1, t.Accuracy, "선택 후 최종 평가"),
        ("Gap (Train-Valid)", v.CV_F1-v.F1, v.CV_Accuracy-v.Accuracy, "CV 결측으로 Gap 계산 불가"),
        ("Gap (Valid-Test)", v.F1-t.F1, v.Accuracy-t.Accuracy, "클래스 구성도 달라 과적합으로 단정 불가"),
        ("Gap (Target-Test)", np.nan, 0.951-t.Accuracy, "Target Accuracy 95.1%; F1 목표값은 미제시"),
    ]:
        rows.append({"구분": label, "F1-Score": f1, "Accuracy": acc, "비고": note})
    return pd.DataFrame(rows)
