from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import json
import numpy as np
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def ensure_dirs() -> Tuple[Path, Path]:
    models = Path("./models"); models.mkdir(parents=True, exist_ok=True)
    metrics = Path("./exports/metrics"); metrics.mkdir(parents=True, exist_ok=True)
    return models, metrics

def save_clustering_artifacts(
    E_norm: np.ndarray,
    labels: np.ndarray,
    metas: List[Dict],
    model,
    algo: str,
    model_params: Optional[Dict] = None
) -> Dict[str, str]:
    models_dir, metrics_dir = ensure_dirs()

    # 1) Salva rótulos e embeddings normalizados
    labels_path = metrics_dir / "cluster_labels.npy"
    np.save(labels_path, labels)

    Enorm_path = Path("./final_embeddings_norm.npy")
    np.save(Enorm_path, E_norm)

    # 2) Salva metadados dos itens (arquivo e papel)
    meta_json = metrics_dir / "cluster_metas.json"
    with meta_json.open("w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)

    # 3) Salva o modelo de clusterização (quando possível)
    model_path = models_dir / f"{algo}_model.joblib"
    try:
        dump(model, model_path)
        saved_model = str(model_path)
    except Exception:
        saved_model = ""

    # 4) Salva um meta do modelo
    meta_model = {
        "algo": algo,
        "model_path": saved_model,
        "n_items": int(E_norm.shape[0]),
        "n_features": int(E_norm.shape[1]),
        "n_clusters_found": int(len(set(labels)) - (1 if (-1 in labels) else 0)),
        "has_noise": bool((-1 in labels)),
        "params": model_params or {},
    }
    model_meta_path = models_dir / f"{algo}_meta.json"
    with model_meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta_model, f, ensure_ascii=False, indent=2)

    return {
        "labels": str(labels_path),
        "embeddings_norm": str(Enorm_path),
        "metas": str(meta_json),
        "model": saved_model,
        "model_meta": str(model_meta_path),
    }

def train_cluster_classifier(
    E_norm: np.ndarray,
    labels: np.ndarray,
    out_name: str = "cluster_classifier_logreg"
) -> Dict[str, str]:
    """
    Treina um classificador supervisionado (LogisticRegression) para prever o cluster (descartando ruído -1).
    Salva modelo, classes e relatório.
    """
    models_dir, metrics_dir = ensure_dirs()

    mask = labels != -1
    X = E_norm[mask]
    y = labels[mask]

    # precisa de pelo menos 2 classes
    if X.shape[0] < 5 or len(set(y.tolist())) < 2:
        report_path = metrics_dir / f"{out_name}_report.txt"
        with report_path.open("w", encoding="utf-8") as f:
            f.write("Insuficiente para treinar classificador (menos de 5 amostras úteis ou <2 classes).\n")
        return {"classifier": "", "classes": "", "report": str(report_path)}

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=1000, n_jobs=None)
    clf.fit(Xtr, ytr)

    ypred = clf.predict(Xte)
    rep = classification_report(yte, ypred, digits=3)

    # salva modelo e classes
    clf_path = models_dir / f"{out_name}.joblib"
    dump(clf, clf_path)
    classes_path = models_dir / f"{out_name}_classes.npy"
    np.save(classes_path, clf.classes_)

    # salva relatório
    report_path = metrics_dir / f"{out_name}_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write(rep)

    return {"classifier": str(clf_path), "classes": str(classes_path), "report": str(report_path)}