"""app/ml/core_classes.py — Kern-ML-Klassen (portiert aus push-balancer-server.py).

Enthält die Pure-Python-Implementierungen für GBRT und Hilfsobjekte.
Keine externen Abhängigkeiten außer stdlib und optional numpy/sklearn.
"""
from __future__ import annotations

import logging
import math
import random
import time
from collections import defaultdict

log = logging.getLogger("push-balancer")

# numpy ist optional (nur für _SklearnModelWrapper benötigt)
try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]


# ── _GBRTNode ────────────────────────────────────────────────────────────

class _GBRTNode:
    """Ein Knoten im Entscheidungsbaum."""
    __slots__ = ("feature_idx", "threshold", "left", "right", "value", "gain")

    def __init__(self):
        self.feature_idx = -1
        self.threshold = 0.0
        self.left = None
        self.right = None
        self.value = 0.0  # Leaf-Value (Mean der Residuen)
        self.gain = 0.0


# ── _GBRTTree ────────────────────────────────────────────────────────────

class _GBRTTree:
    """Ein einzelner Regressions-Baum fuer Gradient Boosting."""

    def __init__(self, max_depth=5, min_samples_leaf=10, n_bins=255):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_bins = n_bins
        self.root = None

    def fit(self, X, residuals, sample_indices=None, sample_weights=None):
        """Trainiert den Baum auf Residuen.

        Args:
            X: Feature-Matrix (Liste von Listen)
            residuals: Ziel-Residuen (Liste von Floats)
            sample_indices: Optionale Subsample-Indices
            sample_weights: Optionale Gewichte pro Sample
        """
        if sample_indices is None:
            sample_indices = list(range(len(X)))
        self.n_features = len(X[0]) if X else 0
        self._weights = sample_weights
        self.root = self._build_tree(X, residuals, sample_indices, depth=0)

    def predict_one(self, x):
        """Prediction fuer einen einzelnen Feature-Vektor."""
        node = self.root
        while node is not None:
            if node.feature_idx < 0:  # Leaf
                return node.value
            if x[node.feature_idx] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return 0.0

    def _build_tree(self, X, residuals, indices, depth):
        """Rekursiver Baum-Aufbau mit Histogram-basiertem Splitting und Sample-Gewichtung."""
        node = _GBRTNode()
        w = self._weights  # Sample-Gewichte (oder None)

        # Leaf: gewichteter Mean der Residuen
        if w:
            w_sum = sum(w[i] for i in indices)
            node.value = sum(residuals[i] * w[i] for i in indices) / w_sum if w_sum > 0 else 0.0
        else:
            vals = [residuals[i] for i in indices]
            node.value = sum(vals) / len(vals) if vals else 0.0

        # Stopp-Kriterien
        if depth >= self.max_depth or len(indices) < self.min_samples_leaf * 2:
            node.feature_idx = -1
            return node

        best_gain = 0.0
        best_feat = -1
        best_thresh = 0.0

        n = len(indices)

        for f_idx in range(self.n_features):
            # Histogram-basiertes Splitting mit Gewichten
            if w:
                f_vals = [(X[i][f_idx], residuals[i], w[i]) for i in indices]
            else:
                f_vals = [(X[i][f_idx], residuals[i], 1.0) for i in indices]
            f_vals.sort(key=lambda x: x[0])

            total_wsum = sum(v[2] * v[1] for v in f_vals)
            total_w = sum(v[2] for v in f_vals)

            left_wsum = 0.0
            left_w = 0.0
            left_n = 0

            # Schritt-Groesse: nicht jeden Wert testen, sondern n_bins gleichmaessig verteilt
            step = max(1, n // self.n_bins)

            for pos in range(0, n - 1, step):
                # Alle Elemente bis pos in Links
                for k in range(left_n, pos + 1):
                    left_wsum += f_vals[k][2] * f_vals[k][1]
                    left_w += f_vals[k][2]
                left_n = pos + 1

                # Kein Split bei gleichen Feature-Werten
                if f_vals[pos][0] == f_vals[min(pos + 1, n - 1)][0]:
                    continue

                right_n = n - left_n
                if left_n < self.min_samples_leaf or right_n < self.min_samples_leaf:
                    continue

                right_wsum = total_wsum - left_wsum
                right_w = total_w - left_w
                if left_w <= 0 or right_w <= 0:
                    continue

                # Gewichtete Varianz-Reduktion
                gain = (left_wsum * left_wsum / left_w +
                        right_wsum * right_wsum / right_w -
                        total_wsum * total_wsum / total_w)

                if gain > best_gain:
                    best_gain = gain
                    best_feat = f_idx
                    best_thresh = (f_vals[pos][0] + f_vals[min(pos + 1, n - 1)][0]) / 2.0

        if best_feat < 0 or best_gain <= 0:
            node.feature_idx = -1
            return node

        # Split anwenden
        left_idx = [i for i in indices if X[i][best_feat] <= best_thresh]
        right_idx = [i for i in indices if X[i][best_feat] > best_thresh]

        if len(left_idx) < self.min_samples_leaf or len(right_idx) < self.min_samples_leaf:
            node.feature_idx = -1
            return node

        node.feature_idx = best_feat
        node.threshold = best_thresh
        node.gain = best_gain
        node.left = self._build_tree(X, residuals, left_idx, depth + 1)
        node.right = self._build_tree(X, residuals, right_idx, depth + 1)
        return node

    def path_contributions(self, x, n_features):
        """TreeSHAP-artige Feature-Contributions via Root-to-Leaf Pfad.

        An jedem Split: contributions[feature] += child.value - node.value
        Summe aller Contributions = leaf.value - root.value
        """
        contributions = [0.0] * n_features
        node = self.root
        if node is None:
            return contributions
        while node is not None and node.feature_idx >= 0:
            fidx = node.feature_idx
            if x[fidx] <= node.threshold:
                child = node.left
            else:
                child = node.right
            if child is not None:
                contributions[fidx] += child.value - node.value
            node = child
        return contributions

    def to_dict(self):
        """Serialisiert den Baum als JSON-faehiges Dict."""
        return self._node_to_dict(self.root) if self.root else {}

    def _node_to_dict(self, node):
        if node is None:
            return None
        if node.feature_idx < 0:
            return {"v": round(node.value, 6)}
        return {
            "f": node.feature_idx,
            "t": round(node.threshold, 6),
            "v": round(node.value, 6),
            "l": self._node_to_dict(node.left),
            "r": self._node_to_dict(node.right),
        }

    @staticmethod
    def from_dict(d):
        """Deserialisiert einen Baum aus Dict."""
        tree = _GBRTTree()
        tree.root = _GBRTTree._node_from_dict(d)
        return tree

    @staticmethod
    def _node_from_dict(d):
        if d is None:
            return None
        node = _GBRTNode()
        if "f" in d:
            # Interner Knoten
            node.feature_idx = d["f"]
            node.threshold = d["t"]
            node.value = d.get("v", 0.0)  # Backward-kompatibel
            node.left = _GBRTTree._node_from_dict(d.get("l"))
            node.right = _GBRTTree._node_from_dict(d.get("r"))
        else:
            # Leaf
            node.feature_idx = -1
            node.value = d.get("v", 0.0)
        return node


# ── GBRTModel ────────────────────────────────────────────────────────────

class GBRTModel:
    """Gradient Boosted Regression Trees — reines Python, kein numpy/sklearn.

    Trainiert ein Ensemble von Regressionsbaeumen via Gradient Boosting
    mit Histogram-basiertem Splitting fuer Performance.
    """

    def __init__(self, n_trees=300, max_depth=6, learning_rate=0.08,
                 min_samples_leaf=8, subsample=0.85, n_bins=255,
                 loss="huber", huber_delta=1.5, log_target=False):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.n_bins = n_bins
        self.loss = loss
        self.huber_delta = huber_delta
        self.log_target = log_target
        self.trees = []
        self.initial_prediction = 0.0
        self.feature_names = []
        self.feature_importance_ = {}
        self.train_metrics = {}

    def fit(self, X, y, feature_names=None, val_X=None, val_y=None,
            sample_weights=None):
        """Trainiert das GBRT-Modell.

        Args:
            X: Feature-Matrix (Liste von Listen), jede Zeile = 1 Sample
            y: Zielwerte (Liste von Floats)
            feature_names: Optionale Feature-Namen
            val_X, val_y: Optionale Validierungs-Daten fuer Early Stopping
            sample_weights: Optionale Gewichte pro Sample (Liste von Floats)
        """
        n = len(X)
        if n == 0:
            return

        self.feature_names = feature_names or [f"f{i}" for i in range(len(X[0]))]

        # Log-Target-Transformation
        if self.log_target:
            y = [math.log1p(v) for v in y]
            if val_y:
                val_y = [math.log1p(v) for v in val_y]

        # Sample Weights normalisieren (Durchschnitt = 1.0)
        if sample_weights:
            w_mean = sum(sample_weights) / len(sample_weights)
            self._sample_weights = [w / w_mean for w in sample_weights] if w_mean > 0 else [1.0] * n
        else:
            self._sample_weights = [1.0] * n

        # Gewichteter Mittelwert als Initial Prediction
        w_total = sum(self._sample_weights)
        self.initial_prediction = sum(y[i] * self._sample_weights[i] for i in range(n)) / w_total
        self.trees = []

        # Aktuelle Predictions
        predictions = [self.initial_prediction] * n
        val_predictions = [self.initial_prediction] * len(val_X) if val_X else []

        # Feature Importance Tracking
        feat_gain = defaultdict(float)

        best_val_mae = float('inf')
        best_n_trees = 0
        rounds_no_improve = 0
        early_stopped = False
        rng = random.Random(42)
        delta = self.huber_delta

        for t in range(self.n_trees):
            # Residuen mit Huber-Gradient
            if self.loss == "huber":
                residuals = []
                for i in range(n):
                    r = y[i] - predictions[i]
                    if abs(r) <= delta:
                        residuals.append(r)
                    else:
                        residuals.append(delta * (1.0 if r > 0 else -1.0))
            else:
                residuals = [y[i] - predictions[i] for i in range(n)]

            # Subsampling
            if self.subsample < 1.0:
                sample_size = max(1, int(n * self.subsample))
                sample_idx = rng.sample(range(n), sample_size)
            else:
                sample_idx = None

            tree = _GBRTTree(max_depth=self.max_depth,
                            min_samples_leaf=self.min_samples_leaf,
                            n_bins=self.n_bins)
            tree.fit(X, residuals, sample_idx, sample_weights=self._sample_weights)
            self.trees.append(tree)

            # Update Predictions
            for i in range(n):
                predictions[i] += self.learning_rate * tree.predict_one(X[i])

            # Feature Importance (aus Split-Gains)
            self._collect_importance(tree.root, feat_gain)

            # Early Stopping auf Validation-Set
            if val_X:
                for i in range(len(val_X)):
                    val_predictions[i] += self.learning_rate * tree.predict_one(val_X[i])
                # Val-MAE im Original-Raum berechnen
                if self.log_target:
                    val_mae = sum(abs(math.expm1(val_predictions[i]) - math.expm1(val_y[i]))
                                  for i in range(len(val_y))) / len(val_y)
                else:
                    val_mae = sum(abs(val_predictions[i] - val_y[i])
                                  for i in range(len(val_y))) / len(val_y)
                if val_mae < best_val_mae - 0.001:
                    best_val_mae = val_mae
                    best_n_trees = t + 1
                    rounds_no_improve = 0
                else:
                    rounds_no_improve += 1
                if rounds_no_improve >= 30 and t + 1 >= 50:
                    early_stopped = True
                    best_n_trees = max(best_n_trees, 40)  # Minimum 40 Bäume
                    log.info(f"[GBRT] Early stopping bei Baum {t+1}, "
                             f"beste Stelle: Baum {best_n_trees} (val_mae={best_val_mae:.4f})")
                    # Bäume nach dem besten Punkt entfernen
                    if best_n_trees > 0 and best_n_trees < len(self.trees):
                        self.trees = self.trees[:best_n_trees]
                    break

        # Early Stopping Attribute speichern
        self.best_n_trees = best_n_trees if best_n_trees > 0 else len(self.trees)
        self.early_stopped = early_stopped

        # Train-Metriken im Original-Raum
        if self.log_target:
            orig_preds = [math.expm1(p) for p in predictions]
            orig_y = [math.expm1(v) for v in y]
        else:
            orig_preds = predictions
            orig_y = y

        train_mae = sum(abs(orig_preds[i] - orig_y[i]) for i in range(n)) / n
        train_rmse = math.sqrt(sum((orig_preds[i] - orig_y[i]) ** 2 for i in range(n)) / n)

        # R² berechnen
        y_mean = sum(orig_y) / n
        ss_res = sum((orig_y[i] - orig_preds[i]) ** 2 for i in range(n))
        ss_tot = sum((orig_y[i] - y_mean) ** 2 for i in range(n))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        self.train_metrics = {
            "mae": round(train_mae, 4),
            "rmse": round(train_rmse, 4),
            "r2": round(r2, 4),
            "r2_residual": round(r2, 4),
            "n_trees_used": len(self.trees),
            "n_trees_requested": self.n_trees,
            "early_stopped": early_stopped,
            "early_stopped_at": self.best_n_trees if early_stopped else None,
            "n_samples": n,
            "loss": self.loss,
            "log_target": self.log_target,
        }

        if val_X:
            self.train_metrics["val_mae"] = round(best_val_mae, 4)

        # Normalize Feature Importance
        total_gain = sum(feat_gain.values()) or 1.0
        self.feature_importance_ = {
            self.feature_names[k] if k < len(self.feature_names) else f"f{k}":
            round(v / total_gain, 4)
            for k, v in sorted(feat_gain.items(), key=lambda x: -x[1])
        }

        log.info(f"[GBRT] Training: {len(self.trees)} Baeume, MAE={train_mae:.3f}, "
                 f"RMSE={train_rmse:.3f}, R²={r2:.3f}, n={n}")

    def fit_incremental(self, X, y, n_new_trees=10):
        """Fuegt inkrementell neue Baeume hinzu ohne bestehendes Modell zurueckzusetzen.

        Berechnet Residuen auf neuen Daten und trainiert flachere Baeume.
        """
        n = len(X)
        if n == 0 or not self.trees:
            return

        # Aktuelle Vorhersagen fuer neue Daten
        predictions = [self.predict_one(x) for x in X]

        # Residuen
        residuals = [y[i] - predictions[i] for i in range(n)]

        rng = random.Random(int(time.time()))
        inc_depth = max(2, self.max_depth - 1)
        inc_min_leaf = max(self.min_samples_leaf + 5, 15)

        for _ in range(n_new_trees):
            # Subsampling
            if self.subsample < 1.0 and n > 1:
                sample_size = max(1, int(n * self.subsample))
                sample_idx = rng.sample(range(n), sample_size)
            else:
                sample_idx = None

            tree = _GBRTTree(max_depth=inc_depth,
                             min_samples_leaf=inc_min_leaf,
                             n_bins=self.n_bins)
            tree.fit(X, residuals, sample_idx)
            self.trees.append(tree)

            # Update Residuen
            for i in range(n):
                predictions[i] += self.learning_rate * tree.predict_one(X[i])
            residuals = [y[i] - predictions[i] for i in range(n)]

        log.info(f"[GBRT] Inkrementell: +{n_new_trees} Baeume "
                 f"(total={len(self.trees)}, depth={inc_depth}, n={n})")

    def predict(self, X):
        """Prediction fuer mehrere Samples. Returns Liste von Floats."""
        return [self.predict_one(x) for x in X]

    def predict_one(self, x):
        """Prediction fuer einen einzelnen Feature-Vektor."""
        pred = self.initial_prediction
        for tree in self.trees:
            pred += self.learning_rate * tree.predict_one(x)
        if self.log_target:
            pred = math.expm1(pred)
        return max(0.01, pred)

    def predict_with_uncertainty(self, x):
        """Prediction mit Unsicherheitsschaetzung aus Baum-Varianz."""
        tree_preds = []
        cumulative = self.initial_prediction
        for tree in self.trees:
            contrib = self.learning_rate * tree.predict_one(x)
            cumulative += contrib
            tree_preds.append(cumulative)

        pred = cumulative
        if self.log_target:
            pred = math.expm1(pred)

        # Varianz der letzten 50 Baeume als Unsicherheits-Mass
        if len(tree_preds) > 50:
            recent = tree_preds[-50:]
            if self.log_target:
                recent = [math.expm1(r) for r in recent]
            mean_recent = sum(recent) / len(recent)
            var = sum((p - mean_recent) ** 2 for p in recent) / len(recent)
            std = math.sqrt(var)
        else:
            std = 0.5  # Default-Unsicherheit

        return {
            "predicted": max(0.01, pred),
            "std": round(std, 4),
            "confidence": round(max(0.1, min(0.99, 1.0 - std / max(1.0, abs(pred)))), 3),
        }

    def shap_values(self, x):
        """Berechnet TreeSHAP-artige Feature-Contributions fuer einen einzelnen Sample.

        Aggregiert path_contributions ueber alle Baeume, gewichtet mit learning_rate.
        Returns: Dict mit base_value, shap_values (feature_idx→contribution), prediction.
        """
        n_features = len(x)
        total_contributions = [0.0] * n_features
        for tree in self.trees:
            contribs = tree.path_contributions(x, n_features)
            for i in range(n_features):
                total_contributions[i] += self.learning_rate * contribs[i]

        raw_prediction = self.initial_prediction + sum(total_contributions)
        if self.log_target:
            prediction = math.expm1(raw_prediction)
            base_value = math.expm1(self.initial_prediction)
        else:
            prediction = raw_prediction
            base_value = self.initial_prediction

        shap_dict = {}
        for i, c in enumerate(total_contributions):
            if abs(c) > 1e-6:
                fname = self.feature_names[i] if i < len(self.feature_names) else f"f{i}"
                shap_dict[fname] = round(c, 5)

        return {
            "base_value": round(base_value, 5),
            "shap_values": shap_dict,
            "prediction": round(max(0.01, prediction), 5),
        }

    def feature_importance(self, top_n=20):
        """Top-N Feature Importance als sortierte Liste."""
        items = sorted(self.feature_importance_.items(), key=lambda x: -x[1])
        return [{"name": k, "importance": v} for k, v in items[:top_n]]

    def _collect_importance(self, node, feat_gain):
        """Sammelt Split-Gains rekursiv."""
        if node is None or node.feature_idx < 0:
            return
        feat_gain[node.feature_idx] += node.gain
        self._collect_importance(node.left, feat_gain)
        self._collect_importance(node.right, feat_gain)

    def to_json(self):
        """Serialisiert das gesamte Modell als JSON-faehiges Dict."""
        d = {
            "type": "GBRT",
            "n_trees": len(self.trees),
            "initial_prediction": round(self.initial_prediction, 6),
            "learning_rate": self.learning_rate,
            "feature_names": self.feature_names,
            "trees": [t.to_dict() for t in self.trees],
            "metrics": self.train_metrics,
            "feature_importance": self.feature_importance(20),
        }
        if self.log_target:
            d["log_target"] = True
        if self.loss != "mse":
            d["loss"] = self.loss
            d["huber_delta"] = self.huber_delta
        if hasattr(self, "conformal_radius"):
            d["conformal_radius"] = round(self.conformal_radius, 6)
        if hasattr(self, "blend_alpha"):
            d["blend_alpha"] = round(self.blend_alpha, 4)
        if hasattr(self, "best_n_trees"):
            d["best_n_trees"] = self.best_n_trees
        if hasattr(self, "early_stopped"):
            d["early_stopped"] = self.early_stopped
        return d

    @staticmethod
    def from_json(data):
        """Deserialisiert ein GBRT-Modell aus JSON."""
        model = GBRTModel()
        model.initial_prediction = data["initial_prediction"]
        model.learning_rate = data["learning_rate"]
        model.feature_names = data["feature_names"]
        model.log_target = data.get("log_target", False)
        model.loss = data.get("loss", "mse")
        model.huber_delta = data.get("huber_delta", 1.5)
        if "conformal_radius" in data:
            model.conformal_radius = data["conformal_radius"]
        if "blend_alpha" in data:
            model.blend_alpha = data["blend_alpha"]
        if "best_n_trees" in data:
            model.best_n_trees = data["best_n_trees"]
        if "early_stopped" in data:
            model.early_stopped = data["early_stopped"]
        model.trees = [_GBRTTree.from_dict(td) for td in data["trees"]]
        model.train_metrics = data.get("metrics", {})
        # Feature Importance aus gespeichertem JSON wiederherstellen
        fi_list = data.get("feature_importance", [])
        if fi_list and model.feature_names:
            for fi_item in fi_list:
                name = fi_item.get("name", "")
                if name in model.feature_names:
                    idx = model.feature_names.index(name)
                    model.feature_importance_[idx] = fi_item.get("importance", 0)
        return model


# ── _SklearnModelWrapper ─────────────────────────────────────────────────

class _SklearnModelWrapper:
    """Wrapper um sklearn GradientBoostingRegressor für API-Kompatibilität mit GBRTModel."""

    def __init__(self, sklearn_model, feature_names):
        self.sklearn_model = sklearn_model
        self.feature_names = list(feature_names)
        self.trees = list(range(sklearn_model.n_estimators_))  # Dummy für len()
        self.train_metrics = {}
        self.feature_importance_ = {}
        self._is_sklearn = True
        # Feature Importance extrahieren
        importances = sklearn_model.feature_importances_
        total = sum(importances)
        if total > 0:
            for i, fname in enumerate(feature_names):
                if importances[i] > 0:
                    self.feature_importance_[fname] = float(importances[i] / total)

    def predict(self, X):
        if np is None:
            return [0.0] * len(X)
        return self.sklearn_model.predict(np.array(X, dtype=np.float64)).tolist()

    def predict_one(self, x):
        if np is None:
            return 0.0
        return float(self.sklearn_model.predict(np.array([x], dtype=np.float64))[0])

    def predict_with_uncertainty(self, x):
        if np is None:
            return {"predicted": 0.0, "confidence": 0.5, "std": 0.0}
        x_arr = np.array([x], dtype=np.float64)
        pred = float(self.sklearn_model.predict(x_arr)[0])
        # Uncertainty via Baum-Varianz (schneller als staged_predict)
        std = 0.0
        try:
            n_trees = self.sklearn_model.n_estimators_
            if n_trees > 50:
                # Nur erste + letzte 25 Bäume vergleichen statt alle zu iterieren
                lr = self.sklearn_model.learning_rate
                # Schnelle Näherung: Fraction of total prediction from recent trees
                tree_preds = []
                for tree_idx in range(max(0, n_trees - 25), n_trees):
                    tp = float(self.sklearn_model.estimators_[tree_idx, 0].predict(x_arr)[0])
                    tree_preds.append(tp * lr)
                if tree_preds:
                    std = math.sqrt(sum(t ** 2 for t in tree_preds) / len(tree_preds))
        except Exception:
            std = abs(pred) * 0.12
        confidence = max(0.1, min(0.95, 1.0 - std / max(1.0, abs(pred))))
        return {"predicted": pred, "confidence": round(confidence, 3), "std": round(std, 4)}

    def shap_values(self, x):
        """Feature-Contributions via batched Leave-One-Out auf Top-20 Features."""
        if np is None:
            return {"base_value": 0.0, "shap_values": {}, "prediction": 0.0}
        x_arr = np.array(x, dtype=np.float64).reshape(1, -1)
        pred = float(self.sklearn_model.predict(x_arr)[0])
        shap_dict = {}
        top_feats = sorted(self.feature_importance_.items(), key=lambda kv: -kv[1])[:20]
        if not top_feats:
            return {"base_value": pred, "shap_values": {}, "prediction": pred}
        try:
            # Batch: 20 modifizierte Kopien auf einmal predicten
            n = len(top_feats)
            x_batch = np.tile(x_arr, (n, 1))  # n Kopien
            feat_indices = []
            for i, (fname, _) in enumerate(top_feats):
                fidx = self.feature_names.index(fname)
                x_batch[i, fidx] = 0.0
                feat_indices.append((fname, i))
            preds_without = self.sklearn_model.predict(x_batch)  # 1 Batch-Call
            for fname, i in feat_indices:
                shap_dict[fname] = pred - float(preds_without[i])
        except Exception:
            for fname, imp in top_feats:
                shap_dict[fname] = imp * 0.5
        return {"base_value": pred, "shap_values": shap_dict, "prediction": pred}

    def feature_importance(self, top_n=20):
        items = sorted(self.feature_importance_.items(), key=lambda x: -x[1])
        return [{"name": k, "importance": v} for k, v in items[:top_n]]

    def to_json(self):
        return {
            "type": "sklearn_GBR",
            "n_trees": len(self.trees),
            "feature_names": self.feature_names,
            "metrics": self.train_metrics,
            "feature_importance": self.feature_importance(20),
            "conformal_radius": getattr(self, "conformal_radius", None),
            "blend_alpha": getattr(self, "blend_alpha", None),
        }


# ── _LGBMModelWrapper ────────────────────────────────────────────────────

class _LGBMModelWrapper:
    """Wrapper um LightGBM-Modell für API-Kompatibilität mit GBRTModel.

    Portiert aus push-balancer-server.py. Wird von gbrt_train() genutzt wenn
    LightGBM verfügbar ist.
    """

    _is_lgbm = True

    def __init__(self, lgbm_model, feature_names):
        self.lgbm_model = lgbm_model
        self.feature_names = list(feature_names)
        self.trees = list(range(getattr(lgbm_model, "n_estimators_", 200)))
        self.train_metrics = {}
        self.conformal_radius = 1.0
        self.blend_alpha = 1.0
        # Feature Importance (normalisiert auf 0–1)
        raw_imp = lgbm_model.feature_importances_
        total = sum(raw_imp) if sum(raw_imp) > 0 else 1
        self.feature_importance_ = {
            f: float(raw_imp[i]) / total
            for i, f in enumerate(feature_names)
        }

    def predict(self, X):
        if np is None:
            return [0.0] * len(X)
        return self.lgbm_model.predict(np.array(X, dtype=np.float64)).tolist()

    def predict_one(self, x):
        if np is None:
            return 0.0
        return float(self.lgbm_model.predict(np.array([x], dtype=np.float64))[0])

    def predict_with_uncertainty(self, x):
        pred = self.predict_one(x)
        # LightGBM: keine Baum-Varianz ohne Quantile-Modell — feste Konfidenz
        return {"predicted": pred, "confidence": 0.7, "std": self.conformal_radius * 0.8}

    def shap_values(self, x):
        """Leave-One-Out Näherung auf Top-20 Features (kein SHAP-Paket nötig)."""
        base = self.predict_one(x)
        top_feat = sorted(self.feature_importance_.items(), key=lambda t: -t[1])[:20]
        contributions = {}
        for fname, _ in top_feat:
            if fname not in self.feature_names:
                continue
            idx = self.feature_names.index(fname)
            x_mod = list(x)
            x_mod[idx] = 0.0
            contributions[fname] = {"shap_values": {fname: round(base - self.predict_one(x_mod), 4)}}
        # Kompatibles Format: {"shap_values": {name: val, ...}}
        flat = {fname: v["shap_values"][fname] for fname, v in contributions.items()}
        return {"shap_values": flat}

    def feature_importance(self, top_n=20):
        return sorted(self.feature_importance_.items(), key=lambda t: -t[1])[:top_n]

    def to_json(self):
        return {
            "type": "lgbm_GBR",
            "n_trees": len(self.trees),
            "feature_names": self.feature_names,
            "metrics": self.train_metrics,
            "feature_importance": [{"name": k, "importance": v}
                                   for k, v in self.feature_importance(20)],
            "conformal_radius": getattr(self, "conformal_radius", None),
            "blend_alpha": getattr(self, "blend_alpha", None),
        }


# ── Isotonic Regression (PAVA) fuer Kalibrierung ────────────────────────

def _isotonic_regression_pava(predicted, actual):
    """Pool Adjacent Violators Algorithm fuer Isotonische Regression.

    Args:
        predicted: Sortierte Predictions (Liste von Floats)
        actual: Zugehoerige Actual-Werte (Liste von Floats)
    Returns:
        calibrated: Kalibrierte Werte (Liste von Floats)
    """
    n = len(predicted)
    if n == 0:
        return []

    # Sortiere nach predicted
    paired = sorted(zip(predicted, actual), key=lambda x: x[0])
    y = [p[1] for p in paired]

    # PAVA: Pool Adjacent Violators
    blocks = [[i] for i in range(n)]
    block_avg = [y[i] for i in range(n)]

    i = 0
    while i < len(blocks) - 1:
        if block_avg[i] > block_avg[i + 1]:
            # Merge blocks
            merged = blocks[i] + blocks[i + 1]
            merged_avg = sum(y[j] for j in merged) / len(merged)
            blocks[i] = merged
            block_avg[i] = merged_avg
            del blocks[i + 1]
            del block_avg[i + 1]
            # Gehe zurueck um weitere Violators zu finden
            if i > 0:
                i -= 1
        else:
            i += 1

    # Ergebnis zurueckschreiben
    result = [0.0] * n
    for block, avg in zip(blocks, block_avg):
        for idx in block:
            result[idx] = avg

    return result


class IsotonicCalibrator:
    """Isotonische Regression fuer Post-Processing-Kalibrierung."""

    def __init__(self):
        self.breakpoints = []  # [(predicted, calibrated), ...]

    def fit(self, predicted, actual):
        """Trainiert den Kalibrator."""
        if len(predicted) < 10:
            return

        # Sortiere nach predicted
        paired = sorted(zip(predicted, actual), key=lambda x: x[0])
        preds = [p[0] for p in paired]
        acts = [p[1] for p in paired]

        calibrated = _isotonic_regression_pava(preds, acts)

        # Breakpoints extrahieren (alle einzigartigen Stufen)
        self.breakpoints = []
        prev_cal = None
        for pred, cal in zip(preds, calibrated):
            if prev_cal is None or abs(cal - prev_cal) > 0.001:
                self.breakpoints.append((pred, cal))
                prev_cal = cal
        # Letzten Punkt sicherstellen
        if preds:
            self.breakpoints.append((preds[-1], calibrated[-1]))

    def calibrate(self, predicted):
        """Kalibriert einen einzelnen Predicted-Wert."""
        if not self.breakpoints:
            return predicted

        # Lineare Interpolation zwischen Breakpoints
        if predicted <= self.breakpoints[0][0]:
            return self.breakpoints[0][1]
        if predicted >= self.breakpoints[-1][0]:
            return self.breakpoints[-1][1]

        for i in range(len(self.breakpoints) - 1):
            p1, c1 = self.breakpoints[i]
            p2, c2 = self.breakpoints[i + 1]
            if p1 <= predicted <= p2:
                if abs(p2 - p1) < 1e-10:
                    return c1
                t = (predicted - p1) / (p2 - p1)
                return c1 + t * (c2 - c1)

        return predicted

    def to_dict(self):
        return {"breakpoints": [(round(p, 4), round(c, 4)) for p, c in self.breakpoints]}

    @staticmethod
    def from_dict(d):
        cal = IsotonicCalibrator()
        cal.breakpoints = [(p, c) for p, c in d.get("breakpoints", [])]
        return cal


# ── CharNGramTFIDF ───────────────────────────────────────────────────────

class CharNGramTFIDF:
    """Character N-Gram TF-IDF fuer semantische Titel-Aehnlichkeit ohne LLM.

    Faengt morphologische Varianten: Kanzler/Bundeskanzler, Transfer/Transfergeruecht.
    """

    def __init__(self, n_range=(2, 5), max_features=5000):
        self.n_range = n_range
        self.max_features = max_features
        self.vocab = {}        # ngram → index
        self.idf = {}          # ngram → idf score
        self.n_docs = 0

    def _extract_ngrams(self, text):
        """Extrahiert Character N-Grams aus einem Text."""
        text = text.lower().strip()
        ngrams = defaultdict(int)
        for n in range(self.n_range[0], self.n_range[1] + 1):
            for i in range(len(text) - n + 1):
                ng = text[i:i + n]
                ngrams[ng] += 1
        return ngrams

    def fit(self, documents):
        """Trainiert IDF auf einer Liste von Dokumenten (Titeln)."""
        self.n_docs = len(documents)
        if self.n_docs == 0:
            return

        # Document Frequency zaehlen
        df = defaultdict(int)
        for doc in documents:
            ngrams = self._extract_ngrams(doc)
            for ng in ngrams:
                df[ng] += 1

        # Top-Features nach DF sortiert (nicht zu selten, nicht zu haeufig)
        min_df = max(2, self.n_docs * 0.001)
        max_df = self.n_docs * 0.8
        filtered = {ng: count for ng, count in df.items()
                    if min_df <= count <= max_df}

        # Top max_features nach DF
        sorted_ngrams = sorted(filtered.items(), key=lambda x: -x[1])[:self.max_features]
        self.vocab = {ng: idx for idx, (ng, _) in enumerate(sorted_ngrams)}

        # IDF berechnen
        self.idf = {}
        for ng, idx in self.vocab.items():
            self.idf[ng] = math.log(self.n_docs / (df[ng] + 1)) + 1

    def transform_one(self, text):
        """Transformiert einen Text in einen TF-IDF-Vektor (sparse dict)."""
        ngrams = self._extract_ngrams(text)
        vec = {}
        norm = 0.0
        for ng, count in ngrams.items():
            if ng in self.vocab:
                tf = 1 + math.log(count) if count > 0 else 0
                tfidf = tf * self.idf.get(ng, 0)
                vec[self.vocab[ng]] = tfidf
                norm += tfidf * tfidf
        # L2-Normalisierung
        if norm > 0:
            norm = math.sqrt(norm)
            vec = {k: v / norm for k, v in vec.items()}
        return vec

    def cosine_similarity(self, vec1, vec2):
        """Cosine Similarity zwischen zwei sparse Vektoren."""
        common = set(vec1.keys()) & set(vec2.keys())
        if not common:
            return 0.0
        return sum(vec1[k] * vec2[k] for k in common)

    def to_dict(self):
        return {"vocab": self.vocab, "idf": self.idf, "n_docs": self.n_docs,
                "n_range": self.n_range, "max_features": self.max_features}

    @staticmethod
    def from_dict(d):
        tfidf = CharNGramTFIDF(n_range=tuple(d.get("n_range", (2, 5))),
                                max_features=d.get("max_features", 5000))
        tfidf.vocab = d.get("vocab", {})
        tfidf.idf = d.get("idf", {})
        tfidf.n_docs = d.get("n_docs", 0)
        return tfidf
