
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
from sklearn.preprocessing import MinMaxScaler, label_binarize


def run_model(file, target):
    print(f"\n{'='*40}\n{file}")
    df = pd.read_csv(file).dropna()
    X = pd.get_dummies(df.drop(columns=[target]))
    y = df[target]
    classes = sorted(y.unique())

    X_scaled = MinMaxScaler().fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    model = LogisticRegression(max_iter=2000, class_weight='balanced')
    model.fit(Xtr, ytr)
    yp = model.predict(Xte)

    yb = label_binarize(yte, classes=classes)
    if yb.shape[1] == 1: yb = np.hstack([1 - yb, yb])
    prauc = np.mean([average_precision_score(yb[:, i], model.predict_proba(Xte)[:, i])
                     for i in range(yb.shape[1]) if yb[:, i].sum() > 0])

    print(f"Accuracy : {accuracy_score(yte, yp):.4f}")
    print(f"F1 Score : {f1_score(yte, yp, average='weighted'):.4f}")
    print(f"PR-AUC   : {prauc:.4f}")


run_model("data.csv", "label")
run_model("Fertilizer_Prediction.csv", "Fertilizer Name")
