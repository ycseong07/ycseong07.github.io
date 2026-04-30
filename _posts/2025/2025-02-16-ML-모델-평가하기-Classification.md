---
layout: post
title: "ML 모델 평가하기 (Classification)"
date: 2025-02-16 09:00:00 +0900
category: blog
tags: []
---

훈련이 완료된 머신러닝 모델의 성능을 평가할 때 단순히 validation set, test set 성능으로만 판단하기엔 부족한 부분이 있습니다. 이번 글에서는 분류 모델이 데이터의 의미 있는 패턴을 학습했는지 확인하거나, 우연한 상관관계에 의한 결과인건지, 특정 데이터 분포에만 특화되어 있는지 등을 검증하기 위한 방법을 소개합니다. 아래는 예제 코드를 위한 라이브러리 리스트입니다.

```python
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import joblib
import warnings

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, confusion_matrix
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.base import clone

from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

# model, X_train, y_train, X_test, y_test 변수는 이미 정의된 상태라고 가정
```

# 1. Y Randomization

Y Randomization은 모델이 학습한 패턴이 진짜 의미 있는 것인지, 아니면 우연히 얻어진 상관관계에 의한 것인지 판별하고자 할 때 사용합니다. 통계에 익숙하신 분들은 p-value를 통한 검증방법과 비슷하다고 생각하시면 감이 쉽게 잡힐 수도 있습니다(0.05와 같은 rule of thumb이 있는 것은 아니지만). 예측해야 할 target 변수를 무작위로 뒤섞은 뒤에도 비슷한 수준의 정확도나 AUC가 나온다면, 이는 모델이 진짜 패턴이 아니라 우연적 상관관계를 학습했을 가능성이 크다는 뜻입니다. 반대로 뒤섞은 데이터에서는 성능이 크게 떨어지고, 원본 데이터에서만 높은 성능을 보인다면 모델이 의미 있는 정보를 찾아냈다고 볼 수 있습니다.

```python
def y_randomization(model, X_train, y_train, X_test, y_test, n_randomizations=30, random_seed=42):
    np.random.seed(random_seed)
    
    original_model = clone(model)
    original_model.fit(X_train, y_train)
    
    y_pred_proba = original_model.predict_proba(X_test)[:, 1]
    
    original_auc = roc_auc_score(y_test, y_pred_proba)
    
    random_aucs = []
    for i in range(n_randomizations):
        y_train_random = np.random.permutation(y_train)
        model_random = clone(model)
        model_random.fit(X_train, y_train_random)

        y_pred_random_proba = model_random.predict_proba(X_test)[:, 1]
        
        score = roc_auc_score(y_test, y_pred_random_proba)
        random_aucs.append(score)
    
    plt.figure(figsize=(6, 4))
    sns.histplot(random_aucs, kde=True, color="skyblue")
    plt.axvline(original_auc, color="red", linestyle="--", label="Original AUC")
    plt.title("AUC Distribution")
    plt.xlabel("AUC")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return original_auc, random_aucs

orig_auc, rand_aucs = y_randomization(model, X_train, y_train, X_test, y_test, n_randomizations=100, random_seed=1234)
```

# 2. Adversarial Validation

Adversarial Validation은 훈련 데이터와 테스트 데이터가 비슷한 분포를 가지고 있는지 확인하기 위해, 가벼운 이진 분류 모델을 사용해보는 방법입니다. 만약 분류기의 AUC가 0.5에 가깝게 나온다면, 이 둘을 구별하기가 어렵다는 뜻이므로 학습/테스트 데이터 간 분포 차이가 작고, 검증셋의 평가와 테스트셋의 평가 간 차이가 크지 않을 것이라고 예상해볼 수 있습니다. 반면 학습 데이터와 테스트 데이터가 눈에 띄게 다른 경우, 모델이 테스트 환경에서 기대만큼 좋은 성능을 내지 못할 수 있음을 의미합니다. 

```python
def adversarial_validation(X_train, X_test, n_splits = 5, seed = 42):
    X_train_np = X_train.values
    X_test_np = X_test.values

    X_adv = np.concatenate([X_train_np, X_test_np], axis=0)
    y_adv = np.concatenate([np.zeros(X_train_np.shape[0]),np.ones(X_test_np.shape[0])])

    adv_model = HistGradientBoostingClassifier(max_iter=1000, random_state=seed) # 결측치 허용하는 모델
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = cross_val_score(adv_model, X_adv, y_adv, cv=skf, scoring="roc_auc")

    return {"fold_aucs": aucs, "adversarial_auc": float(np.mean(aucs))}

adv_val_result = adversarial_validation(X_train, X_test, n_splits=3, seed=1234)
fold_aucs = adv_val_result["fold_aucs"]
adversarial_auc_mean = adv_val_result["adversarial_auc"]

plt.figure(figsize=(6, 4))
sns.boxplot(y=fold_aucs, color="lightblue")
sns.stripplot(y=fold_aucs, color="red", alpha=0.7)
plt.title(f"Mean AUC: {adversarial_auc_mean:.4f}")
plt.ylabel("AUC Score")
plt.tight_layout()
plt.show()

```

# 3. Perturbation Test

Perturbation Test는 모델의 강건성(Robustness)을 확인하는 데 초점을 둡니다. 모델을 훈련시킬 때는 데이터를 많이 정제하지만, 테스트 환경에서는 데이터에 노이즈가 섞여 들어오는 경우가 훨씬 많습니다. 이때 모델이 입력 특성에 약간의 변동만 있어도 성능이 크게 떨어진다면, 특정 데이터나 패턴에 과도하게 의존하고 있을 수 있다는 뜻일 수 있습니다. 따라서 각 컬럼에 일정 비율(예시 코드에서는 10%)로 무작위 노이즈를 섞어 넣어 보고, 그 전후로 모델 성능이 얼마나 변하는지를 살펴볼 수 있습니다. 성능이 상대적으로 안정적으로 유지된다면, 모델이 다양한 상황에서 견고하게 동작할 가능성이 높다고 판단할 수 있습니다.

```python
def raw_perturbation(X, perturb_size=0.1):
    X_perturbed = X.copy()
    for col in X.columns:
        if np.issubdtype(X[col].dtype, np.number):
            std = X[col].std()
            noise = np.random.normal(0, perturb_size * std, size=X.shape[0])
            X_perturbed[col] += noise
    return X_perturbed

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    recall = recall_score(y, y_pred)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return {'accuracy': accuracy, 'f1': f1, 'auc': auc, 'recall': recall, 'specificity': specificity}

def evaluate_models(models, model_names, X_test, y_test, perturb_func, perturb_size=0.1, n_iterations=10, random_seed=42):
    results = {name: {'accuracy': [], 'f1': [], 'auc': [], 'recall': [], 'specificity': []} for name in model_names}
    for i in range(n_iterations):
        np.random.seed(random_seed + i)
        X_test_perturbed = perturb_func(X_test, perturb_size=perturb_size)
        for model, name in zip(models, model_names):
            metrics = evaluate_model(model, X_test_perturbed, y_test)
            for metric_name, value in metrics.items():
                results[name][metric_name].append(value)
    return results
    
    
results_raw = evaluate_models([model], ['Model'], X_test, y_test, raw_perturbation, perturb_size=0.1, n_iterations=10, random_seed=1234)

for metric, values in results_raw['Model'].items():
    print(f"{metric}: mean={np.mean(values):.4f}, std={np.std(values):.4f}")

plt.figure(figsize=(6, 4))
sns.histplot(results_raw['Model']['accuracy'], binwidth=0.05, kde=True, color="lightblue")
plt.title("Accuracy Distribution")
plt.xlabel("Accuracy")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
```

# 4. Permutation Test

Permutation Test는 모델에서 각 특성이 예측에 얼마나 기여하고 있는지를 직관적으로 살펴보는 방법입니다. 일반적으로 트리 기반 모델에서만 feature importance를 체크할 수 있다고 아시는 분들도 있지만, 이 방법을 사용하면 트리기반 모델이 아니더라도 비슷하게 feature importance를 추출해볼 수 있습니다. 한 번에 하나의 열을 무작위로 섞은 다음, 해당 열이 섞이기 전후로 모델의 예측 성능이 얼마나 떨어지는지를 측정합니다. 이렇게 모든 열에 대해 성능 변화를 측정해 보면, 성능 저하 폭이 큰 열일수록 모델이 해당 특성에 크게 의존하고 있음을 추측할 수 있습니다. 반대로 어떤 열을 섞어도 성능 차이가 거의 없다면, 해당 열은 모델이 예측하는 데 그다지 중요한 정보가 아니라고 볼 수 있습니다. 모든 열에 대해 이 작업을 수행해야하기 때문에 일반적으로 리소스가 많이 들기 때문에, 아래의 예제 코드는 병렬처리를 가능하게 한 코드로 작성했습니다.

```python
def compute_feature_importance(model, X, y, col, baseline_score, n_repeats, random_seed):
    np.random.seed(random_seed)
    
    score_drops = []
    for i in range(n_repeats):
        X_permuted = X.copy()
        X_permuted[col] = np.random.permutation(X_permuted[col].values)
        
        y_pred_proba = model.predict_proba(X_permuted)[:, 1]
        score = roc_auc_score(y, y_pred_proba)
        score_drops.append(baseline_score - score)

    return col, np.mean(score_drops)

def permutation_test_feature_importance(model, X, y, n_repeats=30, random_seed=42, n_jobs=-1):
    np.random.seed(random_seed)
    
    y_pred_proba = model.predict_proba(X)[:, 1]
    baseline_score = roc_auc_score(y, y_pred_proba)
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_feature_importance)(
            model, X, y, col, baseline_score, n_repeats, random_seed
        ) 
        for col in X.columns
    )
    
    feature_importances = {}
    for col, drop in results:
        feature_importances[col] = drop
    
    return {"baseline_score": baseline_score, "feature_importances": feature_importances}

perm_test_result = permutation_test_feature_importance(
    model=model,
    X=X_test,
    y=y_test,
    n_repeats=100,
    random_seed=RAND,
    n_jobs=-1
)

baseline_score_pt = perm_test_result["baseline_score"]
feature_importances = perm_test_result["feature_importances"]

print(f"Baseline AUC: {baseline_score_pt:.4f}")

fi_df = pd.DataFrame(list(feature_importances.items()), columns=["feature", "importance_drop"])
fi_df.sort_values("importance_drop", ascending=False, inplace=True)

top_n = 20
fi_top = fi_df.head(top_n)

plt.figure(figsize=(8, 6))
sns.barplot(x="importance_drop", y="feature", data=fi_top, palette="Blues_r")
plt.title(f"Permutation Feature Importance (Top {top_n})")
plt.xlabel("Mean AUC Drop")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
```

실제로 사용할 때는 데이터 형태나 모델 구조에 따라 수정이 필요하겠지만, 전반적인 접근 방식을 파악하는 데는 도움이 되길 바랍니다.