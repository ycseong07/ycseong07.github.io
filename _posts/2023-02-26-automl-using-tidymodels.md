---
layout: post
title: Tidymodels로 AutoML을 구현해보자 (1)
description: 일반인도 사용할 수 있는 AutoML 패키지(stove) 제작에 관한 글입니다. 프로젝트는 아직 진행 중이며, 그간 고민했던 것들과 앞으로 해결해야할 것들을 정리하고, 공유하기 위해 글로 남깁니다.
---

# 간단한 AutoML 아키텍처 만들어보기

약 9개월 전, 처음 접한 "AutoML"이라는 단어는 마치 마법처럼 들렸다. 데이터만 준비하면 ML모델을 자동으로 만들어 비교해 준다는데, 당장 써봐야겠다 싶었다. 당시에는 ML 자체에 대한 지식도 깊지 않아 Optuna나 AutoGluon 같은 저수준의 라이브러리를 사용해 볼 엄두가 나지 않았기 때문에, AWS 클라우드의 SageMaker Autopilot을 사용해보기로 했다. 그리고 샘플 데이터를 넣어보니 신세계가 펼쳐졌다. 데이터 전처리나 하이퍼파라미터 튜닝에 그렇게나 공을 들여가며 아등바등했던 ML 모델링인데, SageMaker는 짧은 시간 안에 여러 모델들을 테스트 한 결과를 비교해주고, 어떤 방식으로 모델링을 진행했는지에 대해서도 보고서 형태로 제공해주었다.

아직 석사생일 때, 연구실 친구들과 서로의 힘듦을 토로하던 순간이 있었다. 당시의 나는 연구를 진행하는 프로세스 자체에 익숙하지 않았고, 연구주제를 결정하는 것도 힘들어 했기 때문에, 이렇게 넋두리를 했다.

> "논문 아이디어 얻으려고 일일이 분석하는 거 너무 귀찮은데, 데이터만 넣으면 자기가 알아서 상관분석, 회귀분석 쫙 돌리고 유의미해보이는 결과만 추려주는 프로그램 같은거 없나?"

당시에는 맥주와 함께 대수롭지 않게 오가는 헛소리 중 하나 쯤으로 생각했던 것 같은데, 몇 년 지나지 않아 비슷한 아이디어가 실현된 것이다. ML 모델을 만드는 연구라면, 일단 데이터를 전처리한 후 SageMaker를 사용해봐도 좋을 것 같았다. 

몇 년이 지나, R을 사용해 AutoML 패키지를 만들어볼 수 있는 기회가 생겼다. 기본적으로는 오픈소스 프레임워크인 Tidymodels를 골자로 구현하고자 했고, 일반인이 사용할 수 있을 정도로 사용하기 쉽게 만들자는 것이 큰 골자였다. Tidymodels의 [agua](https://agua.tidymodels.org/articles/auto_ml.html) 패키지를 사용하면 간단한 AutoML 구현이 가능하기는 하지만, 교차검증을 하지 못한다는 것이 영 아쉬웠다. 간단하게 ML 모델들의 성능을 비교하는 것이 목적일지라도, 교차검증과 하이퍼파라미터 최적화, 결과의 시각화는 가장 우선순위가 높은 구현 대상으로 삼았다.

# 전체적인 구조

<div align="center" class="image-with-caption">
  <figure>
    <img src="/assets/img/illustration/2023-02-26_1.png" alt="image description">
    <figcaption>stove의 모델링 과정</figcaption>
  </figure>
</div>


패키지는 정형데이터에 대해 분류/회귀 모델링을 지원하도록 만들고자 했다. 본 패키지는 오픈소스 프로젝트인 [Statgarten](https://www.r-bloggers.com/2023/03/introduction-to-data-analysis-with-statgarten/)의 일부이며, global preprocessing 이후의 데이터를 다룬다. 본 글에서는 교차검증을 수행하면서 훈련 셋(1)을 다시 훈련(2)/검증 셋으로 나눌 때, (1)에 적용되는 전처리를 global preprocessing, (2)에 적용되는 전처리를 local preprocessing으로 정의했다 (data leakage 고려). Train-Test Split - Recipe 정의 - 모델링 - 결과 출력의 4단계 과정을 통해 진행되도록 설계했다. 그림의 global preprocessing에서는 아래와 같은 전처리를 수행하는 것으로 간주했다.

  - Normalization / Standardization
  - Feature selection
  - 범주형 변수의 경우 One-hot encoding 되어있을 것 (여러 모델에 호환되는 데이터가 필요하므로)
  - String type 값이 있는 열은 Factor type으로 변환되어 있을 것
  - 한 열이 모두 같은 값으로 채워져 있을 경우 제외할 것
  - Date type Column은 제외할 것
  - Target Variable은 분류의 경우 Factor, 회귀의 경우 Numeric type일 것

## 1. Train-Test Split

데이터 분할 단계에서는 사용자가 아래의 설정을 할 수 있도록 구현했다. 

  - Target Variable로 지정할 열 이름
  - 전체 데이터 중 Train set의 비율을 어느 정도로 할지
  - 재현가능한 결과를 위한 Seed 값
  - Target Variable을 예측하기 위해 Feature로 사용할 변수

## 2. 전처리 방법 정의

Tidymodels를 사용해 머신러닝 모델을 만들 때 인상적이었던 것 중 하나는 `recipes`라는 패키지였다. 데이터 전처리 방법을 정의해두는 것을 요리 레시피에 비유한 것인데, 꽤 센스있는 네이밍이라고 생각한다. Under/Over Sampling, Imputation, Scaling 등의 전처리는 Local preprocessing 단계에서 수행하며, 교차검증을 위해 분리한 샘플 각각에 적용되어야 한다. recipes 패키지는 이러한 Local preprocessing 방법을 정의하는 recipe를 생성할 수 있도록 도와준다. 이 단계에서는 현재 다음과 같은 기능을 정의할 수 있도록 구현한 상태이다. 데이터 불균형에 대처하기 위한 언더/오버샘플링 역시 이 부분에서 수행되어야 하지만, 아직 구현하지 못한 상태다.  

  - Imputation 수행 여부
  - Scaling 수행 여부
  - Imputation 수행 시 categorical/numeric 값에 대한 imputation 방법
  - Scaling 수행 시 Scaling 방법

## 3. 모델링

모델링 부분에서는 모델 정의, 교차검증을 통한 베이지안 최적화, 모델 피팅의 작업을 수행한다. 모델 정의 부분에서는 어떤 알고리즘과 엔진을 사용할지, 어떤 하이퍼파라미터를 최적화 대상으로 할지를 정의한다. 교차검증을 통한 베이지안 최적화 부분에서는 우선 교차검증을 위해 Train set을 분류하고, 각 fold에서 베이지안 최적화를 통해 하이퍼파라미터를 탐색한다. 모델 피팅 부분에서는 가장 평가지표가 좋았던 모델을 선정해 피팅하고, Test set의 Target Variable을 예측한다. 

모델링을 지원하는 알고리즘은 아래와 같다. 

  - 분류: Logistic Regression, K-Nearest Neighbor, Naive Bayes, Decision Tree, Random Forest, XGBoost, LightGBM, MLP, SVM
  - 회귀: Linear Regression, K-Nearest Neighbor, Decision Tree, Random Forest, XGBoost, LightGBM, MLP, SVM

사용자가 지정해야하는 사항은 아래와 같다.

  -  분류/회귀 문제 지정
  - 특정 알고리즘을 통해 모델링 할 때 사용할 engine 선택 (예를 들어, Tidymodels에서는 logistic regression의 경우 glm, brulee, glmer, glmnet 등을 engine으로 사용할 수 있다)
  - 교차검증 시 Train set을 분할하는 횟수
  - 베이지안 최적화 시 동시에 테스트할 하이퍼파라미터의 조합 개수 및 최대 서치 반복 횟수
  - 모델의 성능을 평가할 지표

## 4. 결과 출력

최종 모델이 결정되면, 해당 모델의 성능을 가늠하기 위해 아래와 같은 결과를 표와 그래프로써 시각화한다. 평가지표 비교표에서는, 분류모델의 경우 Accuracy, Recall, Specificity, Precision, F1-score, Kappa, MCC 값을 출력하며, 회귀모델의 경우 RMSE, RSQ, MAE, MASE, RPD 값을 출력한다. 

 - 분류: 평가지표 비교표, ROC Curve Plot, Confusion Matrix
 - 회귀: 평가지표 비교표, Regression Plot

# 구현하지 못한 것들

이 패키지는 콘솔 환경에서도 동작 가능하도록 만들어졌지만, 기본적으로는 R Shiny App에 Embedding시킬 목적으로 제작 중이기 때문에 단독으로 사용하도록 만들어진 패키지와 비교하면 지저분한 부분이 많다. 또한 개인적인 역량부족과 더불어 제작할 수 있었던 시간이 상대적으로 짧았던 탓에 아직 디테일한 부분을 많이 신경쓰지 못했고, 아직 미구현된 부분이 많다. 예를 들어, 

  - 모든 함수에는 기본 seed값이 정해져 있는데, 사용 목적에 따라 지정하지 않고 싶을 수도 있을 것 같다. 
  - 현재는 교차검증 시의 전처리방법이 Imputation, Normalization에 한해서만 설정할 수 있다. 
  - 회귀모델의 결과 출력의 경우, RMSE를 비교 플롯을 아직 구현하지 못했다. 

 그 외에도 부족한 점이 많지만, 앞으로 하나씩 보완해나가며 완성해 나가고자 한다.

<p align="center">
  <a href="https://github.com/statgarten/stove">
    <img src="/assets/img/portfolio/stove_logo.png" height="50%" width="50%">
  </a>
</p>
