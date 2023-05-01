---
layout: post
title: Cross Validation과 Data Leakage
tags: [ML]
description: stove 패키지를 개발하면서 왜 굳이 전처리를 교차검증 전후로 나눠서 2번이나 해야하냐는 질문을 많이 받아서, 데이터 누수(Data Leakage) 문제에 대해 정리해봤습니다. 
---
# Data Leakage는 무엇이고, 언제 많이 발생할까?

데이터 누수(Data Leakeage)는 `훈련 데이터 외의 정보가 모델 훈련 과정에 흘러들어오는 것`을 의미한다. 처음 이 개념을 접했을 때는 '도대체 얼마나 모델 훈련 설계를 대충하면 이런 문제가 일어나는걸까'하고 생각했지만, 막상 관련 문제를 찾다보니 여러 논문이 해당 문제로 인해 Reject 되었다는 것, 그리고 이 문제가 Kaggle competition에서도 빈번하게 일어나는 문제라는 것을 알게 되었다. stove 패키지 개발 과정에서도 다음과 같은 질문을 많이 받았다.

> 왜 데이터 분할 전후로 전처리를 두 번이나 해요?

<div align="center" class="image-with-caption">
  <figure>
    <img src="/assets/img/illustration/2023-02-26_1.png" alt="image description">
    <figcaption>stove의 모델링 과정</figcaption>
  </figure>
</div>

그림은 stove 패키지의 전체적인 흐름이다. 데이터 분할 이전의 전처리는 `global preprocessing`, 이후의 전처리는 `local preprocessing`이라는 이름을 붙여 구분했더니, 위와 같은 질문을 몇 번인가 받았던 기억이 있다. 

먼저 위 질문에 대한 대답은, 데이터 누수 문제를 막기 위해서이다. 그리고 처음 언급했던 정의에서, 내 경우(그리고 대부분의 경우) `훈련 데이터 외의 정보`는 **검증 데이터의 정보**를 의미한다. 
ML모델을 훈련시키는 목적은 현재(혹은 기존)의 데이터를 활용해 미래(혹은 새로운)의 데이터를 예측하기 위함이다. 그러나 현재의 데이터를 사용해 모델링을 완료했을 때, 그 모델이 미래의 데이터에 대해 얼마나 잘 작동할지는 알 수 없다. 모델의 성능을 판단하기 위해 우리는 현재 데이터를 훈련 데이터와 테스트 데이터로 나누는 방법을 사용하게 되었다. 

그러나 이 방법은 테스트 데이터에 대해서만 좋은 성능을 보이는 모델로 훈련되는 문제, 즉 과적합 문제를 빈번하게 발생시켰다. 이는 훈련 프로세스 자체의 문제로 볼 수 있다. 그리고 이러한 과적합 문제를 막기 위해 교차 검증(Cross Validation)이라는 방법을 사용하게 되었다. 

<div align="center" class="image-with-caption">
  <figure>
    <img src="/assets/img/illustration/2023-05-01.png" alt="image description">
    <figcaption>데이터 누수 문제는 교차검증이 일반화되면서 빈번하게 발생하기 시작했다</figcaption>
  </figure>
</div>

교차 검증을 포함하는 모델 훈련 프로세스에서, 우리는 훈련 데이터(1)를 다시 한 번 훈련 데이터(2)와 검증 데이터(Validation Data)로 나누게 된다. 나는 이 지점이 데이터 누수 문제를 빈번하게 발생시키는 지점이 아닐까 한다. 데이터 누수 문제는 교차 검증이라는 프로세스 자체가 발생시키는 문제라기 보다는, 분석가의 혼란 때문에 발생하는 것 같다. 혼란을 일으킬만 한 부분은,

- '훈련 데이터'라는 용어가 (1), (2)에 공통으로 쓰이는 부분. 이는 분석가가 전처리를 '훈련 데이터'에 적용할 때 (2)가 아닌 (1)에 적용할 여지를 제공한다. 
- 많은 글들에서 교차검증을 수행함으로써 데이터 누수를 극복할 수 있다고 서술하고 있는 점도 분석가들을 부주의하게 만든다. 

예를 들어 Imputation이나 Oversampling 등을 (1)에 적용한 후 교차 검증을 수행하게 되면, 이미 Imputation이나 Oversampling을 통해 생겨난 값들은 검증 데이터의 정보를 통해 만들어진 값들이기 때문에 훈련 과정에서 데이터 누수를 발생시키고, 검증 결과에서는 성능이 좋았는데, 막상 테스트 데이터에 대해서는 예측을 잘 하지 못하는 모습을 보일 수 있다. 

이를 방지하기 위해 stove 패키지에서는 `recipes`패키지를 사용해 전처리 프로세스를 따로 관리한다. 파이썬의 경우 `Pipeline`을 사용해 동일한 구조를 만들 수 있다. 

정리하면, 훈련 데이터를 k 개의 fold로 나누기 이전과 이후의 전처리 방법은 **값을 변형할 때 다른 값이나 데이터의 분산 등을 사용하는지 여부**로 구분할 수 있다. 

- 다른 값이나 데이터의 분산 등을 사용하지 않는 전처리: 중복값 제거, 원-핫 인코딩, 피처 선택 등  
-> **global preprocessing**
- 다른 값이나 데이터의 분산 등을 사용하는 전처리: Imputation, Scaling, Oversampling 등  
-> **local preprocessing**

문득 기억나는 내용은, Kaggle Competition에서 훈련 데이터 내 값들을 연산해 새로운 피처를 만들어 사용했을 때 데이터 누수 이슈로 질문을 받았다는 내용이 있었는데, 이 경우에도 새로운 피처 생성은 local preprocessing 단계에서 수행해야 했을 것이다. 

 <details>
<summary>▶ References</summary>
<div markdown="1">
- https://towardsdatascience.com/k-fold-cross-validation-are-you-doing-it-right-e98cdf3e6690
- https://www.nature.com/articles/s41597-022-01618-6
- https://towardsdatascience.com/how-to-avoid-data-leakage-while-evaluating-the-performance-of-a-machine-learning-model-ac30f2bb8586
- https://alexforrest.github.io/you-might-be-leaking-data-even-if-you-cross-validate.html
- https://ieeexplore.ieee.org/document/8492368
</div>
</details>