---
layout: post
title: Tidymodels로 AutoML을 구현해보자 (3)
description: 일반인도 사용할 수 있는 AutoML 패키지(stove) 제작에 관한 글입니다. workflow 패키지를 통한 모델링 프로세스 관리, 베이지안 최적화를 다룹니다.
---
# workflows 패키지로 모델링 과정을 관리해보자

workflow 패키지는 ML 모델을 훈련시키는 프로세스를 관리하기 위한 용도로 쓰이며, 파이썬의 `pipeline` 패키지와 비슷하게 쓸 수 있다. 주로는 아래와 같은 목적을 위해 사용하는 것 같다.
- 모델마다 다른 workflow를 적용하고 싶을 때
- 교차검증을 적용해 훈련할 때

우선 첫 번째 경우에 대해 알아보자. 가령 로지스틱회귀는 피처와 타겟의 관계가 선형적이라고  가정하기 때문에 전처리 과정에서 표준화를 적용하는 것이 좋은 선택일 수 있다. 하지만 의사결정트리의 경우에는 정보이득비(information gain ratio)에 따라 노드를 분할하며,  정규화는 정보이득비를 바꾸지 않기 때문에 표준화를 적용할 필요가 없다. 즉, 서로 다른 특징을 가진 모델에 대해 다른 전처리 방법을 적용해야 한다는 것이다. 

먼저, RF 모델을 workflows 함수를 활용해 훈련시키는 과정은 아래와 같다.  RF 모델에 step_log()를 꼭 적용해야하는 것은 아니고, 다만 이전에 전처리가 거의 필요없는 샘플 데이터를 가져왔었기 때문에 recipe를 적용하는 과정을 보여주기 위해 포함시켰다.
```r
# data import 과정은 생략합니다
# Random Forest 모델을 위한 전처리 recipe
preprocessing_recipe <- recipes::recipe(STK ~ ., data = cleaned_data) %>%
  recipes::step_log(TG)

data_preprocessed <- recipes::prep(preprocessing_recipe, training = cleaned_data) %>%
  recipes::bake(new_data = cleaned_data)

# 데이터 분할
set.seed(1234)
dataSplit <- rsample::initial_split(data_preprocessed,
                                    strata = tidyselect::all_of("STK"),
                                    prop = 0.7)

train <- rsample::training(dataSplit)
test <- rsample::testing(dataSplit)

# Random Forest 모델 정의
model_spec <- parsnip::rand_forest() %>%
  parsnip::set_engine("ranger") %>%
  parsnip::set_mode("classification")

# workflow 정의
workflow <- workflows::workflow() %>%
  workflows::add_model(model_spec)

# 모델 fitting
trained_workflow <- workflow %>%
  fit(data = train)

predictions <- trained_workflow %>%
  stats::predict(test, type = "prob")
```

그러나 만약 로지스틱 회귀모델이라면 아래와 같이 상관관계가 너무 높은 피처들을 제외하고, 정규화를 적용하는 레시피를 적용할 수 있다. 

```r
# Logistic Regression 모델을 위한 전처리 recipe
preprocessing_recipe_lm <- recipes::recipe(STK ~ ., data = cleaned_data) %>%
  recipes::step_corr(recipes::all_numeric(), threshold = 0.8) %>%
  recipes::step_normalize(recipes::all_numeric())

data_preprocessed_lm <- recipes::prep(preprocessing_recipe, training = cleaned_data) %>%
  recipes::bake(new_data = cleaned_data)

# Logistic Regression 모델 정의
model_spec <- parsnip::logistic_reg() %>%
  parsnip::set_engine("lr") %>%
  parsnip::set_mode("classification")
```

# Workflow + Bayesian Optimization

두 번째 경우, 즉 교차검증을 수행하는 훈련 과정에 쓰이는 경우를 알아보자. stove 패키지 개발에서 가장 중요하게 생각했던 것은 사용자가 모델의 하이퍼파라미터를 지정할 필요가 없도록 만드는 것이었다. 이를 위해 베이지안 최적화를 적용하고자 했으며, 이를 사용한 예시를 통해 보여주고자 한다. 

```r
# data import 과정은 생략합니다
# 데이터 전처리 (훈련, 테스트 셋 모두 적용)
preprocessing_recipe <- recipes::recipe(STK ~ ., data = cleaned_data) %>%
  recipes::step_log(TG)

data_preprocessed <- recipes::prep(preprocessing_recipe, training = cleaned_data) %>%
  recipes::bake(new_data = cleaned_data)

# 데이터 분할
set.seed(1234)
dataSplit <- rsample::initial_split(data_preprocessed,
                                    strata = tidyselect::all_of("STK"),
                                    prop = 0.7)

train <- rsample::training(dataSplit)
test <- rsample::testing(dataSplit)

# 모델 정의
model_spec <- parsnip::rand_forest(mtry = tune::tune(),
                                   trees = tune::tune(),
                                   min_n = tune::tune()
                                   ) %>%
  parsnip::set_engine("ranger", keep.inbag=FALSE) %>%
  parsnip::set_mode("classification")
```
모델 정의 부분에서, RF모델의 하이퍼 파라미터를 tune() 함수를 통해 tunable하게 만들었다. 이는 사용자가 각 하이퍼파라미터를 지정하지 않고, 베이지안 최적화를 통해 결정할 수 있게 하기 위함이다. 

또한, 베이지안 최적화의 목표는 모델의 성능을 가장 높일 수 있는 하이퍼파라미터 조합을 찾아내는 것이며, 이러한 목적을 위해서는 훈련 중 과적합을 피하기 위해 교차검증을 하는 것이 바람직하다. 이때, `STK` 변수의 경우 클래스 불균형이 존재하기 때문에 SMOTE를 활용한 오버샘플링을 적용하고자 했고, 이는 교차검증을 수행할 때 train fold에만 적용되어야 한다.

```r
oversampling_recipe <- recipes::recipe(STK ~ ., data = train) %>%
  themis::step_smote(STK, seed = 1234)

workflow <- workflows::workflow() %>%
  workflows::add_recipe(oversampling_recipe) %>%
  workflows::add_model(model_spec)

cv_folds <- rsample::vfold_cv(train, v = 5, strata = STK)

param <- workflow %>%
  hardhat::extract_parameter_set_dials() %>%
  recipes::update(mtry = dials::finalize(dials::mtry(), train %>% ungroup() %>% select(-STK)))

tuned_results <-
  workflow %>%
  tune::tune_bayes(cv_folds, initial = 10, iter = 5, param_info = param)
```
위와 같이 오버샘플링을 위한 레시피를 따로 만들어 교차검증 중 train fold에 적용할 수 있다. tune_bayes() 함수의 `initial` 파라미터는 베이지안 최적화 과정에서 사용되는 초기 샘플의 수를 정의하며, `iter` 파라미터는 베이지안 최적화의 반복 횟수를 지정

<div align="center" class="image-with-caption">
  <figure>
    <img src="/assets/img/illustration/2023-04-23_1.png" alt="image description">
    <figcaption>교차검증을 통한 베이지안 최적화 (출처: Shear Strength Prediction of Slender Steel Fiber Reinforced Concrete Beams Using a Gradient Boosting Regression Tree Method)
    </figcaption>
  </figure>
</div>

추가로, param 변수 정의하는 이유는 RF 모델의 하이퍼파라미터 중 mtry는 훈련 데이터의 열 개수에 따라 달리 적용될 수 있어 그 범위가 사전 지정되어있지 않기 때문이다. 이 경우 dials 패키지의 finalize 함수를 통해 범위를 확정지어줘야 한다. 
```r
> dials::mtry()

## # Randomly Selected Predictors (quantitative)
## Range: [1, ?]
```
