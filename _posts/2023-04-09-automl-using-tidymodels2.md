---
layout: post
title: Tidymodels로 AutoML을 구현해보자 (2)
description: 일반인도 사용할 수 있는 AutoML 패키지(stove) 제작에 관한 글입니다. Tidymodels로 간단한 분류모델을 만들어보면서 현재 개발 중인 stove 패키지의 구조를 재정비 해보려고 합니다.
---

# Tidymodels의 문법과 컨셉을 알아보자

Tidymodels는 R을 사용해 통계 모델 혹은 ML 모델을 만들 때 필요한 패키지들을 광범위하게 포괄하는 프레임워크다. R 유저 입장에서는 일관적인 작업 흐름을 통해 모델링을 수행할 수 있고, 데이터 처리에 이제는 보편적으로 쓰이게 된 tidyverse 패키지와 잘 호환되는 구조를 가졌다는 점에서 사용상의 이점이 있다. 이 글에서는 Tidymodels를 사용해 간단한 분류 모델링을 해보고, 문법과 전체적인 컨셉을 되짚어보고자 한다. 

## 1. 패키지 및 데이터 불러오기
dplyr, ggplot2 패키지만 library()로 불러왔고, Tidymodels의 하위 패키지들은 각 단계별로 어떤 패키지의 어떤 함수를 사용했는지 명시하기 위해 library로 불러오지 않고 `패키지명::함수명` 형태로 적었다. 

[datatoys](https://github.com/statgarten/datatoys)라는 패키지는 [공공데이터포털](https://www.data.go.kr/)에서 제공하는 데이터셋을 쉽게 가져올 수 있도록 도와주는 패키지다. 이 글에서는 `bloodTest` 데이터를 사용했고, 뇌혈관질환을 가진 환자를 분류하는 모델을 만들어 보고자 했다. 

```r
# remotes::install_github("statgarten/datatoys")
# install.packages("dplyr")
# install.packages("ggplot2")
# install.packages("tidymodels")
## install.packages("ranger")
library(dplyr)
library(ggplot2)

cleaned_data <- datatoys::bloodTest

# 데이터 출처: https://www.data.go.kr/data/15095107/fileData.do
# SEX : 성별(남성:1, 여성:2)
# AGE_G : 연령(그룹)
# HGB : 혈색소
# TCHOL : 총콜레스테롤
# TG : 중성지방
# HDL : HDL 콜레스테롤
# ANE : 빈혈 진료여부(있음:1, 없음:0)
# IHD : 허혈심장질환 진료여부(있음:1, 없음:0)
# STK : 뇌혈관질환 진료여부(있음:1, 없음:0)

str(cleaned_data) # Categorical variable이 int 형으로 되어있는 경우 factor형으로 변환해줍니다. 

cleaned_data <- cleaned_data %>%
  mutate_at(vars(AGE_G, SEX, ANE, IHD, STK), factor) 

summary(cleaned_data$STK) # STK가 1인 row 수는 62828개 입니다. 

# 적당히 6만개씩 뽑아 balanced data를 생성합니다
cleaned_data <- cleaned_data %>%
  group_by(STK) %>%
  sample_n(60000) # STK(0):STK(1) = 60000:60000
```

## 2. 데이터 전처리

데이터 전처리에는 `recipes`라는 패키지를 사용한다. 결측값 처리, 범주형 변수 인코딩, 변수 정규화 등 다양한 전처리 작업을 수행할 수 있다. 이 패키지는 내가 원하는 전처리를 `recipe`에 적어두고, 그 순서대로 데이터를 `bake`한다는 컨셉을 가지고 있다. 아래 예시에서는 다른 변수들과의 correlation이 0.8 이상인 변수를 제거하고, 데이터 값을 정규화시키는 전처리를 수행했다. `all_numeric()` 함수를 통해 숫자형 타입을 가진 열에 대해서만 해당 전처리를 하도록 지정할 수 있다. 

```r
preprocessing_recipe <- recipes::recipe(STK ~ ., data = cleaned_data) %>%
  recipes::step_corr(recipes::all_numeric(), threshold = 0.8) %>%
  recipes::step_normalize(recipes::all_numeric())

data_preprocessed <- recipes::prep(preprocessing_recipe, training = cleaned_data) %>%
  recipes::bake(new_data = cleaned_data)
```
stove 패키지를 만들 때만 해도 머신러닝을 한다면 당연히 교차검증 과정을 포함해야 한다고 생각했지만, 교차검증을 거친 모델이 그렇지 않은 모델에 비해 얼마나 성능이 나아졌는가에 대한 정보를 얻고 싶어하는 요구도 있었고, 교차검증을 할 만큼 데이터가 충분치 않은 경우에도 ML 모델링을 해보고 싶다는 요구도 있었다. 그래서 이 글에서도 교차검증을 수행하지 않는 예시를 들었고, 향후 stove 에는 교차검증을 수행하지 않는 옵션을 선택할 수 있도록 개발할 예정이다. 만약 Cross Validation을 수행한다면 데이터 분할 전후로 수행해야하는 전처리를 구분해야 하며, 이 내용은 이후의 글에서 다루려고 한다. 

## 3. 데이터 분할
데이터 분할에는 `rsample`패키지를 사용한다. 훈련-테스트 분할 뿐만 아니라, 다양한 샘플링 방법, 교차 검증에도 사용한다. 아래 예시에서는 strata 옵션을 통해 훈련 데이터와 테스트 데이터에서의 클래스 비율을 동일하게 유지시켰다. 

```r
set.seed(1234)
dataSplit <- rsample::initial_split(data_preprocessed,
                                    strata = tidyselect::all_of("STK"), 
                                    prop = 0.7)

train <- rsample::training(dataSplit) 
test <- rsample::testing(dataSplit) 

summary(train$STK) # 0(42000) : 1(42000)
summary(test$STK) # 0(18000) : 1(18000)
```

## 4. 모델 만들기
모델링에는 `parsnip` 패키지를 사용한다. 회귀, 분류, 클러스터링 등에 대한 다양한 모델을 일관적인 문법을 통해 생성할 수 있도록 하며, [사용할 수 있는 모델](https://parsnip.tidymodels.org/reference/index.html#models)들을 구현할 수 있는 engine을 설정하고(set_engine), 어떤 task를 수행할 것인지(set_mode)를 설정해주면 된다. 주의할 점은, 이 패키지의 모델 함수들이 가지고 있는 하이퍼파라미터 리스트가 모든 엔진에서 사용되는 것은 아니라는 점이다. 사용하는 엔진에 따라 지원하지 않는 하이퍼파라미터도 있으며, 동일한 하이퍼파라미터라도 엔진에 따라 지원하는 범위가 다르기도 하다.

모델링 시 하이퍼파라미터의 튜닝은 `tune`패키지를 사용한다. 하지만 이 부분은 이후의 글에서 좀 더 자세히 설명하고자, 이 글의 예시에서는 사용하지 않았다. 아래는 분류를 위한 랜덤포레스트 모델을 만들기 위해 'ranger' 엔진을 사용하는 예시이다. 

```r
model_spec <- parsnip::rand_forest(trees = 1000) %>% 
  parsnip::set_engine("ranger") %>% 
  parsnip::set_mode("classification")

# Train the model
model_fit <- model_spec %>%
  parsnip::fit(STK ~ ., data = train)
```

## 5. 모델을 사용한 시험 데이터 예측

만들어진 모델을 통해 테스트 데이터에 대한 예측 결과를 얻을 수 있다. 이 부분은 R에서 기본적으로 제공하는 `stats` 패키지의 predict() 함수를 사용한다. 모델에 따라 type에 설정할 수 있는 옵션이 다르다.  ranger engine을 사용해 만들어진 랜덤포레스트 모델은 predict의 type option에 'class', 'prob', 'conf_int', 'raw' 옵션을 줄 수 있다. 

```r
predictions <- model_fit %>%
  stats::predict(test, type = "prob") 

# probability 0.5를 기준으로 클래스를 구분하는 열을 추가 생성
predictions <- predictions %>%
  mutate(.pred_class = if_else(.pred_1 > 0.5, 1, 0)) %>%
  mutate_at(vars(.pred_class), factor) 
```

위 예시대로 진행했다면, predict의 type option에 'conf_int'를 설정할 경우 에러가 나는데, 이는 랜덤포레스트 모델을 훈련시킬 때 사용한 부트스트랩 샘플들에 대한 계산 값을 저장하지 않았기 때문이다. 모델 훈련 시 아래처럼 `keep.inbag=TRUE` 옵션을 추가하면 'conf_int' 옵션을 사용할 수 있다. 다만 이 경우 부트스트랩 샘플들에 대한 계산 값을 모두 저장하므로 모델 훈련 시간이 오래 걸리고, 모델의 용량도 더 커진다. 

```r
parsnip::set_engine("ranger", keep.inbag=TRUE)
```

이후 아래 코드를 실행하면 각 클래스에 대한 probability 값의 Confidence Interval을 볼 수 있다. 
```r
predictions_conf <- model_fit %>%
  predict(test, type = "conf_int")
```

## 6. 성능 평가

모델의 성능을 지표화 하기 위한 Confusion Matrix는 다음과 같이 만들 수 있다. 모델을 평가하기 위한 여러 metric은 `yardstick`이라는 패키지를 통해 쉽게 얻을 수 있다.

```r
metrics <- test %>%
  bind_cols(predictions) %>%
  yardstick::metrics(truth = STK, estimate = .pred_class)

tmpDf <- test %>%
  bind_cols(predictions) %>%
  as.data.frame() %>%
  dplyr::select(STK, .pred_class)

confDf <- stats::xtabs(~ tmpDf$.pred_class + tmpDf$STK)

input.matrix <- data.matrix(confDf)
confusion <- as.data.frame(as.table(input.matrix))
colnames(confusion)[1] <- "y_pred"
colnames(confusion)[2] <- "actual_y"
colnames(confusion)[3] <- "Frequency"

confusion
```

위처럼 Confusion Matrix를 생성하고 나면 평가를 위한 metric은 대부분 계산할 수 있지만, 아래처럼 쉽게 값을 얻을 수도 있다.

```r
yardstick::accuracy(tmpDf, STK, .pred_class)
yardstick::sens(tmpDf, STK, .pred_class)
yardstick::spec(tmpDf, STK, .pred_class)
yardstick::precision(tmpDf, STK, .pred_class)
yardstick::f_meas(tmpDf, STK, .pred_class)
yardstick::kap(tmpDf, STK, .pred_class)
yardstick::mcc(tmpDf, STK, .pred_class)
```

그리고 아래처럼 `ggplot2` 패키지를 통해 Confusion Matrix를 그릴 수도 있다. 

```r
plot <- ggplot(confusion, aes(x = actual_y, y = y_pred, fill = Frequency)) +
  geom_tile() +
  geom_text(aes(label = Frequency)) +
  scale_x_discrete(name = "Actual Class") +
  scale_y_discrete(name = "Predicted Class") +
  geom_text(aes(label = Frequency), colour = "black") +
  scale_fill_continuous(high = "#E9BC09", low = "#F3E5AC")

plot
```

또한 아래처럼 ROC Curve를 그릴 수도 있다. 이 예시의 경우 심혈관 질환을 갖지 않은 케이스가 0이고, 가진 케이스가 1이기 때문에, `event_level = "second"` 옵션을 주어야 ROC Curve가 뒤집어지지 않는다. 

```r
roc_curve_result <- test %>%
  bind_cols(predictions) %>%
  ungroup() %>%
  yardstick::roc_curve(truth = STK, estimate = .pred_1, event_level = "second") 


roc_curve_result %>%
  ggplot(
  aes(
    x = 1 - specificity,
    y = sensitivity,
  )
) +
  labs(
    title = "ROC curve",
    x = "False Positive Rate (1-Specificity)",
    y = "True Positive Rate (Sensitivity)"
  ) +
  geom_line(size = 1.1) +
  geom_abline(slope = 1, intercept = 0, size = 0.5) +
  coord_fixed() +
  cowplot::theme_cowplot()
```

Tidymodels 홈페이지의 예시를 따라, 전체적인 문법과 컨셉을 살펴보았다. 이후 글들에서는 교차검증 시 global, local preprocessing의 구분, `tune` 패키지와 bayesian optimization을 통한 하이퍼파라미터 튜닝, `workflows`패키지를 통한 작업 흐름 관리에 대해 작성하면서 AutoML을 위한 기능들에 대해 계속에서 서술하고자 한다.  







