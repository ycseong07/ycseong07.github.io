---
layout: post
title: "High-Dimension and Low-Sample-Size (HDLSS) 데이터에서 ReGNN 활용하기"
date: 2025-01-05 09:00:00 +0900
category: blog
tags: []
---

# 서론

현업에서의 ML/DL 모델링은 점점 더 구체적인 문제를 해결하려는 방향으로 나아가고 있다. 그러다보니 특정 변수를 예측하기 위해 중요한 feature들을 다양하게 수집하게 되고, 데이터는 자연스럽게 HDLSS(High-Dimension and Low-Sample-Size) 특성을 가지는 경우가 많아진다.

HDLSS 데이터를 다룰 때는 Feature의 개수를 줄이는 방향으로 전처리를 하게 되는데, 이 때 가장 먼저 떠오르는 방법은 차원 축소 방법이다(Auto-encoder나 PCA 같은 비지도 학습 방법을 통해 변수를 압축하거나 주요 변수를 선택하는 등). 하지만 이 방법들은 Target에 최적화되지 않은 Latent feature를 생성하며, 결국 예측 성능 향상에는 도움이 되지 않는 경우가 많다. 게다가 변수가 많아 직접적으로 feature selection을 하기 힘들다면, 어떤 대안이 있을까?

아직 논문이 게재되지는 않았지만, 아카이브에 저장되어있는 페이퍼 중 ‘Unveiling Population Heterogeneity in Health Risks Posed by Environmental Hazards Using Regression-Guided Neural Network’ 라는 페이퍼가 있다. 간단하지만 창의적인 방법으로 연구/현업 분야에 도움이 될 수 있는 ReGNN(Regression-Guided Neural Network)이라는 방법론을 소개하고 있다. 간단히 요약하자면, Neural Network와 Regression을 결합하여 target-guided dimention reduction을 수행한다. 페이퍼에서는 많은 상호작용항을 처리하기 위해 이 방법을 고안한 것으로 보이지만, HDLSS를 해소하는 방법으로도 사용될 수 있다.

# ReGNN

ReGNN은 전통적인 Moderated Multiple Regression(MMR)의 구조를 유지하면서도, Neural Network를 결합해 차원축소를 통한 feature를 추출한다. 페이퍼에서는 이 방법을 활용하는 방법을 3단계로 소개한다.

![15609.png](/assets/img/posts/High-Dimension-and-Low-Sample-Size-HDLSS-데이터에서-R/15609.png)

- Neural Network를 기존 Moderated Multiple Regression 모델에 통합하여 학습한다. 이 때, Neural Network는 moderators를 비선형적으로 요약하여 추출한 f(M)를 생성한다. 이 과정은 모든 feature의 비선형 상호작용을 학습하여 다중공선성을 줄이면서도, target 변수 예측에 최적화된 latent feature를 생성하는 효과가 있다.
- 추출된 f(M)을 고정시킨 후 새로운 Regression Model을 학습하고, $c^{\text{int}}$ 의 유의미성 검증을 위한 p-value를 계산한다.
- XAI 도구를 사용해 M → f(M) 영향을 해석한다.

내 경우에는 1,2 단계까지의 내용을 tabular data에 voice data를 결합하여 ML모델을 만들 때 활용해볼 수 있었다. voice data로부터 음성 feature를 추출하기 위해서는 praat이나 opensmile을 사용할 수 있는데, 이를 통해 추출할 수 있는 feature의 개수가 너무 많아 그대로 사용하기는 어려웠다. 이 때 ReGNN을 사용해 target 예측에 최적화된 latent feature를 뽑고, 이를 적용해 다시 모델링을 하는 방법을 사용할 수 있었다. 

- 간단한 구현 코드
    
    ```python
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.linear_model import LinearRegression
    
    class ReGNN(nn.Module):
        def __init__(self, input_dim):
            super(ReGNN, self).__init__()
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, 30),
                nn.GELU(),
                nn.Linear(30, 10),
                nn.GELU(),
                nn.Linear(10, 2),
                nn.GELU(),
                nn.Linear(2, 1)
            )
    
        def forward(self, X, xf):
            f_M = self.mlp(X)
            interaction_term = xf * f_M
            return f_M, interaction_term
    
    input_dim = X_train.shape[1]
    model = ReGNN(input_dim)
    optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.01)
    criterion = nn.MSELoss()
    
    n_epochs = 150 # 조절 필요
    
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
    
        f_M_train, interaction_train = model(X_train, xf_train)
        predictions = interaction_train + y_train.mean()
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            f_M_test, interaction_test = model(X_test, xf_test)
            test_predictions = interaction_test + y_train.mean()
            test_loss = criterion(test_predictions, y_test)
    
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs} - Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")
    
    # Twin Regression
    f_M_train_np = f_M_train.detach().numpy()
    regression_model = LinearRegression()
    regression_model.fit(f_M_train_np, y_train.numpy())
    ```
    

다만 twin regression 결과와 모델링 결과를 비교해봤을 때, p-value를 계산했을 때는 유의미했더라도, predict 결과가 크게 나아지지 않는 경우가 있었다. 결국에는 추출된 음성 feature들을 랜덤하게 조합하여 latent feature를 생성하는 자동화 코드를 만들고, p-value가 유의하면서 predict 성능도 크게 개선되는 경우를 찾아내야 했다.

# 결론

HDLSS 데이터를 사용해 모델을 만들 때에는 feature 수를 축소해야하는 이슈를 해결해야 하고, 설사 그 모델이 높은 성능을 보인다 하더라도 왜 이 모델이 잘 작동하는 것인지 검증할 수 있어야 한다. 그러나 일단 모델이 잘 작동하고나면 모델의 향후 성능이 떨어지지 않도록 하기 위해 재학습 파이프라인을 만드는 데 노력을 더 쏟게 되고, 해석은 뒷전이 되는 경우가 많았다. 그러나 ReGNN은 현업에서도 사용할 수 있을 만큼 간단한 구조를 제안하고 있고, 구현하기에 따라서는 변수를 바꾼 여러 번의 실험을 자동화할 수도 있다. 어딘가는 Auto-encoder를 닮았고, 어딘가는 regression을 닮았지만 각각의 단점을 간단하게 보완한 구조라고 생각한다. 특히, 페이퍼의 의도와는 살짝 다를지도 모르지만, 현업에서의 HDLSS 데이터를 다룰 때의 이슈를 해소해줄 수 있는 좋은 방법론인 듯 하다.

# Reference

- [https://www.arxiv.org/pdf/2409.13205](https://www.arxiv.org/pdf/2409.13205)