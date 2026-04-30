---
layout: post
title: "Poetry, PyCaret을 사용한 모델 배포"
date: 2025-01-19 09:00:00 +0900
category: blog
tags: []
---

최근 빠르게 ML 모델을 빌드하고 배포하는 과정을 여러 번 반복적으로 테스트해야 하는 상황에 직면했습니다. 이를 위해 다양한 라이브러리를 사용해 보았고, 최종적으로는 Poetry와 PyCaret을 사용하는 것이 가장 효율적이었습니다. Poetry는 의존성 충돌을 자동으로 해결해주는 점에서 편리했으며, PyCaret은 코드 템플릿화를 쉽게 할 수 있을 뿐 아니라, 모델 배포를 통한 API 개발까지 지원해주는 점이 좋았습니다. 이번 글에서는 Titanic 데이터를 활용해 이 두 라이브러리를 사용하여 모델을 빌드하고 배포하는 과정을 단계별로 설명하는 튜토리얼 코드를 기록합니다.

# 환경설정

- 가상환경은 poetry로 생성 후 진행함
    - *https://python-poetry.org/docs/basic-usage/*
- poetry 설치 (Ubuntu 22.04 기준)
    
    ```bash
    pip install poetry
    
    ----------------------
    만약 poetry 명령어를 사용할 수 없는 경우,
    echo $PATH
    -> 출력된 경로에 ~/.local/bin이 포함되어있는지 확인
    없다면,
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
    -------------------------
    
    poetry --version
    # Poetry (version 1.8.3)
    
    poetry config virtualenvs.in-project true # 프로젝트 폴더 내 가상환경 파일 설치
    ```
    
- 프로젝트 폴더 생성 후 가상환경 설정
    
    ```bash
    mkdir my_project
    cd my_projec
    poetry init
    ```
    
    - `poetry init` 명령어를 실행하면 종속성 등 설정을 입력해줄 수 있는데, 어차피 프로젝트를 진행하면서 종속성을 맞춰줄 것이므로 일단 모두 넘어감
    - 모두 완료되면 pyproject.toml이 생성되고, 여기에 종속성 정의해줄 수 있음
- 가상환경 생성
    
    ```bash
    poetry install
    ```
    
    - poetry는 의존성을 설치할 때 자동으로 가상환경을 생성함
    - `pyproject.toml` 에 명시된 의존성을 설치하며, 가상환경 생성. 아직 의존성을 정의하지 않은 상태이기 때문에, 의존성 설치 없이 가상환경만 생성함
    - 정상적으로 실행될 경우 poetry.lock 파일 생성됨
- 가상환경 실행

```bash
poetry shell
# 가상환경을 나가고 싶다면 `exit` 입력
```

- 라이브러리 설치
    - 가상환경 내 라이브러리 설치는 `poetry add` 명령어 사용하며, 할 때마다 pyproject.toml 업데이트 됨
    
    ```bash
    poetry add jupyter seaborn pycaret[mlops] ...
    ```
    
- 다만 이런 식으로 라이브러리를 설치할 때 의존성 맞춰주는 작업이 필요한데, 본문의 예시를 위해 최종적으로 수정한 pyproject.toml 파일은 아래와 같음
    
    ```bash
    [tool.poetry]
    name = "pycaret-test"
    version = "0.1.0"
    description = ""
    authors = ["..."]
    readme = "README.md"
    
    [tool.poetry.dependencies]
    python = "^3.10"
    pycaret = {extras = ["mlops"], version = "^3.3.2"}
    kaleido = "0.2.1"
    pandas = "<2.2.0"
    jupyter = "^1.0.0"
    seaborn = "^0.13.2"
    fastapi = "^0.111.0"
    uvicorn = "^0.30.1"
    sktime = {version = "0.26.0", python = ">=3.10,<3.13"}
    
    [build-system]
    requires = ["poetry-core"]
    build-backend = "poetry.core.masonry.api"
    ```
    

# PyCaret을 활용한 모델 배포

목표: Titanic data를 사용해 간단한 모델을 만든 후 배포해보기

- Jupyter Notebook에서 아래 코드를 사용해 간단히 모델 생성

```bash
import pandas as pd
from pycaret.classification import *
import seaborn as sns

data = sns.load_dataset(('titanic'))
data = pd.DataFrame(data)

# 대강 전처리
data['age'].fillna(data['age'].median(), inplace=True)
data['embarked'].fillna(data['embarked'].mode()[0], inplace=True)
data.rename(columns={'class': 'class_'}, inplace=True)

s = setup(data=data, target='survived')

# 모델 학습
# compare_models() # 10분 정도 시간 소요
# model = create_model()
model = create_model('lr')
tuned_model = tune_model(model)
final_model = finalize_model(tuned_model)

# save_model(final_model, 'titanic_survival_model') # titanic_survival_model.pkl 생성
# final_model = load_model('titanic_survival_model')
```

## 모델 배포를 위한 API 생성

- `create_api()`: PyCaret 모델을 기반으로 FastAPI 앱을 생성하여 REST API로 추론 가능하게 함

```python
create_api(final_model, 'titanic_survival_model_api')
```

- 위 명령어가 잘 실행되었을 경우 titanic_survival_model_api.py, titanic_survival_model_api.pkl 파일이 생성됨
- 다만, 자동으로 생성된 titanic_survival_model_api.py 파일은 아래와 같은 문제가 있으므로 수정 필요
    - 만약 기본값으로 *nan* 값이 들어가있다면 에러 발생. numpy를 추가로 import 해주거나, 기본값 변경해줘야 함
    - ‘class’ 라는 열 이름은 사용할 수 없음(예약어)
    - input_model, output_model의 필드 타입 정의가 부실함
        - `pydantic`의 `create_model`을 사용할 때 필드 타입을 명시해주고
        - 데이터프레임 생성시 `dict`의 `by_alias=True` 옵션을 사용하여 JSON 데이터와 필드 이름을 일치시켜줘야함
    - 호스트를 `127.0.0.1`로 설정하면 외부에서 접근할 수 없음. `0.0.0.0`으로 변경
    - 기본 8000번 포트 변경 필요
    - 최종적으로 수정한 titanic_survival_model_api.py
        
        ```python
        # -*- coding: utf-8 -*-
        
        import pandas as pd
        from pycaret.classification import load_model, predict_model
        from fastapi import FastAPI
        import uvicorn
        from pydantic import BaseModel
        
        app = FastAPI()
        
        model = load_model("titanic_survival_model_api")
        
        class InputModel(BaseModel):
            pclass: int = 3
            sex: str = 'male'
            age: float = 22.0
            sibsp: int = 0
            parch: int = 0
            fare: float = 9.0
            embarked: str = 'S'
            class_: str = 'Third'  # 'class'는 예약어이므로 'class_'로 변경
            who: str = 'man'
            adult_male: bool = True
            deck: str = None
            embark_town: str = 'Southampton'
            alive: str = 'no'
            alone: bool = True
        
        class OutputModel(BaseModel):
            prediction: int
        
        @app.post("/predict", response_model=OutputModel)
        def predict(data: InputModel):
            data = pd.DataFrame([data.dict(by_alias=True)])
            predictions = predict_model(model, data=data)
            return {"prediction": predictions["prediction_label"].iloc[0]}
        
        if __name__ == "__main__":
            uvicorn.run(app, host="0.0.0.0", port=8001)
        ```
        
- 아래 명령어 입력 후 192.168.10.201:8001/docs 접속해서 정상 작동하는지 확인
    
    ```bash
    python titanic_survival_model_api.py
    ```
    

## Docker Container 생성

- 학습된 PyCaret 모델을 Docker Container로 배포할 수 있는 환경을 자동으로 생성해줌
    
    ```python
    create_docker('titanic_survival_model_api', *base_image* = "python:3.10", *expose_port* = 8001)
    ```
    
    - 명령어가 정상적으로 실행되면 dockerfile, requirements.txt 파일이 생성되고, 아래 명령어를 통해 도커 이미지 및 컨테이너 생성
    
    ```bash
    docker image build -f "Dockerfile" -t titanic_test:latest .
    docker run -d -p 8001:8001 titanic_test
    docker ps # 컨테이너가 잘 실행되었는지 확인
    ```
    
- 192.168.10.201:8001/docs 접속해서 정상 작동하는지 확인

# 기타 이슈 기록

![ML프레임워크비교.png](/assets/img/posts/Poetry-PyCaret을-사용한-모델-배포/ML%E1%84%91%E1%85%B3%E1%84%85%E1%85%A6%E1%84%8B%E1%85%B5%E1%86%B7%E1%84%8B%E1%85%AF%E1%84%8F%E1%85%B3%E1%84%87%E1%85%B5%E1%84%80%E1%85%AD.png)

- conda는 pip와 독립된 생태계를 사용하기도 하고, 가상환경과 의존성 관리가 분리되어 있는 점이 불편했음. 또한 pipenv에서 의존성이 충돌나던 라이브러리들이 poetry에서는 해결되는 경우가 있었음
- 그러나 poetry가 모든 PyPI 패키지 버전을 지원하지는 않는 듯(pip로 설치되는 버전인데도 poetry로 설치되지 않는 경우가 있음)
- pycaret을 사용할 때, 여러 모델의 성능을 비교하기 위한 커스텀 지표가 validation과정에서는 적용이 안된다거나, 모델링 시 `setup()`의 `use_gpu`, `n_jobs` 옵션이 일부 경우에 적용되지 않는 등의 문제가 있음