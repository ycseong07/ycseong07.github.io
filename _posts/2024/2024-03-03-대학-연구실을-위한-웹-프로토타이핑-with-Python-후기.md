---
layout: post
title: "대학 연구실을 위한 웹 프로토타이핑 (with Python) 후기"
date: 2024-03-03 09:00:00 +0900
category: blog
tags: []
---

*

# 서론

*(서론에서 얘기하는 ‘대학 연구실’은 컴퓨터 관련 학과는 아니지만, 과제 수주로 인해 개발이 필요한 연구실을 염두에 두었습니다)*

대학 연구실과 개발자는 미묘한 관계에 있다. 주로 과제 driven으로 흘러가는 연구실의 특성 상, 정규직 개발자 TO를 내기는 어렵다(일단 대학 차원에서 연구실에 정규직 직원을 뽑도록 해주는 케이스가 거의 없다). 과제 기간동안 특정 프로젝트를 끝마쳐야 하기 때문에, 정규직 개발자를 채용하기보다는 단기 프로젝트를 위한 프리랜서 개발자를 선호하는 경향이 있다. 일단 구인 공고를 낸 후라도, 경력있는 개발자에게 다른 기업 만큼의 연봉을 주기는 부담스럽기 때문에, 주변의 대학원생이나 연구원, 혹은 외주를 줄 수 있는 기관 등을 선택지로 두면서 비용과 결과물의 퀄리티를 저울질하게 된다. 

이런 상황에서, 데이터 분석이나 머신러닝을 통해 연구를 진행하는 연구자들이 많아진 이유인지, 최근 연구자가 직접 프로젝트에 필요한 개발을 하게 되는 경우가 많아진 듯 하다. 다만 나와 주변인들의 경험을 빌려오자면, 연구자의 경우 데이터 분석, 머신러닝 등을 위해 R, Python을 접하고, 보통 대용량 데이터 처리나 빠르게 변하는 기술, 논문 등을 따라가기 위해 애쓴다. 개발이 아닌 분석을 위한 커리큘럼을 통해 프로그래밍 언어을 처음 접했기 때문에, 객체 지향 프로그래밍이나 네트워크 등은 생소한 개념인 경우가 많다.

얼마 전 연고가 있는 연구실에서 “Python을 통한 웹 프로토타이핑”을 주제로 간단한 세미나를 진행하고 왔다. 간단한 개발 결과를 공유하고, 그 과정에서 개발 관련 용어나 지식을 전달하는데 중점을 두었다. 아마 위와 같은 맥락에서, 연구자들이 직면한 개발 관련 문제를 해결하고자 한 시도가 아닐까 싶었다(예를 들어, 연구실에서 개발한 AI 모델이 있고, 간단한 웹 서비스 프로토타입을 제작해 과제 중간 보고에 활용한다거나..) 이 글에서는 해당 세미나의 내용을 요약해 적어두고자 한다.

![                                                                                          구현 결과 미리보기](/assets/img/posts/대학-연구실을-위한-웹-프로토타이핑-with-Python-후기/python_backend_proto.gif)

                                                                                          구현 결과 미리보기

전체 구조는 아래와 같다. 업비트에서 제공하는 REST API를 사용해 마켓 데이터를 호출하고, 이를 파싱하는 서버(FastAPI), 마켓, 로그 및 유저 정보를 저장하기 위한 DB (PostgreSQL), 유저의 행동이 일어나는 웹 페이지(Streamlit)로 구성되어 있다.  

![Untitled](/assets/img/posts/대학-연구실을-위한-웹-프로토타이핑-with-Python-후기/Untitled.png)

# 1. FastAPI로 서버 구현하기

## 1-1. FastAPI 소개

![Untitled](/assets/img/posts/대학-연구실을-위한-웹-프로토타이핑-with-Python-후기/Untitled 1.png)

FastAPI는 API를 만들기 위한 파이썬 웹 프레임워크다. 위 그림에서, 왼쪽 부분은 속도가 왜 빠른지에 대한 답을 찾을 수 있는 구조이다. 베이스로 Cython을 사용하고, 그 위에는 uvicorn이라는 비동기 처리를 지원하는 서버를 사용한다. 그리고 Starlette은 uvicorn을 사용해 서비스를 구축할 수 있는 툴킷이다. 오른쪽 부분은 왜 러닝커브가 짧은지에 대한 답을 얻을 수 있는데, 내부적으로 Pydantic 이라는 라이브러리를 사용해서 입력, 출력 값을 정의하고 검증할 수 있다. 또한 그림에는 나와있지 않지만, Swagger UI를 통해 라이브러리 API 스펙 문서를 자동으로 만들어준다. FastAPI는 Django, Flask와 비교해보더라도, 속도가 가장 빠르며, 비동기 처리도 지원하고, 가장 가볍다. Django의 장점 중 하나는 자체 ORM(Object Relational Mapping)을 제공한다는 점이지만, FastAPI를 사용할떄는 SQLAlchemy라는 라이브러리를 사용하는 대안이 있다. 

## 1-2. 서버 구현

먼저 DB를 연결하지 않은 상태의 server, client 폴더 및 파일을 아래와 같이 만들어 준다.

![Untitled](/assets/img/posts/대학-연구실을-위한-웹-프로토타이핑-with-Python-후기/Untitled 2.png)

- server/main.py에 오른쪽과 같이 코드 입력 후 “uvicorn server.main:app --reload” 입력
- [http://127.0.0.1:8000/](http://127.0.0.1:8000/docs) 에 “Hello World” 메시지가 보이고,
- [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) 에 Swagger UI가 보인다면 잘 실행된 것

## 1-3. 소셜 로그인 껍데기만 구현하기 (Google)

서버가 잘 만들어진 것을 확인하고나면, 이제 소셜 로그인을 위한 라우터(auth.py)를 만들어 본다. 보안에 대한 지식은 부족해서, HS256등을 활용한 보안 부분은 제외하고 Google 계정을 통한 로그인 기능은 겉 부분만 만들어보고자 했다. 

GCP를 이용해 Oauth2.0을 활용하기 위해, 아래 단계를 따라 설정한다. 

- [Google Cloud Console](https://console.cloud.google.com/)에 접속 후 프로젝트 생성
- APIs & Services → Enabled APIs & services → “Google+ API” 검색 후 활성화
- APIs & Services → Credentials 접속
- Create Credentials → OAuth client ID 선택
- ‘Authorized redirect URIs’에 ‘http://localhost:8000/’ 입력 후 Credentials 생성 (’/’나 http’s’ 등 문자 하나에도 민감하니 정확히 입력)
- 생성되는 credential json 파일을 다운 받아두기
- 배포 시 코드 파일에 개인정보가 들어가지 않도록 환경변수로 관리
- 다운받은 credential json 파일에 있는 client_id, client_secret, redirect_uris 정보를 .env 파일에 상수로 정의 후,  dotenv 라이브러리를 활용해 .env 파일의 정보 호출 가능

위 단계가 마무리되면 Google 계정을 통한 로그인 기능을 위한 코드를 auth.py 파일에 작성한다. 

![Google 계정을 통한 로그인 과정](/assets/img/posts/대학-연구실을-위한-웹-프로토타이핑-with-Python-후기/Untitled 3.png)

Google 계정을 통한 로그인 과정

- Google 계정을 통한 로그인 단계 별 주석을 달아 둔 샘플 코드
    
    ```python
    router = APIRouter()
    
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    
    DATABASE_URL = os.getenv("DATABASE_URL") 
    AUTHORIZATION_URL = os.getenv("AUTHORIZATION_URL") 
    TOKEN_URL = os.getenv("TOKEN_URL") 
    GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID") 
    GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET") 
    REDIRECT_URI = os.getenv("REDIRECT_URI") 
    
    scopes = {
        "openid": "OpenID connect to authenticate",
        "profile": "Access to your profile",
        "email": "Access to your email",
    }
    
    oauth2_scheme = OAuth2AuthorizationCodeBearer(
        authorizationUrl=AUTHORIZATION_URL,
        tokenUrl=TOKEN_URL,
        scopes=scopes,
    )
    
    @router.get("/login")
    async def google_login():
        google_login_url = f"https://accounts.google.com/o/oauth2/auth?response_type=code&client_id={GOOGLE_CLIENT_ID}&redirect_uri={REDIRECT_URI}&scope=openid%20profile%20email&access_type=offline"
        return RedirectResponse(url=google_login_url)
    
    @router.get("/login/callback")
    async def google_login_callback(request: Request, code: str = Query(...)):
        async with httpx.AsyncClient() as client:
            # 로그인해서 받은 code를 사용해 Google의 토큰 엔드포인트(TOKEN_URL)에 POST 요청
            token_response = await client.post(
                TOKEN_URL,
                data={
                    "client_id": GOOGLE_CLIENT_ID,
                    "client_secret": GOOGLE_CLIENT_SECRET,
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": REDIRECT_URI
                },
            )
        # 액세스 토큰을 받아 json으로 변환
        token_response_json = token_response.json()
        if token_response.is_error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=token_response_json,
            )
        
        # 액세스 토큰을 사용해 사용자 정보 엔드포인트에 GET 요청 -> 유저 정보 획득
        access_token = token_response_json.get("access_token")
        async with httpx.AsyncClient() as client:
            user_info_response = await client.get("https://www.googleapis.com/oauth2/v1/userinfo", headers={"Authorization": f"Bearer {access_token}"})
        user_info = user_info_response.json()
    
        user_email = user_info.get("email")
        user_email_encoded = quote(user_email)
        final_url = f"http://localhost:8501?user_info={user_email_encoded}"
        return RedirectResponse(url=final_url)
    
    @router.get("/logout")
    async def logout():
        logout_redirect_url = "http://localhost:8501" # streamlit 웹 페이지 포트
        return RedirectResponse(url=logout_redirect_url)
    ```
    

이렇게 작성했을 때, 로그인이 되고 나면 쿼리 파라미터로 유저 정보(user_info)를 가져올 수 있게 된다. auth.py 파일 작성이 되었다면 main.py 파일은 아래와 같이 수정해준다. 다양한 API를 구성하게 되면 main.py에서만 관리하기는 어려워지므로, main.py 파일에는 여러 라우터를 가져오는 역할을 하도록 한다. 

```python
from fastapi import FastAPl
from server.router import auth

app = FastAPI)

app.include_router(auth.router)
```

## 1-4. Upbit REST API 받아오기

업비트에서는 여러 가상화폐에 대한 마켓 정보를 REST API로 제공한다. [업비트 API](https://docs.upbit.com/reference/ticker%ED%98%84%EC%9E%AC%EA%B0%80-%EC%A0%95%EB%B3%B4) [Reference 페이지](https://docs.upbit.com/reference/ticker%ED%98%84%EC%9E%AC%EA%B0%80-%EC%A0%95%EB%B3%B4)를 참조해, 브라우저에 [https://api.upbit.com/v1/ticker?markets=KRW-BTC,KRW-ETH,KRW-XRP](https://api.upbit.com/v1/ticker?markets=KRW-BTC,KRW-ETH,KRW-XRP) url로 접속해보면 response가 출력되는 것을 확인할 수 있다. 이 데이터를 받아오기 위한 utils/get_coin_price.py 파일을 아래와 같이 작성해준다. 

![Untitled](/assets/img/posts/대학-연구실을-위한-웹-프로토타이핑-with-Python-후기/Untitled 4.png)

# 2. Streamlit으로 클라이언트 구현하기

Streamlit은 머신러닝 혹은 데이터 사이언스 팀을 위한 오픈소스 앱 프레임워크다. 하이엔드 웹 페이지를 만들기는 어렵지만, 데이터 시각화나 웹 프로토타이핑에는 유용하게 쓸 수 있다(streamlit에 대한 소개와 장단점, 대시보드를 만드는 과정을 발표한 [영상](https://www.youtube.com/watch?v=a1e0-S2CbmM)이 있으니 참고해주셔도 좋습니다).

로그인이 되면 쿼리 파라미터로 유저 정보를 받아오게 되는데, 이를 통해 로그인 상태를 확인하도록 했다(별도의 보안 기능이 전혀 없는 상태라는 것을 다시 한 번 밝힌다). 로그아웃 기능 역시 별도의 기능 없이 ‘http://localhost:8501’로 리다이렉트 되도록 했다.

- client/app.py 파일 샘플 코드
    
    ```python
    import streamlit as st
    import pandas as pd
    from datetime import datetime
    import requests
    
    st.title('암호화폐 대시보드 프로토타입')
    
    # 로그아웃 API (http://localhost:8501로 리다이렉트)
    logout_button_html = '<a href="http://localhost:8000/logout" target="_self">Logout</a>'
    
    # 쿼리 파라미터에서 유저 정보 확인 (이메일)
    user_info = st.query_params.get_all('user_info')
    if user_info:
        st.session_state['user_info'] = user_info[0]
    
    # 로그인 상태
    if 'user_info' in st.session_state:
        st.success(f'Logged in as {st.session_state["user_info"]}')
        st.markdown(logout_button_html, unsafe_allow_html=True)
    
        # 코인 가격 정보 조회 API 
        def get_current_prices():
            ...
            return data
    
        # session_state 변수 초기화    
        if 'price_data' not in st.session_state:
            st.session_state.price_data = pd.DataFrame(columns=['market', 'trade_date', 'trade_timestamp', 'high_price', 'low_price', 'trade_price'])
        if 'latest_price' not in st.session_state:
            st.session_state.latest_price = pd.DataFrame()
    
        # 현재 시세 확인 후 session_state.latest_price에 저장
        if st.button('현재 시세 확인'):
            data = get_current_prices()
            price_data = pd.DataFrame(data)
            st.session_state.latest_price = price_data[['market', 'trade_date', 'trade_timestamp', 'high_price', 'low_price', 'trade_price']]
    
        # 현재 시세 확인 테이블 출력
        st.header("암호화폐 시세")
        if not st.session_state.latest_price.empty:
            st.table(st.session_state.latest_price)
    
        st.header("비트코인 가격 기록")
    
        # 버튼을 누르면 BTC 정보 기록 한 줄 씩 concat
        if st.button('BTC 정보 기록'):
            if not st.session_state.latest_price.empty:
                btc_data = st.session_state.latest_price[st.session_state.latest_price['market'] == 'KRW-BTC']
                if not btc_data.empty:
                    st.session_state.price_data = pd.concat([st.session_state.price_data, btc_data], ignore_index=True)
    
        # BTC 정보 기록 테이블
        st.table(st.session_state.price_data)
    
    # 비 로그인 상태
    else:
        login_button_html = '<a href="http://localhost:8000/login" target="_blank">Login with Google</a>'
        st.markdown(login_button_html, unsafe_allow_html=True)
        if not user_info:
            st.error('Not logged in or logged out')
    ```
    

# 3. PostgreSQL 연동하고 로그 저장하기

## 3-1. PostgreSQL 설치 및 설정

PostgreSQL과 DBeaver를 설치한 후 진행했다. 먼저, 아래와 같이 PostgreSQL 사용자, 스키마, 테이블을 생성하고 권한을 부여한다.

```bash
# 사용자 확인
postgres=# \du

# test/testpass 사용자 만들기
postgres=# CREATE ROLE test WITH LOGIN PASSWORD 'testpass';

# DB 생성 권한 부여
postgres=# ALTER ROLE test CREATEDB;

# 사용자 확인
postgres=# \du

# test 사용자로 재접속
postgres=# exit
psql postgres -U test

# db 만들기
CREATE DATABASE testdb;

# 해당 db에 대한 모든 권한 부여
GRANT ALL PRIVILEGES ON DATABASE test TO test;
\list

# PostgreSQL에서는 Database -> Schema -> Table 개념
\dt (테이블 리스트 보기)
1dn (스키마 리스트 보기)
```

## 3-2. 데이터 저장 구현

폴더 구조는 아래와 같이 각각의 기능을 기준으로 구분해두었다. 각 폴더에는 아래와 같은 역할을 하는 파일이 포함되도록 했다(다만 이렇게 미리 구조를 잡아둔 대로 코드가 깔끔하게 짜여지지는 않아서, 디자인 패턴이나 아키텍처에 대해 알아보고싶은 생각이 들었다). 

- models: DB 테이블에 대응하는 SQLAlchemy 모델을 정의
    
    ```python
    # 예시 (log_models.py)
    
    from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey
    from sqlalchemy.sql import func
    from sqlalchemy.orm import relationship
    from server.database.database import Base
    
    # 인증 로그 테이블 모델
    class Auth(Base):
        __tablename__ = 'auth'
        __table_args__ = {'schema': 'log'}
    
        id = Column(Integer, primary_key=True)
        user_id = Column(String(50))
        email = Column(String(50))
        full_name = Column(String(20))
        access_time = Column(DateTime, default=func.now())
    ```
    
- schemas: Pydantic을 사용해 입력 데이터의 정의 및 검증
    
    ```python
    # 예시 (log_schemas.py)
    from pydantic import BaseModel
    
    class AuthCreate(BaseModel):
        user_id: str
        email: str
        full_name: str
    ```
    
- crud: CRUD 작업을 위한 함수 정의
    
    ```python
    # 예시 (log_crud.py)
    from sqlalchemy.orm import Session
    from server.models.log_models import Auth
    from server.schemas.log_schemas import AuthCreate
    from datetime import datetime
    
    # 유저 인증 로그 생성
    def create_user(db: Session, user: AuthCreate):
        db_user = Auth(user_id=user.user_id, email=user.email, full_name=user.full_name)
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user
    ```
    
- database: DB 연결과 세션 관리
    
    ```python
    # 예시 (database.py)
    from sqlalchemy import create_engine
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import Session, sessionmaker
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    DATABASE_URL = os.getenv("DATABASE_URL") 
    
    # echo=True: SQL 로그 활성화
    engine = create_engine(DATABASE_URL, echo=True)
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    Base = declarative_base()
    
    try:
        with engine.connect() as conn:
            print("DB 연결 성공")
    except Exception as e:
        print(f"DB 연결 실패: {e}")
    
    def get_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()
    ```
    
- router:  특정 기능 또는 리소스에 대한 API 라우트 정의
    
    ```python
    # 예시 (log_router.py)
    from fastapi import APIRouter, Depends
    from sqlalchemy.orm import Session
    from server.database.database import get_db
    from server.models.log_models import UserLog
    from server.schemas.log_schemas import UserAction
    
    router = APIRouter()
    
    @router.post("/user-action/")
    async def log_user_action(action: UserAction, db: Session = Depends(get_db)): 
        new_log = UserLog(user_id=action.user_id, action_type=action.action_type)
        db.add(new_log)
        db.commit()
        return {"유저 행동 로그 기록 성공"}
    ```
    

![Untitled](/assets/img/posts/대학-연구실을-위한-웹-프로토타이핑-with-Python-후기/Untitled 5.png)

예를 들어, 사용자가 로그인한 로그를 DB에 저장하는 작업은 아래와 같은 순서로 이루어진다.

- 로그인 request가 auth.py를 통해 ‘/login/callback’ 엔드포인트 호출
- 엔드포인트에서 log_crud.py에 정의된 create_user 함수를 호출하여 사용자 접속 로그를 생성
- 이 때, create_user 함수는 SQLAlchemy 모델(log_models.py)을 사용하여 DB에 저장
- log_schemas.py에 정의된 Pydantic 스키마를 사용해 API 응답으로 보낼 데이터를 검증함

위 구조를 따라 유저 접속 로그, 유저 행위 로그(클라이언트 로그), 마켓 정보 로그(1분 간격으로 마켓 정보 저장)를 DB에 저장하도록 구현해두었다. 

- 유저 접속 로그
    
    ```python
    # auth.py 파일 수정
    
    async def google_login_callback(request: Request, code: str = Query(...), db: Session = Depends(get_db)):
    ...
    		user_info = user_info_response.json()
        
        user_data = AuthCreate(
            user_id=user_info["id"],
            email=user_info["email"],
            full_name=user_info["name"] 
        )
        create_user(db, user=user_data)
    ...
    ```
    
- 유저 행위 로그
    
    ```python
    # app.py 파일 수정
    ...
        def log_user_action(user_id, action_type):
            response = requests.post("http://localhost:8000/user-action/", json={"user_id": user_id, "action_type": action_type})
            if response.status_code != 200:
                print("사용자 행위 정보를 가져오는 데 실패했습니다")
    ...
    ```
    
- 마켓 정보 로그
    
    ```python
    # get_coin_price.py 파일 수정
    
    import requests
    from server.database.database import SessionLocal
    from server.crud.market_crud import save_coin_prices
    
    def get_coin_prices():
        markets = "KRW-BTC,KRW-ETH,KRW-XRP"
        url = f"https://api.upbit.com/v1/ticker?markets={markets}"
        response = requests.get(url)
        data = response.json()
        return data
    
    def fetch_and_save_coin_prices():
        coin_prices = get_coin_prices()
        db = SessionLocal()
        try:
            save_coin_prices(db, coin_prices)
        finally:
            db.close()
    
    # main.py에 스케줄러 설정
    
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    
    ...
    
    scheduler = AsyncIOScheduler()
    scheduler.start()
    scheduler.add_job(fetch_and_save_coin_prices, 'interval', minutes=1)
    
    ...
    
    app.include_router(market_router.router, prefix="/market", tags=["market"])
    
    ```
    

# 4. 마치며

해당 프로젝트는 [github repo](https://github.com/ycseong07/python_backend_prototype)에 공개해두었다. 개인적으로는 개발 아키텍처나 로그인 구현, API 구현 등에 대해 많은 공부가 되었다. 나중에는 로그인 정보의 암호화, 아키텍처 개선 후 리팩터링 등의 작업을 해보면 좋을 것 같고, 1분이 아닌 1초 간격으로 데이터를 가져올 때 발생할 문제에 대해서도 개선해보고자 한다.