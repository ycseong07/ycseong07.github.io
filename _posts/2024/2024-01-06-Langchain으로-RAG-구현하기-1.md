---
layout: post
title: "Langchain으로 RAG 구현하기 (1)"
date: 2024-01-06 09:00:00 +0900
category: blog
tags: []
---

*

# 서론

어떤 스타트업에서 '바나나와 코끼리'라는 플랫폼을 만들었다고 가정해봅니다. '바나나와 코끼리'는 온라인 친구 매칭 플랫폼으로, 그 이름으로부터 서비스 내용을 유추하기란 쉽지 않습니다. 만약 이 서비스에 대해 안내하기 위한 챗봇을 오픈소스 LLM을 통해 구현하고자 한다면, LLM에 '바나나와 코끼리'라는 플랫폼에 대한 정보를 추가로 학습시켜줘야 할 필요가 있습니다. 그렇지 않으면 챗봇은 '바나나와 코끼리 서비스를 사용하는 방법을 알려줘' 라는 질문에 대해 알 수 없다거나, 혹은 할루시네이션이 포함된 답변을 제공할 것이기 때문이죠. 이슈는 '어떻게 기반 모델(Foundation Model)에 데이터를 추가적으로 적용시킬 수 있을지'가 될 것입니다. 

Fine-tuning을 통해 모델을 추가적으로 학습시키고자 하는 경우, 그 비용을 줄이기 위한 여러 방법이 논의되어 왔습니다. PEFT나 QLoRa가 대표적이죠. 그러나 아무리 비용 최적화를 하더라도 A100 수준의 GPU는 필요할 것이고, 스타트업의 서비스라는 점을 고려하면 기능이 추가되거나 변화하는 빈도가 잦을 것입니다. 따라서 최신 정보들을 빠르게 업데이트하기 위해서는 모델을 추가적으로 학습시키는 것이 아니라, 모델에게 정해진 데이터를 검색할 수 있도록 하는 Retrieval-Augmented Generation(RAG)를 사용하는 것이 바람직해 보입니다(최근에는 [모델 개발 시 Context 길이를 늘리는 방향보다 검색을 잘 하도록 개발하는 방향의 나을 것이라는 취지의 논문](https://arxiv.org/pdf/2310.03025.pdf)도 나옴). 이 방법은 Fine-tuning에 비해 비용이나 시간이 훨씬 덜 소요되기도 합니다. 물론 실제 RAG 구현 시에는 

- 더 발전된 모델인 Fusion-in-Decoder(FiD)나 Atlas를 고려하기
- 키워드 검색과 벡터 검색을 함께 사용하기
- Vector DB 대신 Knowledge Graph를 사용하기

등의 세부적인 커스터마이징이 이뤄지겠지만, 이 글에서는 기본적인 RAG 모델의 구현에 대해서만 다뤄보겠습니다.

# Retrieval-Augmented Generation(RAG)이 뭔가요?

‘Retrieval-Augmented’를 풀어서 이해해보면, 외부 데이터로부터 정보를 검색(Retrieval)하고, 그 정보를 바탕으로 더 나은(Augmented) 답변을 생성한다는 의미입니다. 즉, RAG는 외부 데이터베이스(기본적으로는 Vector DB)에서 관련 정보를 검색하고, 이 정보를 기반 모델이 답변을 생성하는 과정에 통합시키는 방법론이자 모델로 정의할 수 있습니다.

![출처: Retrieval-augmented generation for knowledge-intensive nlp tasks (Lewis et al., 2020)](/assets/img/posts/Langchain으로-RAG-구현하기-1/Untitled.png)

출처: Retrieval-augmented generation for knowledge-intensive nlp tasks (Lewis et al., 2020)

그림에서처럼, RAG 모델은 크게 Retriever와 Generator로 구분됩니다. Retriever는 주어진 질문과 관련된 문서들(Passages)을 검색하고, Generator는 질문과 Passages를 바탕으로 답변을 생성합니다. Generator에 여러 Passages가 들어왔을 때, 이들을 병합하는 방법은 두 가지가 있습니다.

- RAG-Sequence Model: 질문 벡터와 유사한 문서들을 먼저 선택 → 각 문서와 질문 벡터 결합 → 결합된 데이터를 이용해 여러 응답 생성 → 문서와 질문의 유사도를 기반으로 응답들을 가중 평균하여 최종 응답 생성
- RAG-Token Model: 최종 응답에 대해 각 토큰을 생성할 때 마다 다른 문서를 검색 → 문서들과 질문 벡터를 결합 → 각각에 대한 출력 토큰의 확률 계산 → 문서와 질문의 유사도를 기반으로 토큰 출력 확률을 가중 평균하여 응답 생성

RAG 모델 자체의 모습은 논문에서 소개된 바와 같고, 실제로 구현을한다면 아래 아키텍처와 비슷한 모습으로 구현이 될 것입니다.

![Architecture of Chatbot with RAG 
(출처: [https://www.newsletter.swirlai.com/p/sai-notes-08-llm-based-chatbots-to](https://www.newsletter.swirlai.com/p/sai-notes-08-llm-based-chatbots-to))](Langchain%EC%9C%BC%EB%A1%9C%20RAG%20%EA%B5%AC%ED%98%84%ED%95%98%EA%B8%B0%20(1)/IMG_0108.png)

Architecture of Chatbot with RAG 
(출처: [https://www.newsletter.swirlai.com/p/sai-notes-08-llm-based-chatbots-to](https://www.newsletter.swirlai.com/p/sai-notes-08-llm-based-chatbots-to))

# Langchain Tutorial + RAG Model 써보기

Langchain은 LLM과 관련된 기술들을 간단히 구현할 수 있는 프레임워크입니다. LLM을 이용한 앱을 만들 때, Langchain을 사용하면 복잡한 파이프라인을 ~~치트 수준으로~~ 간단히 구현할 수 있고, RAG 역시 마찬가지입니다. 이 글에서는 Langchain과 OpenAI API를 통해 gpt 3.5 모델에 외부데이터를 추가적으로 학습시키는 과정을 소개합니다.

```python
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import Chroma
from langchain.storage import LocalFileStore
from langchain.chains import RetrievalQA

model = ChatOpenAI()
data_loader = UnstructuredFileLoader ("files/wiki.txt")
cache_dir = LocalFileStore("./.cache/")
```

먼저 OpenAI API Key를 .env 파일에 정의해줍니다(`OPENAI_API_KEY="sk-..."`). 그리고 Langchain을 통해 ChatOpenAI 모델을 호출합니다. 따로 모델을 지정해주지 않는다면, 기본 모델은 `gpt-3.5-turbo` 모델을 사용하게 됩니다. 

추가로 학습시켜 줄 데이터는 [우리나라의 집단주의에 대한 나무위키 텍스트](https://namu.wiki/w/%EC%A7%91%EB%8B%A8%EC%A3%BC%EC%9D%98)를 `./files/wiki.txt` 로 저장해두었습니다. `wiki.txt` 를 학습시키기 위해서는 자연어를 벡터로 변환하는 임베딩 과정이 필요합니다. 매번 임베딩을 반복하지 않도록 캐싱해두는 폴더도 함께 지정해줍니다.

```python
splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator="\n", 
    chunk_size=500,
    chunk_overlap=50
)

docs = data_loader.load_and_split(text_splitter=splitter)
embeddings = OpenAIEmbeddings()
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
```

예제를 위해 사용한 `wiki.txt` 가 길지는 않지만, 긴 텍스트를 학습시킨다는 가정 하에 텍스트를 분할해줍니다. 한 덩어리에 500자를 할당하고(`chunk_size`), 문장이 중간에서 잘릴 경우 의미없는 벡터들이 학습되기 때문에, 앞 덩어리의 끝 부분 50자는 겹치도록(`chunk_overlap`) 해줍니다. 만약 줄바꿈이 적절히 들어가있는 정제가 잘 된 텍스트라면 줄바꿈(`\n`)을 기준으로 문장을 자르도록 할 수도 있습니다(`separator`). 이 경우 분할된 문서들이 지정한 chunk size에 딱 맞지는 않습니다(동작 방식이 직관적이지는 않은 듯).

`from_tiktoken_encoder` 는 글자가 아닌 토큰을 기준으로 문서를 자르는 기능을 하며, `OpenAIEmbeddings` 는 통해 각 토큰을 표현하는 벡터를 만들어 줍니다. 임베딩 과정은 과금이 되는 부분이니, `CacheBackedEmbeddings` 에 임베딩 결과, 캐싱 디렉토리를 넣어주고 추가적인 과금을 막아줍니다. 

```python
vectorstore = Chroma.from_documents(docs, cached_embeddings)
retriever = vectorstore.as_retriever()
```

Langchain에서 제공하는 대표적인 Vector Store는 Chroma입니다. 다른 Vector Store를 사용하면 성능이 달리지기도 하는데, 여기서는 Chroma를 사용합니다. 캐싱된 임베딩을 받아 Vector Store에 저장하고, Vector Store를 Retriever로 변환해 `retriever` 변수에 할당합니다.

```python
chain = RetrievalQA.from_chain_type(
    llm=model,
    chain_type="map_reduce",
    retriever=retriever,
)

chain.run("한국의 집단주의에 대해 설명해줘")
# AIMessage(content='한국의 집단주의는 개인보다는 집단의 이익과 조화를 중시하는 사고방식을 말합니다. 한국 사회에서는 집단의 일원으로서의 역할과 책임을 강조하며, 개인의 욕구나 성취보다는 집단의 안정과 조화를 추구하는 경향이 있습니다. 이로 인해 한국인들은 식당에서도 한 가지 메뉴로 통일하려고 하며, 음식을 나눠먹는 일이 많습니다. 또한 혼자 음식을 먹는 것이 타인에게 나누어 주지 않는 것으로 간주되어 좋지 않게 여겨집니다. 이러한 집단주의는 가족, 친구, 동료와의 관계에서도 나타날 수 있으며, 개인의 선택이나 다른 음식을 먹는 것에 대해 신기해하는 경향이 있습니다. 이러한 문화적 배경으로 인해 한국 사회에서는 집단의 일원으로서의 역할과 집단의 조화를 중시하는 특징을 볼 수 있습니다.')

# ChatOpenAI() 기본 모델 답변과 비교
# '한국의 집단주의는 한국 사회에서 중요한 가치 중 하나로 여겨지는 개념입니다. 이는 개인의 이익보다는 집단의 이익을 우선시하는 사고방식을 의미합니다.\n\n한국 사회에서는 집단이 개인보다 더 큰 가치를 갖는다고 여깁니다. 따라서 개인의 목표나 행동은 종종 집단의 이익과 조화를 이루도록 조절되며, 개인의 권리와 의무는 집단의 안정과 번영을 위해 제한될 수 있습니다. 이러한 이유로 한국 사회에서는 개인의 자유보다는 집단의 안정과 조화를 중시하는 경향이 있습니다.\n\n한국의 집단주의는 한국인들이 다른 사람들과의 관계를 중요하게 여기고, 공동체 의식을 강조하는데 영향을 받았습니다. 가족, 동료, 직장 동료 등과의 관계는 매우 중요하며, 이들과의 상호작용을 통해 개인의 정체성이 형성됩니다. 따라서 한국 사회에서는 개인의 행동이 집단의 평판에 큰 영향을 미치기도 합니다.\n\n한국의 집단주의는 일부로는 안정적이고 화합적인 사회를 형성하는 데 도움이 되는 장점이 있습니다. 그러나 때로는 개인의 창의성과 독립적인 사고를 억압하거나 집단의 압력에 의해 개인의 권리와 자유가 제한될 수도 있습니다. 또한, 집단의 이익을 우선시하는 태도가 개인의 성장과 사회적 변화를 제한할 수도 있습니다.\n\n요약하자면, 한국의 집단주의는 집단의 이익을 우선시하는 사고방식으로, 한국 사회에서 중요한 가치 중 하나입니다. 이는 개인의 이익보다는 집단의 안정과 조화를 중시하며, 개인의 자유와 권리는 집단의 이익을 위해 제한될 수 있습니다.'
```

마지막으로 Langchain에서 제공하는 LCEL Chains 중 하나인 ****RetrievalQA를 불러와 chain을 정의해줍니다. 만들어진 chain에 질문을 넣어주면 RAG 만들기 튜토리얼은 끝입니다. 결과를 보면, 추가적으로 학습시켜준 데이터에 기반한 답변이 출력된 것을 확인할 수 있습니다. 다만 chain 내의 프롬프트를 변경하는 등의 세세한 커스터마이징을 위해서, 그리고 좀 더 정밀한 아키텍처를 만들기 위해서는 직접 chain을 구현해볼 필요가 있습니다. 다음 글에서는 LCEL을 통해 직접 MapReduce chain을 구현해 보겠습니다. 

- References
    
    Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive nlp tasks. *Advances in Neural Information Processing Systems*, *33*, 9459-9474.
    
    Xu, P., Ping, W., Wu, X., McAfee, L., Zhu, C., Liu, Z., ... & Catanzaro, B. (2023). Retrieval meets long context large language models. *arXiv preprint arXiv:2310.03025*.
    [https://stackoverflow.blog/2023/10/18/retrieval-augmented-generation-keeping-llms-relevant-and-current/](https://stackoverflow.blog/2023/10/18/retrieval-augmented-generation-keeping-llms-relevant-and-current/)
    
    [https://medium.com/@bijit211987/optimizing-rag-for-llms-apps-53f6056d8118](https://medium.com/@bijit211987/optimizing-rag-for-llms-apps-53f6056d8118)
    
    [https://towardsdatascience.com/embeddings-knowledge-graphs-the-ultimate-tools-for-rag-systems-cbbcca29f0fd](https://towardsdatascience.com/embeddings-knowledge-graphs-the-ultimate-tools-for-rag-systems-cbbcca29f0fd)
    
    [https://bea.stollnitz.com/blog/rag/](https://bea.stollnitz.com/blog/rag/)[https://medium.com/@bijit211987/optimizing-rag-for-llms-apps-53f6056d8118](https://medium.com/@bijit211987/optimizing-rag-for-llms-apps-53f6056d8118)
    
    [https://api.python.langchain.com/en/latest/chains/langchain.chains.retrieval_qa.base.RetrievalQA.html#langchain.chains.retrieval_qa.base.RetrievalQA](https://api.python.langchain.com/en/latest/chains/langchain.chains.retrieval_qa.base.RetrievalQA.html#langchain.chains.retrieval_qa.base.RetrievalQA)