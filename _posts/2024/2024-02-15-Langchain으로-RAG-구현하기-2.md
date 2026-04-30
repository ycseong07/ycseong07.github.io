---
layout: post
title: "Langchain으로 RAG 구현하기 (2)"
date: 2024-02-15 09:00:00 +0900
category: blog
tags: []
---

*

# 서론

이전 글에서 Langchain을 사용해 RAG 방법론을 구현해보았습니다. 구현 후 아쉬웠던 부분은, Chain이 함수 하나로 추상화 되어있다는 점이었습니다. 물론 구현하기 편하기는 하지만, 프롬프트를 커스텀하거나 다른 기능들을 체인에 추가하기 위해 LCEL을 사용해 좀 더 low-level code를 짜보고자 했습니다. 자료를 찾다가 [노마드 코더의 풀스택 GPT](https://nomadcoders.co/fullstack-gpt) 강의를 보게 되었는데, 이 강의에서 구현한 중간 코드가 꽤 깔끔해서 해당 코드를 참조해 정리해 보았습니다.

# **LangChain Expression Language (LCEL)이란?**

Langchain에서 사용하는 ‘Chain’이라는 용어는 LLM이 최종 답변을 출력하기까지 필요한 기능들을 파이프처럼 이은 일련의 과정을 얘기합니다. `Prompt → LLM` 의 과정이 가장 작은 체인이라고 생각해도 좋을 것 같습니다. LangChain Expression Language (LCEL)는 Langchain에서 제공하는 기능들을 조합한 Chain을 마치 블록처럼 쉽게 분해, 조립할 수 있도록 설계한 프레임워크라고 볼 수 있습니다(LLM 분야의 scikit-learn 이라고 이해해도 무방합니다).

예를 들어, Chain을 구성하는데 필요한 기능들을 `prompt`, `model`, `output_parser` 라고 가정했을 때, LCEL은 `prompt`에서 생성된 결과가 `model`로, `model`에서 생성된 결과가 다시 `output_parser` 로 가는 파이프라인을 구축할 수 있도록 해줍니다.

```python
prompt = ...
model = ...
output_parser = ...

chain = prompt | model | output_parser
```

# Map Reduce Chain

- 이전 글의 코드
    
    ```python
    from langchain.chat_models import ChatOpenAI
    from langchain.document_loaders import UnstructuredFileLoader
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.storage import LocalFileStore
    from langchain.chains.retrieval_qa.base import RetrievalQA
    
    model = ChatOpenAI()
    data_loader = UnstructuredFileLoader ("files/wiki.txt")
    cache_dir = LocalFileStore("./.cache/")
    
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n", 
        chunk_size=500,
        chunk_overlap=50
    )
    
    docs = data_loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = Chroma.from_documents(docs, cached_embeddings)
    
    retriever = vectorstore.as_retriever()
    
    # 이 글에서는 아래 부분을 분리해서 직접 구현해봅니다
    chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="map_reduce",
        retriever=retriever,
    )
    
    chain.run("한국의 집단주의에 대해 설명해줘")
    ```
    

이전 글의 코드에서는 chain이 RetrievalQA를 통해 구현되어 있습니다. 이 경우, `chain_type="map_reduce"` 옵션을 통해  Map Reduce Chain을 사용할 수 있습니다. Map Reduce Chain은 긴 텍스트를 분할(Map)해 LLM에 입력하고, 각 출력값들을 통합(Reduce)해 다시 LLM에 입력하여 최종 답변을 출력시키는 방법입니다. 

![Map Reduce Chain의 구조](/assets/img/posts/Langchain으로-RAG-구현하기-2/Untitled.png)

Map Reduce Chain의 구조

`map_reduce` 외에도, chain_type 옵션의 기본값인 `stuff`나 `refine`, `map-rerank`를 사용할 수도 있습니다. 다만 이 글에서는 `map_reduce` 에 대해서만 구현합니다. 

- chain_type 간단 설명
    - `stuff`: 프롬프트에 있는 문서의 모든 텍스트를 한 번에 사용
    - `map_reduce`: 텍스트를 분할해 LLM에 입력하고, 각 출력값들을 모아 다시 LLM에 입력
    - `refine`: 텍스트를 분할하고, n 번째 텍스트를 LLM에 입력한 결과와 n+1 번째 텍스트를 함께 LLM에 입력
    - `map-rerank`: 텍스트를 분할해 LLM에 입력하고, 각 답변들의 정확도에 대해 점수를 매겨 가장 높은 점수를 받은 답변을 기반으로 최종 답변 생성

# Map Reduce Chain 구현

## Map 단계

먼저 입력 텍스트를 분할해 Map Prompt를 생성하고, 모델에 넘기도록 하는 부분입니다. Langchain을 사용할 때, 프롬프트에는 구체적인 지시를 `system`에, 질문은 `human`에 입력해줍니다. Map Prompt는 분할된 문서들을 요약하도록 요청하는 프롬프트입니다. `map_chain`은 `map_prompt`의 결과를 LLM에 넘기도록 정의해줍니다. 

```python
map_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            질문에 답하기 위해 필요한 내용이 제시된 문장들 내에 포함되어 있는지 확인하세요. 만약 포함되어있다면, 요약본을 반환해주세요. 만약 관련된 내용이 없다면 다음 문장들을 그대로 반환해주세요 : ''
            -------
            {context}
            """,
        ),
        ("human", "{question}"),
    ]
)

map_chain = map_prompt | model
```

`map_prompt`와 `map_chain`을 이용해 분할된 문서들을 요약한 문서들을 생성합니다. 요약할 문서들과 질문을 입력으로 받는 `map_docs`함수를 정의하고, `map_chain`의 결과를 두 줄 간격으로 이어 붙여 리턴합니다. chain을 중간 단계에서 실행해주어야 할 때는 `invoke`를 사용해줍니다.`RunnablePassthrough`는 앞서 입력받은 값을 그대로 전달하는 기능이고, `RunnableLambda`는 마치 함수를 lambda를 통해 실행하듯 `map_docs` 함수를  실행하도록 합니다.

```python
def map_docs(inputs):
    documents, question = inputs["documents"], inputs["question"]
    return "\n\n".join(
        map_chain.invoke({"context": doc.page_content, "question": question}).content
        for doc in documents
    )

map_results = {
    "documents": retriever,
    "question": RunnablePassthrough(),
} | RunnableLambda(map_docs)
```

## Reduce 단계

`reduce_prompt`는 `map_results`를 종합해 최종 답변을 생성하도록 작성해줍니다. `reduce_chain` 은 안에 `map_chain`이 포함되어 있는 구조입니다. `map_results` 를 context로, 사용자의 질문을 question으로 받고, 그 결과를 `reduce_prompt`에 전달해 LLM에 입력하도록 합니다. 

```python
reduce_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            주어진 문장들을 이용해 최종 답변을 작성해주세요. 만약 주어진 문장들 내에 답변을 위한 내용이 포함되어있지 않다면, 답변을 꾸며내지 말고, 모른다고 답해주세요.
            ------
            {context}
            """,
        ),
        ("human", "{question}"),
    ]
)

reduce_chain = {"context": map_results, "question": RunnablePassthrough()} | reduce_prompt | model

reduce_chain.invoke("한국의 집단주의에 대해 설명해줘")
```

이렇게 Map Reduce Chain을 구현하면 프롬프트를 변경하면서 테스트하거나, Chain 중간에 자잘한 기능이나 동작을 추가할 수 있게됩니다. 

- 최종 코드
    
    ```python
    from langchain.chat_models import ChatOpenAI
    from langchain.document_loaders import UnstructuredFileLoader
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.storage import LocalFileStore
    
    model = ChatOpenAI()
    data_loader = UnstructuredFileLoader ("files/wiki.txt")
    cache_dir = LocalFileStore("./.cache/")
    
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n", 
        chunk_size=500,
        chunk_overlap=50
    )
    
    docs = data_loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    
    vectorstore = Chroma.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    
    map_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                질문에 답하기 위해 필요한 내용이 제시된 문장들 내에 포함되어 있는지 확인하세요. 만약 포함되어있다면, 요약본을 반환해주세요. 만약 관련된 내용이 없다면 다음 문장들을 그대로 반환해주세요 : ''
                -------
                {context}
                """,
            ),
            ("human", "{question}"),
        ]
    )
    
    map_chain = map_prompt | model
    
    def map_docs(inputs):
        documents, question = inputs["documents"], inputs["question"]
        return "\n\n".join(
            map_chain.invoke({"context": doc.page_content, "question": question}).content
            for doc in documents
        )
    
    map_results = {
        "documents": retriever,
        "question": RunnablePassthrough(),
    } | RunnableLambda(map_docs)
    
    reduce_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                주어진 문장들을 이용해 최종 답변을 작성해주세요. 만약 주어진 문장들 내에 답변을 위한 내용이 포함되어있지 않다면, 답변을 꾸며내지 말고, 모른다고 답해주세요.
                ------
                {context}
                """,
            ),
            ("human", "{question}"),
        ]
    )
    
    reduce_chain = {"context": map_results, "question": RunnablePassthrough()} | reduce_prompt | model
    
    reduce_chain.invoke("한국의 집단주의에 대해 설명해줘")
    ```
    
- References
    - [https://python.langchain.com/docs/use_cases/summarization#option-2.-map-reduce](https://python.langchain.com/docs/use_cases/summarization#option-2.-map-reduce)
    - [https://python.langchain.com/docs/expression_language/](https://python.langchain.com/docs/expression_language/)
    - [https://nomadcoders.co/fullstack-gpt](https://nomadcoders.co/fullstack-gpt)