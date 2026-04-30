---
layout: post
title: "[UDEMY 강의 리뷰] 랭체인 - LangChain 으로 LLM 기반 애플리케이션 개발하기"
date: 2024-03-31 09:00:00 +0900
category: blog
tags: []
---

# 서론

최근 LLM을 사용한 앱을 프로토타이핑할 때 LangChain을 사용해보고 있었는데, 공식 Documents와 관련 포스팅을 찾아보는 것 이상의 공부가 필요함을 느꼈다. LLM 모델 자체에 대한 지식과, LLM을 앱, 나아가 서비스로 만들기 위한 지식은 사뭇 결이 달라서, 개발을 직접 할 때는 이전에 알고 있던 모델에 대한 지식이 거의 도움이 되지 않았고, 이를 구현하고 실제로 활용하기 위해 새로운 지식을 다시금 익혀야 했다. 이런 경우에는 커리큘럼이 짜여져 있는 강의를 쭉 듣는 것이 정보를 처음부터 체계적으로 정리할 때 도움이 되는 경우가 많았고, 마침 글또에서 UDEMY 강의 쿠폰을 지원해주는 프로모션이 있어 ‘[랭체인 - LangChain 으로 LLM 기반 애플리케이션 개발하기](https://www.udemy.com/course/langchain-korean/?referralCode=1D782720957BBADE7628&utm_medium=udemyads&utm_source=wj-krweb&utm_campaign=udemykorea_course&utm_content=langchain-korean&utm_term=202312&couponCode=ACCAGE0923)’ 강의를 수강하기로 했다.

# 리뷰

이 강의는 LLM을 고도화하는 방법들 중 Prompt Enginnering, RAG에 대해 다룬다. 수강하면서 가장 좋았던 점은 강의자(Edwn Marco)가 직접 만든 그림 자료들이 풍부하고, 깔끔하게 편집해 보여준다는 점이었다. LLM 앱을 만들 때 데이터가 처리되는 복잡한 흐름을 이해하는 것이 어려웠는데, 그런 부분들을 말로만 설명하는 게 아니라 구체적인 도표나 시각자료를 통해 이해시켜주어서, 굉장히 친절하게 느껴졌다. 

개인적으로 가장 유용했던 부분은 ReAct Agent를 직접 구현하는 부분이었다. ReAct Agent란 Prompt Engineering을 위해 만들어진 ReAct(Reason + Act) 프레임워크를 활용한 에이전트이고, ReAct는 CoT(Chain of Thought)를 보완한 프롬프트 엔지니어링 기법(Question - Thought - Action - Observation 구조를 갖는다)이다. LangChain은 복잡한 LLM 파이프라인을 간단하게 만들어주는 좋은 도구임에는 틀림없지만, 그 만큼 추상화가 많이 되어있어 ReAct, 혹은 다른 기능을 사용할 때마다 ‘이 기능이 도대체 어떻게 작동하는 걸까?’하는 의문이 따라다녔다. 그래서 강의에서 설명해주는 자세한 내용이 꽤 도움이 되었고, 강의자가 직접 구현해주는 것이 ReAct 프레임워크를 이해하는 데 큰 도움이 되었다.

![Untitled](/assets/img/posts/UDEMY-강의-리뷰-랭체인---LangChain-으로-LLM-기반-애플리케이션-개발하/Untitled.png)

이 강의의 또 한 가지 장점은 지금까지 알려진 Prompt Enginnering 방법을(적어도 내가 봐 왔던 자료들 중에서는) 최대한 짧고 명확하게 정리해준다는 점이다. 최근 서점에 Prompt Enginnering에 대한 서적이 많이 많이 나와있지만, 이 강의의 [섹션 0: 프롬프트 엔지니어링 이론] 부분을 보면 따로 책이나 자료를 준비해 공부하지 않아도 될 정도로 필요한 내용을 가르쳐준다.

# 마치며

새로운 것을 익힐 때는 이미 앞서간 사람들의 방식을 한 번 답습하는 것이 큰 도움이 된다. 그것이 온전히 내 지식이 아니더라도, 책이나 강의를 쭉 따라가고 나면 스스로 전체적인 로드맵을 그릴 수 있는 상태가 된다. 물론 이 강의를 듣고 현업에서 LLM 앱을 개발할 수 있게 되는 것은 아니다. 그러나 아무 준비가 되어있지 않을 상태로 마라톤의 출발선에 서는 것과, 미리 코스를 익혀두고 워밍업을 충분히 한 후 출발선에 서는 것과는 큰 차이가 있다. 향후 LLM을 활용한 앱 개발 후기를 주제로 글을 포스팅할 것을 다짐하며 리뷰를 마친다.

* 이 글은 글또 9기에서 활동하는 기간동안 UDEMY 강의 쿠폰을 지원받아 작성된 글입니다

![Untitled](/assets/img/posts/UDEMY-강의-리뷰-랭체인---LangChain-으로-LLM-기반-애플리케이션-개발하/Untitled 1.png)