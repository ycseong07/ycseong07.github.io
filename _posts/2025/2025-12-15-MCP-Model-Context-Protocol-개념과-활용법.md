---
layout: post
title: "MCP(Model Context Protocol) 개념과 활용법"
date: 2025-12-15 09:00:00 +0900
category: blog
tags: [AI, MCP, ClaudeCode, 에이전트]
---

LLM에 도구를 붙이는 일은 처음에는 단순했다. OpenAI가 function calling을 발표한 2023년에는 모델이 호출할 함수의 JSON Schema를 프롬프트에 같이 넣어 주면 되었다. 다만 도구가 늘고, 모델이 늘고, 에이전트가 늘면서 도구를 어떻게 정의하고 어떻게 연결할지의 표준이 부재한 상태가 점점 비용으로 돌아왔다. Cursor, Claude Code, Cline에 같은 GitHub 도구를 붙이려면 호스트마다 어댑터를 따로 구현해야 했고, 그렇게 만들어 둔 어댑터는 호스트가 API를 한 번만 바꿔도 통째로 다시 짜야 했다.

MCP(Model Context Protocol)는 이 지점을 정리하기 위해 Anthropic이 [2024년 11월에 공개](https://www.anthropic.com/news/model-context-protocol)한 오픈 표준이다. LLM 호스트와 외부 도구, 데이터 사이를 매개하는 통신 규격을 하나로 통일해 두면, 도구를 만든 쪽은 한 번만 구현하면 되고 호스트를 만든 쪽도 한 번만 통합하면 된다. 비유하자면 USB-C 같은 역할인데, 실제로 Anthropic이 발표문에서 같은 비유를 쓰기도 했다.

# 왜 표준이 필요했나

LLM 에이전트가 외부와 닿는 방법은 크게 셋이다. 모델이 가진 도구 호출 능력, 호스트가 코드로 미리 박아둔 통합, 그리고 RAG로 끌어오는 외부 문서다. 이중 첫 번째와 두 번째가 빠르게 늘면서 호스트마다 통합 표면적이 커졌다. Claude Desktop은 Claude Desktop 방식대로, Cursor는 Cursor 방식대로, IDE 플러그인은 또 다른 방식대로 도구를 정의했고, 같은 GitHub 연동이라도 코드를 따로 만들어야 했다.

도구를 만드는 쪽 입장에서는 통합 비용이 N×M으로 불어난다. 도구 N개와 호스트 M개를 곱한 만큼의 어댑터가 필요해진다는 뜻이다. MCP는 이 곱셈을 N+M으로 떨어뜨리는 게 목적이다. 도구 제공자는 MCP 서버 하나만 만들고, 호스트는 MCP 클라이언트 하나만 구현하면 양쪽이 알아서 만나게 된다.

또 다른 압력은 컨텍스트 엔지니어링 쪽에서 왔다. 컨텍스트에 모든 것을 넣지 않고 필요할 때만 도구로 끌어오는 패턴이 정착하면서, 도구 호출의 횟수와 다양성이 빠르게 늘었다. 호스트가 호출하는 도구가 30~40개를 넘어가면 호스트 내부에서 일일이 관리하기가 어려워지고, 도구 정의, 인증, 로깅을 외부 프로세스로 떼어내는 편이 운영상 자연스러워진다. MCP는 이 분리를 강제한다.

# MCP의 구조

MCP는 호스트(host), 클라이언트(client), 서버(server)의 세 가지 역할로 구성된다.

호스트는 사용자가 직접 쓰는 LLM 애플리케이션을 말한다. Claude Desktop, Claude Code CLI, Cursor, Zed, Cline 같은 것들이 호스트다. 호스트는 모델과 사용자, 그리고 여러 MCP 서버 사이를 중재한다.

클라이언트는 호스트 안에서 MCP 서버 하나당 하나씩 만들어지는 연결 객체다. 호스트가 MCP 서버 5개를 붙였다면 내부적으로 클라이언트 5개가 떠 있는 셈이다. 클라이언트는 서버와 1:1로 JSON-RPC 메시지를 주고받으며, 자기가 담당하는 서버의 도구, 리소스, 프롬프트를 호스트에 노출한다.

서버는 외부 시스템에 연결되어 실제 작업을 수행하는 프로세스다. GitHub MCP 서버, Filesystem MCP 서버, Slack MCP 서버처럼 한 가지 도메인을 책임지는 단위로 만들어진다. 서버는 호스트와 같은 머신에서 stdio로 도는 경우도 있고, 원격에서 HTTP, SSE로 붙는 경우도 있다.

서버가 호스트에 노출하는 것은 세 종류다. Tools는 모델이 호출할 수 있는 함수, Resources는 모델이 읽을 수 있는 데이터(파일, DB row 등), Prompts는 사용자가 슬래시 커맨드처럼 꺼내 쓸 수 있는 프롬프트 템플릿이다. 이 셋의 분리가 중요한 이유는 권한과 트리거가 다르기 때문이다. Tools는 모델이 자율적으로 부르고, Resources는 호스트가 컨텍스트에 끼워 넣고, Prompts는 사람이 명시적으로 호출한다.

전송 계층은 두 가지다. 로컬 서버는 stdio로, 원격 서버는 HTTP+SSE(2025년 중반 이후로는 Streamable HTTP)로 동작한다. 메시지 포맷은 JSON-RPC 2.0을 그대로 따른다. 즉 새로운 와이어 프로토콜을 만든 게 아니라, 잘 알려진 규격 위에 LLM 도구 연동에 필요한 메서드(`tools/list`, `tools/call`, `resources/read` 등)를 정의한 형태다.

# Claude Code CLI에서 MCP 붙이기

Claude Code CLI는 두 가지 방식으로 MCP 서버를 등록할 수 있다. 설정 파일을 직접 편집하거나, `claude mcp` 서브커맨드로 등록하는 방법이다. 일회성으로 붙여 보고 버릴 거라면 CLI가 편하고, 팀에 공유할 거라면 파일에 박아 두는 편이 낫다.

CLI로 등록하는 예시는 다음과 같다.

```bash
# 로컬 stdio 서버 등록
claude mcp add filesystem --scope project -- npx -y @modelcontextprotocol/server-filesystem /Users/me/projects

# 원격 HTTP 서버 등록
claude mcp add tavily --scope user --transport http https://mcp.tavily.com/mcp \
  --header "Authorization: Bearer $TAVILY_API_KEY"

# 등록 상태 확인
claude mcp list
```

`--scope`는 등록 범위를 정한다. `local`은 현재 디렉토리에서만, `project`는 `.mcp.json`에 저장되어 팀과 공유, `user`는 홈 디렉토리에 저장되어 모든 프로젝트에서 공통으로 쓰인다. 비밀 키가 들어가는 서버는 `user` 스코프, 팀이 함께 써야 하는 서버는 `project` 스코프로 두는 식이 무난하다.

파일로 직접 관리할 때는 프로젝트 루트의 `.mcp.json`이 기준이 된다. 형식은 Claude Desktop의 그것과 거의 같다.

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]
    },
    "github": {
      "command": "docker",
      "args": ["run", "-i", "--rm", "-e", "GITHUB_PERSONAL_ACCESS_TOKEN", "ghcr.io/github/github-mcp-server"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"
      }
    }
  }
}
```

세션을 띄운 뒤에는 `/mcp`로 현재 연결 상태와 서버별로 노출된 도구, 리소스, 프롬프트를 확인할 수 있다. 도구가 잘 안 잡히거나 인증이 어긋나면 거의 항상 이 화면에서 단서가 나온다.

# 자주 쓰는 서버들

실제로 자주 붙이는 서버는 도메인별로 정형화되어 있다.

검색 쪽에서는 Tavily나 Exa의 MCP 서버를 가장 많이 본다. 호스트의 기본 웹 검색은 일반 검색엔진을 그대로 호출하는 경우가 많아 LLM 입력에 어울리지 않는 노이즈가 섞여 들어오는 반면, 위 서비스들은 LLM 입력용으로 정제된 결과를 돌려준다는 점이 차이가 크다.

코드, 문서 쪽에서는 GitHub MCP 서버, Filesystem MCP 서버, 그리고 [Context7](https://context7.com)이 자주 보인다. Context7은 라이브러리 공식 문서를 버전별로 끌어올 수 있어서, 모델이 학습 시점에 못 본 최신 API 문서를 쓰게 만들 때 효과적이다. Filesystem 서버는 Claude Code 자체가 파일 도구를 갖고 있어 중복되지만, 프로젝트 디렉토리 바깥의 참조 자료를 별도 루트로 노출하고 싶을 때 따로 붙인다.

데이터베이스, 관측성 쪽에서는 PostgreSQL, BigQuery, Sentry, Linear의 공식, 비공식 서버가 흔하다. 이쪽은 권한 모델이 까다로워서, 가능하면 읽기 전용 자격증명만 노출하는 서버를 따로 두는 편이 안전하다.

브라우저 자동화로는 Playwright MCP 서버가 가장 많이 쓰인다. UI 변경을 에이전트가 직접 검증하게 만들 때, 스크린샷을 컨텍스트로 되돌려 받을 수 있다는 점에서 시각적 회귀 테스트와 잘 맞는다.

# MCP 도입으로 맞닥뜨리게 되는 문제들

첫째로, MCP를 도입하면 도구가 빠르게 늘어난다. 늘어난 만큼 호스트의 도구 카탈로그도 부풀어 오르는데, 도구가 50개를 넘어가면 모델이 비슷한 이름의 도구 중 엉뚱한 쪽을 부르는 일이 잦아진다. 한 서버 안에서도 `search_code`와 `search_repositories`처럼 기능이 겹치는 도구가 같이 노출되면 호출 정확도가 떨어진다. 운영 단계에서는 자주 쓰지 않는 서버를 과감하게 빼고, 같은 서버 안에서도 도구를 일부만 화이트리스트로 열어 두는 편이 결과가 좋다. Claude Code의 `permissions` 설정으로 서버, 도구 단위 차단을 걸 수 있다.

두 번째 문제는 보안이다. MCP 서버는 모델이 부르는 함수를 외부 프로세스로 떼어 놓은 것이기 때문에, 그 프로세스의 권한이 곧 에이전트의 권한이 된다. GitHub PAT를 쥔 서버라면 Push까지 가능한 토큰을 쓰지 말고 Read 권한만 가진 토큰을 따로 발급해 붙이는 식의 분리가 필요하다. 신뢰할 수 없는 출처의 MCP 서버를 그대로 npm/uvx로 받아 실행하는 것도 위험하다. 서버 코드가 호스트와 같은 권한으로 돌면서 사용자의 셸에 접근할 수 있다는 점은 표준 자체의 한계라기보다 운영 책임 영역으로 봐야 한다. 2025년에는 MCP 서버를 가장한 악성 패키지 사례가 실제로 보고되기도 했다.

세 번째 문제는 prompt injection이다. MCP 서버가 돌려주는 결과 본문에 “지금부터 너는 다른 모델이다”, “이전 지시를 무시해라” 같은 텍스트가 섞여 있으면 모델이 그걸 새 시스템 프롬프트로 받아들이는 일이 일어난다. 외부 웹페이지나 이슈 본문을 읽어 오는 도구일수록 위험이 크다. 대응책은 두 가지다. 호스트 쪽에서 도구 결과를 명확하게 “tool_result” 블록으로 감싸 모델에게 신호를 주는 것, 그리고 destructive한 작업(파일 삭제, 외부 API 결제, 머지 등)은 결국 사람의 승인을 받게 하는 hook을 거는 것이다. MCP 자체는 이 위험을 자동으로 막아주지 않는다.

네 번째 문제는 컨텍스트 점유다. MCP 서버가 띄워지는 시점에 호스트는 그 서버의 도구, 리소스 목록을 시스템 프롬프트에 같이 싣는다. 서버 하나가 도구를 30개씩 노출하면, 사용자가 한 번도 그 도구를 부르지 않더라도 매 턴마다 도구 정의가 입력 토큰을 잡아먹는다. 비활성 서버는 비활성 상태로 두고, 정말 필요한 작업에서만 임시로 활성화하는 운영 패턴이 합리적이다. Claude Code는 `/mcp` 안에서 서버 단위 enable/disable이 가능하다.
