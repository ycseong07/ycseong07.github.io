---
layout: post
title: ".claude/settings.json 퍼미션 시스템 이해하기"
date: 2026-03-21 09:00:00 +0900
category: blog
tags: [AI, ClaudeCode, settings, 퍼미션]
---

Claude Code를 사용가는 거의 모든 사용자들이 맞닥뜨리는 문제가 있다. 처음에는 모든 도구 호출을 일일이 승인하다가, 같은 명령을 몇 번쯤 허용하고 나면 'Allow always'를 누르게 되고, 결국에는 `--dangerously-skip-permissions` 플래그를 검색하기 시작한다. 그리고 며칠 뒤, 에이전트가 잘못된 디렉토리에서 `git reset --hard`를 돌리거나, 운영 DB에 붙은 클라이언트로 마이그레이션을 다시 실행하는 사고가 한 번쯤 난다.

이 흐름의 본질은 퍼미션 시스템을 '귀찮은 게이트'로 다루는 데 있다. 이를 해결할 수 있는 파일이 Claude Code의 `settings.json` 파일이다. 이 파일은 어떤 도구 호출을 자동으로 허용하고, 어떤 호출은 매번 묻고, 어떤 호출은 아예 막을지를 선언해두는 정책 파일이라고 볼 수 있다. 이 글에서는 `settings.json`의 위치와 우선순위, 퍼미션 모드, 룰 작성법, 그리고 실전에서 자주 쓰는 패턴을 정리한다.

# settings.json 위치

Claude Code는 설정을 한 파일에 모아두지 않고 네 위치에 분산한다. 각각이 다른 신뢰 경계를 표현하기 때문이다.

가장 위에는 엔터프라이즈 정책 파일이 있다. macOS는 `/Library/Application Support/ClaudeCode/managed-settings.json`, Linux/WSL은 `/etc/claude-code/managed-settings.json`에 위치하고, 사용자가 임의로 풀 수 없는 조직 차원의 제약을 박아두는 자리다. 특정 명령을 무조건 막아야 한다면 여기에 들어간다.

그 다음이 사용자 설정인 `~/.claude/settings.json`이다. 모든 프로젝트에 공통으로 적용되는 개인 환경설정이 들어간다. 자주 쓰는 패키지 매니저 명령이나 git read-only 명령처럼, 어느 프로젝트에서 켜도 어차피 허용할 만한 것들이 이 자리에 들어가야 자연스럽다.

프로젝트 단위로는 두 파일이 있다. `.claude/settings.json`은 팀과 공유되도록 git에 커밋하는 파일이고, `.claude/settings.local.json`은 개인 환경에만 남기고 싶은 설정을 두는 파일이다. 후자는 기본 `.gitignore`에 자동으로 추가된다. 팀이 공유해야 할 코딩 컨벤션 관련 허용 규칙은 전자에, 본인 머신에만 있는 비밀키 경로나 사이드 도구 같은 건 후자에 두면 충돌이 줄어든다.

우선순위는 엔터프라이즈 > 명령행 인자 > `.claude/settings.local.json` > `.claude/settings.json` > `~/.claude/settings.json` 순이다. 같은 키가 여러 파일에 있다면 위 순서대로 덮어 쓴다. 다만 퍼미션 룰만큼은 단순한 덮어쓰기가 아니라 누적된다. `deny`에 한 번이라도 걸리면 어느 레이어에서든 막히고, `allow`는 모든 레이어의 합집합으로 동작하는 식이다. 즉 상위 레이어에서 `deny`로 박아둔 규칙을 하위 레이어가 `allow`로 풀어줄 수는 없다.

# 퍼미션 모드

`settings.json`의 `permissions.defaultMode` 키가 세션의 기본 모드를 결정한다. CLI에서 `--permission-mode`로 한 번만 다르게 띄울 수도 있고, 세션 안에서는 Shift+Tab으로 plan 모드와 acceptEdits 모드를 토글할 수 있다.

`default`는 가장 보수적인 모드다. 도구 호출 중 `allow`로 명시되지 않은 것이 나오면 사용자에게 매번 묻는다. 처음 프로젝트를 들여다볼 때 적합하다.

`acceptEdits`는 파일 편집 도구(Read, Edit, Write, NotebookEdit 등)를 자동 승인하되 그 외 도구는 default와 동일하게 다룬다. 코드 작성을 본격적으로 맡길 때 가장 자주 켜는 모드다. Bash나 외부 호출은 여전히 사용자 확인을 거치므로, 파일 편집 루프만 빠르게 돌리고 싶을 때 적절하다.

`plan` 모드는 모든 쓰기 도구를 막고 읽기 전용 도구만 허용한다. 모델이 작업 계획을 세우는 단계에서 코드를 건드리지 못하게 강제할 때 쓴다. 큰 리팩토링을 시작하기 전에 plan 모드로 의도를 충분히 확인하고, 합의된 계획만 acceptEdits로 옮겨 실행하는 흐름이 안정적이다.

마지막으로 `bypassPermissions`는 모든 퍼미션을 무시하는 모드다. 일회성 자동화 스크립트나 격리된 컨테이너 안에서 빠른 실험을 돌릴 때만 쓰는 게 안전하다. 메인 작업 환경의 `defaultMode`로 두면 사실상 하네스 없는 자율 에이전트가 되어버린다. CLI 플래그 `--dangerously-skip-permissions`도 같은 효과를 내는데, 이 플래그가 위험한 이름을 가진 데에는 이유가 있다.

# 퍼미션 룰의 구조

퍼미션은 `permissions` 키 아래의 `allow`, `ask`, `deny` 세 배열로 표현한다. 각 배열은 도구 패턴 문자열의 리스트다. 같은 호출이 `deny`에 매칭되면 무조건 차단되고, `allow`에 매칭되면 자동 통과되며, `ask`에 매칭되거나 어디에도 매칭되지 않으면 사용자에게 묻는다.

```json
{
  "permissions": {
    "defaultMode": "acceptEdits",
    "allow": [
      "Read",
      "Edit",
      "Bash(git status)",
      "Bash(git diff:*)",
      "Bash(npm run test:*)"
    ],
    "ask": [
      "Bash(git push:*)",
      "Bash(npm publish:*)"
    ],
    "deny": [
      "Bash(rm -rf:*)",
      "Bash(git push --force:*)",
      "Read(./.env)",
      "Read(./secrets/**)"
    ]
  }
}
```

도구 이름만 적으면 그 도구의 모든 호출에 매칭된다. `Read`라고만 두면 어떤 경로의 Read든 자동 허용이라는 뜻이다. 괄호 안의 인자는 도구마다 다른 매칭 규칙을 따른다.

Bash 룰은 명령행 문자열에 대한 prefix 매칭이다. `Bash(npm run test:*)`는 `npm run test`로 시작하는 명령 전체를 허용한다. 콜론과 별표를 붙이지 않으면 정확히 그 문자열만 허용되니, `Bash(git status)`는 `git status` 단독 호출만 통과시키고 `git status -s`는 다시 묻는다. 실제 운영에서는 거의 모든 룰을 `:*`로 끝맺게 된다. 다만 prefix 매칭의 한계를 이해해 두는 게 중요하다. `Bash(npm run test:*)`로 허용해 두면 `npm run test && rm -rf /`도 prefix가 일치해 통과한다. 셸이 명령을 분리해서 실행하는 일까지 퍼미션이 막아주지는 않는다는 뜻이다. 그래서 위험한 명령은 `allow`로 푸는 대신 `deny`에 따로 박는 편이 안전하다.

Edit, Read 같은 파일 도구의 룰은 gitignore-style glob을 따른다. `Read(./src/**)`는 `src` 디렉토리 하위 전체를, `Read(~/.zshrc)`는 홈 디렉토리의 특정 파일을 가리킨다. 절대경로(`/`로 시작), 홈 상대경로(`~/`), 워크스페이스 상대경로(`./`)가 모두 지원된다. `.env`나 비밀키 같은 파일은 `deny` 쪽에 미리 넣어두는 패턴이 거의 표준이다.

WebFetch는 도메인 단위로 끊는다. `WebFetch(domain:docs.anthropic.com)` 형태로 도메인을 명시한다. 검색 결과를 그대로 따라가는 호출이 늘다 보면 잘 모르는 도메인을 모델이 부르는 일이 생기는데, allow 리스트로 신뢰 도메인만 열어 두는 편이 안전하다.

MCP 도구는 `mcp__<server>__<tool>` 형태로 노출되고, 서버 단위 또는 도구 단위로 룰을 작성할 수 있다. `mcp__github`만 적으면 GitHub MCP 서버가 노출하는 모든 도구를 의미하고, `mcp__github__create_pull_request`처럼 적으면 단일 도구만 가리킨다. 다만 MCP 도구는 prefix 매칭이나 인자 매칭을 지원하지 않는다. 같은 도구라도 위험한 호출과 안전한 호출을 구분하고 싶다면, MCP 서버 쪽에서 도구를 분리해 노출하거나 hook 단계에서 거르는 편이 현실적이다.

# 그 외 자주 쓰는 키들

`settings.json`은 퍼미션 외에도 몇 개의 키를 더 받는다. 운영에 직접 영향을 주는 것들만 추리면 다음과 같다.

`additionalDirectories`는 워크스페이스 바깥의 디렉토리를 추가로 노출한다. 모노레포에서 인접 패키지를 함께 보게 하거나, 참조용 문서가 다른 위치에 있을 때 유용하다. 다만 여기에 노출된 경로는 퍼미션 규칙도 똑같이 받기 때문에, `Read(/path/to/extra/**)` 형태로 명시적으로 허용해 줘야 한다.

`env`는 세션이 시작될 때 주입할 환경변수를 정의한다. 비밀키를 직접 박지 말고 `${VAR}` 형태의 expansion만 두는 패턴이 안전하다. `apiKeyHelper`도 비슷한 목적인데, 외부 명령을 실행해 키를 받아오게 한다. 보통은 1Password CLI나 macOS Keychain을 호출하는 헬퍼를 두는 식이다.

`hooks`는 별도로 다룰 만한 주제지만 퍼미션과의 관계는 짚어둘 필요가 있다. `PreToolUse` hook은 모델의 도구 호출이 실제 실행되기 직전에 끼어들어 호출을 막거나 변형할 수 있다. 퍼미션이 정적인 룰 매칭이라면, hook은 코드로 평가되는 동적 게이트다. `Bash(*)`를 일괄 allow한 뒤, hook으로 'main 브랜치에서의 직접 push만 거절' 같은 조건문 검사를 거는 식의 조합이 가능하다.

# allow로 둬도 되는 것들

룰 설계의 핵심은 결국 'allow에 둘 것'과 'ask로 남길 것'의 경계를 어디에 그을지에 달려 있다. 너무 좁게 잡으면 매번 묻느라 작업이 끊기고, 너무 넓게 잡으면 사고 가능성이 커진다. 사용해 본 경험으로는 다음 기준이 무난했다.

읽기 전용 명령은 거의 다 allow로 둬도 무방하다. `git status`, `git diff`, `git log`, `ls`, `cat`, `pwd`, `which` 같은 명령은 부작용이 없다. `find`나 `grep`도 해당된다. 다만 `curl`, `wget`은 외부로 데이터를 보낼 수 있어 ask로 두는 편이 낫다.

빌드, 테스트, 린트는 경험상 프로젝트 단위 `.claude/settings.json`에 박아둬도 된다. `npm run test:*`, `npm run lint:*`, `npm run build` 같은 것들이 매번 사용자 확인을 거치면 acceptEdits 모드의 의미가 사라진다.

git 쓰기 작업은 보수적으로 다룬다. `git add`, `git commit`, `git checkout`은 로컬에 한정되니 allow에 둬도 큰 문제는 없지만, `git push`, `git reset --hard`, `git rebase`, `git branch -D`는 ask 또는 deny에 두는 편이 안전하다. 특히 force push는 deny에 명시적으로 박아두면 모델이 우회할 여지가 줄어든다.

파일 편집은 `acceptEdits` 모드와 deny 룰의 조합으로 다루는 게 깔끔하다. 기본은 자동 통과시키되, `.env`, 시크릿 디렉토리, `.git/` 내부, CI 워크플로 파일처럼 사고 영향이 큰 경로만 deny에 따로 박는다. CI 워크플로의 경우 모델이 디버깅 목적으로 `--no-verify`나 빌드 스킵 같은 옵션을 끼워 넣는 일이 종종 있다.

MCP 도구는 서버 단위로 한 번 검토하고 도구 단위로 좁혀 가는 편이 좋다. 처음에는 `mcp__<server>` 전체를 ask로 두고, 자주 쓰는 도구만 allow로 옮기면 운영 중 문제가 생긴 도구를 빠르게 격리할 수 있다.

# 내가 겪은 오류

내가 처음 설정할 때 겪었던 실수들을 정리해둔다.

첫 번째는 prefix 매칭이다. 앞서 언급한 대로 `Bash(npm run test:*)` 같은 룰은 명령 시작 부분만 본다. 셸 구문(`&&`, `;`, `|`, 백틱)으로 명령이 결합된 경우 뒤쪽이 어떤 명령이든 통과한다. 이 점 때문에 운영에서는 위험 명령을 deny에 명시하는 작업과, hook으로 셸 메타문자 사용을 검출하는 작업이 같이 가야 한다.

두 번째는 deny 우선의 구조를 잊어버리는 것이다. 전역 `~/.claude/settings.json`에 `Bash(git push:*)`를 deny로 박아뒀다가, 특정 프로젝트에서만 이를 풀고 싶다고 `.claude/settings.json`의 `allow`에 같은 항목을 추가하는 일이 있다. 이때는 deny가 이긴다. 풀고 싶다면 사용자 설정 쪽의 deny를 수정해야 한다.

세 번째는 `bypassPermissions` 모드의 위험성이다. 격리 컨테이너에서만 쓰려고 만든 모드인데, 일단 손에 익으면 메인 환경에서도 그냥 켜고 쓰는 사례가 늘어난다. 이 모드에서는 `deny` 룰조차 무시되므로, 본인이 의도하지 않은 명령이 실행될 가능성이 항상 열려 있다. 자율 실행이 필요한 상황이라면 권한을 푸는 대신 worktree나 컨테이너 격리를 함께 가져가는 편이 안전하다.

네 번째는 settings.local.json을 git에 올리는 실수다. 기본 `.gitignore`에 들어 있긴 하지만, 직접 만든 프로젝트의 경우 빠져 있을 수 있다. 본인 환경 전용 비밀키 경로나 자동 승인 룰이 팀원 환경에 적용되는 일은 가능하면 피해야 한다.

# 정리

퍼미션 시스템은 모델이 무엇을 할 수 있는지의 경계를 정적으로 선언하는 역할을 한다. 어디까지 자동 허용할지를 정하는 작업은 곧 'AI에게 어디까지 위임할 것인가'의 결정이고, `settings.json`은 그 결정을 코드처럼 관리할 수 있게 해준다. 다만 정적 룰만으로 모든 케이스를 막을 수는 없다. 셸 결합이나 의미 단위의 위험 판단처럼 코드 평가가 필요한 영역은 hook이 채우게 되고, 둘은 같은 하네스의 안쪽과 바깥쪽을 나눠 맡는 관계다.

간단히 정리하면, 첫째, `defaultMode`는 작업의 위험도에 맞춰 plan/default/acceptEdits 사이에서 고르고 `bypassPermissions`는 격리 환경에서만 쓴다. 둘째, allow/ask/deny는 누적되며 deny가 항상 이긴다는 점을 인지하고, 위험 명령은 allow로 푸는 대신 deny에 박아 둔다. 셋째, 정적 룰로 표현되지 않는 동적 조건은 hook으로 옮긴다. 이 세 축을 분리해서 다루기 시작하면 설정 파일이 깔끔해지고, 같은 사고가 두 번 반복될 여지도 줄어든다.
