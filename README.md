# makefun-aggregator-02

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/dockiemhiep/makefun-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/dockiemhiep/makefun-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

Single-file Node 22 live free-model aggregator (`main.ts`, zero external dependencies) serving both OpenAI `POST /v1/chat/completions` and Anthropic `POST /v1/messages` (JSON and Server-Sent Events) plus auth-aware `GET /v1/models`. Parses the trial brief (`$MAKEFUN_PROMPT_FILE`), registers its bound port and exact free-model catalog with `$MAKEFUN_ENV_URL`, routes each prompt to the highest-rated eligible OpenRouter `:free` model through the trial gateway, validates token and tool fidelity before committing responses, normalizes streaming chunk/event order and token usage across wires, records upstream refusal trails in `x-makefun-upstream-failures`, and fails over deterministically across live and cassette-replay runs.

## Run it

```sh
# Print the registered free-model catalog JSON and exit 0
MAKEFUN_PROMPT_FILE=/path/to/prompt.txt node main.ts --list

# Run a live trial against the make.fun environment and gateway
MAKEFUN_ENV_URL=http://127.0.0.1:8080 \
MAKEFUN_PROMPT_FILE=/path/to/prompt.txt \
OPENAI_BASE_URL=http://127.0.0.1:8081/v1 \
OPENAI_API_KEY=trial-token \
node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=r027wqk6&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>
