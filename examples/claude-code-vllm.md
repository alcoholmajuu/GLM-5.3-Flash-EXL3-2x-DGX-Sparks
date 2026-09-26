# Claude Code on the GLM-5.3-Flash vLLM server

vLLM implements the Anthropic Messages API (`/v1/messages`), so Claude Code
can use this server as a drop-in backend — same SSH tunnel as opencode,
no extra proxy needed.

Verified with Claude Code 2.1.282 against the pinned build
(`0.1.dev20051+g487ecf187`, image `exl3v15-topkfix`): streaming, thinking
blocks, and tool_use round-trips all work with the serving flags already in
`scripts/launch_tp2.py` (`--enable-auto-tool-choice --tool-call-parser glm47
--reasoning-parser deepseek_r1`). That build also strips Claude Code's
per-request attribution header, so `--enable-prefix-caching` keeps hitting.

## Model selection (additive, nothing is overwritten)

The wrapper does **not** remap the `opus`/`sonnet`/`haiku` aliases. Instead:

- `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1` makes the `/model` picker
  list every model the server serves (`GET /v1/models`), appended after the
  built-in rows. Serve more names with additional `--served-model-name`
  values and they appear automatically.
- The session starts on the first served model (via `ANTHROPIC_DEFAULT_MODEL`,
  the lowest-priority default), so a `--model` flag, `ANTHROPIC_MODEL`, or a
  model you saved earlier still wins.
- Inside a session, `/model <served-name>` (or the picker) switches models;
  press `s` in the picker to apply it to the current session only.
- The wrapper points `CLAUDE_CONFIG_DIR` at `~/.claude-glm53`, so pressing
  `Enter` in the picker (which saves the model as default) writes to that
  separate config dir and never touches your normal `~/.claude/settings.json`.

Handy commands:

```sh
scripts/run_claude_code.sh --list-models          # what the server serves
scripts/run_claude_code.sh --model glm53-exl3     # pin for this launch
GLM53_MODEL=glm53-exl3 scripts/run_claude_code.sh # pin for these launches
```

## Quick start

```sh
# Opens the tunnel if port 8000 is not already listening (reuses it if it is),
# sets the env vars below, then execs claude. Host comes from GLM53_SSH_HOST
# or RANK0_HOST in .env.
scripts/run_claude_code.sh "your prompt"
```

## Manual setup

```sh
ssh -f -N -L 8000:127.0.0.1:8000 rank0   # same tunnel as opencode

ANTHROPIC_BASE_URL=http://localhost:8000 \
ANTHROPIC_API_KEY=dummy \
ANTHROPIC_AUTH_TOKEN=dummy \
CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1 \
ANTHROPIC_DEFAULT_MODEL=glm53-exl3 \
CLAUDE_CODE_MAX_CONTEXT_TOKENS=1048576 \
claude
```

| Variable | Value |
| --- | --- |
| `ANTHROPIC_BASE_URL` | Tunnel endpoint; server binds loopback on rank0 |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` | Any value (`AUTH_TOKEN` must be set) |
| `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY` | `1` — fill the `/model` picker from `/v1/models` |
| `ANTHROPIC_DEFAULT_MODEL` | Session default when nothing else selects a model |
| `CLAUDE_CODE_MAX_CONTEXT_TOKENS` | The server's real `--max-model-len` |

`CLAUDE_CODE_MAX_CONTEXT_TOKENS` should match the launch: `1048576` for the
1M agent config (`--max-model-len 1048576 --enable-prefix-caching`), `18432`
for the default recipe launch. Claude Code assumes a 200k window for unknown
model names otherwise and auto-compacts early. Appending `[1m]` to the model
name is equivalent; the suffix is stripped client-side and never reaches the
server.

### Optional: alias pinning instead of discovery

If you would rather have the `opus`/`sonnet`/`haiku` aliases resolve to the
local model (and hide that they do), pin them explicitly — this is opt-in
because it replaces what those aliases point at:

```sh
export ANTHROPIC_DEFAULT_OPUS_MODEL=glm53-exl3
export ANTHROPIC_DEFAULT_SONNET_MODEL=glm53-exl3
export ANTHROPIC_DEFAULT_HAIKU_MODEL=glm53-exl3
```

Without any pinning, background tasks (title generation, classifiers, …) run
on the session model because `ANTHROPIC_AUTH_TOKEN` is set, so leaving the
aliases alone costs nothing on this server.

## Notes

- One `ANTHROPIC_BASE_URL` per session routes **every** request to that
  backend. The built-in `claude-*` picker rows stay visible but a vLLM
  server answers them with a NotFound error — pick a served name instead.
- Model names must not contain `/` (Claude Code limitation; `glm53-exl3` is fine).
- Thinking output is exposed as Anthropic `thinking` blocks (from
  `--reasoning-parser`); tool calls come back as `tool_use` blocks with
  `stop_reason: "tool_use"`.
- The isolated `~/.claude-glm53` config dir starts empty (fresh trust prompt
  per project, no shared history). Copy your `~/.claude/settings.json` into
  it if you want the same permissions, or run with `--shared-config` /
  `GLM53_SHARED_CONFIG=1` to use the normal config.
- With discovery disabled (`GLM53_NO_DISCOVERY=1`) the wrapper instead adds a
  single picker entry via `ANTHROPIC_CUSTOM_MODEL_OPTION` (v2.1.78+), which
  also skips client-side model validation.
- Claude Code >= 2.1.154 sends non-standard roles in the messages array; this
  vLLM build normalizes them (upstream issue #44000). On older vLLM builds
  you would see 400 `Input should be 'user' or 'assistant'`.
- To persist env outside the wrapper, use `~/.claude/settings.json` (`"env"`
  section) — but note that `ANTHROPIC_BASE_URL` there applies to all your
  Claude Code usage, not just this server.
