# Compression Pipeline Design

## Summary

Mini Agent will refactor context compression into a staged pipeline that runs before each LLM API call. The first implementation keeps `compact` and `summarize` in `mini_agent/summarizer.py`, but separates them into different classes:

- `MessageCompactor`: deterministic compression that does not call an LLM.
- `MessageSummarizer`: existing LLM-based semantic summary fallback.
- `CompressionPipeline`: orchestration layer that runs compact steps first, then invokes summarization only if the request still exceeds budget.

The first version implements the first three layers from the proposed diagram: Tool Result Budget, Snip, and Micro-Compact. Context Collapse is left as an explicit future extension point. Auto-Compact continues to use the existing `MessageSummarizer`.

## Goals

- Reduce token pressure before API calls without immediately invoking LLM summarization.
- Preserve active tool-call chains so assistant `tool_calls` and `tool` results never become inconsistent.
- Keep deterministic compression separate from semantic summarization.
- Keep the first implementation small by using `mini_agent/summarizer.py` instead of adding a new module.
- Add tests that prove token usage decreases and protected messages remain intact.

## Non-Goals

- Do not implement real tool-result spill-to-disk in the first version.
- Do not implement Context Collapse read-time projection in the first version.
- Do not change checkpoint file format.
- Do not add embeddings, vector search, or background compression.
- Do not remove the existing `MessageSummarizer` fallback.

## Current State

The current call flow in `Agent.run()` is:

```text
RequestContextBuilder.build(...)
MessageSummarizer.summarize_if_needed(...)
RequestContextBuilder.build(...)
llm.generate(...)
```

This means Mini Agent currently has two compression behaviors:

- `RequestContextBuilder` selects recent messages within a history budget and protects the active tool chain.
- `MessageSummarizer` calls the LLM to summarize old execution rounds when token usage exceeds `token_limit`.

There is no unified deterministic compaction stage before summarization. Large tool results can force the system into LLM summarization even when cheaper deterministic clipping would be enough.

## Proposed Architecture

### MessageCompactor

`MessageCompactor` owns deterministic transformations. It accepts full message history plus token budget settings and returns a compacted message list with metadata describing which layers changed the history.

Responsibilities:

- Estimate token usage using existing helpers from `context_budget.py`.
- Clip oversized tool results.
- Remove old non-protected history when budget pressure remains.
- Apply age-based clipping to older tool results.
- Preserve message order and tool-call pairing.

It must not call the LLM.

### MessageSummarizer

`MessageSummarizer` keeps the existing LLM summary behavior. It remains the final fallback when deterministic compaction is insufficient.

The current `summarize_if_needed(...)` API can remain available for compatibility, but `CompressionPipeline` should become the preferred caller inside `Agent.run()`.

### CompressionPipeline

`CompressionPipeline` orchestrates the full pre-request process:

```text
measure request
if under threshold:
  return original messages

run MessageCompactor:
  layer 1: Tool Result Budget
  layer 2: Snip
  layer 3: Micro-Compact

measure request again
if still over limit:
  call MessageSummarizer

return compacted messages
```

`Agent.run()` should call this pipeline before building the final request for `llm.generate(...)`.

## Layer Design

### Layer 1: Tool Result Budget

Purpose: prevent one large tool result from dominating the context.

Behavior:

- Inspect `role="tool"` messages.
- If a tool result exceeds a per-message token budget, replace its content with a head/tail clipped version.
- Include a clear marker showing the original result was compacted.
- Preserve:
  - `role`
  - `name`
  - `tool_call_id`
  - success/error semantic text already present in content

First-version marker:

```text
[Tool result compacted: original estimated tokens=<N>, retained head/tail excerpts]
```

No disk spill happens in version one. The "stored to disk" part of the diagram becomes a future implementation detail behind this same layer.

### Layer 2: Snip

Purpose: remove old history only after large tool results have been clipped.

Behavior:

- Drop oldest non-system messages until the request is closer to budget.
- Preserve:
  - the initial system message
  - harness summary system messages
  - the latest user turn
  - any active tool-call chain
  - assistant/tool pairs that would otherwise become invalid if only one side is removed

This layer overlaps with existing `RequestContextBuilder` selection. The first implementation should keep `RequestContextBuilder` as the final request shaper, but put obvious old-message snipping in `MessageCompactor` so the persistent agent history can also shrink before summarization.

### Layer 3: Micro-Compact

Purpose: reduce old tool outputs without losing the shape of the conversation.

Behavior:

- Walk tool results from newest to oldest.
- Keep newer tool results more intact.
- Clip older tool results more aggressively.
- Do not alter assistant tool-call messages.
- Do not clip tool results in the active tool chain.

Initial retention tiers:

```text
newest 2 tool results: keep up to 1200 tokens each
next 4 tool results: keep up to 600 tokens each
older tool results: keep up to 250 tokens each
```

These constants should live on `MessageCompactor` so tests can override them without adding config surface area yet.

### Layer 4: Context Collapse

Not implemented in the first version.

The code should reserve a named method or comment-level extension point for future read-time projection. It should not change runtime behavior.

### Layer 5: Auto-Compact

Purpose: semantic fallback when deterministic compaction is not enough.

Behavior:

- Reuse existing `MessageSummarizer`.
- Preserve the existing harness summary message format:

```text
[Harness Execution Summary]
```

- Preserve active tool rounds exactly as the current summarizer does.

## Budget Rules

The pipeline should use two thresholds:

- `near_limit_ratio = 0.85`: run deterministic compaction when estimated request tokens exceed 85% of `token_limit`.
- `hard_limit_ratio = 1.0`: invoke LLM summarization if compacted request tokens still exceed `token_limit`.

The estimate should include:

- selected request messages
- system prompt layers
- tool schemas

The implementation can reuse `RequestContextBuilder.build(...)` to produce the measured request view, then use `estimate_messages_tokens(...)` and `estimate_tool_tokens(...)` from `context_budget.py`.

## Agent Integration

`Agent.__init__` creates:

```python
self.compression_pipeline = CompressionPipeline(
    compactor=MessageCompactor(token_limit=self.token_limit),
    summarizer=self.message_summarizer,
    request_context_builder=self.request_context_builder,
    token_limit=self.token_limit,
    renderer=self.renderer,
)
```

`Agent.run()` changes from direct summarizer use to:

```python
self.messages = await self.compression_pipeline.compress_before_request(
    messages=self.messages,
    api_total_tokens=self.api_total_tokens,
    tools=tool_list,
)
```

Then it builds `request_messages` as it does today.

## Error Handling

- Deterministic compaction should not raise on malformed content. If content is not a string, leave it unchanged.
- If LLM summarization fails, preserve existing fallback behavior from `MessageSummarizer`: return the local summary text.
- If compaction cannot reduce enough, still call the API with the best available compacted context after summarization.

## Testing Strategy

Add `tests/test_compression.py` for deterministic pipeline behavior.

Required tests:

- Large tool result is clipped and keeps `tool_call_id`.
- Active tool chain is preserved raw.
- Snip removes older user/assistant history before latest user turn.
- Micro-Compact clips older tool results more aggressively than newer results.
- Pipeline does not call `MessageSummarizer` when deterministic compaction brings the request below limit.
- Pipeline calls `MessageSummarizer` when deterministic compaction is insufficient.

Keep existing tests:

- `tests/test_summarizer.py`
- `tests/test_request_context.py`
- `tests/test_agent.py`

Expected verification command:

```bash
uv run pytest tests/test_compression.py tests/test_summarizer.py tests/test_request_context.py tests/test_agent.py -q
```

Before completion, run:

```bash
uv run pytest -q
```

## Documentation Updates

Update `README_CN.md` and `docs/ARCHITECTURE_AND_DEVELOPMENT_CN.md` after implementation to describe the staged compression flow:

```text
Tool Result Budget -> Snip -> Micro-Compact -> Auto-Compact fallback
```

Context Collapse should be documented as future work, not current behavior.
