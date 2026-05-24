# Context Governance Eval Design

## Goal

Make context governance measurable instead of relying on simple smoke tasks. The eval must prove that compression pressure occurred, that current task semantics survived, and that stale context did not contaminate final outputs.

## Design

The suite uses real-model eval runs because context governance depends on the Agent loop, tool evidence, request construction, and model behavior. YAML task metadata can now carry fixture files and per-task agent overrides. The real benchmark runner converts those fixtures into workspace files before each case and applies `agent_overrides.token_limit` so a task can intentionally force compression without changing global config.

The runner records context governance metadata from final Agent history:

- `compression_triggered`
- `compression_markers`
- `token_limit`
- `final_message_count`

Scoring adds three opt-in rules:

- `metadata_contains` checks nested execution metadata, including compression evidence.
- `output_excludes` rejects stale markers in the final answer.
- `file_excludes` rejects stale markers in generated artifacts.

## Evaluation Strategy

The context suite avoids trivial tasks. Each case creates pressure using long fixture files, then checks one specific governance behavior:

- latest user instruction overrides stale embedded instructions
- needles survive large-read pressure
- grounded results use workspace evidence
- multi-step state survives distractor context
- final artifacts are still correct after compression boundaries

The suite intentionally scores both positive requirements and negative stale markers. Passing requires a model to follow the current task, use tools, generate artifacts, and leave evidence that compression actually happened.

## Limitations

This does not prove perfect semantic preservation. It proves the implemented Agent can preserve task-critical semantics under representative pressure cases. A model can still fail individual tasks because of model behavior rather than harness behavior; comparing GPT, DeepSeek, and Claude on the same suite makes that distinction visible.
