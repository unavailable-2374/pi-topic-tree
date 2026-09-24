# pi-topic-tree

Topic-tree memory for the [Pi coding agent](https://github.com/earendil-works/pi).

Long conversations get expensive because every request re-sends the whole
linear history. This extension classifies each turn into a topic branch, keeps
a rolling summary per topic, and hands the model a small **projection** of the
conversation instead. Nothing is deleted: the session log is never rewritten,
and the model reads any earlier turn back on demand with the `recall_topic`
tool.

## Benchmark

15 real long coding-agent sessions (tool results included), DeepSeek V4.1
Flash, Pi 0.87.1:

| | Plain Pi | topic-tree (default) |
| --- | --- | --- |
| First compaction, 128k window | median turn 4 (1–15), 2–42 per session | 13 of 15 sessions never compact |
| First compaction, 200k window | median turn 8 (1–17) | never (15/15) |
| Questions about earlier details, right where plain Pi compacts | 58.1% correct | **86.0%** (full history: 86.0%) |
| Context sent at that point | 33,891 tokens (after compaction) | 25,013 tokens |

Compaction summaries keep goals and decisions but drop exact values like
hashes, ports and counts; topic-tree keeps the history and reads it back with
`recall_topic` when needed. The trade-off: a question that needs hidden
detail costs several recall round trips. Setup, per-condition numbers,
costs, classification accuracy and caveats are in
[BENCHMARKS.md](BENCHMARKS.md).

## Install

```bash
pi install git:github.com/unavailable-2374/pi-topic-tree
```

or try it for one run without installing:

```bash
pi -e git:github.com/unavailable-2374/pi-topic-tree --topic-tree
```

The extension is **off by default**. Turn it on per run with `--topic-tree`, or
for every run with `PI_TOPIC_TREE=1` in the environment.

## How it works

- **Classification.** After each prompt is handled, turns that have left the
  verbatim window are classified in the background (batched, never adding
  latency to the conversation) into topics such as "Database migration" or
  "Deployment". Topics can nest. Standing rules the user states ("never touch
  the release config", "reply in English") are collected into a pinned
  `global` node that is always sent.
- **Projection.** Only the request sent to the model changes. Two modes:
  - `thin` (default): from the first turn, older work is replaced by an
    index and pulled back with `recall_topic` on demand. With tool-loop
    folding (the default, below) every user message stays verbatim and older
    assistant and tool steps fold into a segment index; with folding off, the
    model sees the standing rules, a one-line-per-topic index and the last
    few turns.
  - `topic`: once the context passes a share of the window, the current
    topic's turns stay verbatim and other topics are replaced by their
    summaries.
- **Tool-loop folding** (thin mode, on by default). An agent working through
  one request can run hundreds of tool calls without a new user turn, so one
  turn alone can outgrow the window. Older tool-call segments of 8 calls each
  (`--topic-tree-fold-tool-calls <n>`, `0` turns folding off) fold into a
  deterministic index line (what was called on what, how many errors). Every user message and every message injected by other extensions
  stays verbatim; the latest call of each non-standard "controller" tool stays
  whole with its result; unresolved tool errors are quoted in the index. Folds
  happen in discrete steps, so between folds each request is a byte-identical
  extension of the previous one and the provider's prefix cache keeps hitting.
- **Recall.** `recall_topic` has three actions: `list` (topics and summaries),
  `get` (the turns of a topic like `t3`, or one hidden segment like `s1k2x`
  verbatim with its tool results), and `search` (BM25 over latin words and CJK
  bigrams across the whole conversation).

State lives in the session file as custom entries (`topic-tree.node`,
`topic-tree.assign`) and is rebuilt from the current branch on every use, so
forks, resumes and `/tree` navigation stay consistent.

## Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--topic-tree` | off | Enable the extension (or `PI_TOPIC_TREE=1`) |
| `--topic-tree-mode <thin\|topic>` | `thin` | Projection mode |
| `--topic-tree-model <provider/id>` | session model | Model used for classification and summaries |
| `--topic-tree-recent-turns <n>` | 2 | thin: completed turns kept verbatim before the current one |
| `--topic-tree-classify-every <n>` | 4 | thin: classify once this many turns have left the verbatim window |
| `--topic-tree-fold-tool-calls <n>` | 8 | thin: fold long tool loops, `n` tool calls per segment (`0` = off) |
| `--topic-tree-fold-every <n>` | 3 | thin: segments hidden per fold |
| `--topic-tree-min-turns <n>` | 8 | topic: user turns before classification starts |
| `--topic-tree-trigger-ratio <r>` | 0.5 | topic: context share of the window at which projection starts |
| `--topic-tree-target-ratio <r>` | 0.35 | topic: context share the projection aims for |
| `--topic-tree-keep-turns <n>` | 3 | topic: most recent user turns always kept verbatim |

A cheap, fast model for `--topic-tree-model` is usually enough. Reasoning
models work too; output ceilings are sized for their hidden thinking.

## Notes

- In short conversations the projection saves little, because the fixed
  system prompt and tool definitions dominate. The savings come in long
  sessions and long tool loops.
- Classification errors are recoverable by design: a failed or malformed
  classifier reply keeps turns on the last active topic (or a catch-all
  "General" topic), and every turn remains searchable.
- Other extensions can label a turn (e.g. why they re-prompted the agent) with
  `pi.appendEntry("topic-tree.turn-label", { turnKey, label })` while that turn
  is still open; the label shows in the fold index.

## Development

```bash
npm install --legacy-peer-deps
npm test
npm run typecheck
```

## License

MIT
