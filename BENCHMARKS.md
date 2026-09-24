# Benchmarks

Four questions matter for a context projection that hides history:

1. **When does compaction kick in?** How many turns does a session last
   before Pi has to compact it, with and without topic-tree?
2. **Recall after compaction:** at the turn where plain Pi compacts, how well
   does each approach answer questions about what came before?
3. **Recall in conversation:** later questions about earlier details, on the
   conversation text alone.
4. **Classification:** does the classifier put turns into the right topic?

All runs use **DeepSeek V4.1 Flash** as the answering model, the classifier
and the grader, and the same 15 real long working sessions (software
engineering with a coding agent, mostly Chinese, 21 to 326 user turns,
17,585 tool calls in total, 0.24M to 3.7M tokens per session with tool
results). The sessions are private and not included here. Sections 1 and 2
were measured on 2026-09-24 against this package's code and Pi 0.87.1;
sections 3 and 4 on 2026-09-23 against the code it was extracted from.

## 1. When compaction kicks in

**Setup.** Each session was replayed request by request, tool calls and
tool results included. At every model call we computed what Pi would send:
the whole history for plain Pi, the projection for topic-tree. Pi compacts
when a request exceeds the model's context window minus 16,384 reserved
tokens. Token counts are Pi's estimator, corrected per session by the
provider's actual count (1.27–1.57× the estimate for this material). The
fixed prompt was measured in Pi: 1,730 tokens, 1,983 with `recall_topic`.
The topic index was rebuilt from reference topic labels.

| Context window | Plain Pi: first compaction | Plain Pi: compactions per session | topic-tree, folding off | topic-tree default (folding on) |
| --- | --- | --- | --- | --- |
| 128k | 15/15 sessions, median **turn 4** (1–15) | 2–42 | 12/15 compact, 10 of them at the same turn as plain Pi | **13/15 never compact**; one at turn 3 (same as plain Pi: a single tool result there is huge), the 326-turn session at turn 57 (plain Pi: turn 1, then 41 more) |
| 200k | 15/15 sessions, median **turn 8** (1–17) | 1–23 | 4/15 compact | **15/15 never compact** |
| 1M | 2/15 sessions (turns 22 and 68) | 0–3 | never | never |

With folding on, the largest request in a session was 40k–165k tokens
(median 55k), against 0.24M–3.7M tokens of history. Without folding, one
turn's tool loop alone often outgrows a 128k window, because the current
turn is always sent whole. This is why folding is the default.

## 2. Recall right after compaction

**Setup.** For each session, the history was cut at the exact request where
plain Pi first compacts with a 128k window (turn 1 to 15, about 112k
tokens). Pi's own compaction code (`prepareCompaction` + `compact`, default
settings: keep about 20k recent tokens, summarise the rest) produced the
compacted context. The same history was then projected by topic-tree, with
topics classified by the extension's classifier. A model wrote 86 questions
(up to 6 per session) about specific facts in stretches of work that Pi's
compaction had summarised away: a partition name, a commit hash, a port, a
test count. Each question was answered in four ways; tool-using conditions
could call the extension's real `recall_topic`.

| Condition | Correct | Context sent (mean) | Input tokens per question (mean / median) |
| --- | ---: | ---: | ---: |
| Full history, no compaction (reference; needs a >128k window) | 86.0% | 112,355 | 108,284 / 105,129 |
| Plain Pi after compaction | 58.1% | 33,891 | 40,763 / 31,644 |
| **topic-tree default (folding on)** | **86.0%** | **25,013** | 161,668 / 88,741 |
| topic-tree, folding off | 82.6% | 72,736 | 158,846 / 100,302 |

- **Compaction loses a third of the details** that full history still has
  (58% vs 86%). The summary keeps goals and decisions but drops exact
  values.
- **topic-tree matches full history** (86.0% both) while sending a quarter
  of its context on ordinary turns. The model called `recall_topic` on 77
  of 86 questions, 4.2 calls on average.
- **Recall is where the cost goes.** Every question here needs something
  hidden, so each one pays for several recall round trips: 162k input
  tokens on average (89k median), more than just sending full history. On
  ordinary turns that need no recall, topic-tree sends about 25k tokens
  where plain Pi sends the whole history until it compacts.
- With folding off, the projection itself still exceeded the window on 20
  of 86 questions (4 sessions whose single turns are huge), so in real use
  those would have been compacted too. On the 66 that fit, folding off
  scored 54/66, folding on 56/66, plain compaction 37/66.

**Caveats.** Questions were generated from single stretches of work and
graded by a model (2 answers the grader could not parse were graded by
hand). The evaluation loop allows up to 12 recall rounds, then asks for an
answer without tools. Messages are sent to the model as plain text (tool
calls and results inline) in every condition.

## 3. Recall in conversation (text only)

This earlier run used only the conversation text: each turn's user message
and the assistant's prose, clipped, without tool calls or tool results. The
histories are therefore far smaller than in sections 1 and 2, and no
compaction is forced; "budget" below is a share of each history's size.

**Setup.** At three points in each of the 15 sessions (45 snapshots), a model
read an older turn (at least 6 turns back, so outside any verbatim window) and
wrote a follow-up question about a specific fact in it: a commit hash, a
setting, a file path, a number. That gave 113 probes: 26 about the topic the
conversation was on at that point, 87 about another topic. Each probe was
answered under several context strategies, and a separate model call graded
each answer against the reference as correct / partial / wrong.

"Input tokens" are the provider-counted input tokens for answering one
question, **including** every `recall_topic` round trip. They do not include
the background classification cost, which is listed separately below.

| Strategy | Correct | Input tokens (mean) | vs. full history |
| --- | ---: | ---: | ---: |
| Full history (no projection) | 94.7% | 36,378 | 100% |
| **thin + `recall_topic`** (without tool-loop folding) | **97.3%** | **12,905** | **35%** |
| topic projection + `recall_topic` (budget 30%) | 92.0% | 23,561 | 65% |
| topic projection + `recall_topic` (budget 15%) | 95.6% | 17,328 | 48% |
| topic projection, no recall tool (budget 30%) | 32.7% | 18,384 | 51% |
| Pi's built-in compaction (budget 30%) | 16.8% | 6,023 | 17% |
| Truncate to the most recent turns (budget 30%) | 19.5% | 10,685 | 29% |

By probe type:

| Strategy | Same topic (n=26) | Other topic (n=87) |
| --- | ---: | ---: |
| Full history | 92.3% | 95.4% |
| thin + `recall_topic` | 92.3% | 98.9% |
| topic projection + `recall_topic` (30%) | 84.6% | 94.3% |
| Pi's built-in compaction (30%) | 19.2% | 16.1% |
| Truncate (30%) | 38.5% | 13.8% |

What this shows:

- **Summaries alone do not preserve details.** Compaction, truncation and the
  topic projection without the recall tool answer only 17–33% of these
  questions. Exact facts (hashes, paths, numbers) are exactly what a summary
  drops.
- **Recall on demand does.** With `recall_topic`, thin mode matched or beat
  full history (97.3% vs 94.7%) at about a third of the input. A likely
  reason: the recalled turn arrives next to the question, while in full
  history it is buried tens of thousands of tokens back.
- The model used the tool on 112 of 113 probes in thin mode, about 2.2 calls
  per question.
- A second run with 76 fresh probes drawn from more recent turns gave full
  history 92.1%, thin + recall 94.7%, topic + recall 97.4%.

**Background cost.** Classification and topic summaries ran at about 1,500
input and 750 output tokens (hidden reasoning included) per conversation turn,
across 919 turns. That is paid once per turn in the background, not per
request.

**Caveats.** The questions are generated from single turns and graded by a
model, so they test factual lookup, not multi-turn reasoning. The thin
condition was assembled by the evaluation script following the extension's
layout (standing rules, topic index, last 2 turns) rather than by running the
extension inside Pi. Answer quality depends on the model actually calling
`recall_topic`; a model that rarely calls tools will do worse than shown.

## 4. Topic classification

**Setup.** Every turn of the same 15 sessions was labelled by a reference
annotator (a stronger model reading the whole conversation first) with the
topic it belongs to. Each case offers the classifier the candidate topics the
extension's lexical ranker would offer, and asks it to pick one or declare a
new topic. The classifier used the extension's own prompt
(`buildClassifyPrompt`).

| Case type | n | Accuracy |
| --- | ---: | ---: |
| Correct topic is among the candidates | 540 | 87.8% |
| Turn really opens a new topic | 31 | 38.7% |
| New topic, derived (true topic removed from the candidates) | 421 | 35.6% |

The classifier reliably places a turn into an existing topic, but it tends to
**merge a genuinely new direction into an existing topic** instead of opening
one. For this extension that is the safer error: nothing is lost, since every
turn stays retrievable through `recall_topic` search and topic `get`. The cost
is a coarser tree and blurrier topic summaries. The "derived" row is a
pessimistic lower bound: sibling topics from the same project remain in the
candidate list and are often defensible picks.
