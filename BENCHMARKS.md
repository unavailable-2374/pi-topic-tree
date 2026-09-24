# Benchmarks

Two questions matter for a context projection that hides history:

1. **Recall:** when the user later asks about something said many turns ago,
   does the model still answer correctly, and at what input cost?
2. **Classification:** does the classifier put turns into the right topic?

Both were measured on 2026-09-23 on the code this package was extracted from
(same projection, state and classifier prompt), with **DeepSeek V4.1 Flash**
as both the answering model and the classifier. The source material is 15
real long working sessions (software engineering, mostly Chinese, 20 to 336
user turns each). The raw conversations are private and not included here.

## 1. Recall of earlier details

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
| **thin + `recall_topic`** (this package's default) | **97.3%** | **12,905** | **35%** |
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

## 2. Topic classification

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
