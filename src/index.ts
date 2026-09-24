/**
 * Topic-tree memory
 *
 * Classifies each conversation turn into a topic branch and keeps a rolling
 * summary per topic, so the model is handed a small projection of the
 * conversation instead of the whole linear history. The session log is never
 * rewritten: `recall_topic` reads any earlier turn back on demand.
 *
 * Off by default. Enable with `--topic-tree`.
 *
 * Two projections:
 *   thin  (default) from the first turn, older work is replaced by an index
 *         and pulled back with `recall_topic`. Long tool loops fold in
 *         segments (every user message stays verbatim); with folding off,
 *         only standing constraints, a topic index and the last few turns
 *         are sent.
 *   topic once the context passes a share of the window, the current topic's
 *         turns stay verbatim and the other topics are summarised.
 *
 * Flags:
 *   --topic-tree-mode <thin|topic>     projection (default thin)
 *   --topic-tree-recent-turns <n>      thin: completed turns kept verbatim before the current one (default 2)
 *   --topic-tree-classify-every <n>    thin: classify once this many turns have left the window (default 4)
 *   --topic-tree-model <provider/id>   model for classification and summaries (default: session model)
 *   --topic-tree-min-turns <n>         topic: user turns before classification starts (default 8)
 *   --topic-tree-trigger-ratio <r>     topic: context share of the window at which projection starts (default 0.5)
 *   --topic-tree-target-ratio <r>      topic: context share the projection aims for (default 0.35)
 *   --topic-tree-keep-turns <n>        topic: most recent user turns always kept verbatim (default 3)
 *   --topic-tree-fold-tool-calls <n>   thin: fold long tool loops, n tool calls per segment (default 8, 0 = off)
 *   --topic-tree-fold-every <n>        thin: segments hidden per fold when folding tool loops (default 3)
 *
 * PI_TOPIC_TREE=1 in the environment turns it on without the flag.
 *
 * Usage:
 *   pi install git:github.com/unavailable-2374/pi-topic-tree
 *   pi --topic-tree
 */

import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { Model } from "@earendil-works/pi-ai";
import { complete } from "@earendil-works/pi-ai/compat";
import { type ExtensionAPI, type ExtensionContext, estimateTokens } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import {
	buildClassifyPrompt,
	buildSummaryPrompt,
	CLASSIFY_SYSTEM_PROMPT,
	type ClassifyResult,
	type ClassifyTurn,
	fallbackClassification,
	mergeGlobalFacts,
	parseClassifyResponse,
	SUMMARY_SYSTEM_PROMPT,
} from "./classify.ts";
import { projectContext, projectSegments, projectThin } from "./project.ts";
import {
	ASSIGN_ENTRY,
	type AssignEntryData,
	applyAssignments,
	applyNode,
	buildState,
	GLOBAL_TOPIC_ID,
	lexicalScores,
	messagesFromEntries,
	NODE_ENTRY,
	RECALL_TOOL_NAME,
	rankTopics,
	segmentId,
	splitSegments,
	splitTurns,
	type TopicNode,
	type TopicTreeState,
	type Turn,
	topicTurnCounts,
	turnText,
} from "./state.ts";

const CLASSIFY_BATCH = 12;
const MAX_BATCHES_PER_PASS = 4;
const CANDIDATE_TOPICS = 8;
const CLASSIFY_LIMITS = { user: 1200, assistant: 600, toolResult: 0 };
// Tool results stay out of summaries: the assistant's prose reports what they
// showed, while their raw text is mostly protocol output that would crowd out
// the conversation's own content.
const SUMMARY_LIMITS = { user: 2000, assistant: 1500, toolResult: 0 };
const RECALL_LIMITS = { user: 6000, assistant: 6000, toolResult: 2000 };
// One named segment comes back with its tool results at length: a later step
// may depend on the exact bytes of one (a file hash, an error message).
const SEGMENT_RECALL_LIMITS = { user: 6000, assistant: 6000, toolResult: 8000, toolCall: 2000 };
const SUMMARY_INPUT_CHARS = 24000;
// Output ceilings include a reasoning model's hidden thinking: at 2000 tokens
// a reasoning model spent 1985 on thinking and truncated the classifier JSON,
// which silently sent a whole first batch to the fallback topic.
const CLASSIFY_MAX_TOKENS = 8000;
const SUMMARY_MAX_TOKENS = 4000;
const CLASSIFY_ATTEMPTS = 2;

const NUMERIC_FLAGS = {
	"topic-tree-min-turns": { fallback: 8, description: "User turns before topic classification starts" },
	"topic-tree-trigger-ratio": { fallback: 0.5, description: "Context share of the window at which projection starts" },
	"topic-tree-target-ratio": { fallback: 0.35, description: "Context share the topic projection aims for" },
	"topic-tree-keep-turns": { fallback: 3, description: "Most recent user turns always kept verbatim" },
	"topic-tree-recent-turns": {
		fallback: 2,
		description: "Thin mode: completed turns kept verbatim before the current one",
	},
	"topic-tree-classify-every": {
		fallback: 4,
		description: "Thin mode: classify once this many turns have left the verbatim window",
	},
	"topic-tree-fold-tool-calls": {
		fallback: 8,
		description: "Thin mode: fold long tool loops, this many tool calls per segment (default 8, 0 = off)",
	},
	"topic-tree-fold-every": {
		fallback: 3,
		description: "Thin mode: segments hidden per fold when folding tool loops",
	},
} as const;

function textOf(message: { content: unknown }): string {
	const content = message.content;
	if (!Array.isArray(content)) return typeof content === "string" ? content : "";
	return content
		.filter((block): block is { type: "text"; text: string } => block?.type === "text")
		.map((block) => block.text)
		.join("\n");
}

// A package installed both globally and per project would load twice. Whichever loads first owns the session.
const REGISTERED = Symbol.for("pi.topic-tree.registered");

export default function topicTreeExtension(pi: ExtensionAPI) {
	registerTopicTree(pi);
}

export function registerTopicTree(pi: ExtensionAPI, env: NodeJS.ProcessEnv = process.env) {
	const registry = globalThis as { [REGISTERED]?: boolean };
	if (registry[REGISTERED]) return;
	registry[REGISTERED] = true;
	const enabledByEnv = /^(1|true|yes|on)$/i.test(env.PI_TOPIC_TREE?.trim() ?? "");

	pi.registerFlag("topic-tree", {
		description: "Organise long conversations into topic branches and project context by topic",
		type: "boolean",
		default: false,
	});
	pi.registerFlag("topic-tree-model", {
		description: "provider/model-id used for topic classification and summaries (default: session model)",
		type: "string",
	});
	for (const [name, flag] of Object.entries(NUMERIC_FLAGS)) {
		pi.registerFlag(name, { description: flag.description, type: "string" });
	}
	pi.registerFlag("topic-tree-mode", {
		description:
			"Projection: thin (index + recent turns + recall, default) or topic (current topic verbatim + summaries)",
		type: "string",
	});

	const enabled = () => pi.getFlag("topic-tree") === true || enabledByEnv;
	const thinMode = () => pi.getFlag("topic-tree-mode") !== "topic";
	const numberFlag = (name: keyof typeof NUMERIC_FLAGS): number => {
		const raw = pi.getFlag(name);
		const value = Number(raw);
		if (raw !== undefined && raw !== "" && Number.isFinite(value) && value >= 0) return value;
		return NUMERIC_FLAGS[name].fallback;
	};
	const segmentOptions = () => ({
		previousSegments: numberFlag("topic-tree-recent-turns"),
		toolCallsPerSegment: numberFlag("topic-tree-fold-tool-calls"),
		foldEverySegments: numberFlag("topic-tree-fold-every"),
	});

	let running = false;
	let rerun = false;
	let lifetime = new AbortController();

	async function callModel(ctx: ExtensionContext, system: string, prompt: string, maxTokens: number): Promise<string> {
		const flag = pi.getFlag("topic-tree-model");
		const configured = typeof flag === "string" && flag ? flag.split("/") : undefined;
		const model: Model<any> | undefined =
			configured && configured.length >= 2
				? ctx.modelRegistry.find(configured[0], configured.slice(1).join("/"))
				: ctx.model;
		if (!model) throw new Error("no model available for topic-tree");
		const auth = await ctx.modelRegistry.getApiKeyAndHeaders(model);
		if (!auth.ok) throw new Error(auth.error);
		const response = await complete(
			model,
			{
				systemPrompt: system,
				messages: [{ role: "user", content: [{ type: "text", text: prompt }], timestamp: Date.now() }],
			},
			{ apiKey: auth.apiKey, headers: auth.headers, env: auth.env, maxTokens, signal: lifetime.signal },
		);
		if (response.stopReason === "error" || response.stopReason === "aborted") {
			throw new Error(response.errorMessage ?? `topic-tree model call ${response.stopReason}`);
		}
		return textOf(response);
	}

	function persistNode(state: TopicTreeState, node: TopicNode) {
		applyNode(state, node);
		pi.appendEntry<TopicNode>(NODE_ENTRY, node);
	}

	async function classifyBatch(ctx: ExtensionContext, state: TopicTreeState, batch: Turn[]): Promise<ClassifyResult> {
		const turns: ClassifyTurn[] = batch.map((turn) => ({ key: turn.key, text: turnText(turn, CLASSIFY_LIMITS) }));
		const candidateIds = new Set([
			...state.lastTopicIds,
			...rankTopics(state, turns.map((turn) => turn.text).join("\n"), CANDIDATE_TOPICS),
		]);
		const candidates = [...candidateIds]
			.map((id) => state.nodes.get(id))
			.filter((node): node is TopicNode => !!node && node.id !== GLOBAL_TOPIC_ID);
		let failure: unknown;
		for (let attempt = 1; attempt <= CLASSIFY_ATTEMPTS; attempt++) {
			try {
				const reply = await callModel(
					ctx,
					CLASSIFY_SYSTEM_PROMPT,
					buildClassifyPrompt(turns, candidates, state.lastTopicIds),
					CLASSIFY_MAX_TOKENS,
				);
				return parseClassifyResponse(reply, turns, state);
			} catch (error) {
				if (lifetime.signal.aborted) throw error;
				failure = error;
			}
		}
		console.warn(`[topic-tree] classification fell back: ${failure instanceof Error ? failure.message : failure}`);
		return fallbackClassification(turns, state);
	}

	/** Classify every unassigned completed turn on the branch, then refresh the summaries it touched. */
	async function maintain(ctx: ExtensionContext) {
		const entries = ctx.sessionManager.getBranch();
		const state = buildState(entries);
		const { turns } = splitTurns(messagesFromEntries(entries));
		let pending: Turn[];
		if (thinMode()) {
			// Only turns that have left the verbatim window need an index entry, and
			// they are classified in batches to keep the upkeep per turn low.
			const visible = Math.max(0, numberFlag("topic-tree-recent-turns"));
			const hidden = turns.slice(0, Math.max(0, turns.length - visible));
			pending = hidden.filter((turn) => !state.assignments.has(turn.key));
			if (pending.length < numberFlag("topic-tree-classify-every")) return;
		} else {
			if (turns.length < numberFlag("topic-tree-min-turns")) return;
			pending = turns.filter((turn) => !state.assignments.has(turn.key));
		}
		const touched = new Map<string, Turn[]>();
		for (let b = 0; b < MAX_BATCHES_PER_PASS && b * CLASSIFY_BATCH < pending.length; b++) {
			const batch = pending.slice(b * CLASSIFY_BATCH, (b + 1) * CLASSIFY_BATCH);
			const result = await classifyBatch(ctx, state, batch);
			for (const topic of result.newTopics) {
				persistNode(state, { id: topic.ref, parentId: topic.parentId, title: topic.title, summary: "" });
			}
			applyAssignments(state, result.assignments);
			pi.appendEntry<AssignEntryData>(ASSIGN_ENTRY, { assignments: result.assignments });
			if (result.globalFacts.length > 0) {
				const global = state.nodes.get(GLOBAL_TOPIC_ID)!;
				persistNode(state, { ...global, summary: mergeGlobalFacts(global.summary, result.globalFacts) });
			}
			for (const assignment of result.assignments) {
				const turn = batch.find((item) => item.key === assignment.turnKey);
				if (!turn) continue;
				for (const id of assignment.topicIds) touched.set(id, [...(touched.get(id) ?? []), turn]);
			}
		}

		for (const [id, topicTurns] of touched) {
			const node = state.nodes.get(id);
			if (!node) continue;
			let text = topicTurns.map((turn) => turnText(turn, SUMMARY_LIMITS)).join("\n\n");
			if (text.length > SUMMARY_INPUT_CHARS) text = text.slice(-SUMMARY_INPUT_CHARS);
			try {
				const summary = (
					await callModel(ctx, SUMMARY_SYSTEM_PROMPT, buildSummaryPrompt(node, text), SUMMARY_MAX_TOKENS)
				).trim();
				if (summary) persistNode(state, { ...node, summary });
			} catch (error) {
				if (lifetime.signal.aborted) return;
				console.warn(`[topic-tree] summary of ${id} failed: ${error instanceof Error ? error.message : error}`);
			}
		}

		if (pending.length > MAX_BATCHES_PER_PASS * CLASSIFY_BATCH) rerun = true;
	}

	let pass: Promise<void> = Promise.resolve();

	/** Starts a maintenance pass, or folds this request into the running one. Resolves when that pass ends. */
	function schedule(ctx: ExtensionContext): Promise<void> {
		if (running) {
			rerun = true;
			return pass;
		}
		running = true;
		pass = (async () => {
			try {
				do {
					rerun = false;
					await maintain(ctx);
				} while (rerun && !lifetime.signal.aborted);
			} catch (error) {
				if (!lifetime.signal.aborted) {
					console.warn(`[topic-tree] maintenance failed: ${error instanceof Error ? error.message : error}`);
				}
			} finally {
				running = false;
			}
		})();
		return pass;
	}

	let toolRegistered = false;
	let active = false;
	pi.on("session_start", () => {
		lifetime = new AbortController();
		if (!enabled()) return;
		active = true;
		// Only a session that opted in gets the extra tool in its tool list.
		if (!toolRegistered) {
			toolRegistered = true;
			registerRecallTool();
		}
	});

	pi.on("session_shutdown", () => {
		lifetime.abort();
	});

	// Classification runs after the whole prompt has been handled, in the
	// background: it never adds latency to an interactive conversation. A
	// one-shot run (print/json) exits right after this event, so there the pass
	// is awaited — otherwise it would be killed before writing anything.
	pi.on("agent_end", async (_event, ctx) => {
		if (!active) return;
		const done = schedule(ctx);
		if (ctx.mode === "print" || ctx.mode === "json") await done;
	});

	pi.on("context", (event, ctx) => {
		if (!active || !ctx.model) return;
		const state = buildState(ctx.sessionManager.getBranch());
		if (thinMode()) {
			// With tool-loop folding on (the default), every user message stays verbatim at any
			// age, older turns' assistant and tool content folds in discrete,
			// append-only steps (turn boundaries are segment boundaries), and long
			// tool loops inside a turn fold the same way. The provider prefix cache
			// then only misses at a fold. Topics stay available through
			// recall_topic; they are not written into the index, which the
			// background classifier would otherwise rewrite.
			if (numberFlag("topic-tree-fold-tool-calls") > 0) {
				const folded = projectSegments(event.messages, state, segmentOptions());
				return folded ? { messages: folded.messages } : undefined;
			}
			const thin = projectThin(event.messages, state, { previousTurns: numberFlag("topic-tree-recent-turns") });
			return thin ? { messages: thin.messages } : undefined;
		}
		const window = ctx.model.contextWindow;
		const projection = projectContext(event.messages, state, {
			triggerTokens: Math.floor(window * numberFlag("topic-tree-trigger-ratio")),
			targetTokens: Math.floor(window * numberFlag("topic-tree-target-ratio")),
			keepRecentTurns: numberFlag("topic-tree-keep-turns"),
			estimate: (message: AgentMessage) => estimateTokens(message),
		});
		return projection ? { messages: projection.messages } : undefined;
	});

	function registerRecallTool() {
		pi.registerTool({
			name: RECALL_TOOL_NAME,
			label: "Recall Topic",
			description:
				"Read back earlier turns of this conversation that are not shown verbatim. " +
				"action=list lists topics; action=get returns the turns of one topic, or one hidden tool-call segment by its index id (s…); action=search finds turns by keywords across the whole conversation.",
			promptSnippet: "Recall earlier conversation turns by topic or keyword",
			promptGuidelines: [
				"When a topic digest says turns are not shown and you need an exact detail from them, call recall_topic before answering.",
			],
			parameters: Type.Object({
				action: Type.Union([Type.Literal("list"), Type.Literal("get"), Type.Literal("search")]),
				topic: Type.Optional(
					Type.String({
						description: "Topic id (e.g. t3) or a hidden segment id from the segment index (s…) for action=get",
					}),
				),
				query: Type.Optional(Type.String({ description: "Keywords for action=search" })),
				limit: Type.Optional(Type.Number({ description: "Maximum turns to return (default 5)" })),
			}),
			async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
				const entries = ctx.sessionManager.getBranch();
				const state = buildState(entries);
				const messages = messagesFromEntries(entries);
				const { turns } = splitTurns(messages);
				const limit = Math.max(1, Math.min(20, Math.floor(params.limit ?? 5)));
				const reply = (text: string) => ({ content: [{ type: "text" as const, text }], details: undefined });

				// Segment ids (s…) name tool-loop segments in the fold index.
				const wanted = params.topic?.trim();
				if (params.action === "get" && wanted && /^s[0-9a-z]+$/.test(wanted) && !state.nodes.has(wanted)) {
					const size = numberFlag("topic-tree-fold-tool-calls");
					const segments = size > 0 ? splitSegments(messages, size).turns : turns;
					const found = segments.find((segment) => segmentId(segment) === wanted);
					if (!found)
						return reply(`Unknown segment: ${wanted}. Segment ids are listed in the hidden-segment index.`);
					return reply(`Segment ${wanted}, verbatim:\n\n${turnText(found, SEGMENT_RECALL_LIMITS)}`);
				}

				if (params.action === "list") {
					const counts = topicTurnCounts(state);
					const lines = [...state.nodes.values()].map(
						(node) =>
							`[${node.id}] ${node.title}${node.parentId ? ` (under ${node.parentId})` : ""} — ${counts.get(node.id) ?? 0} turn(s)` +
							(node.summary ? `\n  ${node.summary.split("\n")[0].slice(0, 160)}` : ""),
					);
					const unclassified = turns.filter((turn) => !state.assignments.has(turn.key)).length;
					if (unclassified) lines.push(`${unclassified} turn(s) not yet classified; use action=search.`);
					return reply(lines.join("\n") || "No topics yet.");
				}

				if (params.action === "get") {
					if (!params.topic || !state.nodes.has(params.topic))
						return reply(`Unknown topic: ${params.topic ?? "(none)"}`);
					const matching = turns.filter((turn) => state.assignments.get(turn.key)?.includes(params.topic!));
					const selected = matching.slice(-limit);
					const node = state.nodes.get(params.topic)!;
					return reply(
						[
							`Topic [${node.id}] ${node.title}: showing ${selected.length} of ${matching.length} turn(s), oldest first.`,
							...selected.map((turn) => turnText(turn, RECALL_LIMITS)),
						].join("\n\n---\n\n"),
					);
				}

				const query = params.query?.trim();
				if (!query) return reply("action=search needs a query.");
				const texts = turns.map((turn) => turnText(turn, RECALL_LIMITS));
				const scores = lexicalScores(query, texts);
				const lowered = query.toLowerCase();
				const ranked = texts
					.map((text, index) => ({
						index,
						score: scores[index] + (text.toLowerCase().includes(lowered) ? 10 : 0),
					}))
					.filter((item) => item.score > 0)
					.sort((a, b) => b.score - a.score)
					.slice(0, limit)
					.sort((a, b) => a.index - b.index);
				if (ranked.length === 0) return reply(`No earlier turn matches "${query}".`);
				return reply(
					ranked
						.map((item) => {
							const topics = state.assignments.get(turns[item.index].key)?.join(", ") ?? "unclassified";
							return `Turn ${item.index + 1} (topics: ${topics})\n${texts[item.index]}`;
						})
						.join("\n\n---\n\n"),
				);
			},
		});
	}
}
