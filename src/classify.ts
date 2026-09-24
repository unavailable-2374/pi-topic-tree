/**
 * Turn classification: prompt construction and defensive parsing of the
 * classifier's JSON. A response that cannot be used is rejected as a whole so
 * the caller can fall back, never half-applied.
 */

import { GLOBAL_TOPIC_ID, type TopicNode, type TopicTreeState, type TurnAssignment } from "./state.ts";

export interface ClassifyTurn {
	key: string;
	text: string;
}

export interface NewTopic {
	ref: string;
	title: string;
	parentId: string | null;
}

export interface ClassifyResult {
	newTopics: NewTopic[];
	assignments: TurnAssignment[];
	globalFacts: string[];
}

export const CLASSIFY_SYSTEM_PROMPT =
	"You maintain a topic tree for a long working conversation. You assign each conversation turn to the topic " +
	"branch it belongs to. You reply with one JSON object and nothing else.";

export function buildClassifyPrompt(turns: ClassifyTurn[], candidates: TopicNode[], lastTopicIds: string[]): string {
	const topicLines = candidates.length
		? candidates
				.map(
					(node) =>
						`- id=${node.id} parent=${node.parentId ?? "none"} title=${JSON.stringify(node.title)}` +
						(node.summary ? `\n  summary: ${node.summary.slice(0, 400).replace(/\s+/g, " ")}` : ""),
				)
				.join("\n")
		: "(no topics yet)";
	const turnBlocks = turns.map((turn, index) => `<turn index="${index}">\n${turn.text}\n</turn>`).join("\n");
	return `Existing topics (candidates most likely to match):
${topicLines}

Most recently active topic ids: ${lastTopicIds.length ? lastTopicIds.join(", ") : "none"}

Turns to classify, oldest first:
${turnBlocks}

Rules:
- A topic is an area of work, not a single question. A turn about the same subject area as an existing topic belongs to it, even when it asks a new sub-question (naming the cat, feeding the cat and choosing cat litter are all one "cat care" topic).
- Create a new topic only for a genuinely different subject area. When the new area is a distinct part of an existing one, give it that topic as parent; otherwise parent is null.
- A turn may belong to several topics only when it really advances each of them.
- Titles are short noun phrases in the conversation's language naming the area ("Cat care"), never the first question asked in it ("Naming the cat").
- globalFacts: standing rules that apply across all topics, stated once, only when a turn establishes one — constraints and decisions ("never touch the release config") and the user's standing preferences for how answers look ("always give amounts to two decimals", "never answer in tables", "reply in English"). Write each rule itself, in full, so it can be followed without the original turn. Usually empty.
- A turn that only sets such standing rules may be given an empty topics list; never create a topic just to hold rules — they live in globalFacts.

Reply with JSON of this shape:
{"newTopics":[{"ref":"n1","title":"...","parent":null}],
 "assignments":[{"turn":0,"topics":["t3"]},{"turn":1,"topics":["n1"]}],
 "globalFacts":[]}
Every turn index must appear exactly once in assignments. Topic references are existing ids or refs from newTopics; an empty list is allowed only for a turn that just sets standing rules.`;
}

export function extractJsonObject(text: string): unknown {
	const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/);
	const body = fenced ? fenced[1] : text;
	const start = body.indexOf("{");
	const end = body.lastIndexOf("}");
	if (start < 0 || end <= start) throw new Error("no JSON object in classifier reply");
	return JSON.parse(body.slice(start, end + 1));
}

/** Resolves refs to real topic ids (allocated from state.nextId) and validates every reference. */
export function parseClassifyResponse(text: string, turns: ClassifyTurn[], state: TopicTreeState): ClassifyResult {
	const raw = extractJsonObject(text) as {
		newTopics?: { ref?: unknown; title?: unknown; parent?: unknown }[];
		assignments?: { turn?: unknown; topics?: unknown }[];
		globalFacts?: unknown;
	};

	let nextId = state.nextId;
	const refToId = new Map<string, string>();
	const pending: { id: string; title: string; parent: unknown }[] = [];
	for (const topic of raw.newTopics ?? []) {
		if (typeof topic.ref !== "string" || typeof topic.title !== "string" || !topic.title.trim()) {
			throw new Error("malformed new topic");
		}
		const id = `t${nextId++}`;
		refToId.set(topic.ref, id);
		pending.push({ id, title: topic.title.trim().slice(0, 80), parent: topic.parent });
	}
	const resolve = (ref: unknown): string | undefined => {
		if (typeof ref !== "string") return undefined;
		if (refToId.has(ref)) return refToId.get(ref);
		return state.nodes.has(ref) && ref !== GLOBAL_TOPIC_ID ? ref : undefined;
	};
	const newTopics: NewTopic[] = pending.map((topic) => ({
		ref: topic.id,
		title: topic.title,
		parentId: topic.parent == null ? null : (resolve(topic.parent) ?? null),
	}));

	const byIndex = new Map<number, string[]>();
	for (const item of raw.assignments ?? []) {
		const index = typeof item.turn === "number" ? item.turn : Number.NaN;
		if (!Number.isInteger(index) || index < 0 || index >= turns.length || byIndex.has(index)) {
			throw new Error(`invalid turn index ${String(item.turn)}`);
		}
		const topicIds = Array.isArray(item.topics) ? item.topics.map(resolve) : [];
		if (topicIds.some((id) => id === undefined)) {
			throw new Error(`turn ${index} references an unknown topic`);
		}
		byIndex.set(index, [...new Set(topicIds as string[])]);
	}
	if (byIndex.size !== turns.length) throw new Error("classifier left turns unassigned");

	// A turn that only sets standing rules may come back with no topic: it goes
	// with its neighbour in the batch (previous first), else the active topic.
	const lastActive = state.lastTopicIds.filter((id) => state.nodes.has(id) && id !== GLOBAL_TOPIC_ID);
	const assignments: TurnAssignment[] = turns.map((turn, index) => {
		let topicIds = byIndex.get(index)!;
		for (let d = 1; topicIds.length === 0 && d < turns.length; d++) {
			topicIds = byIndex.get(index - d)?.length ? byIndex.get(index - d)! : (byIndex.get(index + d) ?? []);
		}
		if (topicIds.length === 0) topicIds = lastActive;
		if (topicIds.length === 0) throw new Error(`turn ${index} has no topic and nothing to attach it to`);
		return { turnKey: turn.key, topicIds };
	});

	const globalFacts = Array.isArray(raw.globalFacts)
		? raw.globalFacts.filter((fact): fact is string => typeof fact === "string" && fact.trim().length > 0)
		: [];
	return { newTopics, assignments, globalFacts: globalFacts.map((fact) => fact.trim()) };
}

/**
 * Used when the classifier fails: keep the conversation on its last topic, or
 * open a catch-all one. Wrong but recoverable — the turns stay searchable.
 */
export function fallbackClassification(turns: ClassifyTurn[], state: TopicTreeState): ClassifyResult {
	const last = state.lastTopicIds.filter((id) => state.nodes.has(id) && id !== GLOBAL_TOPIC_ID);
	if (last.length > 0) {
		return {
			newTopics: [],
			assignments: turns.map((turn) => ({ turnKey: turn.key, topicIds: last })),
			globalFacts: [],
		};
	}
	const id = `t${state.nextId}`;
	return {
		newTopics: [{ ref: id, title: "General", parentId: null }],
		assignments: turns.map((turn) => ({ turnKey: turn.key, topicIds: [id] })),
		globalFacts: [],
	};
}

// ---------------------------------------------------------------------------
// Summaries
// ---------------------------------------------------------------------------

export const SUMMARY_SYSTEM_PROMPT =
	"You keep a running summary of one topic in a long working conversation. Output only the updated summary.";

export function buildSummaryPrompt(node: TopicNode, newTurnsText: string): string {
	return `Topic: ${node.title} [${node.id}]

<previous-summary>
${node.summary || "(empty)"}
</previous-summary>

<new-turns>
${newTurnsText}
</new-turns>

Update the summary so it covers the previous summary plus the new turns. Keep it under 250 words, in the conversation's language, as terse bullets under these headings (omit an empty heading):
Goal / Decisions / State / Open questions
Keep exact identifiers that later work depends on: file paths, commands, names, numbers, error messages. Drop pleasantries and superseded details.
Summarise the user's work only. Ignore the agent's own protocol bookkeeping — workflow decisions, gate or policy messages, instructions about which tool may be called next.`;
}

/** Appends new standing facts to the global node's summary, dropping exact duplicates and keeping the newest `cap`. */
export function mergeGlobalFacts(summary: string, facts: string[], cap = 40): string {
	const lines = summary
		.split("\n")
		.map((line) => line.trim())
		.filter(Boolean);
	for (const fact of facts) {
		const line = `- ${fact.replace(/^[-*]\s*/, "")}`;
		if (!lines.includes(line)) lines.push(line);
	}
	return lines.slice(-cap).join("\n");
}
