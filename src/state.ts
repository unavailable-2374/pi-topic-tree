/**
 * Topic-tree state: turn segmentation, lexical scoring and reconstruction of
 * the topic index from session entries.
 *
 * Everything here is pure. The durable source of truth is the session log:
 * `topic-tree.node` entries (latest write per id wins) and `topic-tree.assign`
 * entries. Rebuilding from the current branch on every use makes forks and
 * resumes correct without any in-memory cache to invalidate.
 */

import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { SessionEntry } from "@earendil-works/pi-coding-agent";

/** The read-back tool the projections point the model to. */
export const RECALL_TOOL_NAME = "recall_topic";

export const NODE_ENTRY = "topic-tree.node";
export const ASSIGN_ENTRY = "topic-tree.assign";
/**
 * A caller-supplied label for one turn, e.g. why another extension re-prompted
 * the agent: `pi.appendEntry(TURN_LABEL_ENTRY, { turnKey, label })`,
 * where turnKey is `turnKey(userMessage)`. The label appears in the segment
 * index next to the segment that opens the turn. It is accepted only while
 * that turn is the latest one and no assistant message has answered it yet:
 * index text, once a provider has cached it, must never be rewritten, so a
 * late label is ignored rather than changing an emitted line.
 */
export const TURN_LABEL_ENTRY = "topic-tree.turn-label";

export interface TurnLabelData {
	turnKey: string;
	label: string;
}

/** Pinned node for cross-topic constraints and settled decisions. */
export const GLOBAL_TOPIC_ID = "global";

export interface TopicNode {
	id: string;
	parentId: string | null;
	title: string;
	summary: string;
}

export interface TurnAssignment {
	turnKey: string;
	topicIds: string[];
}

export interface AssignEntryData {
	assignments: TurnAssignment[];
}

export interface TopicTreeState {
	nodes: Map<string, TopicNode>;
	/** turnKey -> topic ids */
	assignments: Map<string, string[]>;
	/** Topics of the most recently assigned turn. */
	lastTopicIds: string[];
	nextId: number;
	/** turnKey -> caller label (TURN_LABEL_ENTRY). */
	labels: Map<string, string>;
}

/** One user message plus everything the agent did in response to it. */
export interface Turn {
	key: string;
	messages: AgentMessage[];
}

/** Turns are keyed by the user message timestamp, which both the session log and the LLM projection carry. */
export function turnKey(message: AgentMessage): string {
	return String(message.timestamp);
}

/**
 * Split a message list at user-message boundaries. Everything before the first
 * user message (compaction summary, injected context) is the prefix. Cutting
 * only here never separates a tool call from its result.
 */
export function splitTurns(messages: AgentMessage[]): { prefix: AgentMessage[]; turns: Turn[] } {
	const prefix: AgentMessage[] = [];
	const turns: Turn[] = [];
	for (const message of messages) {
		if (message.role === "user") {
			turns.push({ key: turnKey(message), messages: [message] });
		} else if (turns.length === 0) {
			prefix.push(message);
		} else {
			turns[turns.length - 1].messages.push(message);
		}
	}
	return { prefix, turns };
}

/**
 * An agent working on one request often receives one prompt and then runs a
 * long tool loop, so user-turn boundaries never come. The unit there is a
 * segment: each prompt opens one, and within a turn a new segment starts at
 * the first assistant message after `toolCallsPerSegment` tool results have
 * completed. Cutting only
 * before an assistant message never separates a tool call from its result,
 * and a completed segment never changes as the loop continues, so its key
 * (the timestamp of its first message) stays stable across rebuilds.
 */
export function splitSegments(
	messages: AgentMessage[],
	toolCallsPerSegment: number,
): { prefix: AgentMessage[]; turns: Turn[] } {
	const size = Math.max(1, Math.floor(toolCallsPerSegment));
	const { prefix, turns } = splitTurns(messages);
	const segments: Turn[] = [];
	for (const turn of turns) {
		let current: Turn = { key: turn.key, messages: [] };
		let completed = 0;
		for (const message of turn.messages) {
			if (message.role === "assistant" && completed >= size && current.messages.length > 0) {
				segments.push(current);
				current = { key: turnKey(message), messages: [] };
				completed = 0;
			}
			current.messages.push(message);
			if (message.role === "toolResult") completed++;
		}
		segments.push(current);
	}
	return { prefix, turns: segments };
}

/**
 * Stable id of a segment, derived from its key (the first message's
 * timestamp), so it names the same segment whether it is counted over the
 * whole branch or over a projection that already dropped older turns.
 */
export function segmentId(segment: Turn): string {
	const numeric = Number(segment.key);
	return `s${Number.isFinite(numeric) ? numeric.toString(36) : segment.key}`;
}

export function messagesFromEntries(entries: SessionEntry[]): AgentMessage[] {
	const messages: AgentMessage[] = [];
	for (const entry of entries) {
		if (entry.type === "message") messages.push(entry.message);
	}
	return messages;
}

function contentText(content: unknown): string {
	if (typeof content === "string") return content;
	if (!Array.isArray(content)) return "";
	return content
		.filter((block): block is { type: "text"; text: string } => block?.type === "text")
		.map((block) => block.text)
		.join("\n");
}

function clip(text: string, max: number): string {
	return text.length <= max ? text : `${text.slice(0, max)} …[+${text.length - max} chars]`;
}

export interface TurnTextLimits {
	user: number;
	assistant: number;
	toolResult: number;
	/** Characters of each tool call's arguments (default 200). */
	toolCall?: number;
}

/** Readable rendering of a turn for classification, summarization and recall. */
export function turnText(turn: Turn, limits: TurnTextLimits): string {
	const lines: string[] = [];
	for (const message of turn.messages) {
		switch (message.role) {
			case "user":
				lines.push(`[user] ${clip(contentText(message.content), limits.user)}`);
				break;
			case "assistant": {
				const text = contentText(message.content).trim();
				if (text) lines.push(`[assistant] ${clip(text, limits.assistant)}`);
				for (const block of message.content) {
					if (block.type === "toolCall") {
						lines.push(
							`[tool call] ${block.name} ${clip(JSON.stringify(block.arguments ?? {}), limits.toolCall ?? 200)}`,
						);
					}
				}
				break;
			}
			case "toolResult":
				if (limits.toolResult > 0) {
					lines.push(
						`[tool result${message.isError ? " error" : ""}] ${clip(contentText(message.content), limits.toolResult)}`,
					);
				}
				break;
			case "custom":
				lines.push(`[context] ${clip(contentText(message.content), limits.assistant)}`);
				break;
			default:
				break;
		}
	}
	return lines.join("\n");
}

export function userText(turn: Turn): string {
	return contentText(turn.messages[0]?.role === "user" ? turn.messages[0].content : "");
}

export function emptyState(): TopicTreeState {
	return {
		nodes: new Map([[GLOBAL_TOPIC_ID, { id: GLOBAL_TOPIC_ID, parentId: null, title: "Global", summary: "" }]]),
		assignments: new Map(),
		lastTopicIds: [],
		nextId: 1,
		labels: new Map(),
	};
}

export function applyNode(state: TopicTreeState, node: TopicNode): void {
	state.nodes.set(node.id, { ...node });
	const numeric = node.id.match(/^t(\d+)$/);
	if (numeric) state.nextId = Math.max(state.nextId, Number(numeric[1]) + 1);
}

export function applyAssignments(state: TopicTreeState, assignments: TurnAssignment[]): void {
	for (const assignment of assignments) {
		const topicIds = assignment.topicIds.filter((id) => state.nodes.has(id));
		if (topicIds.length === 0) continue;
		state.assignments.set(assignment.turnKey, topicIds);
		state.lastTopicIds = topicIds;
	}
}

export function buildState(entries: SessionEntry[]): TopicTreeState {
	const state = emptyState();
	let openTurn: string | undefined;
	for (const entry of entries) {
		if (entry.type === "message") {
			if (entry.message.role === "user") openTurn = turnKey(entry.message);
			else if (entry.message.role === "assistant") openTurn = undefined;
			continue;
		}
		if (entry.type !== "custom") continue;
		if (entry.customType === NODE_ENTRY && isTopicNode(entry.data)) {
			applyNode(state, entry.data);
		} else if (entry.customType === ASSIGN_ENTRY && Array.isArray((entry.data as AssignEntryData)?.assignments)) {
			applyAssignments(state, (entry.data as AssignEntryData).assignments);
		} else if (entry.customType === TURN_LABEL_ENTRY) {
			const label = entry.data as TurnLabelData;
			if (
				typeof label?.turnKey === "string" &&
				label.turnKey === openTurn &&
				typeof label.label === "string" &&
				label.label.trim() &&
				!state.labels.has(label.turnKey)
			)
				state.labels.set(label.turnKey, label.label.trim());
		}
	}
	return state;
}

function isTopicNode(value: unknown): value is TopicNode {
	const node = value as TopicNode;
	return typeof node?.id === "string" && typeof node.title === "string" && typeof node.summary === "string";
}

/** Root-first chain of ancestors ending at the topic itself. */
export function topicPath(state: TopicTreeState, topicId: string): TopicNode[] {
	const path: TopicNode[] = [];
	const seen = new Set<string>();
	let current = state.nodes.get(topicId);
	while (current && !seen.has(current.id)) {
		seen.add(current.id);
		path.unshift(current);
		current = current.parentId ? state.nodes.get(current.parentId) : undefined;
	}
	return path;
}

export function topicTurnCounts(state: TopicTreeState): Map<string, number> {
	const counts = new Map<string, number>();
	for (const topicIds of state.assignments.values()) {
		for (const id of topicIds) counts.set(id, (counts.get(id) ?? 0) + 1);
	}
	return counts;
}

// ---------------------------------------------------------------------------
// Lexical scoring. No embedding service exists in this package, so candidate
// recall is BM25 over latin words and CJK character bigrams.
// ---------------------------------------------------------------------------

const CJK = /[\u3400-\u9fff\uf900-\ufaff]/;

export function tokenize(text: string): string[] {
	const tokens: string[] = [];
	for (const word of text.toLowerCase().match(/[a-z0-9_]{2,}/g) ?? []) tokens.push(word);
	const chars = [...text].filter((char) => CJK.test(char));
	for (let i = 0; i + 1 < chars.length; i++) tokens.push(chars[i] + chars[i + 1]);
	return tokens;
}

/** BM25 score of each document against the query. */
export function lexicalScores(query: string, documents: string[]): number[] {
	const k1 = 1.2;
	const b = 0.75;
	const docTokens = documents.map(tokenize);
	const avgLength = docTokens.reduce((sum, tokens) => sum + tokens.length, 0) / Math.max(1, docTokens.length) || 1;
	const docFrequency = new Map<string, number>();
	for (const tokens of docTokens) {
		for (const token of new Set(tokens)) docFrequency.set(token, (docFrequency.get(token) ?? 0) + 1);
	}
	const queryTokens = [...new Set(tokenize(query))];
	return docTokens.map((tokens) => {
		const termFrequency = new Map<string, number>();
		for (const token of tokens) termFrequency.set(token, (termFrequency.get(token) ?? 0) + 1);
		let score = 0;
		for (const token of queryTokens) {
			const tf = termFrequency.get(token);
			if (!tf) continue;
			const df = docFrequency.get(token) ?? 0;
			const idf = Math.log(1 + (documents.length - df + 0.5) / (df + 0.5));
			score += (idf * tf * (k1 + 1)) / (tf + k1 * (1 - b + (b * tokens.length) / avgLength));
		}
		return score;
	});
}

/** Topic ids ranked by lexical relevance to the query, excluding the global node and zero scores. */
export function rankTopics(state: TopicTreeState, query: string, limit: number): string[] {
	const topics = [...state.nodes.values()].filter((node) => node.id !== GLOBAL_TOPIC_ID);
	const scores = lexicalScores(
		query,
		topics.map((node) => `${node.title}\n${node.summary}`),
	);
	return topics
		.map((node, index) => ({ id: node.id, score: scores[index] }))
		.filter((item) => item.score > 0)
		.sort((a, b) => b.score - a.score)
		.slice(0, limit)
		.map((item) => item.id);
}
