/**
 * Context projection: replace off-topic older turns with a topic digest.
 *
 * Only the projection handed to the model changes; the session log is never
 * touched, so every dropped turn stays reachable through `recall_topic`.
 */

import type { AgentMessage } from "@earendil-works/pi-agent-core";
import {
	GLOBAL_TOPIC_ID,
	RECALL_TOOL_NAME,
	rankTopics,
	segmentId,
	splitSegments,
	splitTurns,
	type TopicTreeState,
	type Turn,
	topicPath,
	topicTurnCounts,
	userText,
} from "./state.ts";

export const DIGEST_MESSAGE_TYPE = "topic-tree.digest";

export interface ProjectionOptions {
	/** Projection is a no-op while the context is at or below this many tokens. */
	triggerTokens: number;
	/** Token budget the projected context aims for. */
	targetTokens: number;
	/** Most recent user turns that are always kept verbatim. */
	keepRecentTurns: number;
	estimate: (message: AgentMessage) => number;
}

export interface Projection {
	messages: AgentMessage[];
	droppedTurns: number;
	focusTopicIds: string[];
	digestDetail: DigestDetail;
}

function sumTokens(messages: AgentMessage[], estimate: (message: AgentMessage) => number): number {
	return messages.reduce((sum, message) => sum + estimate(message), 0);
}

/** Topics the conversation is currently about: recent turns' assignments, then the latest prompt's best lexical match. */
export function focusTopics(state: TopicTreeState, recent: Turn[]): string[] {
	const focus = new Set<string>();
	for (const turn of recent) {
		for (const id of state.assignments.get(turn.key) ?? []) focus.add(id);
	}
	if (focus.size === 0) for (const id of state.lastTopicIds) focus.add(id);
	const latest = recent[recent.length - 1];
	if (latest) for (const id of rankTopics(state, userText(latest), 1)) focus.add(id);
	focus.delete(GLOBAL_TOPIC_ID);
	return [...focus];
}

/**
 * How much of each non-focus topic the digest spells out. The current topic is
 * always given in full; the others degrade until the digest fits its share of
 * the budget, and stay reachable through `recall_topic` at every level.
 */
export type DigestDetail = "full" | "brief" | "titles";
const DIGEST_DETAILS: DigestDetail[] = ["full", "brief", "titles"];
const BRIEF_SUMMARY_CHARS = 300;
/** Largest share of the target the digest may take before it degrades. */
const DIGEST_SHARE = 0.35;
/** Recent turns are cut back (to at least one) once they alone pass this share. */
const RECENT_SHARE = 0.5;

function clipSummary(text: string, detail: DigestDetail): string {
	const trimmed = text.trim();
	if (detail === "full" || trimmed.length <= BRIEF_SUMMARY_CHARS) return trimmed;
	return `${trimmed.slice(0, BRIEF_SUMMARY_CHARS)} …`;
}

export function buildDigest(
	state: TopicTreeState,
	focus: string[],
	droppedByTopic: Map<string, number>,
	unclassifiedDropped: number,
	detail: DigestDetail = "full",
): string {
	const sections: string[] = [
		"Earlier turns of this conversation are not shown verbatim; they are organised into topics below. " +
			"Their full text is still available: call `recall_topic` with action `get` and a topic id, or action `search` " +
			"with a query. Recall before relying on a detail you cannot see.",
	];
	const global = state.nodes.get(GLOBAL_TOPIC_ID);
	if (global?.summary.trim()) sections.push(`## Standing constraints and decisions\n${global.summary.trim()}`);

	const shown = new Set<string>();
	const describe = (id: string, heading: string, level: DigestDetail) => {
		if (shown.has(id)) return;
		const path = topicPath(state, id).filter((node) => node.id !== GLOBAL_TOPIC_ID);
		const node = path[path.length - 1];
		if (!node) return;
		shown.add(id);
		const breadcrumb = path.map((item) => item.title).join(" › ");
		const omitted = droppedByTopic.get(id) ?? 0;
		const header = `## ${heading}: ${breadcrumb} [${id}]${omitted ? ` — ${omitted} turn(s) not shown` : ""}`;
		if (level === "titles") {
			sections.push(header);
			return;
		}
		const ancestors = path
			.slice(0, -1)
			.filter((item) => item.summary.trim() && !shown.has(item.id))
			.map((item) => `(${item.id} ${item.title}) ${clipSummary(item.summary, level)}`);
		sections.push([header, ...ancestors, clipSummary(node.summary, level) || "(no summary yet)"].join("\n"));
	};
	for (const id of focus) describe(id, "Current topic", "full");
	for (const id of droppedByTopic.keys()) describe(id, "Other topic", detail);

	const rest = [...state.nodes.values()].filter((node) => node.id !== GLOBAL_TOPIC_ID && !shown.has(node.id));
	if (rest.length > 0)
		sections.push(`## Other known topics\n${rest.map((node) => `- [${node.id}] ${node.title}`).join("\n")}`);
	if (unclassifiedDropped > 0) {
		sections.push(
			`${unclassifiedDropped} older unclassified turn(s) are not shown; use \`recall_topic\` search to find them.`,
		);
	}
	return sections.join("\n\n");
}

/** True when every tool result follows the assistant tool call it answers. */
export function toolPairsIntact(messages: AgentMessage[]): boolean {
	const calls = new Set<string>();
	for (const message of messages) {
		if (message.role === "assistant") {
			for (const block of message.content) if (block.type === "toolCall") calls.add(block.id);
		} else if (message.role === "toolResult" && !calls.has(message.toolCallId)) {
			return false;
		}
	}
	return true;
}

/**
 * Returns undefined whenever projection is unnecessary or unsafe; the caller
 * then passes the context through unchanged.
 */
export function projectContext(
	messages: AgentMessage[],
	state: TopicTreeState,
	options: ProjectionOptions,
): Projection | undefined {
	if (state.assignments.size === 0) return undefined;
	if (sumTokens(messages, options.estimate) <= options.triggerTokens) return undefined;

	const { prefix, turns } = splitTurns(messages);
	const turnCost = (turn: Turn) => sumTokens(turn.messages, options.estimate);
	let keep = Math.max(1, options.keepRecentTurns);
	if (turns.length <= keep) return undefined;
	// Long recent turns would otherwise spend the whole budget before any history.
	while (
		keep > 1 &&
		turns.slice(-keep).reduce((sum, turn) => sum + turnCost(turn), 0) > options.targetTokens * RECENT_SHARE
	) {
		keep--;
	}
	const recent = turns.slice(-keep);
	const older = turns.slice(0, -keep);
	const focus = focusTopics(state, recent);
	const focusSet = new Set(focus);

	const digestMessage = (content: string): AgentMessage => ({
		role: "custom",
		customType: DIGEST_MESSAGE_TYPE,
		content,
		display: false,
		timestamp: older[0].messages[0].timestamp,
	});
	const tally = (keptTurns: Set<Turn>) => {
		const dropped = new Map<string, number>();
		let unclassified = 0;
		let count = 0;
		for (const turn of older) {
			if (keptTurns.has(turn)) continue;
			count++;
			const topics = state.assignments.get(turn.key);
			if (!topics) unclassified++;
			for (const id of topics ?? []) dropped.set(id, (dropped.get(id) ?? 0) + 1);
		}
		return { dropped, unclassified, count };
	};

	// Size the digest before choosing turns, against the worst case (every older
	// turn dropped), at the richest detail that fits its share of the budget.
	const worst = tally(new Set());
	const digestCap = Math.floor(options.targetTokens * DIGEST_SHARE);
	let detail: DigestDetail = "titles";
	for (const level of DIGEST_DETAILS) {
		const size = options.estimate(digestMessage(buildDigest(state, focus, worst.dropped, worst.unclassified, level)));
		if (size <= digestCap) {
			detail = level;
			break;
		}
	}
	const digestReserve = options.estimate(
		digestMessage(buildDigest(state, focus, worst.dropped, worst.unclassified, detail)),
	);
	let budget =
		options.targetTokens -
		digestReserve -
		sumTokens(prefix, options.estimate) -
		recent.reduce((sum, turn) => sum + turnCost(turn), 0);

	// Newest-first: on-topic and not-yet-classified turns are kept while budget lasts.
	// Unclassified turns count as on-topic so a lagging classifier never hides them.
	const kept = new Set<Turn>();
	for (let i = older.length - 1; i >= 0; i--) {
		const turn = older[i];
		const topics = state.assignments.get(turn.key);
		const relevant = !topics || topics.some((id) => focusSet.has(id));
		if (!relevant) continue;
		const cost = turnCost(turn);
		if (cost > budget) continue;
		kept.add(turn);
		budget -= cost;
	}

	const { dropped: droppedByTopic, unclassified: unclassifiedDropped, count: droppedTurns } = tally(kept);
	if (droppedTurns === 0) return undefined;

	// Never larger than the reserve: fewer dropped topics only shrink it.
	const digest = digestMessage(buildDigest(state, focus, droppedByTopic, unclassifiedDropped, detail));
	const projected: AgentMessage[] = [
		...prefix,
		digest,
		...older.filter((turn) => kept.has(turn)).flatMap((turn) => turn.messages),
		...recent.flatMap((turn) => turn.messages),
	];
	if (!toolPairsIntact(projected)) return undefined;
	return { messages: projected, droppedTurns, focusTopicIds: focus, digestDetail: detail };
}

// ---------------------------------------------------------------------------
// Thin projection: tree mode from the first turn. Only the standing
// constraints, a one-line-per-topic index and the last few turns are sent;
// everything older is pulled with `recall_topic`. On long real sessions this
// answered earlier-detail questions at 99% with 35% of full-history input,
// against 94–97% at 48–64% for the topic projection above.
// ---------------------------------------------------------------------------

export const INDEX_MESSAGE_TYPE = "topic-tree.index";

export interface ThinOptions {
	/** Completed turns kept verbatim before the current one. */
	previousTurns: number;
}

export function buildIndex(state: TopicTreeState, unclassified: number): string {
	const counts = topicTurnCounts(state);
	const topics = [...state.nodes.values()]
		.filter((node) => node.id !== GLOBAL_TOPIC_ID)
		.map((node) => {
			const parent = node.parentId && node.parentId !== GLOBAL_TOPIC_ID ? ` (under ${node.parentId})` : "";
			return `- [${node.id}] ${node.title}${parent} — ${counts.get(node.id) ?? 0} turn(s)`;
		});
	const sections = [
		"Earlier turns of this conversation are not shown. Their full text is available: call `recall_topic` " +
			"with action `get` and a topic id, or action `search` with keywords. Recall whenever the user refers to " +
			"something you cannot see, before relying on it.",
	];
	const global = state.nodes.get(GLOBAL_TOPIC_ID)?.summary.trim();
	if (global) sections.push(`## Standing constraints and decisions\n${global}`);
	sections.push(`## Topics so far\n${topics.join("\n") || "(none indexed yet)"}`);
	if (unclassified > 0) {
		sections.push(`${unclassified} earlier turn(s) are not indexed yet; \`recall_topic\` search still finds them.`);
	}
	return sections.join("\n\n");
}

/**
 * Prefix (e.g. a compaction summary), the index, then the last
 * `previousTurns` completed turns and the current turn verbatim. Undefined
 * while there is nothing older to hide.
 */
export function projectThin(
	messages: AgentMessage[],
	state: TopicTreeState,
	options: ThinOptions,
): Projection | undefined {
	const { prefix, turns } = splitTurns(messages);
	const keep = Math.max(0, options.previousTurns) + 1;
	if (turns.length <= keep) return undefined;
	const hidden = turns.slice(0, -keep);
	const unclassified = hidden.filter((turn) => !state.assignments.has(turn.key)).length;
	const index: AgentMessage = {
		role: "custom",
		customType: INDEX_MESSAGE_TYPE,
		content: buildIndex(state, unclassified),
		display: false,
		timestamp: hidden[0].messages[0].timestamp,
	};
	const projected = [...prefix, index, ...turns.slice(-keep).flatMap((turn) => turn.messages)];
	if (!toolPairsIntact(projected)) return undefined;
	return { messages: projected, droppedTurns: hidden.length, focusTopicIds: [], digestDetail: "titles" };
}

// ---------------------------------------------------------------------------
// Tool-loop segments. An agent working through one request is often one
// prompt followed by hundreds of tool calls, so user turns never come; the
// unit is a segment of tool calls (state.ts splitSegments). Standing
// constraints are never folded:
//  - the system prompt is not part of the message list at all;
//  - every message that is not an assistant step or a tool result -- each
//    user prompt and answer, every message injected by another extension, a
//    compaction summary or a turn index -- stays verbatim, in order;
//  - the latest step of each controller tool (anything outside the ordinary
//    work tools below: workflow, sub-agent or receipt tools from other
//    extensions) stays verbatim with its result, because it carries the
//    current instruction or receipt later steps depend on;
//  - tool errors and refusals nothing answered yet are quoted in the index.
// Everything else in older segments becomes one index line per segment, and
// the segment itself stays readable verbatim through `recall_topic`.
//
// Provider prefix caching decides the cost: nearly all input is cache reads,
// and a request only hits the cache as far as it is byte-identical to the
// previous one. So the projection moves in discrete folds, not a sliding
// window: segments are hidden `foldEverySegments` at a time, and between two
// folds every request is the previous one plus the new messages. The index is
// a pure function of the hidden segments (no live classifier state), built
// block by block, so each fold only appends a block to it; stable parts come
// first (prompts and injected messages, then the index) and the verbatim
// segments last.
// ---------------------------------------------------------------------------

export const SEGMENT_INDEX_MESSAGE_TYPE = "topic-tree.segment-index";

export interface SegmentOptions {
	/** Completed segments kept verbatim before the current one. */
	previousSegments: number;
	/** Tool results per segment. */
	toolCallsPerSegment: number;
	/** Segments hidden per fold; between folds the request prefix does not change. */
	foldEverySegments: number;
}

/** How many segments are hidden: whole folds only, so the value moves in steps. */
export function hiddenSegmentCount(segmentCount: number, options: SegmentOptions): number {
	const keep = Math.max(0, options.previousSegments) + 1;
	const step = Math.max(1, Math.floor(options.foldEverySegments));
	return Math.max(0, Math.floor((segmentCount - keep) / step) * step);
}

/** Ordinary work tools, whose older steps fold. Any other tool is treated as a controller. */
const FOLDABLE_TOOLS = new Set([
	"read",
	"write",
	"edit",
	"bash",
	"grep",
	"find",
	"ls",
	"glob",
	"view_image",
	"webfetch",
	"websearch",
	RECALL_TOOL_NAME,
]);

export function foldableTool(name: string): boolean {
	return FOLDABLE_TOOLS.has(name) || name.startsWith("mcp__");
}

/** Each hidden segment is one line; the line is clipped, never the list (it only ever grows by appending). */
const SEGMENT_LINE_CHARS = 200;
/** Unresolved tool errors and refusals carried forward verbatim. */
const PENDING_REFUSALS = 3;

function blockText(content: unknown): string {
	if (typeof content === "string") return content;
	if (!Array.isArray(content)) return "";
	return content
		.filter((block): block is { type: "text"; text: string } => block?.type === "text")
		.map((block) => block.text)
		.join("\n");
}

function callTarget(args: unknown): string {
	if (!args || typeof args !== "object") return "";
	const record = args as Record<string, unknown>;
	for (const key of ["path", "file_path", "filePath", "pattern", "url", "query", "command", "agent", "action"]) {
		const value = record[key];
		if (typeof value === "string" && value.trim()) {
			const flat = value.replace(/\s+/g, " ").trim();
			return flat.length > 60 ? `${flat.slice(0, 60)}…` : flat;
		}
	}
	return "";
}

/** One deterministic line per segment: what was called on what, with repeats folded. No model involved. */
export function describeSegment(segment: Turn): { calls: number; errors: number; actions: string } {
	const actions: { label: string; count: number }[] = [];
	let calls = 0;
	let errors = 0;
	for (const message of segment.messages) {
		if (message.role === "toolResult") {
			if (message.isError) errors++;
			continue;
		}
		if (message.role !== "assistant") continue;
		for (const block of message.content) {
			if (block.type !== "toolCall") continue;
			calls++;
			const target = callTarget(block.arguments);
			const label = target ? `${block.name} ${target}` : block.name;
			const last = actions[actions.length - 1];
			if (last?.label === label) last.count++;
			else actions.push({ label, count: 1 });
		}
	}
	let text = actions
		.map((action) => (action.count > 1 ? `${action.label} ×${action.count}` : action.label))
		.join("; ");
	if (text.length > SEGMENT_LINE_CHARS) text = `${text.slice(0, SEGMENT_LINE_CHARS)} …`;
	return { calls, errors, actions: text };
}

export interface PendingRefusal {
	segment: string;
	toolName: string;
	text: string;
}

/**
 * Error results (refusals by other extensions included) in `block` that nothing answered
 * by the end of `seen`: no later successful result of the same tool. The
 * newest few are carried into the index verbatim, because a refusal not yet
 * acted on is a standing instruction, not history. Judged at the fold that
 * hides the block and frozen there, so the index never changes afterwards.
 */
export function pendingRefusals(seen: Turn[], block: ReadonlySet<Turn>): PendingRefusal[] {
	const lastSuccess = new Map<string, number>();
	const errors: { order: number; segment: Turn; toolName: string; text: string }[] = [];
	let order = 0;
	for (const segment of seen) {
		for (const message of segment.messages) {
			if (message.role !== "toolResult") continue;
			order++;
			if (!message.isError) lastSuccess.set(message.toolName, order);
			else if (block.has(segment))
				errors.push({ order, segment, toolName: message.toolName, text: blockText(message.content) });
		}
	}
	return errors
		.filter((item) => (lastSuccess.get(item.toolName) ?? 0) < item.order)
		.slice(-PENDING_REFUSALS)
		.map((item) => ({ segment: segmentId(item.segment), toolName: item.toolName, text: item.text }));
}

/** Tool names an assistant step calls. */
function stepTools(message: AgentMessage): string[] {
	return message.role === "assistant"
		? message.content.filter((block) => block.type === "toolCall").map((block) => block.name)
		: [];
}

const REPLY_EXCERPT_CHARS = 160;

/** The segment's last assistant prose (its answer, in a conversational turn), clipped. */
function replyExcerpt(segment: Turn): string {
	for (let i = segment.messages.length - 1; i >= 0; i--) {
		const message = segment.messages[i];
		if (message.role !== "assistant") continue;
		const text = blockText(message.content).replace(/\s+/g, " ").trim();
		if (!text) continue;
		return text.length > REPLY_EXCERPT_CHARS ? `${text.slice(0, REPLY_EXCERPT_CHARS)} …` : text;
	}
	return "";
}

function segmentLine(state: TopicTreeState, segment: Turn): string {
	const info = describeSegment(segment);
	const details = [`${info.calls} call(s)`, ...(info.errors ? [`${info.errors} error(s)`] : [])];
	const label = state.labels.get(segment.key);
	const opens = segment.messages[0]?.role === "user" ? `new prompt${label ? ` (${label})` : ""}; ` : "";
	const reply = replyExcerpt(segment);
	return (
		`- ${segmentId(segment)} (${details.join(", ")}): ${opens}${info.actions || "(no tool calls)"}` +
		(reply ? ` — reply: ${reply}` : "")
	);
}

/**
 * Header, then one block per fold, oldest first. A block depends only on its
 * own segments and on the segments hidden by then, so the index of fold k is
 * a prefix of the index of fold k+1. Topic summaries stay out on purpose: the
 * classifier runs in the background and would rewrite the index between folds.
 */
export function buildSegmentIndex(
	state: TopicTreeState,
	segments: Turn[],
	hiddenCount: number,
	foldEvery: number,
): string {
	const step = Math.max(1, Math.floor(foldEvery));
	const parts = [
		"Earlier tool-call segments are not shown verbatim. The prompts and every injected message above are " +
			"unchanged and still apply, as does the latest call of each workflow/controller tool shown after this index. " +
			"The full text of any hidden segment -- its tool calls and results, including file contents, hashes, receipts " +
			`and errors -- is available: call \`${RECALL_TOOL_NAME}\` with action \`get\` and a segment id below, action ` +
			"`search` with keywords, or action `list` for topic summaries. Recall before relying on a detail you cannot " +
			"see; do not redo work only to see its result again.",
	];
	for (let from = 0; from < hiddenCount; from += step) {
		const block = segments.slice(from, Math.min(hiddenCount, from + step));
		const lines = block.map((segment) => segmentLine(state, segment));
		const pending = pendingRefusals(segments.slice(0, from + block.length), new Set(block));
		const refusals = pending.map(
			(item) => `Unresolved at this point (verbatim) — ${item.toolName}, segment ${item.segment}:\n${item.text}`,
		);
		parts.push(
			[
				`## Hidden segments ${segmentId(block[0])} … ${segmentId(block[block.length - 1])}`,
				...lines,
				...refusals,
			].join("\n"),
		);
	}
	return parts.join("\n\n");
}

/**
 * Prefix; every message of the hidden segments that is not a foldable step
 * (prompts, injected messages), in order; the segment index; the latest
 * hidden step of each controller tool with its results; then the visible
 * segments verbatim. Undefined while nothing would be hidden.
 */
export function projectSegments(
	messages: AgentMessage[],
	state: TopicTreeState,
	options: SegmentOptions,
): Projection | undefined {
	const { prefix, turns: segments } = splitSegments(messages, options.toolCallsPerSegment);
	const hiddenCount = hiddenSegmentCount(segments.length, options);
	if (hiddenCount === 0) return undefined;
	const hidden = segments.slice(0, hiddenCount);
	const hiddenMessages = hidden.flatMap((segment) => segment.messages);

	// Latest hidden step per controller tool, kept whole (the call and all its results).
	const pinned = new Set<AgentMessage>();
	const seenControllers = new Set<string>();
	for (let i = hiddenMessages.length - 1; i >= 0; i--) {
		const controllers = stepTools(hiddenMessages[i]).filter((name) => !foldableTool(name));
		if (controllers.length === 0 || controllers.every((name) => seenControllers.has(name))) continue;
		for (const name of controllers) seenControllers.add(name);
		const step = hiddenMessages[i];
		pinned.add(step);
		const ids = new Set(
			step.role === "assistant"
				? step.content.flatMap((block) => (block.type === "toolCall" ? [block.id] : []))
				: [],
		);
		for (const message of hiddenMessages)
			if (message.role === "toolResult" && ids.has(message.toolCallId)) pinned.add(message);
	}
	const standing = hiddenMessages.filter((message) => message.role !== "assistant" && message.role !== "toolResult");
	const pinnedSteps = hiddenMessages.filter((message) => pinned.has(message));
	if (standing.length + pinnedSteps.length === hiddenMessages.length) return undefined;
	const index: AgentMessage = {
		role: "custom",
		customType: SEGMENT_INDEX_MESSAGE_TYPE,
		content: buildSegmentIndex(state, segments, hiddenCount, options.foldEverySegments),
		display: false,
		timestamp: hidden[0].messages[0].timestamp,
	};
	const projected = [
		...prefix,
		...standing,
		index,
		...pinnedSteps,
		...segments.slice(hiddenCount).flatMap((segment) => segment.messages),
	];
	if (!toolPairsIntact(projected)) return undefined;
	return { messages: projected, droppedTurns: hiddenCount, focusTopicIds: [], digestDetail: "titles" };
}
