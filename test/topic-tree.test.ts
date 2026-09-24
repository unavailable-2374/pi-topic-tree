import type { AgentMessage } from "@earendil-works/pi-agent-core";
import { describe, expect, it } from "vitest";
import { SessionManager } from "@earendil-works/pi-coding-agent";
import { fallbackClassification, mergeGlobalFacts, parseClassifyResponse } from "../src/classify.ts";
import {
	DIGEST_MESSAGE_TYPE,
	type ProjectionOptions,
	projectContext,
	projectThin,
	toolPairsIntact,
} from "../src/project.ts";
import {
	ASSIGN_ENTRY,
	applyAssignments,
	applyNode,
	buildState,
	emptyState,
	messagesFromEntries,
	NODE_ENTRY,
	rankTopics,
	splitTurns,
	type TopicTreeState,
	tokenize,
} from "../src/state.ts";

const usage = {
	input: 1,
	output: 1,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 2,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

function user(ts: number, text: string): AgentMessage {
	return { role: "user", content: text, timestamp: ts };
}

function assistant(ts: number, text: string, toolCallId?: string): AgentMessage {
	return {
		role: "assistant",
		content: [
			{ type: "text", text },
			...(toolCallId
				? [{ type: "toolCall" as const, id: toolCallId, name: "bash", arguments: { command: "ls" } }]
				: []),
		],
		api: "anthropic-messages",
		provider: "anthropic",
		model: "claude-test",
		usage,
		stopReason: toolCallId ? "toolUse" : "stop",
		timestamp: ts,
	};
}

function toolResult(ts: number, toolCallId: string, text: string): AgentMessage {
	return {
		role: "toolResult",
		toolCallId,
		toolName: "bash",
		content: [{ type: "text", text }],
		isError: false,
		timestamp: ts,
	};
}

/** One token per message keeps budget arithmetic readable. */
const options = (overrides: Partial<ProjectionOptions> = {}): ProjectionOptions => ({
	triggerTokens: 10,
	targetTokens: 100,
	keepRecentTurns: 2,
	estimate: () => 1,
	...overrides,
});

function stateWith(assignments: Record<number, string[]>): TopicTreeState {
	const state = emptyState();
	applyNode(state, { id: "t1", parentId: null, title: "数据库迁移", summary: "迁移 postgres 表结构" });
	applyNode(state, {
		id: "t2",
		parentId: null,
		title: "Frontend styling",
		summary: "css grid layout for the dashboard",
	});
	applyAssignments(
		state,
		Object.entries(assignments).map(([ts, topicIds]) => ({ turnKey: ts, topicIds })),
	);
	return state;
}

/** Eight turns alternating topics t1 (odd ts) / t2 (even ts); turn 3 carries a tool call. */
function conversation(): AgentMessage[] {
	const messages: AgentMessage[] = [];
	for (let i = 1; i <= 8; i++) {
		const ts = i * 100;
		messages.push(
			user(ts, i === 8 ? "follow-up question 8" : i % 2 ? `migration question ${i}` : `styling question ${i}`),
		);
		if (i === 3) {
			messages.push(assistant(ts + 1, "checking", "call-3"), toolResult(ts + 2, "call-3", "files"));
		}
		messages.push(assistant(ts + 3, `answer ${i}`));
	}
	return messages;
}

const alternating = {
	100: ["t1"],
	200: ["t2"],
	300: ["t1"],
	400: ["t2"],
	500: ["t1"],
	600: ["t2"],
	700: ["t1"],
	800: ["t2"],
};

describe("topic-tree state", () => {
	it("splits at user boundaries and keeps tool calls with their results", () => {
		const { prefix, turns } = splitTurns([
			{ role: "compactionSummary", summary: "s", tokensBefore: 1, timestamp: 1 },
			...conversation(),
		]);
		expect(prefix).toHaveLength(1);
		expect(turns).toHaveLength(8);
		expect(turns[2].messages.map((m) => m.role)).toEqual(["user", "assistant", "toolResult", "assistant"]);
	});

	it("rebuilds nodes (latest write wins) and ignores assignments to unknown topics", () => {
		const session = SessionManager.inMemory();
		session.appendCustomEntry(NODE_ENTRY, { id: "t4", parentId: null, title: "A", summary: "old" });
		session.appendCustomEntry(NODE_ENTRY, { id: "t4", parentId: null, title: "A", summary: "new" });
		session.appendCustomEntry(ASSIGN_ENTRY, {
			assignments: [
				{ turnKey: "1", topicIds: ["t4"] },
				{ turnKey: "2", topicIds: ["t99"] },
			],
		});
		const state = buildState(session.getBranch());
		expect(state.nodes.get("t4")?.summary).toBe("new");
		expect(state.assignments.get("1")).toEqual(["t4"]);
		expect(state.assignments.has("2")).toBe(false);
		expect(state.nextId).toBe(5);
	});

	it("only sees the assignments on the current branch after a fork", () => {
		const session = SessionManager.inMemory();
		session.appendMessage(user(1, "first") as never);
		const forkPoint = session.appendCustomEntry(NODE_ENTRY, { id: "t1", parentId: null, title: "A", summary: "" });
		session.appendCustomEntry(ASSIGN_ENTRY, { assignments: [{ turnKey: "1", topicIds: ["t1"] }] });
		expect(buildState(session.getBranch()).assignments.has("1")).toBe(true);

		session.branch(forkPoint);
		const state = buildState(session.getBranch());
		expect(state.nodes.has("t1")).toBe(true);
		expect(state.assignments.has("1")).toBe(false);
		expect(messagesFromEntries(session.getBranch())).toHaveLength(1);
	});

	it("scores Chinese queries by character bigrams", () => {
		expect(tokenize("数据库迁移")).toEqual(["数据", "据库", "库迁", "迁移"]);
		const state = stateWith({});
		expect(rankTopics(state, "那个迁移脚本还要改吗", 2)).toEqual(["t1"]);
		expect(rankTopics(state, "dashboard grid", 2)).toEqual(["t2"]);
		expect(rankTopics(state, "unrelated words", 2)).toEqual([]);
	});
});

describe("topic-tree projection", () => {
	it("passes the context through unchanged at or below the trigger", () => {
		const messages = conversation();
		expect(
			projectContext(messages, stateWith(alternating), options({ triggerTokens: messages.length })),
		).toBeUndefined();
	});

	it("passes the context through when nothing has been classified yet", () => {
		expect(projectContext(conversation(), emptyState(), options())).toBeUndefined();
	});

	it("passes the context through when every older turn is on a focused topic", () => {
		// Recent turns 700 (t1) and 800 (t2) put both topics in focus, so nothing is off-topic.
		expect(projectContext(conversation(), stateWith(alternating), options())).toBeUndefined();
	});

	it("drops the other topic's older turns when the recent turns stay on one topic", () => {
		const assignments = { ...alternating, 800: ["t1"] };
		const projection = projectContext(conversation(), stateWith(assignments), options({ keepRecentTurns: 1 }));
		expect(projection).toBeDefined();
		const kept = projection!.messages.filter((m) => m.role === "user").map((m) => m.timestamp);
		expect(kept).toEqual([100, 300, 500, 700, 800]);
		const digest = projection!.messages[0];
		expect(digest.role === "custom" && digest.customType).toBe(DIGEST_MESSAGE_TYPE);
		expect(digest.role === "custom" && String(digest.content)).toContain("[t2] — 3 turn(s) not shown");
		expect(toolPairsIntact(projection!.messages)).toBe(true);
	});

	it("brings a topic into focus when the latest prompt matches it lexically", () => {
		const messages = conversation();
		messages[messages.length - 2] = user(800, "back to the dashboard styling");
		const projection = projectContext(
			messages,
			stateWith({ ...alternating, 800: ["t1"] }),
			options({ keepRecentTurns: 1 }),
		);
		expect(projection).toBeUndefined();
	});

	it("keeps unclassified older turns rather than hiding them", () => {
		const { 300: _omitted, ...assignments } = { ...alternating, 800: ["t1"] };
		const state = stateWith(assignments);
		const projection = projectContext(conversation(), state, options({ keepRecentTurns: 1 }));
		const kept = projection!.messages.filter((m) => m.role === "user").map((m) => m.timestamp);
		expect(kept).toContain(300);
	});

	it("drops the oldest on-topic turns first when the budget is short", () => {
		const assignments = { ...alternating, 800: ["t1"] };
		// target 12 - digest reserve 1 - recent turn (2 messages) leaves 9 tokens: turns 700, 500, 300 fit, 100 does not.
		const projection = projectContext(
			conversation(),
			stateWith(assignments),
			options({ keepRecentTurns: 1, targetTokens: 12 }),
		);
		const kept = projection!.messages.filter((m) => m.role === "user").map((m) => m.timestamp);
		expect(kept).toEqual([300, 500, 700, 800]);
	});

	it("refuses to project when the result would orphan a tool result", () => {
		const messages = [...conversation(), toolResult(900, "call-missing", "orphan")];
		const assignments = { ...alternating, 800: ["t1"] };
		expect(projectContext(messages, stateWith(assignments), options({ keepRecentTurns: 1 }))).toBeUndefined();
	});

	it("detects orphaned tool results and accepts intact pairs", () => {
		expect(toolPairsIntact([assistant(1, "x", "a"), toolResult(2, "a", "ok")])).toBe(true);
		expect(toolPairsIntact([toolResult(2, "a", "ok"), assistant(1, "x", "a")])).toBe(false);
	});
});

describe("topic-tree classification parsing", () => {
	const turns = [
		{ key: "100", text: "a" },
		{ key: "200", text: "b" },
	];

	it("resolves new-topic refs to allocated ids and accepts fenced JSON", () => {
		const state = stateWith({});
		const reply =
			'```json\n{"newTopics":[{"ref":"n1","title":"部署","parent":"t1"}],' +
			'"assignments":[{"turn":0,"topics":["t1"]},{"turn":1,"topics":["n1","t2"]}],"globalFacts":["不要改发布配置"]}\n```';
		const result = parseClassifyResponse(reply, turns, state);
		expect(result.newTopics).toEqual([{ ref: "t3", title: "部署", parentId: "t1" }]);
		expect(result.assignments).toEqual([
			{ turnKey: "100", topicIds: ["t1"] },
			{ turnKey: "200", topicIds: ["t3", "t2"] },
		]);
		expect(result.globalFacts).toEqual(["不要改发布配置"]);
	});

	it.each([
		['{"assignments":[{"turn":0,"topics":["t1"]}]}', "unassigned"],
		['{"assignments":[{"turn":0,"topics":["t9"]},{"turn":1,"topics":["t1"]}]}', "unknown topic"],
		['{"assignments":[{"turn":5,"topics":["t1"]},{"turn":1,"topics":["t1"]}]}', "invalid turn index"],
		['{"assignments":[{"turn":0,"topics":["global"]},{"turn":1,"topics":["t1"]}]}', "unknown topic"],
		["not json at all", "no JSON"],
	])("rejects an unusable reply: %s", (reply, message) => {
		expect(() => parseClassifyResponse(reply, turns, stateWith({}))).toThrow(message);
	});

	it("attaches a rules-only turn with no topic to its neighbour, else the active topic", () => {
		const reply =
			'{"newTopics":[{"ref":"n1","title":"部署","parent":null}],' +
			'"assignments":[{"turn":0,"topics":[]},{"turn":1,"topics":["n1"]}],"globalFacts":["金额保留两位小数"]}';
		const result = parseClassifyResponse(reply, turns, stateWith({}));
		expect(result.assignments[0].topicIds).toEqual(["t3"]);
		expect(result.globalFacts).toEqual(["金额保留两位小数"]);
		const alone = parseClassifyResponse(
			'{"assignments":[{"turn":0,"topics":[]},{"turn":1,"topics":[]}]}',
			turns,
			stateWith({ 50: ["t2"] }),
		);
		expect(alone.assignments.map((a) => a.topicIds)).toEqual([["t2"], ["t2"]]);
		expect(() =>
			parseClassifyResponse('{"assignments":[{"turn":0,"topics":[]},{"turn":1,"topics":[]}]}', turns, stateWith({})),
		).toThrow("nothing to attach");
	});

	it("falls back to the last active topic, or opens a catch-all one", () => {
		const active = stateWith({ 50: ["t2"] });
		expect(fallbackClassification(turns, active).assignments.every((a) => a.topicIds[0] === "t2")).toBe(true);
		const fresh = fallbackClassification(turns, emptyState());
		expect(fresh.newTopics).toEqual([{ ref: "t1", title: "General", parentId: null }]);
	});

	it("merges global facts without duplicates", () => {
		expect(mergeGlobalFacts("- a", ["a", "- b", "b"])).toBe("- a\n- b");
	});
});

describe("topic-tree projection budget", () => {
	// Roughly chars/4, like the real estimator, so the digest's size matters.
	const byLength = (m: AgentMessage) => {
		const content = (m as { content?: unknown }).content;
		const text =
			typeof content === "string"
				? content
				: Array.isArray(content)
					? content.map((b: { text?: string }) => b.text ?? "").join("")
					: "";
		return Math.ceil(text.length / 4) + 1;
	};
	const total = (ms: AgentMessage[]) => ms.reduce((sum, m) => sum + byLength(m), 0);

	function manyTopics(turnChars: number, topics: number) {
		const state = emptyState();
		const messages: AgentMessage[] = [];
		for (let i = 1; i <= 40; i++) {
			const topic = `t${(i % topics) + 1}`;
			if (!state.nodes.has(topic)) {
				applyNode(state, { id: topic, parentId: null, title: `Area ${topic}`, summary: "s".repeat(1200) });
			}
			messages.push(user(i, `question ${i} ${"q".repeat(turnChars)}`), assistant(i, "a".repeat(turnChars)));
			applyAssignments(state, [{ turnKey: String(i), topicIds: [topic] }]);
		}
		return { state, messages };
	}

	it("keeps digest, recent and kept turns within the target together", () => {
		const { state, messages } = manyTopics(400, 12);
		const target = Math.floor(total(messages) * 0.3);
		const projection = projectContext(messages, state, {
			triggerTokens: target,
			targetTokens: target,
			keepRecentTurns: 3,
			estimate: byLength,
		})!;
		expect(total(projection.messages)).toBeLessThanOrEqual(target);
		// Twelve 1200-char summaries do not fit 35% of this target: the digest degrades.
		expect(projection.digestDetail).not.toBe("full");
	});

	it("keeps full summaries when they fit (positive control)", () => {
		const { state, messages } = manyTopics(400, 2);
		const target = Math.floor(total(messages) * 0.5);
		const projection = projectContext(messages, state, {
			triggerTokens: target,
			targetTokens: target,
			keepRecentTurns: 3,
			estimate: byLength,
		})!;
		expect(projection.digestDetail).toBe("full");
		expect(total(projection.messages)).toBeLessThanOrEqual(target);
	});

	it("keeps fewer recent turns when they alone would take most of the budget, never fewer than one", () => {
		const { state, messages } = manyTopics(3000, 3);
		const target = 2500;
		const projection = projectContext(messages, state, {
			triggerTokens: target,
			targetTokens: target,
			keepRecentTurns: 3,
			estimate: byLength,
		})!;
		const users = projection.messages.filter((m) => m.role === "user");
		expect(users.length).toBeGreaterThanOrEqual(1);
		expect(users[users.length - 1].timestamp).toBe(40);
		expect(users.length).toBeLessThan(3);
	});
});

describe("topic-tree thin projection", () => {
	it("sends the prefix, the index and the last turns verbatim, from the first hidden turn on", () => {
		const state = stateWith({ 100: ["t1"], 200: ["t2"], 300: ["t1"] });
		applyNode(state, { id: "global", parentId: null, title: "Global", summary: "- never touch the release config" });
		const messages: AgentMessage[] = [
			{ role: "compactionSummary", summary: "older", tokensBefore: 1, timestamp: 1 },
			...conversation(),
		];
		const projection = projectThin(messages, state, { previousTurns: 2 })!;
		const roles = projection.messages.map((m) => m.role);
		expect(roles[0]).toBe("compactionSummary");
		expect(roles[1]).toBe("custom");
		const kept = projection.messages.filter((m) => m.role === "user").map((m) => m.timestamp);
		expect(kept).toEqual([600, 700, 800]);
		const index = String((projection.messages[1] as { content: string }).content);
		expect(index).toContain("never touch the release config");
		expect(index).toContain("[t1] 数据库迁移");
		// Turns 400 and 500 are hidden and unclassified: the index says so.
		expect(index).toContain("2 earlier turn(s) are not indexed yet");
		expect(projection.droppedTurns).toBe(5);
		expect(toolPairsIntact(projection.messages)).toBe(true);
	});

	it("leaves a short conversation untouched (positive control)", () => {
		const messages = conversation().slice(0, 7); // turns 100–300
		expect(projectThin(messages, emptyState(), { previousTurns: 2 })).toBeUndefined();
	});

	it("keeps a tool call and its result together in the current turn", () => {
		const messages = [
			...conversation(),
			user(900, "now run it"),
			assistant(901, "running", "call-9"),
			toolResult(902, "call-9", "ok"),
		];
		const projection = projectThin(messages, emptyState(), { previousTurns: 0 })!;
		expect(projection.messages.map((m) => m.role)).toEqual(["custom", "user", "assistant", "toolResult"]);
		expect(toolPairsIntact(projection.messages)).toBe(true);
	});
});
