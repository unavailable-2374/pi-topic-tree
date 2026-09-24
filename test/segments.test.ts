import type { AgentMessage } from "@earendil-works/pi-agent-core";
import { afterEach, describe, expect, it, vi } from "vitest";

const completions = vi.hoisted(() => ({ prompts: [] as { system: string; prompt: string; maxTokens: number }[] }));
vi.mock("@earendil-works/pi-ai/compat", async (importOriginal) => ({
	...(await importOriginal<Record<string, unknown>>()),
	complete: async (_model: unknown, context: any, options: any) => {
		const prompt = context.messages[0].content[0].text as string;
		completions.prompts.push({ system: context.systemPrompt, prompt, maxTokens: options.maxTokens });
		const turns = [...prompt.matchAll(/<turn index="(\d+)">/g)].map((match) => Number(match[1]));
		const text = context.systemPrompt.includes("topic tree")
			? JSON.stringify({
					newTopics: [{ ref: "n1", title: "Report writing", parent: null }],
					assignments: turns.map((turn) => ({ turn, topics: ["n1"] })),
					globalFacts: [],
				})
			: "Wrote s1 and read it back";
		return { role: "assistant", content: [{ type: "text", text }], stopReason: "stop" };
	},
}));
import { registerTopicTree } from "../src/index.ts";
import { projectSegments, SEGMENT_INDEX_MESSAGE_TYPE, toolPairsIntact } from "../src/project.ts";
import { ASSIGN_ENTRY, applyNode, buildState, emptyState, RECALL_TOOL_NAME, splitSegments } from "../src/state.ts";


const usage = {
	input: 1,
	output: 1,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 2,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

const PROMPT_TEXT =
	"Write section 1 of the report to drafts/s1.md. Acceptance: every number cites its source table; read the file back before finishing.";

function user(ts: number, text: string): AgentMessage {
	return { role: "user", content: [{ type: "text", text }], timestamp: ts };
}

function step(ts: number, calls: { name: string; path: string }[]): AgentMessage {
	return {
		role: "assistant",
		content: calls.map((call, i) => ({
			type: "toolCall" as const,
			id: `c${ts}-${i}`,
			name: call.name,
			arguments: { path: call.path },
		})),
		api: "anthropic-messages",
		provider: "anthropic",
		model: "claude-test",
		usage,
		stopReason: "toolUse",
		timestamp: ts,
	};
}

function result(ts: number, callId: string, text: string, isError = false, toolName = "read"): AgentMessage {
	return { role: "toolResult", toolCallId: callId, toolName, content: [{ type: "text", text }], isError, timestamp: ts };
}

/** `n` single-call write/read steps, the loop shape of an agent drafting a report section. */
function toolLoop(n: number, start = 1000): AgentMessage[] {
	const messages: AgentMessage[] = [];
	for (let i = 0; i < n; i++) {
		const ts = start + i * 10;
		const name = i % 2 === 0 ? "write" : "read";
		messages.push(step(ts, [{ name, path: "drafts/s1.md" }]));
		messages.push(result(ts + 1, `c${ts}-0`, `${name} ok #${i + 1}`, false, name));
	}
	return messages;
}

const segmentOptions = { previousSegments: 2, toolCallsPerSegment: 4, foldEverySegments: 1 };

const REGISTERED = Symbol.for("pi.topic-tree.registered");

afterEach(() => {
	delete (globalThis as { [REGISTERED]?: boolean })[REGISTERED];
	completions.prompts.length = 0;
});

describe("topic-tree segments", () => {
	it("cuts a long tool loop every N completed tool calls, only before an assistant message", () => {
		const messages = [user(1, PROMPT_TEXT), ...toolLoop(10)];
		const { prefix, turns } = splitSegments(messages, 4);
		expect(prefix).toEqual([]);
		expect(turns.map((segment) => segment.messages.filter((m) => m.role === "toolResult").length)).toEqual([4, 4, 2]);
		expect(turns[0].messages[0]).toBe(messages[0]);
		for (const segment of turns.slice(1)) expect(segment.messages[0].role).toBe("assistant");
		expect(toolPairsIntact(turns.flatMap((segment) => segment.messages))).toBe(true);
	});

	it("counts parallel calls and never splits a step from its results", () => {
		const messages = [
			user(1, PROMPT_TEXT),
			step(10, [
				{ name: "read", path: "a" },
				{ name: "read", path: "b" },
				{ name: "read", path: "c" },
			]),
			result(11, "c10-0", "a"),
			result(12, "c10-1", "b"),
			result(13, "c10-2", "c"),
			step(20, [{ name: "write", path: "d" }]),
			result(21, "c20-0", "ok"),
		];
		const { turns } = splitSegments(messages, 2);
		expect(turns.map((segment) => segment.messages.length)).toEqual([5, 2]);
	});

	it("opens a segment at every new prompt and keeps completed segment keys stable", () => {
		const first = [user(1, PROMPT_TEXT), ...toolLoop(9)];
		const before = splitSegments(first, 4).turns.map((segment) => segment.key);
		const resumed = [...first, user(5000, "Continue: fix the receipt"), ...toolLoop(3, 6000)];
		const after = splitSegments(resumed, 4).turns;
		expect(after.slice(0, 2).map((segment) => segment.key)).toEqual(before.slice(0, 2));
		expect(after.map((segment) => segment.messages[0].role)).toEqual(["user", "assistant", "assistant", "user"]);
	});
});

describe("topic-tree tool-loop projection", () => {
	it("keeps the prompt and injected messages verbatim and hides only older tool steps", () => {
		const prompt = user(1, PROMPT_TEXT);
		const repair: AgentMessage = {
			role: "custom",
			customType: "reminder",
			content: "The receipt is missing its verdict; submit it again.",
			display: false,
			timestamp: 1045,
		};
		const loop = toolLoop(20);
		const messages = [prompt, ...loop.slice(0, 9), repair, ...loop.slice(9)];
		const snapshot = JSON.stringify(prompt);
		const projection = projectSegments(messages, emptyState(), segmentOptions)!;
		expect(projection).toBeDefined();
		// Byte-identical: the very same object, serialised the same.
		expect(projection.messages[0]).toBe(prompt);
		expect(JSON.stringify(projection.messages[0])).toBe(snapshot);
		expect(projection.messages).toContain(repair);
		const index = projection.messages.find(
			(m) => m.role === "custom" && m.customType === SEGMENT_INDEX_MESSAGE_TYPE,
		) as { content: string };
		expect(index.content).toContain("recall_topic");
		expect(index.content).toContain("- s1 (4 call(s)): new prompt; write drafts/s1.md; read drafts/s1.md; write");
		// 5 segments of 4; the last two completed plus the current stay verbatim.
		expect(projection.droppedTurns).toBe(2);
		expect(projection.messages.filter((m) => m.role === "toolResult")).toHaveLength(12);
		expect(toolPairsIntact(projection.messages)).toBe(true);
		// The session's messages themselves are untouched.
		expect(messages).toHaveLength(42);
	});

	it("leaves a short tool loop untouched (positive control)", () => {
		const messages = [user(1, PROMPT_TEXT), ...toolLoop(12)];
		expect(projectSegments(messages, emptyState(), segmentOptions)).toBeUndefined();
	});

	it("carries an unresolved refusal from a hidden segment forward verbatim, and drops a resolved one", () => {
		const refusal =
			"Tool bash is not allowed in this session; use read and write to make progress.";
		const messages = [
			user(1, PROMPT_TEXT),
			step(10, [{ name: "bash", path: "x" }]),
			result(11, "c10-0", refusal, true, "bash"),
			step(20, [{ name: "read", path: "gone.md" }]),
			result(21, "c20-0", "ENOENT: no such file", true, "read"),
			...toolLoop(14, 100),
		];
		const projection = projectSegments(messages, emptyState(), segmentOptions)!;
		const index = String((projection.messages[1] as { content: string }).content);
		expect(index).toContain(`Unresolved at this point (verbatim) — bash, segment s1:\n${refusal}`);
		// A later successful read answered the read error.
		expect(index).not.toContain("ENOENT");
	});

	it("keeps the latest step of each controller tool whole, and folds older ones", () => {
		const control = (ts: number, text: string): AgentMessage[] => [
			{ ...(step(ts, [{ name: "task_workflow", path: "decide" }]) as object) } as AgentMessage,
			result(ts + 1, `c${ts}-0`, text, false, "task_workflow"),
		];
		const older = control(20, "decision v1");
		const latest = control(40, "decision v2: next call must be write");
		const messages = [user(1, PROMPT_TEXT), ...older, ...latest, ...toolLoop(16, 100)];
		const projection = projectSegments(messages, emptyState(), segmentOptions)!;
		for (const message of latest) expect(projection.messages).toContain(message);
		for (const message of older) expect(projection.messages).not.toContain(message);
		expect(toolPairsIntact(projection.messages)).toBe(true);
	});

	it("keeps every prompt verbatim and shows a caller's turn label", () => {
		const first = user(1, PROMPT_TEXT);
		const resume = user(5000, "Review returned: section 2 lacks effect sizes, revise drafts/s1.md");
		const messages = [first, ...toolLoop(9), resume, ...toolLoop(14, 6000)];
		const state = emptyState();
		state.labels.set("5000", "review: effect sizes missing");
		const projection = projectSegments(messages, state, segmentOptions)!;
		expect(projection.messages.filter((m) => m.role === "user")).toEqual([first, resume]);
		const index = String(
			(projection.messages.find((m) => m.role === "custom") as { content: string }).content,
		);
		expect(index).toContain("new prompt (review: effect sizes missing)");
		expect(toolPairsIntact(projection.messages)).toBe(true);
	});

	it("keeps the index independent of live classifier state, so a background pass cannot break the cache", () => {
		const messages = [user(1, PROMPT_TEXT), ...toolLoop(20)];
		const options = { previousSegments: 2, toolCallsPerSegment: 4, foldEverySegments: 1 };
		const before = projectSegments(messages, emptyState(), options)!;
		const state = emptyState();
		applyNode(state, { id: "t1", parentId: null, title: "Report writing", summary: "Wrote s1 and read it back" });
		state.assignments.set("1", ["t1"]);
		const after = projectSegments(messages, state, options)!;
		expect(JSON.stringify(after.messages)).toBe(JSON.stringify(before.messages));
	});

	it("folds in discrete steps: between folds each request extends the previous one, and a fold only appends to the index", () => {
		const options = { previousSegments: 2, toolCallsPerSegment: 4, foldEverySegments: 3 };
		const prompt = user(1, PROMPT_TEXT);
		const loop = toolLoop(60);
		let previous: AgentMessage[] | undefined;
		let previousIndex: string | undefined;
		let folds = 0;
		for (let steps = 1; steps <= 60; steps++) {
			const messages = [prompt, ...loop.slice(0, steps * 2)];
			const projected = projectSegments(messages, emptyState(), options)?.messages ?? messages;
			const index = projected.find((m) => m.role === "custom") as { content: string } | undefined;
			if (previous) {
				const sameFold = index?.content === previousIndex;
				if (sameFold) {
					// Byte-identical prefix: the previous request is a prefix of this one.
					expect(JSON.stringify(projected.slice(0, previous.length))).toBe(JSON.stringify(previous));
				} else {
					folds++;
					if (previousIndex) expect(index!.content.startsWith(previousIndex)).toBe(true);
					// Everything before the index is unchanged across the fold.
					expect(projected[0]).toBe(prompt);
				}
			}
			previous = projected;
			previousIndex = index?.content;
		}
		// 15 segments: hidden moves 0 -> 3 -> 6 -> 9 -> 12 only.
		expect(folds).toBe(4);
	});
});

type Handler = (event: any, ctx?: any) => any;

function fakePi(flags: Record<string, unknown> = {}) {
	const handlers = new Map<string, Handler[]>();
	const tools: any[] = [];
	const entries: any[] = [];
	const pi = {
		on: (name: string, handler: Handler) => handlers.set(name, [...(handlers.get(name) ?? []), handler]),
		registerFlag: () => {},
		getFlag: (name: string) => flags[name],
		registerTool: (tool: any) => tools.push(tool),
		appendEntry: (customType: string, data: unknown) => entries.push({ type: "custom", customType, data }),
	};
	const fire = (name: string, event: any, ctx?: any) => (handlers.get(name) ?? []).map((handler) => handler(event, ctx));
	return { pi, handlers, tools, entries, fire };
}

function sessionCtx(messages: AgentMessage[], extra: any[] = []) {
	const branch = [...messages.map((message) => ({ type: "message", message })), ...extra];
	return {
		model: { contextWindow: 128000 },
		mode: "json",
		modelRegistry: { getApiKeyAndHeaders: async () => ({ ok: true, apiKey: "k" }) },
		sessionManager: { getBranch: () => branch, getSessionId: () => "session" },
	};
}

describe("topic-tree extension entry", () => {
	it("stays off without --topic-tree or PI_TOPIC_TREE: no tool, no projection (positive control)", () => {
		const { pi, tools, fire } = fakePi();
		registerTopicTree(pi as never, {});
		fire("session_start", {});
		expect(tools).toHaveLength(0);
		const messages = [user(1, PROMPT_TEXT), ...toolLoop(40)];
		expect(fire("context", { messages }, sessionCtx(messages))).toEqual([undefined]);
	});

	it("turns on from the environment and registers recall_topic", () => {
		const { pi, tools, fire } = fakePi();
		registerTopicTree(pi as never, { PI_TOPIC_TREE: "1" });
		fire("session_start", {});
		expect(tools.map((tool) => tool.name)).toEqual([RECALL_TOOL_NAME]);
	});

	it("registers once when the package is loaded twice", () => {
		const first = fakePi({ "topic-tree": true });
		const second = fakePi({ "topic-tree": true });
		registerTopicTree(first.pi as never, {});
		registerTopicTree(second.pi as never, {});
		expect(second.handlers.size).toBe(0);
	});

	it("folds a long single-turn tool loop when --topic-tree-fold-tool-calls is set, keeping the prompt byte-identical", () => {
		const { pi, fire } = fakePi({ "topic-tree": true, "topic-tree-fold-tool-calls": "4", "topic-tree-fold-every": "1" });
		registerTopicTree(pi as never, {});
		fire("session_start", {});
		const prompt = user(1, PROMPT_TEXT);
		const messages = [prompt, ...toolLoop(40)];
		const [result] = fire("context", { messages }, sessionCtx(messages));
		expect(result.messages[0]).toBe(prompt);
		expect(result.messages.some((m: AgentMessage) => m.role === "custom" && m.customType === SEGMENT_INDEX_MESSAGE_TYPE)).toBe(true);
		expect(result.messages.length).toBeLessThan(messages.length);
		expect(toolPairsIntact(result.messages)).toBe(true);
	});

	it("keeps the turn-only projection when folding is off (positive control)", () => {
		const { pi, fire } = fakePi({ "topic-tree": true });
		registerTopicTree(pi as never, {});
		fire("session_start", {});
		const messages = [user(1, PROMPT_TEXT), ...toolLoop(40)];
		expect(fire("context", { messages }, sessionCtx(messages))).toEqual([undefined]);
	});

	it("returns a hidden segment verbatim through recall_topic, tool results included", async () => {
		const { pi, tools, fire } = fakePi({ "topic-tree": true, "topic-tree-fold-tool-calls": "4" });
		registerTopicTree(pi as never, {});
		fire("session_start", {});
		const messages = [
			user(1, PROMPT_TEXT),
			step(10, [{ name: "bash", path: "sha256sum out.bin" }]),
			result(11, "c10-0", "9f86d081884c7d65  out.bin", false, "bash"),
			...toolLoop(20, 100),
		];
		const ctx = sessionCtx(messages);
		const reply = await tools[0].execute("call", { action: "get", topic: "s1" }, undefined, undefined, ctx);
		expect(reply.content[0].text).toContain("9f86d081884c7d65  out.bin");
		const search = await tools[0].execute("call", { action: "search", query: "sha256sum" }, undefined, undefined, ctx);
		expect(search.content[0].text).toContain("Turn 1");
	});

	it("classifies turns that left the thin window after the agent ends, on the session model", async () => {
		const { pi, entries, fire } = fakePi({ "topic-tree": true, "topic-tree-classify-every": "1" });
		registerTopicTree(pi as never, {});
		fire("session_start", {});
		const messages = [user(1, "first question"), ...toolLoop(1, 10), user(100, "second"), user(200, "third"), user(300, "fourth")];
		await Promise.all(fire("agent_end", {}, sessionCtx(messages)));
		expect(completions.prompts.some((call) => call.system.includes("topic tree"))).toBe(true);
		expect(entries.some((entry) => entry.customType === ASSIGN_ENTRY)).toBe(true);
		expect(buildState(entries as never).nodes.size).toBeGreaterThan(1);
	});
});

