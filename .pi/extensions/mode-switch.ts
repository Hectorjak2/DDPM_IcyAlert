/**
 * Mode Switch — two operating modes for the DDPM_IcyAlert project.
 *
 *   • NORMAL mode (default, GREEN):
 *       Not restricted. read / edit / write / bash all work. The only
 *       guardrail is a permission prompt before RISKY bash commands
 *       (rm -rf, sudo, chmod/chown 777, git push --force, dd, mkfs, disk
 *       redirects, etc). Everything else runs without asking.
 *
 *   • EXPERIMENT mode (YELLOW):
 *       The original thesis guardrails, enforced as code:
 *         1. Bash allowlist: `./utils/hpc_run.sh <name>` (launch), plus a
 *            READ-ONLY analysis lane — the sample evaluator
 *            (`./.venv/bin/python utils/evaluate_samples.py ...`) and ad-hoc
 *            analysis scripts under `analysis/` run through the project venv.
 *            Shell chaining / metacharacters are rejected in all cases.
 *         2. `write`/`edit` only on config.py OR scratch files under
 *            `analysis/` (temporary evaluation scripts). The source pipeline
 *            (main.py, models/, utils/) stays read-only.
 *         3. Name-match invariant: launch arg must equal config.py's
 *            experiment_name.
 *       Only launches count toward the budget; evaluation runs are free.
 *       Plus the live dashboard + findings log.
 *
 * Toggle between modes with SHIFT+TAB (or the /mode command).
 * Reasoning level cycling has been moved off shift+tab to CMD+R
 * (see .pi/../keybindings — app.thinking.cycle = super+r).
 */

import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { readFileSync, writeFileSync, existsSync, mkdirSync } from "node:fs";
import { resolve, dirname, join } from "node:path";

// The one and only launch command (experiment mode). Exactly one arg (the name).
const LAUNCH_RE = /^\s*\.\/utils\/hpc_run\.sh\s+(\S+)\s*$/;
// Extract experiment_name from config.py.
const NAME_RE = /experiment_name\s*:\s*str\s*=\s*["']([^"']+)["']/;

// ---- Analysis lane (experiment mode, read-only w.r.t. the pipeline) --------
// Scratch directory for temporary evaluation scripts the agent may write/edit.
const ANALYSIS_DIR = "analysis";
// Safe argument tail: no shell metacharacters (no ; & | $ ` quotes redirects),
// so these commands cannot smuggle a second command.
const SAFE_TAIL = /^[\w\s./=-]*$/;
// Run the sample evaluator through the project venv.
const EVAL_RE = /^\s*\.\/\.venv\/bin\/python3?\s+utils\/evaluate_samples\.py(\s+[\w\s./=-]+)?$/;
// Run an ad-hoc analysis script that lives under analysis/ through the venv.
const ANALYSIS_RUN_RE = /^\s*\.\/\.venv\/bin\/python3?\s+analysis\/[\w./-]+\.py(\s+[\w\s./=-]+)?$/;

// Risky bash patterns that require an explicit user OK in NORMAL mode.
const RISKY_PATTERNS: { re: RegExp; label: string }[] = [
  { re: /\brm\s+(-[a-z]*r[a-z]*f|-[a-z]*f[a-z]*r|--recursive)\b/i, label: "recursive/forced delete" },
  { re: /\brm\s+-[a-z]*f\b/i, label: "forced delete" },
  { re: /\bsudo\b/i, label: "sudo" },
  { re: /\b(chmod|chown)\b[^\n]*\b777\b/i, label: "chmod/chown 777" },
  { re: /\bgit\s+push\b[^\n]*(--force|-f)\b/i, label: "git force-push" },
  { re: /\bgit\s+(reset\s+--hard|clean\s+-[a-z]*f)/i, label: "destructive git" },
  { re: /\bdd\b[^\n]*\bof=/i, label: "dd write" },
  { re: /\bmkfs\b/i, label: "format filesystem" },
  { re: /\b(shutdown|reboot|halt|poweroff)\b/i, label: "power/reboot" },
  { re: /\bkill(all)?\s+-9\b/i, label: "kill -9" },
  { re: />\s*\/dev\/(sd|nvme|disk|null|zero)/i, label: "device redirect" },
  { re: /:\(\)\s*\{.*\|.*&\s*\}/, label: "fork bomb" },
];

type Mode = "normal" | "experiment";
type Status = "running" | "completed" | "failed";

interface ExperimentRecord {
  name: string;
  status: Status;
  startedAt: number;
  endedAt?: number;
}
interface Finding {
  text: string;
  at: number;
}
interface Note {
  text: string;
  at: number;
}
interface Budget {
  total: number;
  startedAt: number;
}
interface State {
  mode: Mode;
  experiments: ExperimentRecord[];
  findings: Finding[];
  notes: Note[];
  budget: Budget | null;
}

export default function (pi: ExtensionAPI) {
  let statePath = "";
  const state: State = { mode: "normal", experiments: [], findings: [], notes: [], budget: null };
  // Map bash toolCallId -> index in state.experiments for the running run.
  const runningByCall = new Map<string, number>();

  const stripQuotes = (s: string) => s.replace(/^["']|["']$/g, "");

  function loadState(ctx: ExtensionContext) {
    statePath = join(ctx.cwd, ".pi", "experiment-state.json");
    try {
      if (existsSync(statePath)) {
        const parsed = JSON.parse(readFileSync(statePath, "utf8")) as Partial<State>;
        state.mode = parsed.mode === "experiment" ? "experiment" : "normal";
        state.experiments = parsed.experiments ?? [];
        state.findings = parsed.findings ?? [];
        state.notes = parsed.notes ?? [];
        state.budget = parsed.budget ?? null;
        // Any run left "running" from a previous session is stale/unknown.
        for (const e of state.experiments) {
          if (e.status === "running") e.status = "failed";
        }
      }
    } catch {
      /* start fresh on parse error */
    }
  }

  function saveState() {
    if (!statePath) return;
    try {
      mkdirSync(dirname(statePath), { recursive: true });
      writeFileSync(statePath, JSON.stringify(state, null, 2));
    } catch {
      /* non-fatal */
    }
  }

  function readConfigName(ctx: ExtensionContext): string | null {
    try {
      const txt = readFileSync(resolve(ctx.cwd, "config.py"), "utf8");
      const m = txt.match(NAME_RE);
      return m ? m[1] : null;
    } catch {
      return null;
    }
  }

  function isConfigPy(ctx: ExtensionContext, p: string | undefined): boolean {
    if (!p) return false;
    return resolve(ctx.cwd, p) === resolve(ctx.cwd, "config.py");
  }

  // True if p resolves to somewhere inside the scratch analysis/ directory
  // (and does not escape it via ..).
  function isAnalysisPath(ctx: ExtensionContext, p: string | undefined): boolean {
    if (!p) return false;
    const root = resolve(ctx.cwd, ANALYSIS_DIR);
    const target = resolve(ctx.cwd, p);
    return target === root || target.startsWith(root + "/");
  }

  function ts(ms: number): string {
    return new Date(ms).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  // Runs counted toward the current budget: every launch since the budget was set
  // (running, completed, OR failed — failures still consume the budget).
  function usedCount(): number {
    if (!state.budget) return 0;
    const since = state.budget.startedAt;
    return state.experiments.filter((e) => e.startedAt >= since).length;
  }

  // Orange branding for experiment mode (no matching theme key). 256-color 208.
  const orange = (t: string) => `\x1b[38;5;208m${t}\x1b[0m`;

  // ---- UI -------------------------------------------------------------------
  function render(ctx: ExtensionContext) {
    if (!ctx.hasUI) return;
    const theme = ctx.ui.theme;

    if (state.mode === "normal") {
      // Footer: green.
      ctx.ui.setStatus(
        "mode-switch",
        theme.fg("success", "\u{1F7E2} normal") +
          theme.fg("muted", "  \u21E5 shift+tab \u2192 experiment"),
      );
      const lines = [
        theme.fg("success", "\u2500\u2500 Normal Mode \u2500\u2500"),
        theme.fg("muted", "  full access \u00b7 risky bash asks first"),
      ];
      if (state.notes.length > 0) {
        lines.push(theme.fg("accent", `  \u2139 ${state.notes.length} note(s) in context ( /notes )`));
      }
      if (state.findings.length > 0) {
        lines.push(theme.fg("warning", `  \u270E ${state.findings.length} finding(s) ( /findings )`));
      }
      if (state.budget) {
        lines.push(theme.fg("accent", `  \u25F7 budget ${usedCount()}/${state.budget.total} runs this iteration`));
      }
      ctx.ui.setWidget("mode-switch", lines);
      return;
    }

    // EXPERIMENT mode dashboard (yellow).
    const glyph: Record<Status, string> = {
      running: "\u25CF running ",
      completed: "\u2714 done    ",
      failed: "\u2718 failed  ",
    };
    const lines: string[] = [orange("\u2500\u2500 Experiment Mode \u2500\u2500")];
    if (state.experiments.length === 0) {
      lines.push(theme.fg("muted", "  (no experiments yet)"));
    } else {
      for (const e of state.experiments.slice(-8)) {
        const when = e.endedAt ? `${ts(e.startedAt)}\u2192${ts(e.endedAt)}` : `${ts(e.startedAt)}`;
        const glyphStr =
          e.status === "completed"
            ? theme.fg("success", `  ${glyph[e.status]}`)
            : e.status === "failed"
              ? theme.fg("error", `  ${glyph[e.status]}`)
              : orange(`  ${glyph[e.status]}`);
        lines.push(glyphStr + ` ${e.name}  ${theme.fg("dim", `[${when}]`)}`);
      }
    }
    if (state.budget) {
      lines.push(orange(`  \u25F7 budget ${usedCount()}/${state.budget.total} runs this iteration`));
    }
    if (state.notes.length > 0) {
      lines.push(theme.fg("accent", `  \u2139 ${state.notes.length} note(s) in context ( /notes )`));
    }
    if (state.findings.length > 0) {
      lines.push(theme.fg("warning", `  \u270E ${state.findings.length} finding(s) to raise ( /findings )`));
    }
    ctx.ui.setWidget("mode-switch", lines);

    const running = state.experiments.filter((e) => e.status === "running").length;
    const done = state.experiments.filter((e) => e.status === "completed").length;
    const failed = state.experiments.filter((e) => e.status === "failed").length;
    ctx.ui.setStatus(
      "mode-switch",
      orange(`\u{1F9EA} exp-mode`) +
        theme.fg(
          "muted",
          ` | ${running} running, ${done} done, ${failed} failed` +
            (state.budget ? ` \u00b7 budget ${usedCount()}/${state.budget.total}` : "") +
            `  \u21E5 shift+tab \u2192 normal`,
        ),
    );
  }

  function setMode(ctx: ExtensionContext, mode: Mode, announce = true) {
    state.mode = mode;
    saveState();
    render(ctx);
    if (announce && ctx.hasUI) {
      if (mode === "normal") {
        ctx.ui.notify("Normal mode \u2014 full access. Risky bash commands will ask first.", "info");
      } else {
        ctx.ui.notify("Experiment mode \u2014 only config.py edits + ./utils/hpc_run.sh.", "warn");
      }
    }
  }

  pi.on("session_start", async (_event, ctx) => {
    loadState(ctx);
    render(ctx);
  });

  // Inject persisted experiment notes/constraints into context every turn.
  pi.on("before_agent_start", async (event, _ctx) => {
    let block = "";
    if (state.notes.length > 0) {
      block +=
        "\n\n## Known experiment constraints / notes (persisted)\n" +
        state.notes.map((n) => `- ${n.text}`).join("\n");
    }
    if (state.budget) {
      const used = usedCount();
      const remaining = Math.max(0, state.budget.total - used);
      block +=
        "\n\n## Experiment budget (this iteration)\n" +
        `- ${used}/${state.budget.total} runs used \u2014 EVERY launch counts, success or fail \u2014 ${remaining} remaining.\n` +
        (remaining > 0
          ? "- You may launch experiments back-to-back until the budget is spent.\n" +
            `- The moment it reaches ${state.budget.total}/${state.budget.total}, STOP launching and hand back to the user to review this iteration.`
          : "- Budget spent. Do NOT launch more. Summarise the iteration and hand back to the user for review.");
    }
    if (!block) return;
    return { systemPrompt: event.systemPrompt + block };
  });

  // ---- Mode toggle: shift+tab + /mode ---------------------------------------
  pi.registerShortcut("shift+tab", {
    description: "Toggle Normal / Experiment mode",
    handler: async (ctx) => setMode(ctx, state.mode === "normal" ? "experiment" : "normal"),
  });

  pi.registerCommand("mode", {
    description: "Toggle Normal / Experiment mode (shift+tab)",
    handler: async (_args, ctx) => setMode(ctx, state.mode === "normal" ? "experiment" : "normal"),
  });

  // ---- Guardrails -----------------------------------------------------------
  pi.on("tool_call", async (event, ctx) => {
    // ===== BUDGET: deterministic cap on launches, enforced in ANY mode. =====
    if (event.toolName === "bash" && state.budget) {
      const command = (event.input as { command?: string }).command ?? "";
      if (LAUNCH_RE.test(command)) {
        const used = usedCount();
        if (used >= state.budget.total) {
          return {
            block: true,
            reason:
              `Experiment budget reached: ${used}/${state.budget.total} runs used this iteration ` +
              "(every launch counts, success or fail). Do NOT launch more. Stop and hand back to " +
              "the user to review. To continue, the user sets a new budget (/budget <n>) or starts a new session.",
          };
        }
      }
    }

    // ===== NORMAL MODE: unrestricted, but confirm risky bash. =====
    if (state.mode === "normal") {
      if (event.toolName !== "bash") return; // read/edit/write/etc pass through
      const command = (event.input as { command?: string }).command ?? "";
      const hit = RISKY_PATTERNS.find((p) => p.re.test(command));
      if (!hit) return;

      if (!ctx.hasUI) {
        return { block: true, reason: `Risky command blocked (${hit.label}); no UI to confirm.` };
      }
      const choice = await ctx.ui.select(
        `\u26A0\uFE0F Risky command (${hit.label}):\n\n  ${command}\n\nAllow?`,
        ["No", "Yes, run it"],
      );
      if (choice !== "Yes, run it") {
        return { block: true, reason: "Blocked by user." };
      }
      return;
    }

    // ===== EXPERIMENT MODE: original hard guardrails. =====
    if (event.toolName === "write") {
      const p = (event.input as { path?: string }).path;
      if (!isAnalysisPath(ctx, p)) {
        return {
          block: true,
          reason:
            `Experiment mode: \`write\` is only allowed for scratch analysis scripts under \`${ANALYSIS_DIR}/\` (got "${p}"). ` +
            "Edit config.py to change the experiment, or switch to normal mode (shift+tab).",
        };
      }
      return;
    }

    if (event.toolName === "edit") {
      const p = (event.input as { path?: string }).path;
      if (!isConfigPy(ctx, p) && !isAnalysisPath(ctx, p)) {
        return {
          block: true,
          reason:
            `Experiment mode: edits are only allowed on config.py or scratch scripts under \`${ANALYSIS_DIR}/\` (got "${p}"). ` +
            "Record other changes with record_finding, or switch to normal mode (shift+tab).",
        };
      }
      return;
    }

    if (event.toolName === "bash") {
      const cmd = (event.input as { command?: string }).command ?? "";
      // Analysis lane: evaluator + ad-hoc analysis scripts (read-only lane).
      // SAFE_TAIL on the whole command rejects chaining/metacharacters.
      if ((EVAL_RE.test(cmd) || ANALYSIS_RUN_RE.test(cmd)) && SAFE_TAIL.test(cmd)) {
        return; // allowed, does not count toward the budget
      }
      const m = cmd.match(LAUNCH_RE);
      if (!m) {
        return {
          block: true,
          reason:
            "Experiment mode: allowed bash is `./utils/hpc_run.sh <experiment_name>` (launch), " +
            "`./.venv/bin/python utils/evaluate_samples.py <name> [...]` (evaluate), or " +
            "`./.venv/bin/python analysis/<script>.py [...]` (ad-hoc analysis). No shell chaining. " +
            "Use read/grep/find/ls to inspect files, or switch to normal mode (shift+tab).",
        };
      }
      const arg = stripQuotes(m[1]);
      const cfgName = readConfigName(ctx);
      if (cfgName === null) {
        return { block: true, reason: "Experiment mode: could not read experiment_name from config.py." };
      }
      if (arg !== cfgName) {
        return {
          block: true,
          reason:
            `Experiment mode: name mismatch. config.py experiment_name = "${cfgName}", ` +
            `launch argument = "${arg}". Make them match, then launch with the SAME name.`,
        };
      }
      return; // allowed
    }
    // read / grep / find / ls and custom tools pass through.
  });

  // ---- Experiment dashboard tracking ---------------------------------------
  pi.on("tool_execution_start", async (event, ctx) => {
    if (event.toolName !== "bash") return;
    const cmd = (event.args as { command?: string }).command ?? "";
    const m = cmd.match(LAUNCH_RE);
    if (!m) return;
    const name = stripQuotes(m[1]);
    const idx = state.experiments.length;
    state.experiments.push({ name, status: "running", startedAt: Date.now() });
    runningByCall.set(event.toolCallId, idx);
    saveState();
    render(ctx);
  });

  pi.on("tool_execution_end", async (event, ctx) => {
    const idx = runningByCall.get(event.toolCallId);
    if (idx === undefined) return;
    runningByCall.delete(event.toolCallId);
    const rec = state.experiments[idx];
    if (rec) {
      rec.status = event.isError ? "failed" : "completed";
      rec.endedAt = Date.now();
    }
    saveState();
    render(ctx);
    if (ctx.hasUI) {
      ctx.ui.notify(
        `Experiment "${rec?.name}" ${rec?.status}.`,
        rec?.status === "completed" ? "info" : "warn",
      );
    }
  });

  // ---- Findings tool + commands --------------------------------------------
  pi.registerTool({
    name: "record_finding",
    label: "Record finding",
    description:
      "Record an out-of-bounds idea or needed change that cannot be made now " +
      "(anything beyond editing config.py in experiment mode). Raised to the user later.",
    parameters: Type.Object({
      text: Type.String({ description: "The finding to raise later." }),
    }),
    async execute(_id, params, _signal, _onUpdate, ctx) {
      state.findings.push({ text: params.text, at: Date.now() });
      saveState();
      render(ctx);
      return {
        content: [{ type: "text", text: `Recorded finding #${state.findings.length}.` }],
        details: {},
      };
    },
  });

  pi.registerTool({
    name: "record_note",
    label: "Record note",
    description:
      "Record a durable experiment constraint or discovery (e.g. 'batch_size max = 2 at " +
      "current resolution'). Notes persist across sessions and are auto-injected into " +
      "context on every turn, in both modes.",
    parameters: Type.Object({
      text: Type.String({ description: "The constraint/discovery to remember." }),
    }),
    async execute(_id, params, _signal, _onUpdate, ctx) {
      state.notes.push({ text: params.text, at: Date.now() });
      saveState();
      render(ctx);
      return {
        content: [{ type: "text", text: `Recorded note #${state.notes.length}.` }],
        details: {},
      };
    },
  });

  pi.registerTool({
    name: "set_experiment_budget",
    label: "Set experiment budget",
    description:
      "Start an experiment iteration with a fixed number of runs. EVERY launch of " +
      "./utils/hpc_run.sh counts toward the budget (success OR failure). You may launch " +
      "back-to-back until the budget is spent; then launches are hard-blocked until the " +
      "user sets a new budget or starts a new session. Setting a budget resets the count.",
    parameters: Type.Object({
      total: Type.Number({ minimum: 1, description: "Number of experiment runs allowed this iteration (integer >= 1)." }),
    }),
    async execute(_id, params, _signal, _onUpdate, ctx) {
      const total = Math.floor(params.total);
      if (!Number.isFinite(total) || total < 1) {
        return {
          content: [{ type: "text", text: "Budget must be an integer >= 1." }],
          details: {},
        };
      }
      state.budget = { total, startedAt: Date.now() };
      saveState();
      render(ctx);
      return {
        content: [{ type: "text", text: `Experiment budget set: 0/${total} runs used this iteration.` }],
        details: {},
      };
    },
  });

  pi.registerCommand("note", {
    description: "Add a durable experiment constraint/note (auto-injected into context)",
    handler: async (args, ctx) => {
      const text = (args ?? "").trim();
      if (!text) {
        ctx.ui.notify("Usage: /note <constraint or discovery>", "warn");
        return;
      }
      state.notes.push({ text, at: Date.now() });
      saveState();
      render(ctx);
      ctx.ui.notify(`Note #${state.notes.length} added (now in context).`, "info");
    },
  });

  pi.registerCommand("notes", {
    description: "List persisted experiment notes/constraints",
    handler: async (_args, ctx) => {
      if (state.notes.length === 0) {
        ctx.ui.notify("No notes recorded.", "info");
        return;
      }
      const lines = state.notes.map((n, i) => `${i + 1}. [${ts(n.at)}] ${n.text}`);
      ctx.ui.notify(`Experiment notes:\n${lines.join("\n")}`, "info");
    },
  });

  pi.registerCommand("clear-notes", {
    description: "Clear persisted experiment notes",
    handler: async (_args, ctx) => {
      const ok = await ctx.ui.confirm("Clear notes?", `Delete ${state.notes.length} note(s)?`);
      if (!ok) return;
      state.notes = [];
      saveState();
      render(ctx);
      ctx.ui.notify("Notes cleared.", "info");
    },
  });

  pi.registerCommand("findings", {
    description: "List recorded out-of-bounds findings to raise",
    handler: async (_args, ctx) => {
      if (state.findings.length === 0) {
        ctx.ui.notify("No findings recorded.", "info");
        return;
      }
      const lines = state.findings.map((f, i) => `${i + 1}. [${ts(f.at)}] ${f.text}`);
      ctx.ui.notify(`Findings to raise:\n${lines.join("\n")}`, "info");
    },
  });

  pi.registerCommand("experiments", {
    description: "Show the experiment run log",
    handler: async (_args, ctx) => {
      if (state.experiments.length === 0) {
        ctx.ui.notify("No experiments run yet.", "info");
        return;
      }
      const lines = state.experiments.map(
        (e) => `${e.name}: ${e.status} (${ts(e.startedAt)}${e.endedAt ? "\u2192" + ts(e.endedAt) : ""})`,
      );
      ctx.ui.notify(`Experiments:\n${lines.join("\n")}`, "info");
    },
  });

  pi.registerCommand("clear-findings", {
    description: "Clear recorded findings after raising them",
    handler: async (_args, ctx) => {
      const ok = await ctx.ui.confirm("Clear findings?", `Delete ${state.findings.length} finding(s)?`);
      if (!ok) return;
      state.findings = [];
      saveState();
      render(ctx);
      ctx.ui.notify("Findings cleared.", "info");
    },
  });

  pi.registerCommand("budget", {
    description: "Set/show the experiment budget for this iteration (/budget <n>)",
    handler: async (args, ctx) => {
      const text = (args ?? "").trim();
      if (!text) {
        if (!state.budget) {
          ctx.ui.notify("No budget set. Usage: /budget <positive integer>", "info");
          return;
        }
        ctx.ui.notify(`Budget: ${usedCount()}/${state.budget.total} runs used this iteration.`, "info");
        return;
      }
      const n = parseInt(text, 10);
      if (!Number.isFinite(n) || n < 1) {
        ctx.ui.notify("Usage: /budget <positive integer>", "warn");
        return;
      }
      state.budget = { total: n, startedAt: Date.now() };
      saveState();
      render(ctx);
      ctx.ui.notify(`Budget set: 0/${n} runs this iteration.`, "info");
    },
  });

  pi.registerCommand("clear-budget", {
    description: "Clear the experiment budget (unlimited launches)",
    handler: async (_args, ctx) => {
      state.budget = null;
      saveState();
      render(ctx);
      ctx.ui.notify("Budget cleared.", "info");
    },
  });
}
