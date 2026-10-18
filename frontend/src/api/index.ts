// Typed API client. Hand-written for now; `pnpm gen:api` can replace the types from the backend's
// OpenAPI schema once the shapes settle.

export type Dataset = {
  name: string; label: string; kind: "example" | "upload" | "folder"; path: string; files: number;
  workspace: string | null; markers: string[]; size_mb: number;
};
export type Rejected = { name: string; why: string };
/** The data lab's toolkit, defined once on the server: what the toolbar, the sheet and the assistant offer. */
export type Toolkit = {
  plot_types: { id: string; name: string; axes: number; render: string; about: string }[];
  statistics: { id: string; name: string; needs: string[] }[];
  gate_tools: { id: string; name: string; key: string; about: string }[];
  later: string[];
};
export type Experiment = {
  id: string; title: string; data_path: string; created: string; question: string;
  last_opened?: string; archived?: boolean; plan_kept?: boolean; figures?: number;
  /** the workflow runs on its own when new FCS files arrive in the data folder */
  active?: boolean;
};

export type Step = {
  id: string; op: string; kind: "macro" | "model" | "measure" | "view"; title: string;
  inputs: string[]; outputs: string[]; why: string; gate?: string; macro_id?: string;
  /** the step's settings, from its node type */
  params?: Record<string, string>;
  /** a script step's script: its code, where it came from, and the test it passed in the sandbox */
  script?: Script;
};
export type Script = {
  code: string;
  provenance: { what?: string; source?: string | null; version?: string | null; licence?: string | null; adapted?: string };
  test?: { passed: boolean; problems?: string[]; hash?: string; seconds?: number };
};
export type Run = { step: string; op: string; status: string; summary: string; at: string; table?: string };
export type Draft = { id: string; kind: "plan" | "gate" | "view" | "macro" | "report"; payload: any; summary: string; created: string };
export type TableInfo = { name: string; rows: number; columns: string[] };
export type Macro = {
  id: string; title: string; summary: string; code: string; inputs: string[]; outputs: string[];
  provenance: { name: string; url?: string; licence: string; what?: string }[];
};
export type ViewSpec = {
  id: string; title: string; caption: string; type: "bar" | "dot" | "dumbbell" | "line" | "scatter" | "heatmap" | "table" | "anatomy";
  table: string; x?: string; y?: string; color?: string; filter: Record<string, unknown>; agg: string; sort: string;
  columns: string[]; x_label?: string; y_label?: string; log: boolean;
};
export type View = { spec: ViewSpec; rows: Record<string, any>[]; draft: boolean };

export type Snapshot = {
  title: string; question: string | null; data_folder: string;
  samples: { id: string; events: number }[]; channels: string[]; flowjo_workspace: string | null;
  plan: { id: string; op: string; title: string; run: string }[];
  user_gates: string[]; views: { id: string; title: string }[]; macros: { id: string; title: string }[];
  tables: TableInfo[]; uploads: string[]; draft: { kind: string; summary: string } | null; gating_models: string;
};

export type ExperimentState = Experiment & {
  snapshot: Snapshot; plan: Step[]; runs: Run[]; gates: any[]; views: ViewSpec[]; macros: Macro[];
  draft: Draft | null; tables: TableInfo[];
};

export type Channel = { id: string; label: string; short: string; scatter: boolean; group: "scatter" | "marker" | "time"; name: string };
export type StatRow = { sample: string; gate: string; gate_path: string; count: number; pct_parent: number; pct_total: number; [col: string]: string | number | null };
export type Sample = { id: string; file: string; events: number };
export type GateNode = {
  key: string; name: string; parent: string | null; source: "flowjo" | "user" | "model";
  dims: string[]; vertices: number[][]; kind: string; count: number; pct_parent: number; pct_total: number; draft: boolean;
};
export type PlotData = {
  sample: string; x: number[]; y: number[]; label: number[];
  children: { key: string; name: string; source: string; dims: string[]; vertices: number[][] }[];
  parent: string | null; n_parent: number; n_shown: number; stage: string;
  /** where round channel values sit on each axis, in display units: 0, 10², 10³... or 0, 50K, 100K... */
  x_ticks?: { v: number; label: string }[]; y_ticks?: { v: number; label: string }[];
};

export type Memory = { id: string; text: string; kind: "definition" | "preference" | "lab" | "project"; quote: string;
  source: { experiment: string | null; title?: string | null }; created: string; last_used: string | null };
/** One run of the workflow: each step's status as it went, and what started it. */
export type ExecNode = { status: "queued" | "running" | "done" | "cached" | "failed" | "blocked"; title: string; op: string; summary?: string; table?: string | null; seconds?: number };
export type Execution = { id: string; trigger: string; mode: string; node: string | null; started: string; finished: string | null; status: string; nodes: Record<string, ExecNode> };
export type NodeType = { name: string; kind: string; takes: string[]; makes: string; version: string; about: string;
  params?: Record<string, { label: string; enum?: string[]; default?: string; names?: Record<string, string>; choices?: "channels" | "gates" }> };
export type StepImage = { port: string; role: "input" | "output"; kind: string; grid: number[][]; meta: { x?: string; y?: string; parent?: string; recipe?: string; events?: number } };
/** One file gated by the lab's models: per gate the polygon, numbers, which model answered and how sure it was. */
export type AutogateLive = { running: boolean; files: string[]; done: string[]; fresh: string[]; current: string | null; lab: string | null; version: string };
export type AutogateGate = {
  gate: string; parent: string; x: string; y: string; model: string; count: number; parent_count: number; percent_of_parent: number;
  confidence: number | null; low_confidence: boolean; flowjo_pct?: number; f1?: number; error_pp?: number;
  polygon: { display: number[][] | null; raw: number[][] | null }; plot?: { density: number[][] };
};
export type AutogateResult = { sample: string; lab: string | null; model_version: string | null; individual?: string; unstim?: string | null; seconds?: number; gates: AutogateGate[] };
export type Taught = { samples: number; individuals: string[] };
export type LabJob = { id: string; lab: string; status: string; lines: string[]; error?: string;
  summary?: { version: string | null; promoted: boolean; reason: string; graduated: string[] } | null };
export type LabVersion = { version: string; current: boolean; created?: string; notes?: string; metrics?: Record<string, unknown>; parent?: string | null;
  /** gate id -> model folder; the gates this version answers with a trained model */
  gates?: Record<string, string>; confidence_threshold?: number };
export type UiEvent =
  | { type: "refresh" }
  | { type: "execution"; id: string }
  | { type: "memory"; action: "added" | "already" | "forgotten"; id: string; text: string }
  | { type: "progress"; text: string }
  | { type: "open"; page: string }
  | { type: "lab"; fig?: string | null; sample?: string; parent?: string | null; x?: string; y?: string; stage?: string; mode?: string;
      view?: "plot" | "table" | "saved"; sel?: string | null; compare?: boolean; sheetGates?: string[]; sheetCols?: string[];
      sheetSpec?: string[]; sheetFilter?: string; sheetSort?: { key: string; desc: boolean } | null; sheetColour?: string | null;
      sheetChart?: string | null; tool?: string };

export type TraceItem = { tool: string; args: Record<string, unknown>; ok: boolean; summary: string };
/** A page a web search found and the answer cites, as [n] in its text. */
export type Source = { n: number; title: string; url: string };
export type Message = {
  id: string; t: string; role: "user" | "assistant"; text: string; attachments?: string[];
  trace?: TraceItem[]; ui?: UiEvent[]; draft?: { id: string; kind: string; summary: string } | null;
  sources?: Source[]; error?: string | null; seconds?: number;
};

/** What a streamed turn says as it runs: the model thinking, a tool starting and ending, screen changes, the answer. */
export type TurnEvent =
  | { type: "thinking"; round: number }
  | { type: "step"; i: number; tool: string; label: string }
  | { type: "step_done"; i: number; ok: boolean; summary: string }
  | { type: "ui"; events: UiEvent[] }
  | { type: "text"; delta: string }
  | { type: "plan_partial"; steps: Step[] }
  | { type: "message"; message: Message }
  | { type: "error"; error: string };

export type Health = { status: string; llm_provider: string; model: string; llm_key: string };
export type ModelStatus = { available: boolean; message: string; labs?: string[]; registry?: boolean; train?: boolean; infer?: boolean };

class ApiError extends Error {
  constructor(public status: number, message: string) { super(message); }
}

async function req<T>(method: string, path: string, body?: unknown): Promise<T> {
  const init: RequestInit = { method, headers: {} };
  if (body instanceof FormData) init.body = body;
  else if (body !== undefined) {
    init.body = JSON.stringify(body);
    (init.headers as Record<string, string>)["content-type"] = "application/json";
  }
  const r = await fetch(`/api${path}`, init);
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail ?? detail; } catch { /* not json */ }
    throw new ApiError(r.status, typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return r.json() as Promise<T>;
}

/** POST a turn and read its server-sent events as they arrive; resolves with the answer. */
async function stream(path: string, body: unknown, onEvent: (e: TurnEvent) => void): Promise<Message> {
  const r = await fetch(`/api${path}`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok || !r.body) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail ?? detail; } catch { /* not json */ }
    throw new ApiError(r.status, typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  const reader = r.body.getReader(), dec = new TextDecoder();
  let buf = "", answer: Message | null = null;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let cut;
    while ((cut = buf.indexOf("\n\n")) >= 0) {
      const line = buf.slice(0, cut).trim();
      buf = buf.slice(cut + 2);
      if (!line.startsWith("data: ")) continue;
      const e = JSON.parse(line.slice(6)) as TurnEvent;
      if (e.type === "message") answer = e.message;
      else if (e.type === "error") throw new ApiError(500, e.error);
      else onEvent(e);
    }
  }
  if (!answer) throw new ApiError(500, "the turn ended without an answer");
  return answer;
}

const q = (o: Record<string, string | null | undefined>) =>
  "?" + Object.entries(o).filter(([, v]) => v != null && v !== "").map(([k, v]) => `${k}=${encodeURIComponent(v!)}`).join("&");

export const api = {
  health: () => req<Health>("GET", "/health"),
  datasets: () => req<Dataset[]>("GET", "/datasets"),
  experiments: () => req<Experiment[]>("GET", "/experiments"),
  renameProject: (id: string, title: string) => req<Experiment>("PATCH", `/experiments/${id}`, { title }),
  archiveProject: (id: string, archived: boolean) => req<Experiment>("PATCH", `/experiments/${id}`, { archived }),
  projectOpened: (id: string) => req<Experiment>("POST", `/experiments/${id}/opened`),
  create: (dataset: string, title?: string) => req<Experiment & { snapshot: Snapshot }>("POST", "/experiments", { dataset, title }),
  experiment: (id: string) => req<ExperimentState>("GET", `/experiments/${id}`),
  samples: (id: string) => req<{ samples: Sample[]; channels: Channel[]; workspace: string | null }>("GET", `/experiments/${id}/samples`),
  gates: (id: string, sample?: string) => req<{ sample: string; gates: GateNode[] }>("GET", `/experiments/${id}/gates${q({ sample })}`),
  plot: (id: string, p: { sample?: string; x: string; y: string; parent?: string | null; stage?: string }) =>
    req<PlotData>("GET", `/experiments/${id}/plot${q({ sample: p.sample, x: p.x, y: p.y, parent: p.parent ?? undefined, stage: p.stage })}`),
  drawGate: (id: string, g: { name: string; parent: string | null; dims: string[]; vertices: number[][]; kind: string }) =>
    req<{ draft: Draft; gate: GateNode }>("POST", `/experiments/${id}/gates`, g),
  runStep: (id: string, step: string) => req<Run>("POST", `/experiments/${id}/steps/${encodeURIComponent(step)}/run`),
  keep: (id: string, gate?: string) => req<{ kept: Draft }>("POST", `/experiments/${id}/keep${gate ? `?gate=${encodeURIComponent(gate)}` : ""}`),
  undo: (id: string, gate?: string) => req<{ dropped: Draft }>("POST", `/experiments/${id}/undo${gate ? `?gate=${encodeURIComponent(gate)}` : ""}`),
  messages: (id: string) => req<Message[]>("GET", `/experiments/${id}/messages`),
  chat: (id: string, text: string, attachments: string[] = []) => req<Message>("POST", `/experiments/${id}/chat`, { text, attachments }),
  chatStream: (id: string, text: string, attachments: string[], onEvent: (e: TurnEvent) => void) =>
    stream(`/experiments/${id}/chat/stream`, { text, attachments }, onEvent),
  upload: (id: string, files: File[]) => {
    const fd = new FormData();
    files.forEach((f) => fd.append("files", f, f.name));
    return req<{ saved: { name: string; bytes: number }[] }>("POST", `/experiments/${id}/uploads`, fd);
  },
  views: (id: string) => req<View[]>("GET", `/experiments/${id}/views`),
  table: (id: string, name: string) => req<{ name: string; rows: Record<string, any>[] }>("GET", `/experiments/${id}/tables/${name}`),
  models: () => req<ModelStatus>("GET", "/models/status"),
  linkFolder: (path: string) => req<Dataset>("POST", "/datasets/link", { path }),
  memory: () => req<{ memories: Memory[]; paused: boolean }>("GET", "/memory"),
  nodeTypes: () => req<{ types: Record<string, NodeType>; sources: Record<string, string> }>("GET", "/node-types"),
  runWorkflow: (id: string, mode = "all", node?: string) => req<Execution>("POST", `/experiments/${id}/executions`, { mode, node }),
  executions: (id: string) => req<Execution[]>("GET", `/experiments/${id}/executions`),
  execution: (id: string, eid: string) => req<Execution>("GET", `/experiments/${id}/executions/${eid}`),
  stepImages: (id: string, step: string, file?: string) =>
    req<{ files: string[]; file?: string; images: StepImage[] }>("GET", `/experiments/${id}/steps/${encodeURIComponent(step)}/images${q({ file })}`),
  autogateFiles: (id: string) => req<{ lab: string | null; version: string; files: { sample: string; gated: boolean }[] }>("GET", `/experiments/${id}/autogate`),
  autogateResult: (id: string, sample: string) => req<AutogateResult>("GET", `/experiments/${id}/autogate/${encodeURIComponent(sample)}/result`),
  /** the models at work: the run's files, the one being gated now, those done (and done just now) */
  autogateLive: (id: string) => req<AutogateLive>("GET", `/experiments/${id}/autogate/live`),
  autogateReview: (id: string, sample: string) =>
    req<{ gates: string[]; sample: string; parent: string | null; x: string; y: string; gate: string }>("POST", `/experiments/${id}/autogate/${encodeURIComponent(sample)}/review`),
  autogateAccept: (id: string, sample: string) =>
    req<{ lab: string; gates: number; from_scientist: number; individual: string; taught: Taught }>("POST", `/experiments/${id}/autogate/${encodeURIComponent(sample)}/accept`, {}),
  labModels: (lab: string) => req<{ lab: string; versions: LabVersion[]; taught?: Taught }>("GET", `/labs/${encodeURIComponent(lab)}/models`),
  retrainLab: (lab: string, gates?: string[]) => req<LabJob>("POST", `/labs/${encodeURIComponent(lab)}/retrain`, { gates }),
  job: (jid: string) => req<LabJob>("GET", `/jobs/${encodeURIComponent(jid)}`),
  regionGates: (id: string, step: string, file?: string) =>
    req<{ gates: string[]; sample: string; parent: string; x: string; y: string }>("POST", `/experiments/${id}/steps/${encodeURIComponent(step)}/gates`, { file }),
  stepOutput: (id: string, step: string) => req<{ ran: boolean; status?: string; summary?: string; table?: string; at?: string; execution?: string; rows?: Record<string, any>[]; total_rows?: number }>("GET", `/experiments/${id}/steps/${encodeURIComponent(step)}/output`),
  setParams: (id: string, step: string, params: Record<string, string>) => req<Draft>("POST", `/experiments/${id}/steps/${encodeURIComponent(step)}/params`, { params }),
  setActive: (id: string, on: boolean) => req<{ active: boolean }>("POST", `/experiments/${id}/active`, { on }),
  addMemory: (text: string, kind = "preference") => req<Memory>("POST", "/memory", { text, kind }),
  editMemory: (id: string, text: string) => req<Memory>("PATCH", `/memory/${id}`, { text }),
  forgetMemory: (id: string) => req<{ forgotten: Memory }>("DELETE", `/memory/${id}`),
  pauseMemory: (paused: boolean) => req<{ paused: boolean }>("POST", "/memory/pause", { paused }),
  /** Files from the start page as a new dataset, with upload progress (fetch cannot report it). */
  uploadDataset: (files: File[], onProgress: (share: number) => void) =>
    new Promise<Dataset & { rejected: Rejected[] }>((resolve, reject) => {
      const fd = new FormData();
      files.forEach((f) => fd.append("files", f, f.name));
      const x = new XMLHttpRequest();
      x.open("POST", "/api/datasets");
      x.upload.onprogress = (e) => { if (e.lengthComputable) onProgress(e.loaded / e.total); };
      x.onload = () => {
        let body: any = null;
        try { body = JSON.parse(x.responseText); } catch { /* not json */ }
        if (x.status < 300) resolve(body);
        else {
          const d = body?.detail;
          const err = new ApiError(x.status, typeof d === "string" ? d : d?.message ?? x.statusText) as ApiError & { rejected?: Rejected[] };
          err.rejected = d?.rejected ?? [];
          reject(err);
        }
      };
      x.onerror = () => reject(new ApiError(0, "the upload did not reach the server"));
      x.send(fd);
    }),
  stats: (id: string, gates: string[], cols: string[]) =>
    req<{ rows: StatRow[]; columns: { spec: string; label: string }[] }>("GET", `/experiments/${id}/stats${q({ gates: gates.join(","), cols: cols.join(",") })}`),
  toolkit: () => req<Toolkit>("GET", "/toolkit"),
  drawQuadrant: (id: string, g: { parent: string | null; dims: string[]; centre: number[] }) =>
    req<{ draft: Draft; gates: GateNode[] }>("POST", `/experiments/${id}/gates/quadrant`, g),
  labMessages: (id: string) => req<Message[]>("GET", `/experiments/${id}/lab/messages`),
  labChat: (id: string, text: string, screen: Record<string, unknown>) => req<Message>("POST", `/experiments/${id}/lab/chat`, { text, screen }),
  labChatStream: (id: string, text: string, screen: Record<string, unknown>, onEvent: (e: TurnEvent) => void) =>
    stream(`/experiments/${id}/lab/chat/stream`, { text, screen }, onEvent),
};

export { ApiError };
