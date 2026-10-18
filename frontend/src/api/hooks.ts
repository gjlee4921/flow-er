// Server state through react-query. Everything about an experiment is keyed under ["exp", id],
// so one invalidation refreshes the screen after a keep, an undo or an agent turn.
import { useQuery, useQueryClient, type QueryClient } from "@tanstack/react-query";
import { api, type UiEvent } from ".";
import { ui, type Page } from "../lib";

export const useHealth = () => useQuery({ queryKey: ["health"], queryFn: api.health, refetchInterval: 30000 });
export const useSettings = (on = true) => useQuery({ queryKey: ["settings"], queryFn: api.settings, enabled: on });
export const useDatasets = () => useQuery({ queryKey: ["datasets"], queryFn: api.datasets });
export const useExperiments = () => useQuery({ queryKey: ["experiments"], queryFn: api.experiments });
export const useModels = () => useQuery({ queryKey: ["models"], queryFn: api.models });

export const useExperiment = (id: string | null) =>
  useQuery({ queryKey: ["exp", id, "state"], queryFn: () => api.experiment(id!), enabled: !!id, retry: false });
export const useLabMessages = (id: string | null) =>
  useQuery({ queryKey: ["exp", id, "lab-messages"], queryFn: () => api.labMessages(id!), enabled: !!id });
export const useNodeTypes = () => useQuery({ queryKey: ["node-types"], queryFn: api.nodeTypes, staleTime: Infinity });
export const useExecutions = (id: string | null) =>
  useQuery({ queryKey: ["exp", id, "executions"], queryFn: () => api.executions(id!), enabled: !!id });
export const useStepOutput = (id: string | null, step: string | null) =>
  useQuery({ queryKey: ["exp", id, "output", step], queryFn: () => api.stepOutput(id!, step!), enabled: !!id && !!step });

/** Follow a workflow run as it happens: each step's status lands in the UI state (the canvas and the
 *  chat's run card read it), and when it ends the experiment is refreshed. */
export function followRun(exp: string, id: string, qc?: QueryClient) {
  ui.set({ run: { exp, id, status: "running", nodes: {} } });
  const es = new EventSource(`/api/experiments/${exp}/executions/${id}/events`);
  const put = (patch: Partial<NonNullable<ReturnType<typeof ui.get>["run"]>>) =>
    ui.set((s) => ({ run: s.run && s.run.id === id ? { ...s.run, ...patch } : s.run }));
  es.onmessage = (m) => {
    const e = JSON.parse(m.data);
    if (e.type === "start") put({ nodes: e.execution.nodes });
    else if (e.type === "node") ui.set((s) => ({ run: s.run && s.run.id === id ? { ...s.run, nodes: { ...s.run.nodes, [e.id]: e } } : s.run }));
    else if (e.type === "done") {
      put({ status: e.execution.status, nodes: e.execution.nodes });
      es.close();
      if (qc) refresh(qc, exp);
    }
  };
  es.onerror = () => es.close();
}

export const useStepImages = (id: string | null, step: string | null, file?: string, on = true) =>
  useQuery({ queryKey: ["exp", id, "images", step, file], queryFn: () => api.stepImages(id!, step!, file), enabled: !!id && !!step && on });
/** Which files have model gates; `every` (ms) keeps asking while a run gates them one by one. */
export const useAutogateFiles = (id: string | null, on = true, every: number | false = false) =>
  useQuery({ queryKey: ["exp", id, "autogate"], queryFn: () => api.autogateFiles(id!), enabled: !!id && on, refetchInterval: every });
export const useAutogateResult = (id: string | null, sample?: string) =>
  useQuery({ queryKey: ["exp", id, "autogate", sample], queryFn: () => api.autogateResult(id!, sample!), enabled: !!id && !!sample });
export const useRounds = () => useQuery({ queryKey: ["rounds"], queryFn: api.rounds, staleTime: 60000 });
export const useLabRounds = (lab?: string | null) => useQuery({ queryKey: ["rounds", lab], queryFn: () => api.labRounds(lab!), enabled: !!lab, staleTime: Infinity });
export const useLabModels = (lab?: string | null) =>
  useQuery({ queryKey: ["lab", lab, "models"], queryFn: () => api.labModels(lab!), enabled: !!lab });
export const useMemory = () => useQuery({ queryKey: ["memory"], queryFn: api.memory });
export const useToolkit = () => useQuery({ queryKey: ["toolkit"], queryFn: api.toolkit, staleTime: Infinity });
export const useStats = (id: string | null, gates: string[], cols: string[], on = true) =>
  useQuery({ queryKey: ["exp", id, "stats", gates.join(","), cols.join(",")], queryFn: () => api.stats(id!, gates, cols),
    enabled: !!id && on, placeholderData: (prev) => prev });
export const useMessages = (id: string | null) =>
  useQuery({ queryKey: ["exp", id, "messages"], queryFn: () => api.messages(id!), enabled: !!id });
export const useSamples = (id: string | null) =>
  useQuery({ queryKey: ["exp", id, "samples"], queryFn: () => api.samples(id!), enabled: !!id, staleTime: Infinity });
export const useGates = (id: string | null, sample?: string) =>
  useQuery({ queryKey: ["exp", id, "gates", sample], queryFn: () => api.gates(id!, sample), enabled: !!id && !!sample });
export const useViews = (id: string | null) =>
  useQuery({ queryKey: ["exp", id, "views"], queryFn: () => api.views(id!), enabled: !!id });
export const useTable = (id: string | null, name: string | null) =>
  useQuery({ queryKey: ["exp", id, "table", name], queryFn: () => api.table(id!, name!), enabled: !!id && !!name, retry: false });
export const usePlot = (id: string | null, p: { sample?: string; x?: string; y?: string; parent: string | null; stage: string }) =>
  useQuery({
    queryKey: ["exp", id, "plot", p.sample, p.x, p.y, p.parent, p.stage === "raw" ? "raw" : "processed"],
    queryFn: () => api.plot(id!, { sample: p.sample, x: p.x!, y: p.y!, parent: p.parent, stage: p.stage === "raw" ? "raw" : "processed" }),
    enabled: !!id && !!p.sample && !!p.x && !!p.y,
    placeholderData: (prev) => prev,
  });

export const useRefresh = () => {
  const qc = useQueryClient();
  return (id: string | null) => refresh(qc, id);
};
export const refresh = (qc: QueryClient, id: string | null) => qc.invalidateQueries({ queryKey: ["exp", id] });

/** What the agent asked the screen to do after a turn. */
export function apply(events: UiEvent[] = []) {
  for (const e of events) {
    if (e.type === "open") ui.open(e.page as Page);
    else if (e.type === "execution") { const exp = ui.get().exp; if (exp) followRun(exp, e.id); }
    else if (e.type === "lab") {
      const patch: Record<string, unknown> = {};
      if (e.sample) patch.sample = e.sample;
      if ("parent" in e) patch.parent = e.parent ?? null;
      if (e.x) patch.x = e.x;
      if (e.y) patch.y = e.y;
      if (e.stage) patch.stage = e.stage;
      if (e.mode) patch.mode = ["dots", "hist", "contour"].includes(e.mode) ? e.mode : "density";
      if (e.view) patch.view = e.view;
      if ("fig" in e) patch.fig = e.fig ?? null;
      if ("sel" in e) patch.sel = e.sel ?? null;
      if (e.compare != null) patch.compare = e.compare;
      if (e.sheetGates) patch.sheetGates = e.sheetGates;
      if (e.sheetCols) patch.sheetSpec = e.sheetCols.flatMap((c) => [`median:${c}`, `cv:${c}`]);
      if (e.sheetSpec) patch.sheetSpec = e.sheetSpec;
      if (e.sheetFilter != null) patch.sheetFilter = e.sheetFilter;
      if ("sheetSort" in e) patch.sheetSort = e.sheetSort ?? null;
      if ("sheetColour" in e) patch.sheetColour = e.sheetColour ?? null;
      if ("sheetChart" in e) patch.sheetChart = e.sheetChart ?? null;
      if (e.tool) patch.tool = e.tool;
      ui.lab(patch);
    }
  }
}
