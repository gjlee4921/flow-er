// The shell: the plan on the left, the conversation in the middle, the workflow panel on the right.
import { useEffect } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { api } from "./api";
import { useExperiment } from "./api/hooks";
import { Icon } from "./components/icons";
import { Resizer, widthOf } from "./components/Resizer";
import Chat from "./features/agent/Chat";
import MemoryPanel from "./features/shell/MemoryPanel";
import MiniRail from "./features/shell/MiniRail";
import Panel from "./features/shell/Panel";
import Rail from "./features/shell/Rail";
import SettingsPanel from "./features/shell/SettingsPanel";
import Start from "./features/study-sheet/Start";
import { ui, useUi } from "./lib";

const watched = new Set<string>();

export default function App() {
  const qc = useQueryClient();
  const exp = useUi((u) => u.exp);
  const page = useUi((u) => u.page);
  const panelOpen = useUi((u) => u.panelOpen);
  const full = useUi((u) => u.full);
  const railOpen = useUi((u) => u.railOpen);
  const toast = useUi((u) => u.toast);
  const { data: state, error } = useExperiment(exp);

  // note when a project is opened, so the project list leads with it
  useEffect(() => { if (exp) api.projectOpened(exp).then(() => qc.invalidateQueries({ queryKey: ["experiments"] }), () => {}); }, [exp]); // eslint-disable-line react-hooks/exhaustive-deps
  // an experiment that no longer exists (deleted, or a different machine) sends us back to the start
  useEffect(() => { if (error) ui.openProject(null); }, [error]);
  // when a run starts gating with the lab's models, the data lab opens to show them at work (once a run)
  const run = useUi((u) => u.run);
  useEffect(() => {
    if (!run || run.exp !== exp || watched.has(run.id)) return;
    if (Object.values(run.nodes).some((n) => n.op === "autogate" && n.status === "running")) {
      watched.add(run.id);
      ui.lab({ live: true, view: "plot", compare: false });
      ui.open("lab");
    }
  }, [run, exp]);
  // opening an experiment that already has a plan shows the plan, not the empty panel
  useEffect(() => { if (state && ui.get().page === "none" && (state.plan.length || state.draft)) ui.open("flow"); }, [state?.id]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape" || (e.target as Element)?.closest?.("textarea,input,select")) return;
      const s = ui.get();
      if (s.page === "lab") ui.leaveLab();
      else if (s.full) ui.set({ full: false });
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // the full navigator only inside a project, when unfolded; otherwise the strip of project circles
  const mini = !exp || !railOpen;
  const cls = ["ws", panelOpen && exp ? "" : "noviews", full ? "fullviews" : "", page === "lab" ? "labmode" : "", mini ? "minirail" : ""].filter(Boolean).join(" ");
  return (
    <div className={cls} id="ws" aria-label="Analysis workspace">
      <div className="ws-body">
        {mini ? <MiniRail exp={exp} /> : <Rail exp={exp} />}
        {exp ? <Chat key={exp} exp={exp} /> : <Start />}
        <Panel exp={exp} />
        {!mini && <Resizer className="rsz-rail" cssVar="--railw" side="left" label="Sidebar width" measure={widthOf(".ws-rail")} min={200} max={() => 400} />}
        {exp && <Resizer className="rsz-views" cssVar="--panelw" side="right" label="Panel width" measure={widthOf(".ws-views")} min={380}
          max={() => window.innerWidth - (document.querySelector(".ws-rail")?.getBoundingClientRect().width ?? 0) - 400} />}
        {!panelOpen && exp && (
          <button className="btn sm panel-open" onClick={() => ui.set({ panelOpen: true })}><Icon.sidebar />Workflow</button>
        )}
      </div>
      <MemoryPanel />
      <SettingsPanel />
      {toast && <div className="toast" role="status">{toast}</div>}
    </div>
  );
}
