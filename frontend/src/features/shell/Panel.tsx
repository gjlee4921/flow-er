// The workflow panel: a floating card on the right with whatever the plan, the agent or the
// scientist opened. The data lab takes the whole screen and brings its own chrome.
import { useExperiment } from "../../api/hooks";
import { Icon, Sprout } from "../../components/icons";
import { ui, useUi, type Page } from "../../lib";
import DataLab from "../explore/DataLab";
import LiveLab from "../explore/LiveLab";
import { Clean, Models, RawData } from "../pipeline/Pages";
import Pipeline from "../pipeline/Pipeline";
import { Measure, Results } from "../results/Results";

const TITLE: Record<Page, string> = {
  none: "", flow: "Pipeline", lab: "Data lab", results: "Results", measure: "Measure", models: "Learned models", clean: "Clean up", data: "Raw data",
};

export default function Panel({ exp }: { exp: string | null }) {
  const { data: s } = useExperiment(exp);
  const page = useUi((u) => u.page);
  const full = useUi((u) => u.full);
  const watching = useUi((u) => u.lab.live);

  let body: JSX.Element;
  if (!s || page === "none") {
    const read = !!s;
    body = (
      <div className="ws-page" id="pg-none">
        <div className="empty"><div style={{ display: "grid", gap: 10, justifyItems: "center" }}>
          <span className="seedling"><Sprout up={read} /></span>
          <h2>{read ? "Your files are in" : "Nothing to show yet"}</h2>
          <p>{read ? "Tell me what the experiment is for, and the plan grows from here: the steps, the models, then the results."
            : "Pick a folder of data and tell me what the experiment is for. The plan, the models and the results all grow here."}</p>
        </div></div>
      </div>
    );
  } else if (page === "lab") {
    // while a run's models gate the files, the lab shows them at work instead
    body = watching ? <LiveLab s={s} /> : <DataLab s={s} />;
  } else {
    const P = { flow: Pipeline, results: Results, measure: Measure, models: Models, clean: Clean, data: RawData }[page];
    body = <div className={`ws-page${page === "flow" ? " flow-page" : ""}`} id={`pg-${page}`}><P s={s} /></div>;
  }

  return (
    <aside className="ws-views" id="ws-views" aria-label="Workflow">
      <div className="views-h">
        <div className="t"><b id="view-title">{TITLE[page]}</b></div>
        <div className="r">
          <button className="ib sm" id="views-wide" aria-label={full ? "Shrink the panel" : "Full screen"} title={full ? "Shrink" : "Full screen"}
            onClick={() => ui.set({ full: !full })}>{full ? <Icon.shrink /> : <Icon.expand />}</button>
          <button className="ib sm" id="views-close" aria-label="Hide the panel" title="Hide" onClick={() => ui.set({ panelOpen: false, full: false })}><Icon.close /></button>
        </div>
      </div>
      <div className="views-b" id="ws-main">{body}</div>
    </aside>
  );
}
