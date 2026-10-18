// Before there is a project: a dashboard of what is here (projects, the lab's gating models, datasets)
// and a way to bring more data. Starting an analysis lives in the flower at the bottom (Dock): it
// unfolds into the composer, which opens the project and, with a question, plans it at once.
import { useMemo, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { api, type Dataset, type Experiment, type Rejected } from "../../api";
import { useDatasets, useExperiments, useHealth, useLabModels, useModels } from "../../api/hooks";
import { Bloom, Flower, Icon } from "../../components/icons";
import { fmt, lastProject, ui } from "../../lib";
import Dock, { type DockState } from "./Dock";

const KIND = { example: "Example", upload: "Uploaded", folder: "On this computer" } as const;

/** "today", "yesterday", "3 days ago", else the date. */
function ago(iso?: string) {
  if (!iso) return "";
  const days = Math.floor((Date.now() - new Date(iso).getTime()) / 86_400_000);
  return days <= 0 ? "today" : days === 1 ? "yesterday" : days < 14 ? `${days} days ago` : iso.slice(0, 10);
}

const folderOf = (path: string) => path.split(/[\\/]/).filter(Boolean).pop()?.replace(/[_-]+/g, " ") ?? "";

function ProjectCard({ e }: { e: Experiment }) {
  const status = e.figures ? `${e.figures} figure${e.figures === 1 ? "" : "s"}` : e.plan_kept ? "plan kept" : "not planned yet";
  return (
    <button className="db-proj" onClick={() => ui.openProject(e.id)}>
      <span className={`pm-dot ${e.figures ? "fig" : e.plan_kept ? "plan" : ""}`} aria-hidden="true" />
      <b>{e.title}</b>
      {folderOf(e.data_path) !== e.title && <span className="db-sub">{folderOf(e.data_path)}</span>}
      <span className="db-meta"><span>{status}</span><span>{ago(e.last_opened ?? e.created)}</span></span>
    </button>
  );
}

const STATE = { live: "Live", outdated: "Needs retraining", offline: "Offline" } as const;

/** One lab's gating models: a flower that breathes while they are live, goes black and white when
 *  kept corrections are waiting to be learned, and stills when they cannot run; then the details. */
function LabCard({ lab }: { lab: string }) {
  const { data } = useLabModels(lab);
  const versions = data?.versions ?? [];
  const cur = versions.find((v) => v.current);
  const gates = cur?.gates ? Object.keys(cur.gates).length : 0;
  const taught = data?.taught;
  const st = data?.status?.state ?? "offline";
  return (
    <div className={`db-lab ${data ? st : "loading"}`}>
      <div className={`lab-orb ${data ? st : "loading"}`} title={data?.status?.why}><Bloom p={1} /></div>
      <div className="db-lab-h"><b>Lab {lab}</b>{cur && <span className="pill">{cur.version}</span>}
        {data?.status && <span className={`lab-state ${st}`}>{STATE[st]}</span>}</div>
      {data?.status && <p className="db-why">{data.status.why}</p>}
      {!data ? <span className="db-sub">Reading…</span> : (
        <dl className="db-dl">
          <div><dt>Gate models</dt><dd>{gates ? `${gates} trained` : "none (template copy)"}</dd></div>
          <div><dt>Review flag</dt><dd>{cur?.confidence_threshold != null ? `calibrated at ${cur.confidence_threshold}` : "default 0.6 (not calibrated)"}</dd></div>
          <div><dt>Taught</dt><dd>{taught?.samples ? `${taught.samples} file${taught.samples === 1 ? "" : "s"} from ${taught.individuals.length} individual${taught.individuals.length === 1 ? "" : "s"}` : "nothing kept yet"}</dd></div>
          <div><dt>Versions</dt><dd>{versions.length}{cur?.created ? ` · current since ${ago(cur.created)}` : ""}</dd></div>
        </dl>
      )}
    </div>
  );
}

export default function Start() {
  const qc = useQueryClient();
  const { data: datasets, isLoading } = useDatasets();
  const { data: recent = [] } = useExperiments();
  const { data: health } = useHealth();
  const { data: models } = useModels();
  const [chosen, setChosen] = useState<Dataset | null>(null);
  const [text, setText] = useState("");
  const [dock, setDock] = useState<DockState>("rest");
  const [busy, setBusy] = useState(false);
  const [upload, setUpload] = useState<{ share: number; names: string } | null>(null);
  const [rejected, setRejected] = useState<Rejected[]>([]);
  const [path, setPath] = useState("");
  const [folderErr, setFolderErr] = useState("");
  const [over, setOver] = useState(false);
  const [find, setFind] = useState("");
  const [allProjects, setAllProjects] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const choose = (d: Dataset | null) => { setChosen(d); if (d) setDock("open"); };
  const landed = async (d: Dataset) => { await qc.invalidateQueries({ queryKey: ["datasets"] }); choose(d); };

  const send = async (files: File[]) => {
    if (!files.length) return;
    setRejected([]);
    setUpload({ share: 0, names: files.length === 1 ? files[0].name : `${files.length} files` });
    try {
      const d = await api.uploadDataset(files, (share) => setUpload((u) => u && { ...u, share }));
      setRejected(d.rejected);
      await landed(d);
    } catch (e) {
      setRejected((e as { rejected?: Rejected[] }).rejected ?? []);
      ui.toast((e as Error).message);
    } finally {
      setUpload(null);
    }
  };

  const useFolder = async () => {
    setFolderErr("");
    try { await landed(await api.linkFolder(path.trim())); setPath(""); }
    catch (e) { setFolderErr((e as Error).message); }
  };

  const start = async (d: Dataset | null = chosen) => {
    if (!d || busy) return;
    setBusy(true);
    try {
      const e = await api.create(d.name);
      await qc.invalidateQueries({ queryKey: ["experiments"] });
      if (text.trim()) ui.set({ autoSend: text.trim() });
      ui.openProject(e.id);
    } catch (err) {
      ui.toast((err as Error).message);
      setBusy(false);
    }
  };

  const live = useMemo(() => recent.filter((e) => !e.archived), [recent]);
  const shown = useMemo(() => live.filter((e) => e.title.toLowerCase().includes(find.trim().toLowerCase())), [live, find]);
  // the project you had open last, one click away (the app no longer jumps into it by itself)
  const last = live.find((e) => e.id === lastProject());
  const labs = models?.labs ?? [];

  return (
    <main className="ws-chat start-dash">
      <div className="scroll" id="chat-scroll"><div className="thread start-page">
        <header className="st-head">
          <h1>flow-er</h1>
          <p>Your projects, the lab's gating models and the data on this computer. To start an analysis, open the flower at the bottom.</p>
        </header>
        {health && health.llm_key !== "loaded" && (
          <p className="note warn">The agent has no model key yet: set <code>OPENAI_API_KEY</code> in <code>backend/.env</code>.</p>
        )}

        {last && (
          <button className="st-continue" onClick={() => ui.openProject(last.id)}>
            <span className="k">Continue</span><b>{last.title}</b>
            <span className="d">{last.figures ? `${last.figures} figure${last.figures > 1 ? "s" : ""} · ` : ""}{ago(last.last_opened ?? last.created)}</span><Icon.caret />
          </button>
        )}

        <div className="db-cols">
          <section className="st-sec">
            <div className="st-h2row"><h2>Projects</h2>
              {!!live.length && <label className="st-find"><Icon.search /><input id="start-find" value={find} onChange={(e) => setFind(e.target.value)} placeholder="Find a project" aria-label="Find a project" /></label>}
            </div>
            {live.length ? (
              <div className="db-projs">
                {(allProjects ? shown : shown.slice(0, 6)).map((e) => <ProjectCard key={e.id} e={e} />)}
                {!shown.length && <p className="note">No project matches.</p>}
              </div>
            ) : <p className="note">No projects yet. Pick a dataset below and open the flower to start one.</p>}
            {shown.length > 6 && (
              <button className="btn sm plain db-more" onClick={() => setAllProjects(!allProjects)}>{allProjects ? "Show fewer" : `Show all ${shown.length}`}</button>
            )}
          </section>

          <section className="st-sec">
            <h2>Gating models</h2>
            <div className="db-labs">
              {labs.map((l) => <LabCard key={l} lab={l} />)}
              {models && !labs.length && <p className="note">{models.message}</p>}
              {!models && <span className="thinking"><Flower /><span className="shimmer">Checking the models…</span></span>}
            </div>
            {health && <p className="db-sub db-agent"><Bloom p={1} className="xs" /> Agent: {health.llm_key === "loaded" ? `${health.model} ready` : "no key"}</p>}
          </section>
        </div>

        <section className="st-sec">
          <h2>Datasets</h2>
          <div className="ds-grid">
            {isLoading && <span className="thinking"><Flower /><span className="shimmer">Looking in DATA_DIR…</span></span>}
            {datasets?.map((d) => (
              <button key={d.name} data-feeds-dock className={`ds-card${chosen?.name === d.name ? " on" : ""}`} aria-pressed={chosen?.name === d.name}
                onClick={() => choose(d)} onDoubleClick={() => start(d)} title="Choose this data; double-click to open it now">
                <span className={`ds-kind ${d.kind}`}>{KIND[d.kind]}</span>
                <b>{d.label}</b>
                <span>{d.files} FCS file{d.files === 1 ? "" : "s"} · {fmt(d.size_mb, 1)} MB{d.workspace ? " · FlowJo workspace" : ""}</span>
                {!!d.markers.length && <span className="ds-markers">{d.markers.slice(0, 10).map((m) => <i key={m}>{m}</i>)}</span>}
              </button>
            ))}
            {datasets && !datasets.length && <p className="note">No datasets yet. Bring some below.</p>}
          </div>
        </section>

        <section className="st-sec">
          <h2>Bring data</h2>
          <div className="st-ways">
            <div className={`way drop${over ? " over" : ""}`} id="start-drop" role="button" tabIndex={0}
              onClick={() => fileInput.current?.click()} onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") fileInput.current?.click(); }}
              onDragOver={(e) => { if (e.dataTransfer.types.includes("Files")) { e.preventDefault(); setOver(true); } }}
              onDragLeave={() => setOver(false)} onDrop={(e) => { e.preventDefault(); setOver(false); send(Array.from(e.dataTransfer.files)); }}>
              <input ref={fileInput} id="start-files" type="file" multiple hidden accept=".fcs,.wsp,.csv,.tsv,.txt,.md,.pdf,.docx,.xlsx"
                onChange={(e) => { send(Array.from(e.target.files ?? [])); e.target.value = ""; }} />
              <span className="way-ic"><Icon.upload /></span>
              <b>Drop or choose files</b>
              <span>FCS files, their FlowJo workspace, and a protocol or plate map if you have one</span>
              {upload && (
                <div className="way-progress" role="progressbar" aria-valuenow={Math.round(upload.share * 100)} aria-valuemin={0} aria-valuemax={100}>
                  <i style={{ width: `${Math.round(upload.share * 100)}%` }} /><span>{upload.share < 1 ? `Uploading ${upload.names}: ${Math.round(upload.share * 100)}%` : "Checking the files…"}</span>
                </div>
              )}
            </div>
            <form className="way folder" onSubmit={(e) => { e.preventDefault(); useFolder(); }}>
              <span className="way-ic"><Icon.folder /></span>
              <b>Use a folder on this computer</b>
              <span>Read where it is, so large files are not copied</span>
              <div className="way-row">
                <input id="start-path" value={path} onChange={(e) => setPath(e.target.value)} placeholder="~/data/my-experiment" aria-label="Folder path" />
                <button className="btn sm" type="submit" disabled={!path.trim()}>Use folder</button>
              </div>
              {folderErr && <span className="way-err" role="alert">{folderErr}</span>}
            </form>
          </div>
          {!!rejected.length && (
            <div className="st-rejected" role="status"><b>Not taken:</b> {rejected.map((r) => `${r.name} (${r.why})`).join("; ")}</div>
          )}
        </section>
      </div></div>
      <Dock state={dock} setState={setDock} text={text} setText={setText} chosen={chosen} setChosen={choose}
        datasets={datasets ?? []} busy={busy} onStart={() => start()} />
    </main>
  );
}
