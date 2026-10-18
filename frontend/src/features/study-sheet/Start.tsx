// Before there is a project: say what the experiment is for and bring its data. Three ways to bring
// it sit side by side: drop files in, use a folder on this computer (read in place, nothing copied),
// or pick a dataset already here. Start opens the project and, with a question, plans it at once.
import { useMemo, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { api, type Dataset, type Rejected } from "../../api";
import { useDatasets, useExperiments, useHealth } from "../../api/hooks";
import { Flower, Icon } from "../../components/icons";
import { fmt, lastProject, ui } from "../../lib";

const KIND = { example: "Example", upload: "Uploaded", folder: "On this computer" } as const;

export default function Start() {
  const qc = useQueryClient();
  const { data: datasets, isLoading } = useDatasets();
  const { data: recent = [] } = useExperiments();
  const { data: health } = useHealth();
  const [chosen, setChosen] = useState<Dataset | null>(null);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [upload, setUpload] = useState<{ share: number; names: string } | null>(null);
  const [rejected, setRejected] = useState<Rejected[]>([]);
  const [path, setPath] = useState("");
  const [folderErr, setFolderErr] = useState("");
  const [over, setOver] = useState(false);
  const [find, setFind] = useState("");
  const fileInput = useRef<HTMLInputElement>(null);
  const textBox = useRef<HTMLTextAreaElement>(null);

  const choose = (d: Dataset) => { setChosen(d); textBox.current?.focus(); };
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

  const shown = useMemo(() => recent.filter((e) => e.title.toLowerCase().includes(find.trim().toLowerCase())), [recent, find]);
  // the project you had open last, one click away (the app no longer jumps into it by itself)
  const last = recent.find((e) => e.id === lastProject() && !e.archived);

  return (
    <main className="ws-chat">
      <div className="scroll" id="chat-scroll"><div className="thread start-page">
        <header className="st-head">
          <h1>Start an analysis</h1>
          <p>Say what the experiment is for and bring its data. flow-er plans the analysis from there, and nothing changes until you keep it.</p>
        </header>
        {last && (
          <button className="st-continue" onClick={() => ui.openProject(last.id)}>
            <span className="k">Continue</span><b>{last.title}</b>
            <span className="d">{last.figures ? `${last.figures} figure${last.figures > 1 ? "s" : ""} · ` : ""}{last.created.slice(0, 10)}</span><Icon.caret />
          </button>
        )}
        {health && health.llm_key !== "loaded" && (
          <p className="note warn">The agent has no model key yet: set <code>OPENAI_API_KEY</code> in <code>backend/.env</code>.</p>
        )}

        <form className="composer st-comp" onSubmit={(e) => { e.preventDefault(); start(); }}>
          {chosen && (
            <div className="c-atts"><span className="att st-data"><span className="att-k">Data</span><span className="att-n">{chosen.label}</span>
              <span className="att-s">{chosen.files} files</span>
              <button type="button" aria-label="Choose other data" onClick={() => setChosen(null)}>×</button></span></div>
          )}
          <textarea id="start-text" ref={textBox} rows={2} value={text} onChange={(e) => setText(e.target.value)}
            placeholder="What is this experiment for? For example: how CD4 and CD8 T cells respond to stimulation"
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); start(); } }} aria-label="What the experiment is for" />
          <div className="c-row">
            <span className="st-hint">{chosen ? (text.trim() ? "Start opens the project and plans the analysis" : "Start opens the project; add a sentence and it plans too") : "Choose or bring data below"}</span>
            <button className="btn ink" id="start-go" type="submit" disabled={!chosen || busy}>
              {busy ? <Flower className="sm" /> : null}{text.trim() ? "Start and plan" : "Start"}
            </button>
          </div>
        </form>

        <section className="st-sec">
          <h2>Bring the data</h2>
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

        <section className="st-sec">
          <h2>Datasets</h2>
          <div className="ds-grid">
            {isLoading && <span className="thinking"><Flower /><span className="shimmer">Looking in DATA_DIR…</span></span>}
            {datasets?.map((d) => (
              <button key={d.name} className={`ds-card${chosen?.name === d.name ? " on" : ""}`} aria-pressed={chosen?.name === d.name}
                onClick={() => choose(d)} onDoubleClick={() => start(d)} title="Choose this data; double-click to open it now">
                <span className={`ds-kind ${d.kind}`}>{KIND[d.kind]}</span>
                <b>{d.label}</b>
                <span>{d.files} FCS file{d.files === 1 ? "" : "s"} · {fmt(d.size_mb, 1)} MB{d.workspace ? " · FlowJo workspace" : ""}</span>
                {!!d.markers.length && <span className="ds-markers">{d.markers.slice(0, 10).map((m) => <i key={m}>{m}</i>)}</span>}
              </button>
            ))}
            {datasets && !datasets.length && <p className="note">No datasets yet. Drop files above or use a folder on this computer.</p>}
          </div>
        </section>

        {!!recent.length && (
          <section className="st-sec">
            <div className="st-h2row"><h2>Recent projects</h2>
              <label className="st-find"><Icon.search /><input id="start-find" value={find} onChange={(e) => setFind(e.target.value)} placeholder="Find a project" aria-label="Find a project" /></label>
            </div>
            <div className="n-tree recent">
              {shown.slice(0, 12).map((e) => (
                <button key={e.id} className="n-row" onClick={() => ui.openProject(e.id)}>
                  <span className="tw" /><span className="nm">{e.title}</span><span className="val">{e.created.slice(0, 10)}</span>
                </button>
              ))}
              {!shown.length && <p className="note">No project matches.</p>}
            </div>
          </section>
        )}
      </div></div>
    </main>
  );
}
