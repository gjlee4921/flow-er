// Settings, from the profile button at the top of the sidebar: who is signed in, the agent, the lab's
// gating models, where data and results live, and what flow-er remembers. The machine's setup comes
// from backend/.env and is shown read-only here; memory can be paused and managed in place.
import { useEffect } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { api } from "../../api";
import { useLabModels, useMemory, useModels, useSettings } from "../../api/hooks";
import { Bloom, Icon, Mark } from "../../components/icons";
import { ui, useUi } from "../../lib";

const STATE = { live: "Live", outdated: "Needs retraining", offline: "Offline" } as const;

function LabRow({ lab }: { lab: string }) {
  const { data } = useLabModels(lab);
  const st = data?.status?.state;
  return (
    <div className="set-row">
      <span className="set-k"><span className={`set-fl ${st ?? "loading"}`}><Bloom p={1} /></span>Lab {lab}</span>
      <span className="set-v">{st ? `${STATE[st]} · ${data?.status?.why}` : "reading…"}</span>
    </div>
  );
}

function Row({ k, v, mono = false }: { k: string; v: React.ReactNode; mono?: boolean }) {
  return <div className="set-row"><span className="set-k">{k}</span><span className={`set-v${mono ? " mono" : ""}`}>{v}</span></div>;
}

export default function SettingsPanel() {
  const open = useUi((u) => u.settingsOpen);
  const qc = useQueryClient();
  const { data: cfg } = useSettings(open);
  const { data: models } = useModels();
  const { data: mem } = useMemory();
  useEffect(() => {
    if (!open) return;
    qc.invalidateQueries({ queryKey: ["settings"] });
    qc.invalidateQueries({ queryKey: ["memory"] });
    const k = (e: KeyboardEvent) => { if (e.key === "Escape") ui.set({ settingsOpen: false }); };
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [open]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!open) return null;
  const close = () => ui.set({ settingsOpen: false });
  const memories = mem?.memories ?? [];
  const pause = async () => { await api.pauseMemory(!mem?.paused); qc.invalidateQueries({ queryKey: ["memory"] }); };

  return (
    <div className="mem-back" onPointerDown={(e) => { if (e.target === e.currentTarget) close(); }}>
      <section className="mem-panel set-panel" role="dialog" aria-modal="true" aria-labelledby="set-h">
        <header>
          <h2 id="set-h">Settings</h2>
          <button className="ib sm" aria-label="Close" onClick={close}><Icon.close /></button>
        </header>
        <div className="set-me"><Mark /><div><b>Will</b><span>flow-er on this machine</span></div></div>

        <section className="set-sec">
          <h3>What flow-er remembers</h3>
          <p className="set-lead">Things you told the chat about how you work, used in every project. Only your words; never your data.</p>
          <ul className="set-mem">
            {memories.slice(0, 3).map((m) => <li key={m.id}>{m.text}</li>)}
            {!memories.length && <li className="none">Nothing yet.</li>}
          </ul>
          <div className="set-acts">
            <button className="btn sm" onClick={() => ui.set({ settingsOpen: false, memoryOpen: true })}>Manage all {memories.length || ""}</button>
            <label className="mem-pause"><input type="checkbox" checked={!!mem?.paused} onChange={pause} /> Pause remembering</label>
          </div>
        </section>

        <section className="set-sec">
          <h3>Agent</h3>
          {cfg ? <>
            <Row k="Model" v={`${cfg.model || "none"} (${cfg.llm_provider})`} />
            <Row k="Key" v={cfg.llm_key === "loaded" ? "loaded" : <span className="warn">missing: set OPENAI_API_KEY</span>} />
            <Row k="Tool rounds" v={`up to ${cfg.agent_max_rounds} per answer`} />
          </> : <p className="set-lead">Reading…</p>}
        </section>

        <section className="set-sec">
          <h3>Gating models</h3>
          {(models?.labs ?? []).map((l) => <LabRow key={l} lab={l} />)}
          {models && !models.labs?.length && <p className="set-lead">{models.message}</p>}
          {cfg && <Row k="GPU retraining" v={cfg.colab === "not set" ? "on this computer's CPU (COLAB_JOBS_DIR not set)" : `Colab: ${cfg.colab}`} />}
        </section>

        <section className="set-sec">
          <h3>Data and folders</h3>
          {cfg && <>
            <Row k="Data" v={cfg.data_dir} mono />
            <Row k="Models" v={cfg.models_dir} mono />
            <Row k="Projects" v={cfg.workspace_dir} mono />
            <Row k="Folders you can link" v={cfg.import_roots.join(", ")} mono />
            <Row k="Largest upload" v={`${cfg.upload_max_mb.toLocaleString()} MB`} />
          </>}
        </section>

        <footer>
          <span>{cfg ? <>These come from <code>{cfg.env_file}</code>{cfg.env_file_exists ? "" : " (not created yet)"}; change them there and restart the backend.</> : ""}</span>
        </footer>
      </section>
    </div>
  );
}
