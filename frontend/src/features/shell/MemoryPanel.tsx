// What flow-er remembers about the scientist, in one place: every memory in their own words, where it
// came from, and the means to change it, forget it, add one by hand, or pause remembering altogether.
import { useEffect, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { api, type Memory } from "../../api";
import { useMemory } from "../../api/hooks";
import { Icon } from "../../components/icons";
import { ui, useUi } from "../../lib";

const KIND = { definition: "Definition", preference: "Preference", lab: "Lab", project: "Project" } as const;

function Row({ m, onChanged }: { m: Memory; onChanged: () => void }) {
  const [edit, setEdit] = useState(false);
  const [text, setText] = useState(m.text);
  const save = async () => { if (text.trim() && text !== m.text) await api.editMemory(m.id, text.trim()); setEdit(false); onChanged(); };
  const forget = async () => { await api.forgetMemory(m.id); onChanged(); ui.toast("Forgotten."); };
  return (
    <li className="mem-row">
      <span className={`mem-kind ${m.kind}`}>{KIND[m.kind] ?? m.kind}</span>
      <div className="mem-body">
        {edit ? (
          <form onSubmit={(e) => { e.preventDefault(); save(); }}>
            <input autoFocus value={text} onChange={(e) => setText(e.target.value)} onBlur={save} aria-label="Change this memory" />
          </form>
        ) : (
          <button className="mem-text" title="Click to change" onClick={() => setEdit(true)}>{m.text}</button>
        )}
        <span className="mem-src">{m.source.experiment ? <>From “{m.quote}” in {m.source.title ?? "a project"}</> : "Typed in here"} · {m.created.slice(0, 10)}</span>
      </div>
      <button className="ib sm" aria-label={`Forget: ${m.text}`} title="Forget" onClick={forget}><Icon.close /></button>
    </li>
  );
}

export default function MemoryPanel() {
  const open = useUi((u) => u.memoryOpen);
  const qc = useQueryClient();
  const { data } = useMemory();
  const [draft, setDraft] = useState("");
  const changed = () => qc.invalidateQueries({ queryKey: ["memory"] });
  useEffect(() => {
    if (!open) return;
    changed();
    const k = (e: KeyboardEvent) => { if (e.key === "Escape") ui.set({ memoryOpen: false }); };
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [open]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!open) return null;
  const items = data?.memories ?? [];
  const add = async () => { if (draft.trim().length < 3) return; await api.addMemory(draft.trim()); setDraft(""); changed(); };
  const pause = async () => { await api.pauseMemory(!data?.paused); changed(); };
  return (
    <div className="mem-back" onPointerDown={(e) => { if (e.target === e.currentTarget) ui.set({ memoryOpen: false }); }}>
      <section className="mem-panel" role="dialog" aria-modal="true" aria-labelledby="mem-h">
        <header>
          <h2 id="mem-h">What flow-er remembers</h2>
          <button className="ib sm" aria-label="Close" onClick={() => ui.set({ memoryOpen: false })}><Icon.close /></button>
        </header>
        <p className="mem-lead">Things you told the chat about how you work, kept for every project. Only your own words are kept; never your data, files or results. Click one to change it.</p>
        {data?.paused && <p className="mem-paused" role="status">Remembering is paused: nothing new is kept, and what is here is still used.</p>}
        <ul className="mem-list" id="mem-list">
          {items.map((m) => <Row key={m.id} m={m} onChanged={changed} />)}
          {!items.length && <li className="mem-none">Nothing yet. Tell the chat how you define a population or like your results shown, and it is kept here.</li>}
        </ul>
        <form className="mem-add" onSubmit={(e) => { e.preventDefault(); add(); }}>
          <input id="mem-new" value={draft} onChange={(e) => setDraft(e.target.value)} placeholder="Add something to remember, e.g. I report subsets as % of CD3+" aria-label="Add something to remember" />
          <button className="btn sm ink" disabled={draft.trim().length < 3}>Add</button>
        </form>
        <footer>
          <label className="mem-pause"><input type="checkbox" id="mem-pause" checked={!!data?.paused} onChange={pause} /> Pause remembering</label>
          <span>{items.length} kept</span>
        </footer>
      </section>
    </div>
  );
}
