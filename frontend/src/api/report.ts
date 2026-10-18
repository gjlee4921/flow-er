// The results page as one document: the agent's words with every number resolved to a table cell,
// and the methods and data list written by code. Keyed under ["exp", id] so a keep, an undo or a turn
// refreshes it with everything else.
import { useQuery } from "@tanstack/react-query";

export type BoundNumber = {
  display: string; value: number | number[]; table: string; column: string; agg: string | null;
  rows: Record<string, string>[]; n: number; step?: string | null;
};
export type Segment = { text: string } | { number: BoundNumber; ref: string } | { broken: string; ref: string };
export type Report = {
  title: string; summary: Segment[]; key_numbers: ({ label: string } & Segment)[];
  draft: boolean; written: boolean;
  methods: { heading: string; text: string }[];
  data: {
    files: { name: string; bytes: number; sha256: string }[];
    tables: { name: string; rows: number; columns: string[]; step: string | null }[];
    workflow_version: string | null;
    software: { name: string; version: string; licence: string }[];
  };
  references: { text: string; url: string }[];
};

async function getReport(id: string): Promise<Report> {
  const r = await fetch(`/api/experiments/${id}/report`);
  if (!r.ok) throw new Error((await r.json().catch(() => null))?.detail ?? r.statusText);
  return r.json();
}

export const useReport = (id: string | null) =>
  useQuery({ queryKey: ["exp", id, "report"], queryFn: () => getReport(id!), enabled: !!id });
