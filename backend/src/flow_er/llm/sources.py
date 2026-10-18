"""Fetching reference code for a script (spec §3): one file from GitHub or from a PyPI package's source,
capped in size, with its licence read from the repository or the package metadata. A source whose
licence is not allowed comes back refused, without its code, so it cannot be adapted by accident.
"""
from __future__ import annotations

import base64
import io
import json
import re
import tarfile
import urllib.request
import zipfile

from flow_er.workspace.scripts import licence_ok

MAX_FILE = 200_000      # bytes of code handed to the writer at once
MAX_FETCH = 3 << 20     # bytes of one file read from the network
MAX_ARCHIVE = 15 << 20  # bytes of a package download
UA = {"User-Agent": "flow-er (script provenance check)", "Accept": "application/vnd.github+json"}
# PyPI licence words and classifiers, as SPDX ids
SPDX = {"mit": "MIT", "bsd": "BSD-3-Clause", "bsd-3-clause": "BSD-3-Clause", "bsd-2-clause": "BSD-2-Clause", "new bsd": "BSD-3-Clause",
        "apache 2.0": "Apache-2.0", "apache-2.0": "Apache-2.0", "apache software license": "Apache-2.0", "isc": "ISC",
        "python software foundation license": "PSF-2.0", "psf": "PSF-2.0", "gpl": "GPL", "gplv2": "GPL-2.0", "gplv3": "GPL-3.0",
        "gnu general public license v2 (gplv2)": "GPL-2.0", "gnu general public license v3 (gplv3)": "GPL-3.0",
        "gnu lesser general public license v3 (lgplv3)": "LGPL-3.0", "mozilla public license 2.0 (mpl 2.0)": "MPL-2.0"}


def _get(url: str, limit: int) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=20) as r:
        data = r.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"{url} is larger than {limit // 1000} kB")
    return data


def _spdx(text: str | None) -> str | None:
    if not text:
        return None
    t = text.strip().lower()
    if t in SPDX:
        return SPDX[t]
    for k, v in sorted(SPDX.items(), key=lambda kv: -len(kv[0])):
        if k in t:
            return v
    return text.strip()[:40]


def _from_text(text: str) -> str | None:
    """The licence a licence file states, when GitHub could not name it."""
    t = " ".join(text.split()).lower()
    if "gnu lesser general public license" in t:
        return "LGPL-3.0" if "version 3" in t else "LGPL-2.1"
    if "gnu affero general public license" in t:
        return "AGPL-3.0"
    if "gnu general public license" in t:
        return "GPL-3.0" if "version 3" in t else "GPL-2.0"
    if "permission is hereby granted, free of charge" in t:
        return "MIT"
    if "apache license" in t and "version 2.0" in t:
        return "Apache-2.0"
    if "redistribution and use in source and binary forms" in t:
        return "BSD-3-Clause" if "neither the name" in t else "BSD-2-Clause"
    return None


def github(url: str) -> dict:
    m = re.match(r"https?://(?:github\.com/([^/]+)/([^/]+)/(?:blob|raw)/([^/]+)/(.+)|raw\.githubusercontent\.com/([^/]+)/([^/]+)/([^/]+)/(.+))", url)
    if not m:
        raise ValueError("give a link to one file on GitHub (github.com/<owner>/<repo>/blob/<branch>/<path>)")
    owner, repo, ref, path = [g for g in m.groups() if g is not None]
    meta = json.loads(_get(f"https://api.github.com/repos/{owner}/{repo}/license", 200_000))
    spdx = (meta.get("license") or {}).get("spdx_id")
    if spdx in (None, "NOASSERTION"):
        spdx = _from_text(base64.b64decode(meta.get("content") or b"").decode("utf-8", "replace"))
    return {"source": f"https://github.com/{owner}/{repo}/blob/{ref}/{path}", "version": ref, "licence": spdx,
            "licence_from": "the repository's licence file", "fetch": f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}"}


def pypi(package: str, path: str) -> dict:
    info = json.loads(_get(f"https://pypi.org/pypi/{package}/json", 2_000_000))
    i = info["info"]
    spdx = _spdx(i.get("license_expression")) or _spdx(next((c.split("::")[-1] for c in i.get("classifiers", []) if c.startswith("License ::")), None)) or _spdx(i.get("license"))
    sdist = next((u for u in info["urls"] if u["packagetype"] == "sdist"), None)
    if not sdist:
        raise ValueError(f"{package} {i['version']} has no source download on PyPI")
    return {"source": f"https://pypi.org/project/{package}/{i['version']}/ ({path})", "version": i["version"], "licence": spdx,
            "licence_from": "the package metadata on PyPI", "archive": sdist["url"], "member": path}


def fetch(ref: str, lines: str | None = None) -> dict:
    """One file of reference code, with where it came from and its licence. `ref` is a GitHub file link,
    or pypi:<package>/<path inside its source>. `lines` ("120-260") picks part of a large file; a large
    file asked for whole comes back as an index of its functions and classes, to pick from."""
    if ref.startswith("pypi:"):
        package, _, path = ref[5:].partition("/")
        src = pypi(package, path)
    else:
        src = github(ref)
    if why := licence_ok(src["licence"]):
        return {**{k: src[k] for k in ("source", "version", "licence", "licence_from")}, "refused": why}
    if not src["licence"]:
        return {**{k: src[k] for k in ("source", "version", "licence_from")}, "licence": None,
                "refused": "no licence was found for it, so it cannot be reused; write the step from scratch instead."}
    if "fetch" in src:
        text = _get(src["fetch"], MAX_FETCH).decode("utf-8", "replace")
    else:
        blob = _get(src["archive"], MAX_ARCHIVE)
        text = _member(blob, src["archive"], src["member"])
    all_lines = text.splitlines()
    first, last = 1, len(all_lines)
    if lines:
        m = re.match(r"\s*(\d+)\s*-\s*(\d+)\s*$", lines)
        if not m:
            raise ValueError("lines must look like 120-260")
        first, last = max(1, int(m.group(1))), min(len(all_lines), int(m.group(2)))
    part = all_lines[first - 1:last]
    head = {**{k: src[k] for k in ("source", "version", "licence", "licence_from")}, "lines": len(all_lines)}
    if sum(len(x) + 1 for x in part) > MAX_FILE:
        index = [f"{n:5d}  {x.strip()[:100]}" for n, x in enumerate(all_lines, 1) if re.match(r"\s*(def|class)\s", x)]
        return {**head, "too_long": f"{len(all_lines)} lines; ask again with lines, e.g. a function's lines from this index",
                "index": "\n".join(index[:400])}
    return {**head, "shown": f"{first}-{last}", "code": "\n".join(f"{n:5d}  {x}" for n, x in enumerate(part, first))}


def _member(blob: bytes, url: str, path: str) -> str:
    if url.endswith(".zip"):
        z = zipfile.ZipFile(io.BytesIO(blob))
        name = next((n for n in z.namelist() if n.endswith("/" + path) or n == path), None)
        if not name:
            raise ValueError(f"no file {path} in the package")
        data = z.read(name)
    else:
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as t:
            m = next((m for m in t.getmembers() if m.isfile() and (m.name.endswith("/" + path) or m.name == path)), None)
            if not m:
                raise ValueError(f"no file {path} in the package")
            data = t.extractfile(m).read()
    if len(data) > MAX_FETCH:
        raise ValueError(f"{path} is larger than {MAX_FETCH >> 20} MB")
    return data.decode("utf-8", "replace")
