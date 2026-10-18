"""Web search without the network: citations become numbered sources, and searches show in the trace."""
from types import SimpleNamespace as NS

from flow_er.config import settings
from flow_er.llm import client, web


def test_citations_become_numbers_and_a_source_list():
    text = ("FlowKit reads .wsp files. ([flowkit.readthedocs.io](https://flowkit.readthedocs.io/a?utm_source=openai)) "
            "It is BSD-3. ([github.com](https://github.com/whitews/FlowKit/blob/master/LICENSE)) "
            "See the docs again ([flowkit.readthedocs.io](https://flowkit.readthedocs.io/a?utm_source=openai)).")

    docs = "[flowkit.readthedocs.io](https://flowkit.readthedocs.io/a?utm_source=openai)"
    lic = "[github.com](https://github.com/whitews/FlowKit/blob/master/LICENSE)"

    def span(link: str, start: int) -> dict:  # a citation covers the markdown link, as the API marks it
        return {"start": start, "end": start + len(link), "url": link[link.index("](") + 2:-1], "title": "t"}
    cites = [span(docs, text.index(docs)), span(lic, text.index(lic)), span(docs, text.rindex(docs))]
    out, sources = web.numbered([(text, cites)])
    assert out == "FlowKit reads .wsp files. [1] It is BSD-3. [2] See the docs again [1]."
    assert [s["url"] for s in sources] == ["https://flowkit.readthedocs.io/a",
                                           "https://github.com/whitews/FlowKit/blob/master/LICENSE"]
    assert [s["n"] for s in sources] == [1, 2]


def test_text_without_citations_is_left_alone():
    assert web.numbered([("No sources here.", [])]) == ("No sources here.", [])


def test_search_steps_say_what_they_did():
    assert web.search_summary({"type": "search", "query": "FlowKit licence"}) == ("Searching the web", "searched: FlowKit licence")
    assert web.search_summary({"type": "open_page", "url": "https://pypi.org/project/flowkit/?utm_source=openai"})[1] \
        == "opened https://pypi.org/project/flowkit/"


class Item(NS):
    def model_dump(self, exclude_none=True):
        return {k: v for k, v in vars(self).items() if v is not None and not callable(v)}


def test_respond_returns_sources_and_searches(monkeypatch):
    text = "It is BSD-3. ([github.com](https://github.com/x/LICENSE))"
    link = text[text.index("[") :-1]
    ann = NS(type="url_citation", start_index=text.index(link), end_index=text.index(link) + len(link),
             url="https://github.com/x/LICENSE", title="LICENSE")
    out = [Item(type="web_search_call", id="ws1", status="completed", action=Item(type="search", query="flowkit licence")),
           Item(type="message", id="m1", content=[NS(type="output_text", text=text, annotations=[ann])])]
    fake = NS(responses=NS(create=lambda **kw: NS(output=out, output_text=text, usage=NS(input_tokens=1, output_tokens=1))))
    monkeypatch.setattr(client, "_openai", lambda: fake)
    monkeypatch.setattr(settings, "llm_provider", "openai")
    r = client.respond("x", [{"role": "user", "content": "licence?"}], [web.TOOL])
    assert r.text == "It is BSD-3. [1]"
    assert r.sources == [{"n": 1, "title": "LICENSE", "url": "https://github.com/x/LICENSE"}]
    assert r.searches == [{"action": {"type": "search", "query": "flowkit licence"}, "ok": True}]
    assert not r.calls
