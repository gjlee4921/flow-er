"""Plain-language requests routed to tools by the agent (OpenAI in the web app for now).

Each turn has a plain endpoint (one answer at the end) and a streamed one (server-sent events:
the model thinking, each tool starting and finishing, screen changes, then the answer), so the
screen can show the work as it happens.
"""
import json
import logging
import queue
import threading
from collections.abc import Callable, Iterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from flow_er.llm import agent
from flow_er.schemas import ChatIn, LabChatIn
from flow_er.workspace import store

router = APIRouter(tags=["agent"])
log = logging.getLogger(__name__)


def _stream(turn: Callable[[Callable[[dict], None]], dict]) -> StreamingResponse:
    """Run a turn in a thread and pass its events on as they happen; the last one is the answer."""
    q: queue.Queue[dict | None] = queue.Queue()

    def work() -> None:
        try:
            q.put({"type": "message", "message": turn(q.put)})
        except Exception as e:  # the stream must end with something the screen can show
            log.exception("streamed turn failed")
            q.put({"type": "error", "error": str(e)})
        finally:
            q.put(None)

    threading.Thread(target=work, daemon=True).start()

    def events() -> Iterator[str]:
        while (e := q.get()) is not None:
            yield f"data: {json.dumps(e, default=str)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/experiments/{exp_id}/messages")
def messages(exp_id: str) -> list[dict]:
    try:
        return store.messages(exp_id)
    except KeyError:
        raise HTTPException(404, f"no experiment {exp_id}")


@router.post("/experiments/{exp_id}/chat")
def chat(exp_id: str, body: ChatIn) -> dict:
    """One turn. Runs in the threadpool: model calls and FlowKit are blocking."""
    try:
        store.meta(exp_id)
    except KeyError:
        raise HTTPException(404, f"no experiment {exp_id}")
    if not body.text.strip() and not body.attachments:
        raise HTTPException(400, "say something or attach a file")
    return agent.run_turn(exp_id, body.text.strip(), body.attachments)


@router.post("/experiments/{exp_id}/chat/stream")
def chat_stream(exp_id: str, body: ChatIn) -> StreamingResponse:
    """The same turn as /chat, streamed as it happens."""
    try:
        store.meta(exp_id)
    except KeyError:
        raise HTTPException(404, f"no experiment {exp_id}")
    if not body.text.strip() and not body.attachments:
        raise HTTPException(400, "say something or attach a file")
    return _stream(lambda on_event: agent.run_turn(exp_id, body.text.strip(), body.attachments, on_event=on_event))


@router.get("/experiments/{exp_id}/lab/messages")
def lab_messages(exp_id: str) -> list[dict]:
    """The data lab assistant's own conversation, kept apart from the main chat."""
    try:
        return store.messages(exp_id, "lab_chat")
    except KeyError:
        raise HTTPException(404, f"no experiment {exp_id}")


@router.post("/experiments/{exp_id}/lab/chat")
def lab_chat(exp_id: str, body: LabChatIn) -> dict:
    """One turn with the data lab assistant. It moves the lab and reads numbers; it changes nothing."""
    try:
        store.meta(exp_id)
    except KeyError:
        raise HTTPException(404, f"no experiment {exp_id}")
    if not body.text.strip():
        raise HTTPException(400, "say something")
    return agent.run_turn(exp_id, body.text.strip(), thread="lab_chat", screen=body.screen)


@router.post("/experiments/{exp_id}/lab/chat/stream")
def lab_chat_stream(exp_id: str, body: LabChatIn) -> StreamingResponse:
    """The lab assistant's turn, streamed, so the lab moves on screen as each tool runs."""
    try:
        store.meta(exp_id)
    except KeyError:
        raise HTTPException(404, f"no experiment {exp_id}")
    if not body.text.strip():
        raise HTTPException(400, "say something")
    return _stream(lambda on_event: agent.run_turn(exp_id, body.text.strip(), thread="lab_chat", screen=body.screen, on_event=on_event))
