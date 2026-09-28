"""The live monitor's WebSocket, `/api/monitor` (#5, #12, #16).

The client sends binary frames (`ryuk.api.frames`); the service answers each frame it recognises
with a JSON `result`, and tells the client when the active model changes. A receive loop keeps
draining the socket into a one-slot buffer, the latest frame winning, while one worker runs
recognition on a thread, so a frame that arrives while inference is busy replaces the one waiting
and stale frames are dropped. Only one live monitor runs: a new connection supersedes the old one,
which is closed with 4001; with no active model the socket is closed with 4002.
"""

import logging
from contextlib import suppress
from typing import Annotated, Final, Literal

import anyio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import Field, RootModel
from starlette.concurrency import run_in_threadpool
from starlette.types import Message

from ryuk.api.frames import Frame, FrameError, decode_frame, parse_frame
from ryuk.api.schema import ApiModel
from ryuk.watchlist import live
from ryuk.watchlist.service import Watchlist

logger = logging.getLogger(__name__)

SUPERSEDED: Final = 4001
"""Close code: the live monitor was opened in another connection, which took over."""
NO_ACTIVE_MODEL: Final = 4002
"""Close code: nothing can be recognised, because no evaluated recognition model can be active or
the detector's weights are missing. The close reason says which."""

router = APIRouter(tags=["monitor"])


class FaceBox(ApiModel):
    """A face's box in the pixels of the frame it was found in. Not clipped to the frame."""

    x: float
    y: float
    width: float
    height: float


class MatchedPerson(ApiModel):
    id: str
    name: str


class MatchFace(ApiModel):
    outcome: Literal["match"]
    box: FaceBox
    score: float
    person: MatchedPerson
    """The person of interest matched; only a match names anyone."""


class NoMatchFace(ApiModel):
    outcome: Literal["no_match"]
    box: FaceBox
    score: float | None
    """The top candidate's score, under the threshold; None when nobody is on the watchlist."""


class TooSmallFace(ApiModel):
    """A detection too small to use: never scored."""

    outcome: Literal["too_small"]
    box: FaceBox


type Face = Annotated[MatchFace | NoMatchFace | TooSmallFace, Field(discriminator="outcome")]


class FrameResult(ApiModel):
    """Every face in one frame, judged by the active model at its threshold."""

    type: Literal["result"]
    seq: int
    """The frame's sequence number, from its header."""
    captured_at: int
    """The frame's capture time from its header, echoed."""
    width: int
    height: int
    model_key: str
    threshold: float
    faces: list[Face]


class ActiveModelChanged(ApiModel):
    """The active model was switched; the next frame's result is judged by it."""

    type: Literal["active_model_changed"]
    model_key: str
    threshold: float


class MonitorError(ApiModel):
    """A message that could not be used as a frame, or a frame whose recognition failed
    unexpectedly (`internal_error`). The connection stays open."""

    type: Literal["error"]
    seq: int | None
    """The frame's sequence number, when its header could be read."""
    code: str
    detail: str


class MonitorMessage(
    RootModel[
        Annotated[FrameResult | ActiveModelChanged | MonitorError, Field(discriminator="type")]
    ]
):
    """Every message the service sends on `/api/monitor`, by `type`."""


class LiveMonitor:
    """The one live monitor connection. A new one supersedes it."""

    def __init__(self) -> None:
        self._current: _Connection | None = None

    async def serve(self, websocket: WebSocket, watchlist: Watchlist) -> None:
        """Recognise frames from an accepted socket until it closes or is superseded."""
        connection = _Connection(websocket, watchlist)
        previous, self._current = self._current, connection
        if previous is not None:
            await previous.close(SUPERSEDED, "The live monitor was opened in another tab.")
        try:
            await connection.run()
        finally:
            if self._current is connection:
                self._current = None

    async def announce(self, message: ActiveModelChanged) -> None:
        """Send `message` to the live monitor, if one is running."""
        if self._current is not None:
            await self._current.send(message)


@router.websocket("/monitor")
async def monitor(websocket: WebSocket) -> None:
    watchlist: Watchlist | None = websocket.app.state.watchlist
    live_monitor: LiveMonitor = websocket.app.state.monitor
    # Accepted first: a socket closed during the handshake reaches the browser as 1006, and the
    # client needs the code to say why.
    await websocket.accept()
    refusal = "The watchlist is not running." if watchlist is None else watchlist.monitor_refusal()
    if watchlist is None or refusal is not None:
        await websocket.close(NO_ACTIVE_MODEL, refusal)
        return
    await live_monitor.serve(websocket, watchlist)


class _Connection:
    def __init__(self, websocket: WebSocket, watchlist: Watchlist) -> None:
        self._websocket = websocket
        self._watchlist = watchlist
        # The one-slot buffer: only the latest frame waits for the worker.
        self._latest: Frame | None = None
        self._arrived = anyio.Event()
        # The receive loop, the worker and `announce` all send; one message at a time.
        self._sending = anyio.Lock()
        self._closed = False
        # Created before `run` enters it, so a connection superseded straight away still stops.
        self._scope = anyio.CancelScope()

    async def run(self) -> None:
        with self._scope:
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(self._recognise_latest)
                await self._receive_frames()
                tasks.cancel_scope.cancel()

    async def send(self, message: ApiModel) -> None:
        async with self._sending:
            if self._closed:
                return
            try:
                await self._websocket.send_text(message.model_dump_json(by_alias=True))
            except WebSocketDisconnect:
                self._closed = True

    async def close(self, code: int, reason: str) -> None:
        async with self._sending:
            if not self._closed:
                self._closed = True
                # The client may have left first; then there is nobody to tell.
                with suppress(WebSocketDisconnect):
                    await self._websocket.close(code, reason)
        self._scope.cancel()

    async def _receive_frames(self) -> None:
        """Drain the socket into the one-slot buffer until the client disconnects."""
        while True:
            message = await self._websocket.receive()
            if message["type"] == "websocket.disconnect":
                self._closed = True
                return
            try:
                frame = _frame(message)
            except FrameError as error:
                await self.send(_error(error))
                continue
            self._latest = frame  # the latest frame wins; one still waiting is dropped
            self._arrived.set()

    async def _recognise_latest(self) -> None:
        """Recognise the frame waiting in the buffer, one at a time, off the event loop."""
        while True:
            await self._arrived.wait()
            self._arrived = anyio.Event()
            frame, self._latest = self._latest, None
            if frame is not None:
                await self.send(await run_in_threadpool(self._recognise, frame))

    def _recognise(self, frame: Frame) -> FrameResult | MonitorError:
        try:
            return self._result(frame)
        except FrameError as error:
            return _error(error)
        except Exception:
            # One frame that fails must not end the live monitor: the error is logged and
            # answered, and the next frame is recognised as usual.
            logger.exception("Recognising frame %d failed", frame.seq)
            return MonitorError(
                type="error",
                seq=frame.seq,
                code="internal_error",
                detail="The service hit an unexpected error recognising this frame.",
            )

    def _result(self, frame: Frame) -> FrameResult:
        recognition = self._watchlist.recognise(decode_frame(frame))
        return FrameResult(
            type="result",
            seq=frame.seq,
            captured_at=frame.captured_at,
            width=frame.width,
            height=frame.height,
            model_key=recognition.model.id,
            threshold=recognition.threshold,
            faces=[_face(face) for face in recognition.faces],
        )


def _frame(message: Message) -> Frame:
    data: bytes | None = message.get("bytes")
    if data is None:
        raise FrameError("invalid_frame", "A frame is a binary message.")
    return parse_frame(data)


def _face(face: live.LiveFace) -> MatchFace | NoMatchFace | TooSmallFace:
    box = FaceBox(x=face.box.x, y=face.box.y, width=face.box.width, height=face.box.height)
    match face:
        case live.Match(candidate=candidate):
            return MatchFace(
                outcome="match",
                box=box,
                score=candidate.score,
                person=MatchedPerson(id=candidate.person_id, name=candidate.name),
            )
        case live.NoMatch(score=score):
            return NoMatchFace(outcome="no_match", box=box, score=score)
        case live.TooSmall():
            return TooSmallFace(outcome="too_small", box=box)


def _error(error: FrameError) -> MonitorError:
    return MonitorError(type="error", seq=error.seq, code=error.code, detail=error.detail)
