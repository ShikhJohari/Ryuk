# Streaming webcam frames from the browser to FastAPI

Issue: [ShikhJohari/Ryuk#5](https://github.com/ShikhJohari/Ryuk/issues/5). Context: [ADR 0001](../adr/0001-python-inference-service-with-react-client.md) already commits Ryuk to a browser client that captures webcam frames and streams them to the FastAPI service over a WebSocket, so the service never touches camera hardware.

## Recommendation

Binary WebSocket frames, not base64 JSON and not WebRTC/aiortc.

The client captures a frame to canvas, encodes it to JPEG with `canvas.toBlob`, and sends the raw bytes over a single WebSocket connection with a small binary header. The server accepts the connection once, keeps a single-slot "latest frame" buffer, and runs the detector plus recognition model in a worker thread per request via `run_in_threadpool` so the event loop stays free to keep draining the socket. When the worker is busy and a newer frame arrives, the older one is dropped, exactly as `FrameProcessor.java` does with `AtomicReference<BufferedImage>`. Target a 10 fps capture rate by default; the drop rule absorbs the rest.

## Message schema

### Client to server: binary frame message

A single WebSocket binary frame, big-endian, no JSON:

| Bytes | Field | Type |
|---|---|---|
| 0 | message type | `uint8` (`0x01` = frame) |
| 1-4 | sequence number | `uint32` |
| 5-12 | capture timestamp (client, ms epoch) | `uint64` |
| 13-14 | width | `uint16` |
| 15-16 | height | `uint16` |
| 17.. | JPEG bytes | raw, from `toBlob`/`Blob.arrayBuffer()` |

The sequence number is the correlation key. The client never has to wait for an ack to know a frame was dropped: gaps in the sequence numbers coming back in results tell it directly, with no extra field needed.

### Server to client: JSON result message

Results are tiny (boxes, a handful of landmark points, a match, a float, a latency number) so a JSON text frame is fine here; there's no base64 cost to pay because there's no image payload going this direction.

```json
{
  "type": "result",
  "seq": 123456,
  "capturedAt": 1732000000123,
  "processedAt": 1732000000181,
  "latencyMs": 54,
  "faces": [
    {
      "box": { "x": 142, "y": 88, "width": 210, "height": 260 },
      "landmarks": [[178, 150], [230, 148], [204, 190], [180, 225], [228, 223]],
      "match": { "personId": "p_0192", "name": "Jane Doe" },
      "similarity": 0.87
    }
  ]
}
```

`match` and `similarity` are `null` when the embedding didn't clear the recognition threshold. `box` and `landmarks` are in the pixel coordinates of the frame named by `width`/`height` in the corresponding request, not the canvas's on-screen size, so the client can scale to whatever it renders at.

## The backpressure rule

Latest-frame-wins, dropping while busy, is the same rule `FrameProcessor.java` uses for the live monitor: `submit()` does `pending.getAndSet(frame)`, and if that returns a non-null previous frame, it counts as dropped (`/Users/shikhar/Projects/crimdet/src/main/java/com/crimdet/service/FrameProcessor.java:77-84`). Capacity is fixed at one. A single worker (`Executors.newSingleThreadExecutor`) drains the slot and re-kicks itself if another frame arrived while it was busy (`runCycle`, lines 133-161).

The FastAPI side needs the same shape, and needs it for the same reason: a receive loop that keeps draining the socket (so the OS/browser send buffer never backs up and frames never grow stale sitting in a buffer), paired with exactly one in-flight inference call. An `asyncio.Queue` isn't quite the right primitive here, since `put()` on a full bounded queue blocks or raises instead of overwriting. A plain mutable slot does the job, and since a single asyncio event loop is single-threaded, no lock is needed around the slot itself, only around the call into the thread pool:

```python
@app.websocket("/ws/frames")
async def frames(websocket: WebSocket):
    await websocket.accept()
    pending: bytes | None = None
    busy = False

    async def worker() -> None:
        nonlocal pending, busy
        while pending is not None:
            frame, pending = pending, None
            busy = True
            try:
                result = await run_in_threadpool(pipeline.process, frame)
                await websocket.send_json(result)
            finally:
                busy = False

    try:
        async for message in websocket.iter_bytes():
            pending = message  # last write wins, same as AtomicReference.getAndSet
            if not busy:
                asyncio.create_task(worker())
    except WebSocketDisconnect:
        pass
```

This is a sketch of the shape, not production code (no error mapping, no sequence-number bookkeeping shown), but it is the direct FastAPI equivalent: `iter_bytes()` is Starlette's async iterator over incoming binary frames ([starlette.dev/websockets](https://www.starlette.dev/websockets/)), and `run_in_threadpool` is FastAPI's re-export of Starlette's helper, which itself is `anyio.to_thread.run_sync` under a `functools.partial` (`starlette/concurrency.py`, [github.com/encode/starlette](https://github.com/encode/starlette/blob/master/starlette/concurrency.py)). That's the whole mechanism: the blocking detector-plus-recognition call moves to a worker thread and gets awaited, so it never blocks the event loop that's still reading off the socket.

One thing worth calling out because it isn't spelled out in Starlette's docs: a single `WebSocket` connection has one underlying ASGI `receive` callable, and the pattern above only ever has one task calling `iter_bytes()`/`receive()` at a time. That's deliberate, not incidental. Reading concurrently from two tasks on the same connection isn't a supported pattern in the ASGI model Starlette implements; the receive loop above is the single reader, and it hands frames off to the worker rather than the worker calling `receive()` itself.

Keep the pipeline on a single worker, not a thread pool sized for parallelism. With the drop rule in place, there is only ever one inference call in flight, which matches `FrameProcessor`'s single-thread executor and sidesteps the question of whether the detector/recognition models are safe to call concurrently from multiple threads (many aren't, without their own internal locking).

## Frame size and rate budget

### Processing budget (CPU pipeline, detector + recognition at 40-80ms/frame)

A single worker's throughput ceiling is `1000 / latency_ms`:

- At 40ms/frame: ceiling is 25 fps. At 15 fps input, that's 600ms of work per 1000ms window, 60% utilization, no drops expected.
- At 80ms/frame: ceiling is 12.5 fps. At 15 fps input, that's 1200ms of work needed per 1000ms window, over 100% utilization: the worker is saturated and roughly one frame in six gets dropped even with nothing going wrong. At 10 fps input it's 800ms of work per 1000ms, 80% utilization, with headroom for jitter.

Given the ticket's own 40-80ms range, 10 fps is the safer default capture rate: it stays under 100% utilization even at the slow end of that range, and lets the drop rule handle transient spikes (a slow frame, a GC pause, model warm-up) instead of being the primary rate limiter for steady-state traffic.

### Byte counts, measured

`getUserMedia` with `{ video: { width: 640, height: 480 } }` hands the client a `MediaStream`; a `<video>` element or `ImageCapture.grabFrame()` gets a frame from it, and a canvas draw plus `toBlob('image/jpeg', 0.7)` encodes it. Numbers below are from actually encoding a 640x480 synthetic frame built to resemble webcam content (gradient background, a skin-toned ellipse standing in for a face, additive sensor noise) through libjpeg at matching quality settings, not estimates:

| Quality | Raw JPEG bytes | KiB |
|---|---|---|
| 0.5 | 16,844 | 16.4 |
| 0.7 | 29,426 | 28.7 |
| 0.9 | 83,322 | 81.4 |
| 1.0 | 290,704 | 283.9 |

At quality 0.7, the size that matters for this comparison:

- Raw JPEG bytes: 29,426.
- Base64 of those bytes: 39,236, a 1.333x expansion, matching the theoretical 4/3 overhead of base64.
- Wrapped in a minimal JSON envelope (`type`, `seq`, `ts`, `width`, `height`, `data`): 39,348 bytes total. The JSON structure itself adds only ~112 bytes; the base64 step is what costs you, not the JSON.
- Binary equivalent (17-byte fixed header, see schema above, plus the raw JPEG): 29,443 bytes.
- Binary saves 9,905 bytes per frame versus base64 JSON at this quality, a 25.2% reduction.

At the target rates:

| Rate | Binary | Base64 JSON |
|---|---|---|
| 10 fps | 294.4 KB/s (2.36 Mbit/s) | 393.5 KB/s (3.15 Mbit/s) |
| 15 fps | 441.6 KB/s (3.53 Mbit/s) | 590.2 KB/s (4.72 Mbit/s) |

For reference, the uncompressed `ImageData` a canvas would hand back for a 640x480 RGBA frame is 1,228,800 bytes, about 1.2MB; JPEG at quality 0.7 is already a ~42x reduction before the transport question even comes up.

None of these numbers matter for correctness on a browser tab talking to a local or LAN FastAPI service; a few hundred KB/s is not a bandwidth problem on that path. What does matter is CPU: base64 encoding on the client and decoding on the server are extra passes over tens of KB per frame, ten to fifteen times a second, on a pipeline whose whole point is to protect the CPU budget for the detector and recognition model. `canvas.toBlob` already returns a `Blob`; taking `Blob.arrayBuffer()` and sending that over the WebSocket as a binary frame means the client never builds a base64 string in the first place, so there's no encode step to skip on that side either. (`toDataURL`, the older canvas API that returns a base64 data URI directly, also runs synchronously on the main thread, which is worse for frame-rate jitter than `toBlob`'s callback; MDN documents `toBlob` as the async, callback-based path. [developer.mozilla.org/.../HTMLCanvasElement/toBlob](https://developer.mozilla.org/en-US/docs/Web/API/HTMLCanvasElement/toBlob))

## Capture: getUserMedia, canvas, and ImageCapture

`navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, frameRate: { ideal: 10, max: 15 } } })` returns a `Promise<MediaStream>`. Width, height, and frame rate are constraints, not guarantees: the browser can pick the closest match it supports and may crop or scale, and an unsatisfiable combination raises `OverconstrainedError`. Camera access requires a secure context (HTTPS or localhost) and a user permission grant. ([developer.mozilla.org/.../MediaDevices/getUserMedia](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia))

Two ways to pull a still frame out of the stream at the target rate:

- Draw the `<video>` element to an off-screen canvas on a timer, then `canvas.toBlob(callback, 'image/jpeg', 0.7)`. `toBlob` is asynchronous (callback-based, returns `undefined`), takes an optional MIME type (defaults to `image/png`) and an optional quality in `[0, 1]` for lossy formats, with the browser choosing a default quality if omitted or out of range. This is the well-supported, works-everywhere path. ([developer.mozilla.org/.../HTMLCanvasElement/toBlob](https://developer.mozilla.org/en-US/docs/Web/API/HTMLCanvasElement/toBlob))
- `ImageCapture.grabFrame()` on the stream's video track returns a `Promise<ImageBitmap>` directly from the track, skipping the `<video>` element and one canvas draw. It still needs a canvas (or `OffscreenCanvas`) to get from `ImageBitmap` to an encoded JPEG, since `ImageCapture` has no built-in encoder for arbitrary quality/format beyond `takePhoto()`'s device-driven photo capture. ([developer.mozilla.org/.../ImageCapture](https://developer.mozilla.org/en-US/docs/Web/API/ImageCapture))

Either is fine at 10-15 fps; `toBlob` off a `<video>` element is the safer default since `ImageCapture` has had uneven browser support historically and the win here (skipping one canvas draw) doesn't matter at this frame rate. Check current support before committing if this becomes load-bearing.

## FastAPI/Starlette WebSocket handling

The relevant Starlette `WebSocket` methods: `accept()`, `receive_bytes()`/`receive_text()`/`receive_json()`, `send_bytes()`/`send_text()`/`send_json()`, `close()`, and the async iterators `iter_bytes()`/`iter_text()`/`iter_json()` that exit cleanly when `WebSocketDisconnect` is raised. FastAPI's WebSocket support is built directly on this. ([starlette.dev/websockets](https://www.starlette.dev/websockets/), [fastapi.tiangolo.com/advanced/websockets](https://fastapi.tiangolo.com/advanced/websockets/))

For keeping blocking work off the event loop: FastAPI's documented rule is that a path operation (or, by the same mechanism, any `def` function you explicitly hand to the threadpool) declared with plain `def` runs in "an external threadpool that is then awaited, instead of being called directly (as it would block the server)" ([fastapi.tiangolo.com/async](https://fastapi.tiangolo.com/async/)). The mechanism underneath is `starlette.concurrency.run_in_threadpool`, which FastAPI re-exports, and which is a one-line wrapper: `functools.partial(func, *args, **kwargs)` awaited via `anyio.to_thread.run_sync` (confirmed from source, `starlette/concurrency.py`, [github.com/encode/starlette](https://github.com/encode/starlette/blob/master/starlette/concurrency.py)). For the frame handler, that means: keep the WebSocket route itself `async def`, and wrap only the call into the detector/recognition pipeline (`pipeline.process(frame_bytes)`, itself an ordinary synchronous function) in `await run_in_threadpool(pipeline.process, frame_bytes)`. AnyIO's default worker thread limiter caps concurrent threadpool calls (commonly cited as 40 by default), which is far more than the single concurrent call this design ever makes given the drop-stale-frame rule.

## Why not WebRTC (aiortc)

Considered, and rejected for now.

aiortc has no built-in signaling server: setting up an `RTCPeerConnection` still means hand-rolling SDP offer/answer and ICE candidate exchange over some other channel, which in practice is a WebSocket anyway. That means adopting WebRTC here would add a second protocol stack (ICE, DTLS-SRTP, RTP jitter buffering, codec negotiation) on top of the WebSocket connection the system already needs for signaling, not instead of it. On the media side, receiving frames means implementing `MediaStreamTrack.recv()` against aiortc's `RTCRtpReceiver`, typically backed by PyAV/ffmpeg bindings for decode, which is a heavier native dependency than encoding JPEG in the browser and reading bytes off a socket. ([aiortc.readthedocs.io/en/latest/api.html](https://aiortc.readthedocs.io/en/latest/api.html), [aiortc.readthedocs.io/en/latest/helpers.html](https://aiortc.readthedocs.io/en/latest/helpers.html))

WebRTC's actual selling points, adaptive bitrate, codec negotiation, jitter buffering, and NAT traversal for peer-to-peer media, solve problems this pipeline doesn't have. It's a single browser tab talking to one FastAPI service, typically on the same machine or LAN; there's no second peer, no public-internet NAT to cross, and per the byte counts above, the transport cost of a 640x480 JPEG at 10-15 fps is a few hundred KB/s, nowhere near where codec compression would earn its complexity back. The actual bottleneck is the 40-80ms CPU cost of the detector and recognition model, which WebRTC does nothing about. Worse, WebRTC's jitter buffer and frame pacing are built to smooth delivery, which is the opposite of the latest-frame-wins drop rule this design wants; getting "always the newest frame, discard the rest" out of a transport designed to avoid discarding is fighting the stack rather than using it.

This could be revisited if Ryuk grows into multi-camera or multi-peer streaming over an unreliable WAN link, where aiortc's congestion control and NAT traversal would start paying for themselves. That isn't the shape of the current problem.

## Modelling the socket lifecycle in Effect

The client is Effect-based, so the natural building blocks are `Scope` for the connection's lifetime, `Queue` for the one-slot backpressure rule, and forked `Fiber`s for the concurrent send/receive loops.

- **Scope owns the connection.** Opening the WebSocket (and the camera track under it) is an `Effect.acquireRelease`: acquire waits for the socket's open event, release calls `.close()` (and stops the `MediaStreamTrack`). `Scope` guarantees the release step runs once acquired, on success, failure, or interruption, in reverse order of acquisition, which is what closing the socket before releasing the camera, or vice versa, needs. ([effect.website/docs/resource-management/scope](https://effect.website/docs/resource-management/scope))
- **A dropping queue is the client-side mirror of `AtomicReference<BufferedImage>`.** `Queue.dropping<Uint8Array>(1)` gives capacity one, and offering into a full dropping queue discards the new value rather than blocking or replacing (note: this is the opposite overwrite behavior from `AtomicReference.getAndSet`, which keeps the newest and drops the old one). For a genuine latest-frame-wins client-side buffer, prefer a plain `Ref<Option<Uint8Array>>` updated with `Ref.set` on every captured frame (always overwrite, matching `getAndSet` exactly); a `Queue.sliding(1)` is the built-in Effect primitive that gives that same keep-newest-drop-oldest behavior if a queue-shaped API is preferred over a bare `Ref`. ([effect.website/docs/concurrency/queue](https://effect.website/docs/concurrency/queue))
- **Two forked fibers, tied to the scope.** A capture fiber runs the `getUserMedia`/canvas/`toBlob` loop and writes into the slot. A sender fiber loops: take the current slot value, send it, then wait for either the correlated result (matched by sequence number) or a timeout before taking again, so the send rate follows the server's actual pace rather than the capture timer. A receiver fiber (or a `Stream` built from the socket's message events) parses incoming JSON results and updates application state. Both are forked with `Effect.forkScoped`, which ties a fiber's lifetime to a scope rather than to its parent fiber, so closing the connection's scope (component unmount) interrupts both without extra bookkeeping. ([effect.website/docs/concurrency/fibers](https://effect.website/docs/concurrency/fibers))
- **Reconnection is `Effect.retry` around the whole scoped connection**, not custom state. This matches the direction Effect's own `@effect/platform` `Socket` module has taken: its WebSocket client was reworked around a pull-based reader with end-to-end backpressure specifically so that "reconnects [are] as simple as `Effect.retry`" (Effect release notes, "This Week in Effect"). If Ryuk's client already depends on `@effect/platform`, its `Socket` module is worth using directly instead of hand-rolling accept/send/receive over the raw `WebSocket` API, since it already has this pull-based, backpressure-aware shape built in.
- **`Stream` is the right mental model for the inbound result feed**, not the outbound frame feed. `Stream<A, E, R>` models zero-or-more values over time and is a reasonable fit for "the sequence of result messages coming back," consumed with normal `Stream` combinators (`Stream.runForEach`, etc.) inside the receiver fiber. The outbound side is better modelled as the slot-plus-sender-fiber pattern above, since a `Stream` on its own doesn't give you the overwrite-not-enqueue semantics the drop rule needs; that's what the `Ref`/`Queue.sliding` sits underneath it for. ([effect.website/docs/stream/introduction](https://effect.website/docs/stream/introduction))

## Sources

- MDN, [`MediaDevices.getUserMedia()`](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia)
- MDN, [`HTMLCanvasElement.toBlob()`](https://developer.mozilla.org/en-US/docs/Web/API/HTMLCanvasElement/toBlob)
- MDN, [`ImageCapture`](https://developer.mozilla.org/en-US/docs/Web/API/ImageCapture)
- FastAPI, [WebSockets](https://fastapi.tiangolo.com/advanced/websockets/)
- FastAPI, [Concurrency and async/await](https://fastapi.tiangolo.com/async/)
- Starlette, [WebSockets](https://www.starlette.dev/websockets/)
- Starlette source, [`starlette/concurrency.py`](https://github.com/encode/starlette/blob/master/starlette/concurrency.py)
- Starlette source, [`starlette/websockets.py`](https://github.com/encode/starlette/blob/master/starlette/websockets.py)
- aiortc, [API reference](https://aiortc.readthedocs.io/en/latest/api.html)
- aiortc, [Helpers](https://aiortc.readthedocs.io/en/latest/helpers.html)
- Effect, [Queue](https://effect.website/docs/concurrency/queue)
- Effect, [Scope](https://effect.website/docs/resource-management/scope)
- Effect, [Fibers](https://effect.website/docs/concurrency/fibers)
- Effect, [Stream introduction](https://effect.website/docs/stream/introduction)
- `crimdet` source, [`FrameProcessor.java`](https://github.com/ShikhJohari/crimdet/blob/main/src/main/java/com/crimdet/service/FrameProcessor.java) (local: `/Users/shikhar/Projects/crimdet/src/main/java/com/crimdet/service/FrameProcessor.java`)
- Ryuk, [ADR 0001: Python inference service with a React client](../adr/0001-python-inference-service-with-react-client.md)

## Caveats

- The byte counts above come from one synthetic 640x480 frame built to resemble webcam content in entropy (gradient plus noise plus a skin-toned ellipse), encoded with Pillow/libjpeg at quality settings matched to the canvas 0-1 scale. Real webcam frames vary with scene detail, lighting, and sensor noise; treat these as representative order-of-magnitude figures, not guarantees for every frame.
- Browsers vary in exactly how `canvas.toBlob`'s quality parameter maps to encoder settings; MDN documents the parameter but not the mapping, so different browsers can produce different byte counts at the same nominal quality.
- The `crimdet` repository was not checked for a public remote; the citation above assumes it mirrors the local path's structure.
- I could not find Starlette or FastAPI documentation that explicitly states a `WebSocket` connection must have only one concurrent reader; that constraint is inferred from the single-`receive`-callable ASGI model Starlette implements around, not quoted from a doc page that says so directly.
