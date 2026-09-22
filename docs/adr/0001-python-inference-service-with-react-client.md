---
status: accepted
---

# Python inference service with a React client

The course requires the machine learning work in Python, and Shikhar wants a React front end. Ryuk is therefore a FastAPI service that owns every mark-bearing piece (data, models, pipeline, evaluation, persistence) and a separate TanStack Router client that only renders what the service returns. The browser captures webcam frames and streams them to the service over a WebSocket, so the service never touches camera hardware and can be tested as a pure function of frames.

## Considered options

A single Streamlit app was the cheaper path and was rejected for the client. A server-side OpenCV webcam with MJPEG streaming was rejected because it only works on the machine running the service.

## Consequences

Two toolchains (uv and pnpm) and two CI jobs. Any logic that appears in the client is a smell, since the rubric does not credit it.
