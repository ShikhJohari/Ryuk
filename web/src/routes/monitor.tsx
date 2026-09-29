import { useQueryClient, useSuspenseQuery } from "@tanstack/react-query";
import { createFileRoute } from "@tanstack/react-router";
import { type ReactNode, useRef } from "react";
import type { RecognitionModelInfo } from "@/api/models";
import { modelsKey, modelsQueryOptions } from "@/api/models.queries";
import { FaceOverlay } from "@/components/face-overlay";
import { MonitorToolbar } from "@/components/monitor-toolbar";
import { PageHeader } from "@/components/page-header";
import { Button } from "@/components/ui/button";
import {
  type DisconnectReason,
  type LiveMonitorStatus,
  useLiveMonitor,
} from "@/hooks/use-live-monitor";
import type { CameraProblem } from "@/lib/camera";

export const Route = createFileRoute("/monitor")({
  loader: ({ context }) =>
    context.queryClient.ensureQueryData(modelsQueryOptions),
  component: LiveMonitorPage,
});

function LiveMonitorPage() {
  const models = useSuspenseQuery(modelsQueryOptions).data;
  const active = models.find((model) => model.state === "active");

  return (
    <section className="flex flex-col gap-6">
      <PageHeader
        title="Live monitor"
        description="Your webcam, with every face boxed and scored against the watchlist by the active model."
      />
      {active === undefined ? (
        <Notice title="No recognition model can be active">
          <NoActiveModel />
        </Notice>
      ) : (
        <LiveMonitor models={models} active={active} />
      )}
    </section>
  );
}

function LiveMonitor({
  models,
  active,
}: {
  readonly models: ReadonlyArray<RecognitionModelInfo>;
  readonly active: RecognitionModelInfo;
}) {
  const queryClient = useQueryClient();
  const video = useRef<HTMLVideoElement>(null);
  const monitor = useLiveMonitor({
    video,
    onActiveModelChanged: () =>
      void queryClient.invalidateQueries({ queryKey: modelsKey }),
  });
  const { status, result } = monitor;
  const aspectRatio =
    result === null ? "4 / 3" : `${result.width} / ${result.height}`;

  return (
    <>
      <MonitorToolbar
        models={models}
        active={active}
        result={result}
        framesPerSecond={monitor.framesPerSecond}
      />
      <div
        className="relative w-full max-w-[960px] overflow-hidden rounded-md bg-ink"
        style={{ aspectRatio }}
      >
        <video
          ref={video}
          muted
          playsInline
          aria-label="Webcam"
          className="absolute inset-0 size-full object-contain"
        />
        {status.kind === "running" && result !== null ? (
          <FaceOverlay result={result} />
        ) : null}
        <StatusNotice status={status} onRestart={monitor.restart} />
      </div>
    </>
  );
}

function StatusNotice({
  status,
  onRestart,
}: {
  readonly status: LiveMonitorStatus;
  readonly onRestart: () => void;
}) {
  switch (status.kind) {
    case "running":
      return null;
    case "starting":
      return <VideoNotice title="Starting the camera…" />;
    case "no_camera":
      return (
        <VideoNotice title="No camera">
          <p>{cameraMessage(status.problem)}</p>
        </VideoNotice>
      );
    case "no_active_model":
      return (
        <VideoNotice title="No recognition model can be active">
          <NoActiveModel />
        </VideoNotice>
      );
    case "superseded":
      return (
        <VideoNotice title="The live monitor moved to another tab">
          <p>Only one live monitor runs at a time, so this one stopped.</p>
          <Button variant="secondary" onClick={onRestart}>
            Monitor here
          </Button>
        </VideoNotice>
      );
    case "stalled":
      return (
        <VideoNotice title="The service stopped answering">
          <p>
            No result has come back for a while, so nothing on screen would be
            current. Results show again as soon as it answers; if they do not,
            check the service, then reconnect.
          </p>
          <Button variant="secondary" onClick={onRestart}>
            Reconnect
          </Button>
        </VideoNotice>
      );
    case "camera_lost":
      return (
        <VideoNotice title="The camera stopped">
          <p>
            It was disconnected, turned off, or access to it was withdrawn, so
            nothing is being sent. Check it, then try again.
          </p>
          <Button variant="secondary" onClick={onRestart}>
            Try again
          </Button>
        </VideoNotice>
      );
    case "disconnected": {
      const { title, body } = disconnectMessage(status.reason);
      return (
        <VideoNotice title={title}>
          <p>{body}</p>
          <Button variant="secondary" onClick={onRestart}>
            Reconnect
          </Button>
        </VideoNotice>
      );
    }
  }
}

function disconnectMessage(reason: DisconnectReason): {
  readonly title: string;
  readonly body: string;
} {
  switch (reason) {
    case "unreachable":
      return {
        title: "Could not connect to the service",
        body: "Check that the service is running, and that this page is open at 127.0.0.1 or localhost: the service refuses a live monitor from anywhere else. Then reconnect.",
      };
    case "lost":
      return {
        title: "Lost the connection to the service",
        body: "Check that the service is running, then reconnect.",
      };
    case "refused":
      return {
        title: "The service refused the live monitor",
        body: "It only accepts one opened from a page on this machine. Open Ryuk at 127.0.0.1 or localhost, then reconnect.",
      };
    case "frame_too_large":
      return {
        title: "The service refused a frame as too large",
        body: "Frames are normally far under its limit. Reconnect; if it happens again, check the camera's resolution.",
      };
    case "service_error":
      return {
        title: "The service hit an unexpected error",
        body: "Check the service's log, then reconnect.",
      };
    case "invalid_url":
      return {
        title: "The service's address is not valid",
        body: "VITE_API_BASE_URL is not a URL the live monitor can connect to. Fix it, rebuild the client, then reconnect.",
      };
  }
}

function cameraMessage(problem: CameraProblem): string {
  switch (problem) {
    case "denied":
      return "Camera access was refused. Allow it in the browser's site settings, then reload the page.";
    case "missing":
      return "No camera was found. Connect one, then reload the page.";
    case "busy":
      return "The camera is in use by another application. Close it, then reload the page.";
    case "unsupported":
      return "This browser cannot open a camera here. Use a current browser on localhost.";
  }
}

function NoActiveModel() {
  return (
    <p>
      The live monitor needs an evaluated recognition model with its weights on
      this machine, and none has them. Fetch the weights with{" "}
      <code>ryuk weights fetch</code> and restart the service.
    </p>
  );
}

/** A state shown in place of the video. */
function VideoNotice({
  title,
  children,
}: {
  readonly title: string;
  readonly children?: ReactNode;
}) {
  return (
    <div
      role="status"
      // An ink focus ring would be invisible on the ink stage.
      className="absolute inset-0 flex flex-col items-start justify-center gap-4 bg-ink p-10 text-paper [&_button]:border-paper [&_button]:text-paper [&_button]:focus-visible:outline-paper"
    >
      <h2 className="font-serif text-[26px] leading-tight">{title}</h2>
      {children === undefined ? null : (
        <div className="flex max-w-[56ch] flex-col items-start gap-4 text-paper/80">
          {children}
        </div>
      )}
    </div>
  );
}

function Notice({
  title,
  children,
}: {
  readonly title: string;
  readonly children: ReactNode;
}) {
  return (
    <div
      role="status"
      className="flex max-w-[64ch] flex-col gap-3 border-rule border-y py-6"
    >
      <h2 className="font-serif text-[26px] text-ink leading-tight">{title}</h2>
      <div className="text-muted-foreground">{children}</div>
    </div>
  );
}
