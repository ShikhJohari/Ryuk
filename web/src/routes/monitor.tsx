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
    case "disconnected":
      return (
        <VideoNotice title="Lost the connection to the service">
          <p>Check that the service is running, then reconnect.</p>
          <Button variant="secondary" onClick={onRestart}>
            Reconnect
          </Button>
        </VideoNotice>
      );
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
      className="absolute inset-0 flex flex-col items-start justify-center gap-4 bg-ink p-10 text-paper [&_button]:border-paper [&_button]:text-paper"
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
