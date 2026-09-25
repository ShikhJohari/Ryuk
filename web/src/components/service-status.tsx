import { useQuery } from "@tanstack/react-query";
import { healthQueryOptions } from "@/api/health.queries";
import { cn } from "@/lib/utils";

function StatusDot({ className }: { readonly className: string }) {
  return (
    <span
      aria-hidden="true"
      className={cn("inline-block size-2 shrink-0 rounded-full", className)}
    />
  );
}

/** Whether the service answers its health check, polled in the background. */
export function ServiceStatus() {
  const health = useQuery(healthQueryOptions);

  return (
    <output aria-live="polite" className="flex items-center gap-2 text-sm">
      {health.isPending ? (
        <>
          <StatusDot className="bg-field" />
          <span className="text-muted-foreground">Checking service</span>
        </>
      ) : health.isError ? (
        <>
          <StatusDot className="bg-destructive" />
          <span>Service unreachable</span>
        </>
      ) : (
        <>
          <StatusDot className="bg-ok" />
          <span>Service connected</span>
          <span className="text-muted-foreground">v{health.data.version}</span>
        </>
      )}
    </output>
  );
}
