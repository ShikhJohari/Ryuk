import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import type { EnrollmentWarning, WarningCode } from "@/api/problem";
import { warningsOf } from "@/lib/problems";

type Held<V> = {
  readonly input: V;
  readonly warnings: ReadonlyArray<EnrollmentWarning>;
};

/**
 * An upload that the service may answer with `409 warnings`. The warnings
 * are held, with the upload that raised them, for a dialog whose confirm
 * resends that same upload with every listed code acknowledged.
 */
export function useAcknowledgedMutation<V, T>({
  mutationFn,
  onSuccess,
}: {
  readonly mutationFn: (
    input: V,
    acknowledgedWarnings: ReadonlyArray<WarningCode>,
  ) => Promise<T>;
  readonly onSuccess: (result: T) => void | Promise<void>;
}) {
  const [held, setHeld] = useState<Held<V> | null>(null);
  const mutation = useMutation({
    mutationFn: ({
      input,
      acknowledged,
    }: {
      readonly input: V;
      readonly acknowledged: ReadonlyArray<WarningCode>;
    }) => mutationFn(input, acknowledged),
    onSuccess: async (result) => {
      setHeld(null);
      await onSuccess(result);
    },
    onError: (error, { input }) => {
      const warnings = warningsOf(error);
      setHeld(warnings === null ? null : { input, warnings });
    },
  });

  return {
    /** Any failure other than warnings, for an inline message. */
    error: held === null ? mutation.error : null,
    warnings: held?.warnings ?? null,
    isPending: mutation.isPending,
    submit: (input: V) => mutation.mutate({ input, acknowledged: [] }),
    acknowledge: () => {
      if (held !== null) {
        mutation.mutate({
          input: held.input,
          acknowledged: held.warnings.map((warning) => warning.code),
        });
      }
    },
    dismiss: () => {
      setHeld(null);
      mutation.reset();
    },
    reset: mutation.reset,
  };
}
