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
 *
 * A request in flight cannot be taken back: while `isPending`, `reset` and
 * `dismiss` do nothing, and the dialogs using this disable their controls.
 */
export function useAcknowledgedMutation<V, T>({
  mutationFn,
  onSuccess,
  onSuccessWhileMounted,
}: {
  readonly mutationFn: (
    input: V,
    acknowledgedWarnings: ReadonlyArray<WarningCode>,
  ) => Promise<T>;
  /**
   * Runs on every success, even once the component has gone, since the
   * change was made: keep caches true here.
   */
  readonly onSuccess?: (result: T) => void | Promise<void>;
  /**
   * Runs on success only while the component is still mounted, after
   * `onSuccess` has settled: close, navigate.
   */
  readonly onSuccessWhileMounted?: (result: T) => void;
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
      await onSuccess?.(result);
    },
    onError: (error, { input }) => {
      const warnings = warningsOf(error);
      setHeld(warnings === null ? null : { input, warnings });
    },
  });
  const send = (input: V, acknowledged: ReadonlyArray<WarningCode>) =>
    mutation.mutate(
      { input, acknowledged },
      // Per call, so it never runs after the component has unmounted.
      { onSuccess: onSuccessWhileMounted },
    );
  const reset = () => {
    if (!mutation.isPending) {
      mutation.reset();
    }
  };

  return {
    /** Any failure other than warnings, for an inline message. */
    error: held === null ? mutation.error : null,
    warnings: held?.warnings ?? null,
    isPending: mutation.isPending,
    submit: (input: V) => send(input, []),
    acknowledge: () => {
      if (held !== null) {
        send(
          held.input,
          held.warnings.map((warning) => warning.code),
        );
      }
    },
    dismiss: () => {
      if (!mutation.isPending) {
        setHeld(null);
        mutation.reset();
      }
    },
    reset,
  };
}
