import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useId, useState } from "react";
import {
  canBeActive,
  type RecognitionModelInfo,
  setActiveModel,
} from "@/api/models";
import { modelsKey } from "@/api/models.queries";
import type { FrameResult } from "@/api/monitor";
import { formatScore } from "@/lib/format";
import { problemMessage } from "@/lib/problems";
import { runQuery } from "@/lib/runtime";
import { Button } from "./ui/button";
import { Dialog } from "./ui/dialog";

type MonitorToolbarProps = {
  readonly models: ReadonlyArray<RecognitionModelInfo>;
  readonly active: RecognitionModelInfo;
  /** The latest result, whose model and threshold judged the faces on screen. */
  readonly result: FrameResult | null;
  /** Results received in the last second; null while none are coming. */
  readonly framesPerSecond: number | null;
};

/** The operating point: the active model, its threshold and the frame rate achieved. */
export function MonitorToolbar({
  models,
  active,
  result,
  framesPerSecond,
}: MonitorToolbarProps) {
  const selectId = useId();
  const [choice, setChoice] = useState<RecognitionModelInfo | null>(null);
  // Only evaluated models with their weights are offered: nothing runs on an
  // unmeasured threshold.
  const offered = models.filter(canBeActive);
  // Right after a switch, the boxes on screen may still be the old model's.
  const judgedBy =
    models.find((model) => model.id === result?.modelKey) ?? active;
  const threshold = result?.threshold ?? judgedBy.threshold;

  return (
    <div className="flex flex-wrap items-end justify-between gap-6 border-rule border-y py-4">
      <dl className="flex gap-10">
        <Reading label="Active model" value={judgedBy.name} />
        <Reading
          label="Threshold"
          value={threshold === null ? "—" : formatScore(threshold)}
        />
        <Reading
          label="Frame rate"
          value={framesPerSecond === null ? "—" : `${framesPerSecond} fps`}
        />
      </dl>
      {offered.length > 1 ? (
        <div className="flex flex-col gap-1">
          <label htmlFor={selectId} className="section-label">
            Switch model
          </label>
          <select
            id={selectId}
            value={active.id}
            onChange={(event) =>
              setChoice(
                offered.find((model) => model.id === event.target.value) ??
                  null,
              )
            }
            className="h-11 rounded-[22px] border border-ink bg-transparent px-4 text-ink"
          >
            {offered.map((model) => (
              <option key={model.id} value={model.id}>
                {model.name}
              </option>
            ))}
          </select>
        </div>
      ) : null}
      {choice === null ? null : (
        <SwitchModelDialog
          active={active}
          choice={choice}
          onDone={() => setChoice(null)}
        />
      )}
    </div>
  );
}

function Reading({
  label,
  value,
}: {
  readonly label: string;
  readonly value: string;
}) {
  return (
    <div className="flex flex-col gap-1">
      <dt className="section-label">{label}</dt>
      <dd className="font-serif text-[22px] text-ink leading-tight">{value}</dd>
    </div>
  );
}

function SwitchModelDialog({
  active,
  choice,
  onDone,
}: {
  readonly active: RecognitionModelInfo;
  readonly choice: RecognitionModelInfo;
  readonly onDone: () => void;
}) {
  const queryClient = useQueryClient();
  const change = useMutation({
    mutationFn: () => runQuery(setActiveModel(choice.id)),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: modelsKey }),
  });
  const threshold =
    choice.threshold === null
      ? ""
      : ` at its threshold of ${formatScore(choice.threshold)}`;

  return (
    <Dialog
      title={`Switch to ${choice.name}?`}
      // A sighting belongs to one active model, so the service ends every
      // open one when the model changes.
      description={`From the next frame, every face is scored by ${choice.name}${threshold}. Switching ends every open sighting; later ones are logged under ${choice.name}. Nobody needs enrolling again.`}
      onClose={onDone}
      dismissible={!change.isPending}
    >
      {change.error === null ? null : (
        <p role="alert" className="mb-4 text-destructive">
          {problemMessage(change.error)}
        </p>
      )}
      <div className="flex justify-end gap-3">
        <Button
          variant="secondary"
          onClick={onDone}
          disabled={change.isPending}
        >
          Keep {active.name}
        </Button>
        <Button
          // Per call, so it never runs after the toolbar has gone.
          onClick={() => change.mutate(undefined, { onSuccess: onDone })}
          disabled={change.isPending}
        >
          Switch model
        </Button>
      </div>
    </Dialog>
  );
}
