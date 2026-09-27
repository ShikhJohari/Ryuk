import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useId, useState } from "react";
import {
  canBeActive,
  type RecognitionModelInfo,
  setActiveModel,
} from "@/api/models";
import { modelsKey } from "@/api/models.queries";
import { formatScore } from "@/lib/format";
import { problemMessage } from "@/lib/problems";
import { runQuery } from "@/lib/runtime";
import { Button } from "./ui/button";
import { Dialog } from "./ui/dialog";

type MonitorToolbarProps = {
  readonly models: ReadonlyArray<RecognitionModelInfo>;
  readonly active: RecognitionModelInfo;
  /** Results received in the last second; null before the first. */
  readonly framesPerSecond: number | null;
};

/** The operating point: the active model, its threshold and the frame rate achieved. */
export function MonitorToolbar({
  models,
  active,
  framesPerSecond,
}: MonitorToolbarProps) {
  const selectId = useId();
  const [choice, setChoice] = useState<RecognitionModelInfo | null>(null);
  // Only evaluated models are offered: nothing runs on an unmeasured threshold.
  const evaluated = models.filter(canBeActive);

  return (
    <div className="flex flex-wrap items-end justify-between gap-6 border-rule border-y py-4">
      <dl className="flex gap-10">
        <Reading label="Active model" value={active.name} />
        <Reading
          label="Threshold"
          value={
            active.threshold === null ? "—" : formatScore(active.threshold)
          }
        />
        <Reading
          label="Frame rate"
          value={framesPerSecond === null ? "—" : `${framesPerSecond} fps`}
        />
      </dl>
      {evaluated.length > 1 ? (
        <div className="flex flex-col gap-1">
          <label htmlFor={selectId} className="section-label">
            Switch model
          </label>
          <select
            id={selectId}
            value={active.id}
            onChange={(event) =>
              setChoice(
                evaluated.find((model) => model.id === event.target.value) ??
                  null,
              )
            }
            className="h-11 rounded-[22px] border border-ink bg-transparent px-4 text-ink"
          >
            {evaluated.map((model) => (
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
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: modelsKey });
      onDone();
    },
  });
  const threshold =
    choice.threshold === null
      ? ""
      : ` at its threshold of ${formatScore(choice.threshold)}`;

  return (
    <Dialog
      title={`Switch to ${choice.name}?`}
      description={`From the next frame, every face is scored by ${choice.name}${threshold}. Nobody needs enrolling again.`}
      onClose={onDone}
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
        <Button onClick={() => change.mutate()} disabled={change.isPending}>
          Switch model
        </Button>
      </div>
    </Dialog>
  );
}
