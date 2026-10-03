import {
  createContext,
  type ReactNode,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Button } from "./button";

/**
 * How long a toast stays, in ms: long enough to read it and reach its
 * action. The time stops while the pointer is on the toast or focus is in
 * it, and starts afresh when both have left.
 */
export const TOAST_DURATION_MS = 10_000;

export type ToastAction = {
  readonly label: string;
  readonly onAction: () => void;
};

export type ToastOptions = {
  readonly message: string;
  /** Such as Undo; taking it also dismisses the toast. */
  readonly action?: ToastAction;
  /**
   * What the toast is about, such as a person of interest's ID, so it can be
   * dismissed when that changes: an Undo for someone since purged.
   */
  readonly tag?: string;
};

type Toasts = {
  /** Shows a toast in place of any other. */
  readonly show: (toast: ToastOptions) => void;
  /**
   * Dismisses the toast showing, its action not taken: any toast, or only
   * one with `tag`.
   */
  readonly dismiss: (tag?: string) => void;
};

type ShownToast = ToastOptions & { readonly id: number };

const ToastContext = createContext<Toasts | null>(null);

/** Raises toasts from anywhere under a `ToastProvider`. */
export function useToast(): Toasts {
  const toasts = useContext(ToastContext);
  if (toasts === null) {
    throw new Error("useToast must be used inside a ToastProvider");
  }
  return toasts;
}

/**
 * Brief news of something done, such as "Removed Ada Lovelace." with Undo,
 * one at a time along the bottom of the page. The status region is always
 * there, so a screen reader announces each toast as it appears.
 */
export function ToastProvider({ children }: { readonly children: ReactNode }) {
  const [toast, setToast] = useState<ShownToast | null>(null);
  const lastId = useRef(0);

  const show = useCallback((options: ToastOptions) => {
    lastId.current += 1;
    setToast({ ...options, id: lastId.current });
  }, []);
  const dismiss = useCallback(
    (tag?: string) =>
      setToast((shown) =>
        tag === undefined || shown?.tag === tag ? null : shown,
      ),
    [],
  );
  // Only that toast: a newer one may already have replaced it.
  const dismissOne = useCallback(
    (id: number) => setToast((shown) => (shown?.id === id ? null : shown)),
    [],
  );
  const toasts = useMemo(() => ({ show, dismiss }), [show, dismiss]);

  return (
    <ToastContext value={toasts}>
      {children}
      <div
        role="status"
        className="pointer-events-none fixed inset-x-0 bottom-6 z-50 flex justify-center px-6"
      >
        {toast === null ? null : (
          <ToastCard
            // A new toast is a new card, so its time starts afresh.
            key={toast.id}
            toast={toast}
            onDismiss={dismissOne}
          />
        )}
      </div>
    </ToastContext>
  );
}

function ToastCard({
  toast,
  onDismiss,
}: {
  readonly toast: ShownToast;
  readonly onDismiss: (id: number) => void;
}) {
  const card = useRef<HTMLDivElement>(null);
  // What had focus before the operator moved into the toast.
  const returnFocus = useRef<HTMLElement | null>(null);
  const [hovered, setHovered] = useState(false);
  const [focused, setFocused] = useState(false);
  const { id, message, action } = toast;

  useEffect(() => {
    if (hovered || focused) {
      return;
    }
    const timer = setTimeout(() => onDismiss(id), TOAST_DURATION_MS);
    return () => clearTimeout(timer);
  }, [hovered, focused, id, onDismiss]);

  const close = () => {
    // Focus in the toast would drop to the page as it goes.
    const back = returnFocus.current;
    if (card.current?.contains(document.activeElement) && back?.isConnected) {
      back.focus();
    }
    onDismiss(id);
  };

  return (
    // Hover and focus pause the timer; the keys it handles belong to its buttons.
    // biome-ignore lint/a11y/noStaticElementInteractions: see above
    <div
      ref={card}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      onFocus={(event) => {
        const from = event.relatedTarget;
        if (!event.currentTarget.contains(from)) {
          returnFocus.current = from instanceof HTMLElement ? from : null;
        }
        setFocused(true);
      }}
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget)) {
          setFocused(false);
        }
      }}
      onKeyDown={(event) => {
        if (event.key === "Escape") {
          close();
        }
      }}
      className="pointer-events-auto flex max-w-[560px] items-center gap-4 rounded-md border border-ink bg-ink py-2 pr-2 pl-5 text-paper"
    >
      <p>{message}</p>
      <div className="flex items-center gap-1">
        {action === undefined ? null : (
          <Button
            variant="secondary"
            className="h-9 border-paper px-4 text-paper hover:bg-paper/15 focus-visible:outline-paper"
            onClick={() => {
              // Gone first, so the action runs once even if it throws.
              close();
              action.onAction();
            }}
          >
            {action.label}
          </Button>
        )}
        <button
          type="button"
          aria-label="Dismiss"
          onClick={close}
          className="flex size-9 items-center justify-center rounded-full text-paper/80 hover:bg-paper/15 hover:text-paper focus-visible:outline-2 focus-visible:outline-paper focus-visible:outline-offset-2"
        >
          <span aria-hidden="true">×</span>
        </button>
      </div>
    </div>
  );
}
