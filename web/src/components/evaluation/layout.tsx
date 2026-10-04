import { type ReactNode, useId } from "react";
import { TableCaption } from "@/components/ui/table";
import { cn } from "@/lib/utils";

/** One of the page's sections, a region named by its heading. */
export function EvaluationSection({
  title,
  children,
}: {
  readonly title: string;
  readonly children: ReactNode;
}) {
  const headingId = useId();
  return (
    <section aria-labelledby={headingId} className="flex flex-col gap-6">
      <h2
        id={headingId}
        className="border-ink border-t pt-6 font-normal font-serif text-[28px] text-ink leading-tight"
      >
        {title}
      </h2>
      {children}
    </section>
  );
}

/** A part of a section with a heading of its own. */
export function Subsection({
  title,
  children,
}: {
  readonly title: string;
  readonly children: ReactNode;
}) {
  return (
    <div className="flex flex-col gap-4">
      <h3 className="font-normal font-serif text-[21px] text-ink leading-tight">
        {title}
      </h3>
      {children}
    </div>
  );
}

/** What a section shows until the command that measures it has run. */
export function NotMeasured({ command }: { readonly command: string }) {
  return (
    <p className="text-muted-foreground">
      Not measured yet: run <code className="text-ink">{command}</code>.
    </p>
  );
}

/** A table's caption: "Table N." in ink, then its title and what it holds. */
export function NumberedCaption({
  number,
  children,
}: {
  readonly number: number;
  readonly children: ReactNode;
}) {
  return (
    <TableCaption className="max-w-[80ch]">
      <span className="text-ink">Table {number}.</span> {children}
    </TableCaption>
  );
}

/** A table's or figure's notes, each its own paragraph. */
export function Notes({
  children,
  className,
}: {
  readonly children: ReactNode;
  readonly className?: string;
}) {
  return (
    <ul
      className={cn(
        "flex max-w-[80ch] flex-col gap-1.5 text-muted-foreground text-sm",
        className,
      )}
    >
      {children}
    </ul>
  );
}

/** Prose around the tables, at reading width. */
export function Lede({ children }: { readonly children: ReactNode }) {
  return <p className="max-w-[72ch] text-muted-foreground">{children}</p>;
}

/** A body row's first cell, naming what the row is of: a model, a measure, a group. */
export function RowHeader({
  children,
  className,
  rowSpan,
}: {
  readonly children: ReactNode;
  readonly className?: string;
  /** The rows it heads, when more than one. */
  readonly rowSpan?: number;
}) {
  return (
    <th
      scope={rowSpan === undefined ? "row" : "rowgroup"}
      rowSpan={rowSpan}
      className={cn(
        "whitespace-nowrap px-3 py-2.5 text-left align-middle font-normal text-ink",
        className,
      )}
    >
      {children}
    </th>
  );
}

/** A number column: right-aligned, as the report sets numbers in a column. */
export const numberClass = "text-right";
