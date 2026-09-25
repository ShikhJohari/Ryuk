type PageHeaderProps = {
  /** Optional small uppercase label above the title. */
  readonly label?: string;
  readonly title: string;
  readonly description: string;
};

export function PageHeader({ label, title, description }: PageHeaderProps) {
  return (
    <header>
      {label === undefined ? null : (
        <p className="section-label mb-2">{label}</p>
      )}
      <h1 className="font-normal font-serif text-[38px] text-ink leading-[1.1]">
        {title}
      </h1>
      <p className="mt-3 max-w-[64ch] text-muted-foreground">{description}</p>
    </header>
  );
}
