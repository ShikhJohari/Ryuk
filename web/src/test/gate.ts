/**
 * Holds a mocked response until the test opens it, so the test can act while
 * a request is in flight.
 */
export function gate() {
  let open: () => void = () => undefined;
  const opened = new Promise<void>((resolve) => {
    open = resolve;
  });
  return { opened, open };
}
