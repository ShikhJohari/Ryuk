import { describe, expect, it } from "vitest";
import { encodeFrame } from "./monitor";

const hex = (buffer: ArrayBuffer) =>
  [...new Uint8Array(buffer)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");

describe("encodeFrame", () => {
  it("writes the header the service unpacks with struct '>BIQHH', then the JPEG", () => {
    const message = encodeFrame({
      seq: 0x01020304,
      capturedAt: 1_732_000_000_123,
      width: 640,
      height: 360,
      jpeg: new Uint8Array([0xff, 0xd8]).buffer,
    });

    // struct.pack(">BIQHH", 1, 0x01020304, 1732000000123, 640, 360).hex()
    expect(hex(message)).toBe(`${"010102030400000193433ea87b02800168"}ffd8`);
  });
});
