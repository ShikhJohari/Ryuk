/**
 * Compile-time type equality, strict enough to catch a renamed, added or
 * removed field, optional versus required, and readonly drift. Used to pin
 * the hand-written Effect schemas to the types generated from openapi.json.
 */
export type Equals<A, B> =
  (<T>() => T extends A ? 1 : 2) extends <T>() => T extends B ? 1 : 2
    ? true
    : false;

/** Fails to compile unless `T` is exactly `true`. */
export type Assert<T extends true> = T;

/** `T` as one object type, so an intersection compares equal to its flat form. */
export type Simplify<T> = { [K in keyof T]: T[K] };
