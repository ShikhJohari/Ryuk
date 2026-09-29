import { notFound } from "@tanstack/react-router";
import { isNotFound } from "./problems";

/**
 * A route loader's data, awaited: the service's `not_found` becomes the
 * router's not-found page, and any other failure the route's error page.
 */
export async function orNotFound<A>(loading: Promise<A>): Promise<A> {
  try {
    return await loading;
  } catch (error) {
    throw isNotFound(error) ? notFound() : error;
  }
}
