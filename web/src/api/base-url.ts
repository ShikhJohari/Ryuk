/**
 * The service's base URL for what the browser loads itself, outside
 * `ApiClient`: enrolled photos and the live monitor's socket. The same
 * `VITE_API_BASE_URL` `ApiClientLive` reads; empty means this page's origin.
 */
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";
