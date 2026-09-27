/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Where the service is; unset, requests stay relative and go through the Vite proxy. */
  readonly VITE_API_BASE_URL?: string;
}
