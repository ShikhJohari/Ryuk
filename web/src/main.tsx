import { QueryClient } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./app";
import { createAppRouter } from "./router";
import "./styles.css";

const queryClient = new QueryClient();
const router = createAppRouter({ queryClient });

const rootElement = document.getElementById("root");
if (rootElement === null) {
  throw new Error("index.html is missing the #root element");
}

createRoot(rootElement).render(
  <StrictMode>
    <App queryClient={queryClient} router={router} />
  </StrictMode>,
);
