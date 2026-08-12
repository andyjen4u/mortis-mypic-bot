import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { DecisionDashboard } from "./DecisionDashboard";
import "./globals.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <DecisionDashboard />
  </StrictMode>,
);
