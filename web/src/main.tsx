import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { lazy, Suspense } from "react";

const App = location.pathname.startsWith("/research")
  ? lazy(async () => {
      await import("./styles.css");
      return import("./App");
    })
  : location.pathname.startsWith("/games")
    ? lazy(() => import("./games/GameApp"))
    : lazy(() => import("./pet/PetApp"));

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Suspense fallback={null}><App /></Suspense>
  </StrictMode>,
);

