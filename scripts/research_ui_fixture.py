"""Ephemeral browser-test server. All submitted judgments are test fixtures."""

import json
import tempfile
from pathlib import Path

import uvicorn

from ecocompanion.benchmark.generation import generate_canonical_cases
from ecocompanion.research.annotation import AnnotationStore
from ecocompanion.research.api import annotation_app
from ecocompanion.research.identity import IdentityMap, observable_case

if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="ecotriage-TEST-ONLY-") as directory:
        case = observable_case(generate_canonical_cases()[0], IdentityMap())
        store, tokens = AnnotationStore.create(
            Path(directory) / "TEST-ONLY.sqlite3", (case,), manual_digest="TEST-ONLY"
        )
        # Captured only by the browser-test parent, never a real pilot credential.
        print(json.dumps(tokens), flush=True)
        uvicorn.run(annotation_app(store), host="127.0.0.1", port=8766, access_log=False)
