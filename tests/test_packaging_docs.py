"""Documentation links must work in both checkouts and unpacked source archives."""

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit


def test_readme_and_linked_documents_have_local_targets():
    """Omitting a linked document from the sdist breaks this packaged test."""
    root = Path(__file__).resolve().parents[1]
    pending = [root / "README.md"]
    visited = set()
    missing = []
    while pending:
        document = pending.pop()
        if document in visited:
            continue
        visited.add(document)
        for destination in re.findall(r"\]\(([^)]+)\)", document.read_text()):
            url = urlsplit(destination)
            if url.scheme or url.netloc or not url.path:
                continue
            target = (document.parent / unquote(url.path)).resolve()
            if not target.exists():
                missing.append(f"{document.relative_to(root)}: {destination}")
            elif target.suffix == ".md":
                pending.append(target)
    assert not missing, "Broken local documentation links:\n" + "\n".join(missing)
