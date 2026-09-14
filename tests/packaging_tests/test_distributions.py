"""Check the actual distributions after `uv build`, including PyPI description text."""

import tarfile
import zipfile
from pathlib import Path, PurePosixPath

PROJECT = Path(__file__).resolve().parents[2]


def _artifact(pattern: str) -> Path:
    artifacts = list((PROJECT / "dist").glob(pattern))
    assert len(artifacts) == 1, "Run uv build with a clean dist directory first"
    return artifacts[0]


def _check_public_text(text: str) -> None:
    for internal_marker in (
        "local review proposal",
        "/path/to/goodmem-langchain",
        "docs/integration-review.md",
        "uncommitted",
        "nothing in this proposal",
    ):
        assert internal_marker not in text.lower(), internal_marker


def test_sdist_contains_distributable_files_and_public_documentation() -> None:
    allowed_roots = {
        "langchain_goodmem",
        "tests",
        "examples",
        ".gitignore",
        "README.md",
        "CHANGELOG.md",
        "LICENSE",
        "pyproject.toml",
        "uv.lock",
        "PKG-INFO",
    }
    with tarfile.open(_artifact("*.tar.gz")) as archive:
        members = {
            str(
                PurePosixPath(member.name).relative_to(
                    PurePosixPath(member.name).parts[0]
                )
            ): member
            for member in archive.getmembers()
            if member.isfile()
        }
        assert {PurePosixPath(name).parts[0] for name in members} <= allowed_roots
        for name in ("README.md", "CHANGELOG.md", "PKG-INFO"):
            stream = archive.extractfile(members[name])
            assert stream is not None
            _check_public_text(stream.read().decode("utf-8"))
        assert "langchain_goodmem/retrievers.py" in members
        readme = archive.extractfile(members["README.md"])
        assert readme is not None and len(readme.read().decode().split()) <= 500


def test_wheel_contains_current_package_and_public_pypi_description() -> None:
    with zipfile.ZipFile(_artifact("*.whl")) as archive:
        for name in archive.namelist():
            root = PurePosixPath(name).parts[0]
            assert root == "langchain_goodmem" or root.endswith(".dist-info"), name
        metadata = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        assert len(metadata) == 1
        _check_public_text(archive.read(metadata[0]).decode("utf-8"))
        sources = list((PROJECT / "langchain_goodmem").rglob("*.py"))
        assert {name for name in archive.namelist() if name.endswith(".py")} == {
            source.relative_to(PROJECT).as_posix() for source in sources
        }
        for source in sources:
            assert (
                archive.read(source.relative_to(PROJECT).as_posix())
                == source.read_bytes()
            )
