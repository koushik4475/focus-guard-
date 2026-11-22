"""Check that the project's declared runtime and diagnostic dependencies import.

Run after installing requirements.txt:
    python verify_env.py

The command exits non-zero if any required import is unavailable, making it
suitable for local setup checks and automation.
"""
import importlib
from importlib import metadata
import sys
from typing import Callable, List, Tuple

# Distribution name and import module. Keep this list aligned with
# requirements.txt and the optional camera/TTS diagnostic in test_capture.py.
PACKAGES: Tuple[Tuple[str, str], ...] = (
    ("opencv-python", "cv2"),
    ("numpy", "numpy"),
    ("gTTS", "gtts"),
    ("pygame", "pygame"),
    ("mediapipe", "mediapipe"),
    ("scipy", "scipy"),
    ("PyQt6", "PyQt6"),
    ("protobuf", "google.protobuf"),
    ("pyttsx3", "pyttsx3"),
)


def check_dependencies(
    importer: Callable[[str], object] = importlib.import_module,
) -> List[Tuple[str, str]]:
    """Return (distribution, error) pairs for dependencies that cannot import."""
    failures = []
    for distribution, module_name in PACKAGES:
        try:
            module = importer(module_name)
            version = getattr(module, "__version__", None)
            if version is None:
                try:
                    version = metadata.version(distribution)
                except metadata.PackageNotFoundError:
                    version = "installed (version unavailable)"
            print(f"{distribution} (module {module_name}): OK, version={version}")
        except Exception as exc:
            failures.append((distribution, str(exc)))
            print(f"{distribution} (module {module_name}): FAIL, import error: {exc}")
    return failures


def main() -> int:
    print("Python:", sys.executable)
    failures = check_dependencies()
    if failures:
        print(f"\nDependency check failed: {len(failures)} package(s) unavailable.")
        return 1
    print("\nAll required dependencies are available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
