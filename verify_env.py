"""verify_env.py
Simple script to verify that required packages import and print versions.
Run with the project's venv active or via the configured interpreter.
"""
import sys
import importlib

packages = [
    ("opencv-python", "cv2"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("playsound", "playsound"),
    ("gtts", "gtts"),
    ("dlib", "dlib"),
    ("imutils", "imutils"),
]

print("Python:", sys.executable)

for pkg_name, modname in packages:
    try:
        mod = importlib.import_module(modname)
        ver = getattr(mod, "__version__", None)
        if ver is None:
            # try importlib.metadata
            try:
                import importlib.metadata as md
            except Exception:
                import importlib_metadata as md
            try:
                ver = md.version(pkg_name)
            except Exception:
                ver = "unknown"
        print(f"{pkg_name} (module {modname}): OK, version={ver}")
    except Exception as e:
        print(f"{pkg_name} (module {modname}): FAIL, import error: {e}")

print("\nDone.")
