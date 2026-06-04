import pathlib
import sys

def normalize_label_file(label_path: pathlib.Path) -> None:
    """Replace any class ID in a YOLO label file with 0.

    Each line in a YOLO label file is of the form:
        <class_id> <cx> <cy> <w> <h>
    This function rewrites the file so that <class_id> is always 0.
    """
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except Exception as e:
        print(f"[WARN] Unable to read {label_path}: {e}", file=sys.stderr)
        return

    new_lines = []
    for line in lines:
        if not line.strip():
            continue  # skip empty lines
        parts = line.split()
        if len(parts) < 5:
            # malformed line, keep as‑is but warn
            print(f"[WARN] Malformed line in {label_path}: '{line}'", file=sys.stderr)
            new_lines.append(line)
            continue
        # Replace class id with 0
        parts[0] = "0"
        new_lines.append(" ".join(parts))

    try:
        label_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    except Exception as e:
        print(f"[ERROR] Failed to write {label_path}: {e}", file=sys.stderr)

def main():
    # Determine the path to the dataset directory (ml/license-plates-ec-combined)
    # __file__ is .../ml/training/normalize_labels.py
    # parents[1] gives the 'ml' directory; then we append the dataset name.
    root = pathlib.Path(__file__).resolve().parents[1] / "license-plates-ec-combined"
    label_dirs = [root / "train" / "labels", root / "valid" / "labels", root / "test" / "labels"]
    for ld in label_dirs:
        if not ld.is_dir():
            print(f"[INFO] No label directory: {ld}")
            continue
        for label_file in ld.glob("*.txt"):
            normalize_label_file(label_file)
    print("Normalisation complete.")

if __name__ == "__main__":
    main()
