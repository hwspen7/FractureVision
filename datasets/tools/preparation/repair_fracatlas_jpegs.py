import csv
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageFile


DATASETS_DIR = Path(__file__).resolve().parents[2]
RAW_ROOT = DATASETS_DIR / "fracatlas_raw" / "FracAtlas"
DATASET_ROOT = DATASETS_DIR / "fracatlas_yolo"

SOURCE_CSV = RAW_ROOT / "dataset.csv"
MANIFEST = DATASET_ROOT / "split_manifest.csv"
REPORT = DATASET_ROOT / "jpeg_repair_report.json"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def jpeg_has_end_marker(path: Path) -> bool:
    if path.stat().st_size < 2:
        return False

    with path.open("rb") as file:
        file.seek(-2, 2)
        return file.read() == b"\xff\xd9"


def strict_validation_error(path: Path) -> str | None:
    ImageFile.LOAD_TRUNCATED_IMAGES = False

    try:
        with Image.open(path) as image:
            image.verify()

        with Image.open(path) as image:
            image.load()

        if not jpeg_has_end_marker(path):
            return "missing JPEG end marker"

    except Exception as error:
        return str(error)

    return None


def repair_jpeg(source: Path, target: Path) -> tuple[int, int]:
    temporary = target.with_name(target.name + ".repairing")

    if temporary.exists():
        raise FileExistsError(
            f"Temporary repair file already exists: {temporary}"
        )

    try:
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        with Image.open(source) as image:
            image.load()
            original_size = image.size

            if image.mode not in {"L", "RGB"}:
                clean_image = image.convert("RGB")
            else:
                clean_image = image.copy()

        clean_image.save(
            temporary,
            format="JPEG",
            quality=100,
            subsampling=0,
        )

        error = strict_validation_error(temporary)
        if error:
            raise RuntimeError(
                f"Repaired image is still invalid: {source}: {error}"
            )

        with Image.open(temporary) as repaired:
            repaired.load()
            repaired_size = repaired.size

        if repaired_size != original_size:
            raise RuntimeError(
                f"Image dimensions changed: {source}: "
                f"{original_size} -> {repaired_size}"
            )

        temporary.replace(target)
        return repaired_size

    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    source_rows = {
        row["image_id"]: row
        for row in read_csv(SOURCE_CSV)
    }
    manifest_rows = read_csv(MANIFEST)

    repaired_images = []
    repaired_by_split = Counter()

    for row in manifest_rows:
        image_id = row["image_id"]
        split = row["split"]

        source_row = source_rows.get(image_id)
        if source_row is None:
            raise KeyError(
                f"Image is missing from dataset.csv: {image_id}"
            )

        category = (
            "Fractured"
            if source_row["fractured"] == "1"
            else "Non_fractured"
        )

        source = RAW_ROOT / "images" / category / image_id
        target = DATASET_ROOT / "images" / split / image_id

        if not source.exists():
            raise FileNotFoundError(f"Missing source image: {source}")

        if not target.exists():
            raise FileNotFoundError(f"Missing output image: {target}")

        error = strict_validation_error(source)
        if error is None:
            continue

        width, height = repair_jpeg(source, target)
        repaired_by_split[split] += 1

        repaired_images.append(
            {
                "image_id": image_id,
                "split": split,
                "source_error": error,
                "width": width,
                "height": height,
            }
        )

    removed_caches = []

    for cache_path in sorted(
        (DATASET_ROOT / "labels").glob("*.cache")
    ):
        cache_path.unlink()
        removed_caches.append(str(cache_path))

    report = {
        "checked_images": len(manifest_rows),
        "repaired_images": len(repaired_images),
        "repaired_by_split": dict(repaired_by_split),
        "removed_yolo_caches": removed_caches,
        "images": repaired_images,
    }

    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(
        {
            "checked_images": report["checked_images"],
            "repaired_images": report["repaired_images"],
            "repaired_by_split": report["repaired_by_split"],
            "removed_cache_count": len(removed_caches),
        },
        ensure_ascii=False,
        indent=2,
    ))

    print(f"\nRepair report: {REPORT}")
    print("Raw FracAtlas files were not modified.")


if __name__ == "__main__":
    main()