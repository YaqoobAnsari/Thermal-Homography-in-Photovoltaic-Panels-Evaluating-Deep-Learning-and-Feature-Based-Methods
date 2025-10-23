"""
Thermal Dataset Preprocessor (keep ALL steps, 16-bit output, clean rebuild)
---------------------------------------------------------------------------
- Removes existing Preprocessed Images folder before processing (clean rebuild)
- Mirrors day/time/height structure; assigns folders round-robin to heights
- Keeps ALL steps:
    1) Robust normalization/stretch (per-image by default, optional global fixed)
    2) Glare suppression
    3) CLAHE (8-bit domain; mapped back to 16-bit)
    4) Bilateral filter
    5) Shadow correction (brightness-preserving, outlier-tamed)
    6) Sharpen
- Writes exactly ONE file per input: enhanced_<original>.TIFF (uint16)

Usage (best default):
    python preprocess_dataset.py --no-confirm

Global fixed stretch (consistent look):
    python preprocess_dataset.py --no-confirm --use-fixed-stretch --fixed-low 0.10 --fixed-high 0.35
"""

from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import shutil
import logging
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------- I/O helpers ----------------
def _to_gray(img: np.ndarray) -> np.ndarray:
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] >= 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def read_tiff_any(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    if img is None:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read {path}")
    img = _to_gray(img)

    # Normalize to uint16 domain as input baseline
    if img.dtype == np.uint16:
        return img
    if img.dtype == np.uint8:
        return (img.astype(np.uint16) << 8)
    f = img.astype(np.float32)
    mn, mx = float(f.min()), float(f.max())
    if mx <= mn + 1e-12:
        return np.zeros_like(img, dtype=np.uint16)
    f = (f - mn) / (mx - mn)
    return (np.clip(f, 0.0, 1.0) * 65535.0).astype(np.uint16)

def find_tiffs(folder: Path):
    pats = ['*.tif', '*.tiff', '*.TIF', '*.TIFF']
    out = []
    for p in pats:
        out += list(folder.glob(p))
    if out:
        return out
    # sometimes raws live under 'enhanced' subfolder
    enh = folder / 'enhanced'
    if enh.exists():
        tmp = []
        for p in pats:
            tmp += list(enh.glob(p))
        return tmp
    return []


# ---------------- Processing core ----------------
def _odd_between(v: int, lo: int, hi: int) -> int:
    v = max(lo, min(hi, v))
    if v % 2 == 0:
        v += 1 if v < hi else -1
    return v

class ThermalPreprocessorFull:
    """
    Full pipeline in float [0..1], with CLAHE in 8-bit domain (round-trip):
      1) robust stretch (per-image OR fixed global)
      2) glare suppression
      3) CLAHE (8-bit) -> map back to float
      4) bilateral filter (8-bit domain used internally; mapped back)
      5) shadow correction (brightness-preserving)
      6) sharpen
      -> uint16 output
    """

    def __init__(self,
                 stretch_low: float = 1.0,
                 stretch_high: float = 99.5,
                 use_fixed_stretch: bool = False,
                 fixed_low_value: float = None,
                 fixed_high_value: float = None,
                 clahe_clip: float = 0.018,
                 clahe_tiles=(8, 8),
                 bilateral_d: int = 9,
                 bilateral_sigma_color: float = 0.015,  # fraction of 255 used for 8-bit
                 bilateral_sigma_space: int = 8):
        self.p_low = stretch_low
        self.p_high = stretch_high
        self.use_fixed = use_fixed_stretch
        self.fixed_lo = fixed_low_value
        self.fixed_hi = fixed_high_value
        self.clahe_clip = clahe_clip
        self.clahe_tiles = clahe_tiles
        self.bilat_d = bilateral_d
        self.bilat_sig_color = bilateral_sigma_color
        self.bilat_sig_space = bilateral_sigma_space

    # ----- stage 1: robust normalization/stretch -----
    def _robust_stretch(self, f: np.ndarray) -> np.ndarray:
        if self.use_fixed and self.fixed_lo is not None and self.fixed_hi is not None:
            lo, hi = float(self.fixed_lo), float(self.fixed_hi)
        else:
            lo = float(np.percentile(f, self.p_low))
            hi = float(np.percentile(f, self.p_high))
        if hi <= lo + 1e-12:
            mn, mx = float(f.min()), float(f.max())
            if mx <= mn + 1e-12:
                return np.zeros_like(f)
            lo, hi = mn, mx
        return np.clip((f - lo) / (hi - lo), 0.0, 1.0)

    # ----- stage 2: glare suppression (float) -----
    def _suppress_glare(self, f: np.ndarray) -> np.ndarray:
        mean, std = float(f.mean()), float(f.std())
        if std < 1e-8:
            return f
        thr = mean + 2.0 * std
        g = np.clip(f, 0.0, thr)
        m = float(g.max())
        return g / m if m > 1e-12 else g

    # ----- stage 3: CLAHE (8-bit) -----
    def _clahe_8bit(self, f: np.ndarray) -> np.ndarray:
        u8 = (np.clip(f, 0.0, 1.0) * 255.0).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=self.clahe_tiles)
        u8 = clahe.apply(u8)
        return u8.astype(np.float32) / 255.0

    # ----- stage 4: bilateral (8-bit kernel) -----
    def _bilateral(self, f: np.ndarray) -> np.ndarray:
        u8 = (np.clip(f, 0.0, 1.0) * 255.0).astype(np.uint8)
        u8 = cv2.bilateralFilter(
            u8,
            d=self.bilat_d,
            sigmaColor=int(self.bilat_sig_color * 255.0),
            sigmaSpace=self.bilat_sig_space
        )
        return u8.astype(np.float32) / 255.0

    # ----- stage 5: shadow correction (float) -----
    def _shadow_correct(self, f: np.ndarray) -> np.ndarray:
        h, w = f.shape
        k = _odd_between(max(h, w) // 20, 15, 101)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        bg = cv2.morphologyEx(f, cv2.MORPH_OPEN, kernel)
        bg_mean = float(bg.mean())
        if bg_mean < 1e-4:
            return f
        eps = 1e-6
        out = f / (bg + eps)
        out *= bg_mean  # preserve global brightness

        # tame extreme highs
        p99 = float(np.percentile(out, 99.5))
        if p99 > 1e-6:
            out = np.clip(out / p99, 0.0, 1.0)
            out *= min(1.0, bg_mean / max(p99, 1e-6))
        return np.clip(out, 0.0, 1.0)

    # ----- stage 6: sharpen (float) -----
    def _sharpen(self, f: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(f, (0, 0), 2.0)
        # mild unsharp mask to avoid ringing on thermal texture
        out = 1.5 * f - 0.5 * blurred
        return np.clip(out, 0.0, 1.0)

    # ----- full pipeline -----
    def preprocess(self, img_u16: np.ndarray) -> np.ndarray:
        f = img_u16.astype(np.float32) / 65535.0
        f = self._robust_stretch(f)
        f = self._suppress_glare(f)
        f = self._clahe_8bit(f)
        f = self._bilateral(f)
        f = self._shadow_correct(f)
        f = self._sharpen(f)
        return (np.clip(f, 0.0, 1.0) * 65535.0).astype(np.uint16)


# ---------------- Main driver ----------------
def main():
    parser = argparse.ArgumentParser(description="Thermal preprocessor (16-bit, full steps, clean rebuild)")
    parser.add_argument("--source", type=str, default=r"D:\kuyfavsksuyvsakuvcsa\Dataset")
    parser.add_argument("--target", type=str, default=r"D:\kuyfavsksuyvsakuvcsa\Preprocessed Images")
    parser.add_argument("--date-prefix", type=str, default="2024-12")

    # Stretch controls
    parser.add_argument("--use-fixed-stretch", action="store_true",
                        help="Use global fixed stretch instead of per-image percentiles.")
    parser.add_argument("--fixed-low", type=float, default=None, help="Fixed low in float [0..1].")
    parser.add_argument("--fixed-high", type=float, default=None, help="Fixed high in float [0..1].")
    parser.add_argument("--stretch-low", type=float, default=1.0, help="Per-image lower percentile.")
    parser.add_argument("--stretch-high", type=float, default=99.5, help="Per-image upper percentile.")

    # Flow control
    parser.add_argument("--no-confirm", action="store_true")
    args = parser.parse_args()

    src = Path(args.source)
    dst = Path(args.target)

    if not src.exists():
        print(f"ERROR: {src} not found")
        return

    print("=" * 70)
    print("DATASET PREPROCESSOR (16-bit, full steps, clean rebuild)")
    print("=" * 70)
    print(f"Source: {src}")
    print(f"Target: {dst}")
    print("Existing 'Preprocessed Images' folder will be DELETED and rebuilt.")
    if args.use_fixed_stretch:
        print(f"Global stretch: [{args.fixed_low}, {args.fixed_high}]")
    else:
        print(f"Per-image stretch percentiles: low={args.stretch_low}, high={args.stretch_high}")

    if not args.no_confirm:
        resp = input("\nProceed? (yes/no): ").strip().lower()
        if resp != "yes":
            print("Cancelled.")
            return

    # --- Clean rebuild: remove old folder then create fresh ---
    if dst.exists():
        logger.info("Deleting existing Preprocessed Images folder...")
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    # Preprocessor with chosen stretch mode
    tp = ThermalPreprocessorFull(
        stretch_low=args.stretch_low,
        stretch_high=args.stretch_high,
        use_fixed_stretch=args.use_fixed_stretch,
        fixed_low_value=args.fixed_low,
        fixed_high_value=args.fixed_high,
    )

    heights = ["10cm", "20cm", "30cm", "40cm"]
    total, processed, failed = 0, 0, 0

    # Build day->time->folder map and process
    for day_folder in sorted(src.iterdir()):
        if not day_folder.is_dir() or not day_folder.name.startswith(args.date_prefix):
            continue

        for time_folder in sorted(day_folder.iterdir()):
            if not time_folder.is_dir():
                continue

            # Gather subfolders that contain tiffs
            folders = []
            for sub in sorted(time_folder.iterdir()):
                if not sub.is_dir():
                    continue
                tiffs = find_tiffs(sub)
                if tiffs:
                    folders.append((sub, tiffs))

            if not folders:
                continue

            # Assign folders round-robin to heights and process
            for idx, (sub_folder, tiffs) in enumerate(folders):
                height = heights[idx % len(heights)]
                out_dir = dst / day_folder.name / time_folder.name / height
                out_dir.mkdir(parents=True, exist_ok=True)

                logger.info(f"Processing {day_folder.name}/{time_folder.name}/{sub_folder.name} "
                            f"→ {height} ({len(tiffs)} images)")
                total += len(tiffs)

                for t in tqdm(tiffs, desc=f"  {sub_folder.name}", leave=False):
                    try:
                        img16 = read_tiff_any(t)
                        out_path = out_dir / f"enhanced_{t.name}"
                        proc16 = tp.preprocess(img16)
                        ok = cv2.imwrite(str(out_path), proc16)
                        if not ok:
                            failed += 1
                            logger.error(f"Failed to write: {out_path}")
                        else:
                            processed += 1
                    except Exception as e:
                        failed += 1
                        logger.error(f"Failed {t.name}: {e}")

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"Total images found:        {total}")
    print(f"Successfully processed:    {processed}")
    print(f"Failed:                    {failed}")
    print(f"Output root:               {dst}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
