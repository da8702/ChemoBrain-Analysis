#!/usr/bin/env python3
"""
Convert Leica *_RAW_ch02.tif images → QuickNII-ready PNGs
and generate a matching XML series file:

* Down-scales any dimension > max_side (default 4000 px)
* Rotates each slice by --rotate degrees clockwise (default 90)
* Writes minimal QuickNII XML:
  <?xml …?>
  <series name='PREFIX_series'>
      <slice filename='…' nr='1' width='W' height='H'/>
      …
  </series>
"""
import os, re, subprocess, argparse, shutil, sys, xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image   # pip install pillow

# ───────────────────────────────────────────────────────────
def magick_convert(src: Path, dst: Path, max_side=4000, rotate=90):
    cmd = [
        "magick", str(src),
        "-auto-level",
        "-contrast-stretch", "0.3%",
        "-depth", "8",
        "-colorspace", "sRGB",
        "-resize", f"{max_side}x{max_side}>",
    ]
    if rotate:
        cmd += ["-rotate", str(rotate)]
    cmd += ["-strip", "-quality", "92", f"PNG24:{dst}"]
    subprocess.run(cmd, check=True)

# ───────────────────────────────────────────────────────────
def prep_series(folder, prefix, max_side=4000, rotate=90):
    folder = Path(folder)

    if not shutil.which("magick"):
        sys.exit("ImageMagick not found — install with:  brew install imagemagick")

    patt = re.compile(rf"^{re.escape(prefix)}.*_RAW_ch02\.tif$", re.I)
    tifs = sorted([f for f in folder.iterdir() if patt.match(f.name)])
    if not tifs:
        raise FileNotFoundError(f"No {prefix}*_RAW_ch02.tif files in {folder}")

    out_dir = folder / f"{prefix}_QuickNII_PNG"
    out_dir.mkdir(exist_ok=True)

    slices = []
    for idx, tif in enumerate(tifs, start=1):
        sec = f"s{idx:03d}"
        png_name = f"{tif.stem}_{sec}.png"
        dst = out_dir / png_name
        magick_convert(tif, dst, max_side=max_side, rotate=rotate)

        with Image.open(dst) as im:
            w, h = im.size
        slices.append((png_name, idx, w, h))

    series_el = ET.Element("series", {"name": f"{prefix}_series"})
    for fname, nr, w, h in slices:
        ET.SubElement(series_el, "slice", {
            "filename": fname,
            "nr": str(nr),
            "width": str(w),
            "height": str(h)
        })

    if hasattr(ET, "indent"):
        ET.indent(series_el, space="    ")
    xml_path = out_dir / f"{prefix}_series.xml"
    ET.ElementTree(series_el).write(xml_path, encoding="utf-8", xml_declaration=True)

    print("Converted images →", out_dir)
    print("QuickNII XML    →", xml_path)

# ───────────────────────────────────────────────────────────
def prep_tilescan_series(folder, base_prefix, max_side=4000, rotate=90):
    """
    Process all GoodCis<number>...Merging.tif images in the folder (not ...Merging_RAW_ch02.tif),
    grouping by unique GoodCis<number> prefix. Prints debug info for matching.
    """
    folder = Path(folder)

    if not shutil.which("magick"):
        sys.exit("ImageMagick not found — install with:  brew install imagemagick")

    # Regex to match GoodCis<number>...Merging.tif, but not ...Merging_RAW_ch02.tif
    patt = re.compile(rf"^({re.escape(base_prefix)}\d+).*Merging\.tif$", re.I)
    exclude_patt = re.compile(r"Merging_RAW_ch02\.tif$", re.I)

    prefix_to_files = {}
    print(f"\n[DEBUG] Checking files in {folder} for pattern: {base_prefix}<number>...Merging.tif\n")
    for f in folder.iterdir():
        if not f.is_file():
            continue
        if exclude_patt.search(f.name):
            print(f"[DEBUG] Excluded (RAW_ch02): {f.name}")
            continue
        m = patt.match(f.name)
        if m:
            prefix = m.group(1)
            prefix_to_files.setdefault(prefix, []).append(f)
            print(f"[DEBUG] MATCH: {f.name}  (prefix: {prefix})")
        else:
            print(f"[DEBUG] No match: {f.name}")

    if not prefix_to_files:
        raise FileNotFoundError(f"No {base_prefix}<number>*Merging.tif files in {folder}")

    for prefix, tifs in prefix_to_files.items():
        tifs = sorted(tifs)
        out_dir = folder / f"{prefix}_QuickNII_PNG"
        out_dir.mkdir(exist_ok=True)

        slices = []
        for idx, tif in enumerate(tifs, start=1):
            sec = f"s{idx:03d}"
            png_name = f"{tif.stem}_{sec}.png"
            dst = out_dir / png_name
            magick_convert(tif, dst, max_side=max_side, rotate=rotate)

            with Image.open(dst) as im:
                w, h = im.size
            slices.append((png_name, idx, w, h))

        series_el = ET.Element("series", {"name": f"{prefix}_series"})
        for fname, nr, w, h in slices:
            ET.SubElement(series_el, "slice", {
                "filename": fname,
                "nr": str(nr),
                "width": str(w),
                "height": str(h)
            })

        if hasattr(ET, "indent"):
            ET.indent(series_el, space="    ")
        xml_path = out_dir / f"{prefix}_series.xml"
        ET.ElementTree(series_el).write(xml_path, encoding="utf-8", xml_declaration=True)

        print(f"Converted images for {prefix} →", out_dir)
        print(f"QuickNII XML for {prefix}    →", xml_path)

# ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Prep Leica *_RAW_ch02.tif images for QuickNII or tilescan Merging.tif images.")
    ap.add_argument("folder",  help="Folder with image files")
    ap.add_argument("prefix",  help="File prefix (GoodControl, etc.)")
    ap.add_argument("--max_side", type=int, default=4000,
                    help="Longest allowed pixel dimension (default 4000)")
    ap.add_argument("--rotate", type=int, choices=[0, 90, 180, 270],
                    default=90, help="Rotate each slice clockwise (default 90°)")
    ap.add_argument("--tilescan", action="store_true",
                    help="Process tilescan Merging.tif images instead of *_RAW_ch02.tif")
    args = ap.parse_args()

    if args.tilescan:
        prep_tilescan_series(args.folder, args.prefix,
                             max_side=args.max_side, rotate=args.rotate)
    else:
        prep_series(args.folder, args.prefix,
                    max_side=args.max_side, rotate=args.rotate)
        