import time
print('Importing dependencies...')
s_import = time.time()

import os
import math
import gc
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings("ignore")

# === Utility Functions ===

def decimal_2_dec(decimal):
    sign = '+' if decimal >= 0 else '-'
    deg = math.floor(abs(decimal))
    minute = math.floor((abs(decimal) - deg) * 60)
    sec = ((abs(decimal) - deg) * 60 - minute) * 60
    return f"{sign}{int(deg)}°{int(minute)}'{sec:.2f}\""

def decimal_2_ra(decimal):
    hours = int(decimal / 15)
    minutes = int((decimal / 15 - hours) * 60)
    seconds = (decimal / 15 - hours - minutes/60)*3600
    return f"{hours:02d}:{minutes:02d}:{seconds:06.2f}"

def ask_yes_no(question):
    while True:
        ans = input(f"{question} (yes/no): ").strip().lower()
        if ans in ('yes','y'): return True
        if ans in ('no','n'): return False
        print("Please answer yes or no.")

# === Import Timer End ===
e_import = time.time()
print(f"Dependencies imported. Took {e_import - s_import:.2f} seconds.")

# === User Input ===

nbands = int(input("How many bands to image? "))
bands_info = {}
for i in range(nbands):
    name = input(f"Name of band #{i+1} (e.g., '115','Y'): ").strip()
    path = input(f"Full path to FITS file for band '{name}': ").strip()
    hdr_ext = int(input(f"  → Header extension number for '{name}': "))
    data_ext = int(input(f"  → Data extension number for '{name}': "))
    bands_info[name] = {'path': path, 'hdr_ext': hdr_ext, 'data_ext': data_ext}

catalog_path = input("Path to catalog file (csv/tsv) to use: ").strip()
sep = '\t' if catalog_path.lower().endswith('tsv') else ','
catalog = pd.read_csv(catalog_path, sep=sep)
assert all(col in catalog.columns for col in ['RA','DEC','ID']), \
    "Catalog must contain columns: RA, DEC, ID"
print(f"Catalog loaded: {len(catalog)} entries.")

do_search = ask_yes_no("Perform search/filter on catalog?")
if do_search:
    ncrit = int(input("How many criteria to apply? "))
    for _ in range(ncrit):
        col = input("  → Column: ").strip()
        lo = float(input(f"    Min {col}: "))
        hi = float(input(f"    Max {col}: "))
        catalog[col] = pd.to_numeric(catalog[col], errors='coerce')
        catalog = catalog[(catalog[col] >= lo) & (catalog[col] <= hi)]
    print(f"Filtering complete. {len(catalog)} remain.")

        # === Masks ===

    use_existing = ask_yes_no("Have masks already been created?")
    s_mask = time.time()
    masks = {}
    mask_dir = os.path.join('images', 'masks')
    os.makedirs(mask_dir, exist_ok=True)

    for band, info in bands_info.items():
        base = os.path.splitext(os.path.basename(info['path']))[0]
        mask_path = os.path.join(mask_dir, f"{base}_mask.fits")
        
        if use_existing and os.path.exists(mask_path):
            masks[band] = fits.getdata(mask_path)
        else:
            data = fits.getdata(info['path'], ext=info['data_ext'])
            mask = (data > 0).astype(np.uint8)
            hdr = fits.getheader(info['path'], ext=info['hdr_ext'])
            hdr['EXTNAME'] = 'MASK'
            fits.writeto(mask_path, mask, hdr, overwrite=True)
            masks[band] = mask
            print(f"Mask saved to {mask_path}")

    e_mask = time.time()
    print(f"Mask load/creation complete. Took {e_mask - s_mask:.2f} seconds.")

    mask_cut = int(input("Mask cutout radius in pixels: "))

    # === Band Info Prep ===

    band_info = {
        band: {
            'wcs': WCS(fits.getheader(info['path'], ext=info['hdr_ext'])),
            'shape': fits.getdata(info['path'], ext=info['data_ext']).shape,
            'mask': masks[band],
        }
        for band, info in bands_info.items()
    }

    # === Filtering Valid Objects ===

    _global_band_info = None
    _global_mask_cut = None

    def init_worker(binfo, mcut):
        global _global_band_info, _global_mask_cut
        _global_band_info = binfo
        _global_mask_cut = mcut

    def is_valid_object(row):
        ra, dec = row['RA'], row['DEC']
        for band, info in _global_band_info.items():
            wcs, shape, mask = info['wcs'], info['shape'], info['mask']
            #print(f'RA:{ra}\tDEC:{dec}')
            xpix, ypix = wcs.all_world2pix(ra, dec, 0)
            #print(f'x_bef:{xpix}\ty_bef:{ypix}')
            x, y = int(round(xpix.item())), int(round(ypix.item()))
            #print(f'x:{x}\ty:{y}')
            ylo, yhi = y - _global_mask_cut, y + _global_mask_cut
            xlo, xhi = x - _global_mask_cut, x + _global_mask_cut
            if ylo < 0 or yhi >= shape[0] or xlo < 0 or xhi >= shape[1]:
                #print(f"Bounds fail: {(xlo,xhi,ylo,yhi)} vs shape {shape}")
                return False
            if not np.all(mask[ylo:yhi+1, xlo:xhi+1]):
                #print(f"Mask fail at ({x},{y}) in {band}")
                return False
        return True

    if len(catalog) > 10000:
        with Pool(processes=cpu_count(), initializer=init_worker, initargs=(band_info, mask_cut)) as pool:
            results = pool.map(is_valid_object, [row for _, row in catalog.iterrows()])
        filtered_catalog = catalog[results]
    else:
        _global_band_info = band_info
        _global_mask_cut = mask_cut
        filtered_catalog = catalog[catalog.apply(is_valid_object, axis=1)]

    print(f"Object filtering complete. {len(filtered_catalog)} remain.")

    if ask_yes_no("Save filtered catalog?"):
        os.makedirs('objects', exist_ok=True)
        fname = input("Filename under 'objects/': ").strip()
        filtered_catalog.to_csv(os.path.join('objects', fname), index=False)

# === Imaging ===

user_input = input("How many objects to image? (type 'all'): ").strip()
to_process = filtered_catalog if user_input == 'all' else filtered_catalog.head(int(user_input))

imsave_fits = ask_yes_no("Save FITS cutouts?")
make_collage = ask_yes_no("Generate collage images?")
use_pixels = ask_yes_no("Use pixel cutout size?")
radius = int(input("Edge half-width in pixels: ")) if use_pixels else None
scale = float(input("Plate scale (arcsec/pixel): ")) if not use_pixels else None
s_image = time.time()
os.makedirs('output/FITS_Cutouts', exist_ok=True)
os.makedirs('output/Collages', exist_ok=True)

for _, row in to_process.iterrows():
    ra, dec, objid = row['RA'], row['DEC'], row['ID']
    rpix = radius if use_pixels else int(round(row['rad'] / scale * 1.5))
    imgs, wcs_dict = {}, {}

    for band, info in bands_info.items():
        data = fits.getdata(info['path'], ext=info['data_ext'])
        imgs[band] = data
        wcs = band_info[band]['wcs']
        wcs_dict[band] = wcs

    if make_collage:
        fig, axes = plt.subplots(1, len(imgs), figsize=(4*len(imgs), 4))
        if len(imgs) == 1: axes = [axes]
        for ax, (band, img) in zip(axes, imgs.items()):
            w = wcs_dict[band]
            xpix, ypix = w.all_world2pix(ra, dec, 1)
            x, y = int(round(xpix.item())), int(round(ypix.item()))
            stamp = img[y-rpix:y+rpix, x-rpix:x+rpix]
            ax.imshow(stamp, origin='lower', cmap='gray')
            ax.set_title(f"{objid}_{band}")
        plt.tight_layout()
        plt.savefig(f"output/Collages/{objid}_collage.png")
        plt.close()

    if imsave_fits:
        for band, img in imgs.items():
            w = wcs_dict[band]
            xpix, ypix = w.all_world2pix(ra, dec, 1)
            x, y = int(round(xpix.item())), int(round(ypix.item()))
            stamp = img[y-rpix:y+rpix, x-rpix:x+rpix]
            fits.writeto(f"output/FITS_Cutouts/{objid}_{band}.fits", stamp, overwrite=True)
e_image = time.time()
print(f"Image processing complete. Took {e_image - s_image:.2f} seconds.")
print("All tasks completed.")