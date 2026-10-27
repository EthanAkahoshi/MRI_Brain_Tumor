"""Turn IXI T1 volumes (healthy volunteers) into 2D axial PNG slices.

Reads datasets/IXI/download/IXI-T1.tar, writes datasets/IXI/slices/*.png and
datasets/IXI/slices.csv (path, subject, slice). Slices from one subject are
near-duplicates of each other, so keep them grouped by subject when splitting.
Source: https://brain-development.org/ixi-dataset/ (CC BY-SA 3.0).
"""
import csv
import os
import tarfile

import cv2
import nibabel as nib
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
TAR = os.path.join(ROOT, 'datasets', 'IXI', 'download', 'IXI-T1.tar')
VOLUMES = os.path.join(ROOT, 'datasets', 'IXI', 'T1')
SLICES = os.path.join(ROOT, 'datasets', 'IXI', 'slices')
# Depths below the top of the head, in mm. IXI volumes include the neck, so "widest
# cross-section" lands on the neck/skull base; measured from the crown, 30-60 mm is
# the cerebrum and ventricles, the level most tumour training slices show.
DEPTHS_MM = (30, 45, 60)


def extract():
    os.makedirs(VOLUMES, exist_ok=True)
    with tarfile.open(TAR) as tar:
        tar.extractall(VOLUMES, filter='data')  # 'data' rejects absolute paths and '..'


def axial_slices(path):
    image = nib.as_closest_canonical(nib.load(path))  # RAS: x = right, y = anterior, z = superior
    data = np.asarray(image.dataobj, dtype=np.float32)
    vx, vy, vz = image.header.get_zooms()[:3]
    top = np.percentile(data, 99.5)
    mask = data > 0.1 * top
    areas = mask.sum(axis=(0, 1))
    crown = int(np.flatnonzero(areas >= 0.05 * areas.max())[-1])  # highest slice with real head area
    picks = [crown - round(d / vz) for d in DEPTHS_MM]
    picks = [k for k in picks if 0 <= k < data.shape[2]]

    for k in picks:
        s = np.flipud(data[:, :, k].T)  # rows = anterior at top, columns = x
        s = np.clip(s / top, 0, 1) * 255
        h, w = s.shape
        # Square pixels: scale by voxel size so the head keeps its true shape
        size = (round(w * vx / min(vx, vy)), round(h * vy / min(vx, vy)))
        yield int(k), cv2.resize(s.astype(np.uint8), size, interpolation=cv2.INTER_AREA)


def main():
    if not os.path.isdir(VOLUMES) or not os.listdir(VOLUMES):
        extract()
    os.makedirs(SLICES, exist_ok=True)
    rows = []
    for name in sorted(os.listdir(VOLUMES)):
        if not name.endswith('.nii.gz'):
            continue
        subject = name.split('-')[0]  # e.g. IXI002
        try:
            for k, png in axial_slices(os.path.join(VOLUMES, name)):
                out = os.path.join(SLICES, f'{subject}_z{k:03d}.png')
                cv2.imwrite(out, png)
                rows.append({'path': os.path.relpath(out, ROOT).replace(os.sep, '/'), 'subject': subject, 'slice': k})
        except Exception as error:  # one unreadable volume shouldn't stop the rest
            print('skipped', name, error)
    with open(os.path.join(ROOT, 'datasets', 'IXI', 'slices.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['path', 'subject', 'slice'])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} slices from {len({r['subject'] for r in rows})} subjects")


if __name__ == '__main__':
    main()
