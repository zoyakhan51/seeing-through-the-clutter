import csv
import json

import numpy as np
from PIL import Image

from tangle import config
from tangle.paths import DATA


def load_scenes(name,limit=None):
    """Read one dataset and group its rows by scene with the clean version first"""
    folder = DATA/config.DATASETS[name]["folder"]
    with open(folder/"metadata.csv",encoding="utf-8",newline="") as f:
        rows = list(csv.DictReader(f))
    scenes = {}
    for row in rows:
        scenes.setdefault(int(row["scene_id"]),[]).append(row)
    out = []
    for scene_id in sorted(scenes):
        versions = sorted(scenes[scene_id],key=lambda r:int(r["difficulty"]))
        assert versions[0]["version"]=="clean",f"scene {scene_id} has no clean version first"
        out.append({"scene_id":scene_id,"split":versions[0]["split"],"rows":versions})
    return out[:limit] if limit else out


def load_image(name,row):
    """Load one picture as an rgb array of bytes"""
    return np.asarray(Image.open(DATA/config.DATASETS[name]["folder"]/row["file_name"]).convert("RGB"))


def enlarge(image,family="qwen"):
    """Bring a png picture to the size the model reads by repeating pixels"""
    size = config.PROCESSED_SIZE[family]
    if size==config.IMAGE_SIZE:
        return image
    # nearest pixel so every 32 px square becomes exactly one 48 px block and no pixel spills into the next square
    source = np.arange(size)*config.IMAGE_SIZE//size
    return image[source][:,source]


def patch_cells(family="qwen"):
    """Give the row and column on the 16 pixel grid of every patch in the order the processor lists them"""
    if family=="gemma":
        # gemma lists the patches of the 768 px picture row by row
        n = config.PROCESSED_SIZE["gemma"]//config.PATCH
        p = np.arange(n*n)
        return np.stack([p//n,p%n],1)
    # the processor lists patches token by token and each token holds a two by two block of patches
    p = np.arange(config.PATCHES)
    token = p//4
    sub = p%4
    rows = 2*(token//config.GRID)+sub//2
    cols = 2*(token%config.GRID)+sub%2
    return np.stack([rows,cols],1)


def patch_squares(family="qwen"):
    """Give the square that every patch belongs to in the order the processor lists the patches"""
    cells = patch_cells(family)
    side = config.PATCH_SIDE[family]
    return (cells[:,0]//side)*config.GRID+cells[:,1]//side


def changed_cells(clean,image,size):
    """Mark every cell of the given size where at least one pixel differs between two pictures"""
    n = clean.shape[0]//size
    return (clean!=image).any(2).reshape(n,size,n,size).any((1,3))


def bbox_cells(bbox,size):
    """Mark every cell of the given size that overlaps a bounding box"""
    n = config.IMAGE_SIZE//size
    x0,y0,x1,y1 = bbox
    start = np.arange(n)*size
    rows = (start<=y1)&(start+size-1>=y0)
    cols = (start<=x1)&(start+size-1>=x0)
    return rows[:,None]&cols[None,:]


def bbox_patches(bbox,family="qwen"):
    """Mark every 16 pixel patch of the picture the model reads that shows a pixel of a bounding box"""
    size = config.PROCESSED_SIZE[family]
    # the png pixel each pixel of the read picture comes from
    source = np.arange(size)*config.IMAGE_SIZE//size
    x0,y0,x1,y1 = bbox
    rows = ((source>=y0)&(source<=y1)).reshape(-1,config.PATCH).any(1)
    cols = ((source>=x0)&(source<=x1)).reshape(-1,config.PATCH).any(1)
    return rows[:,None]&cols[None,:]


def masks(clean,image,objects,family="qwen"):
    """Split the squares and patches of a picture into object background and clutter parts"""
    cells = patch_cells(family)
    targets = [o for o in objects if o["role"]=="target"]
    changed_sq = changed_cells(clean,image,config.SQUARE).reshape(-1)
    # patches are compared on the picture the model reads because gemma reads it larger than the png
    changed_patch = changed_cells(enlarge(clean,family),enlarge(image,family),config.PATCH)[cells[:,0],cells[:,1]]
    target_sq = np.zeros(config.TOKENS,bool)
    target_patch = np.zeros(len(cells),bool)
    per_target = []
    for o in targets:
        sq = bbox_cells(o["bbox"],config.SQUARE).reshape(-1)
        patch = bbox_patches(o["bbox"],family)[cells[:,0],cells[:,1]]
        target_sq |= sq
        target_patch |= patch
        per_target.append({"id":o["id"],"shape":o["shape"],"squares":np.flatnonzero(sq&~changed_sq),"patches":np.flatnonzero(patch&~changed_patch)})
    return {
        "untouched":~changed_sq,
        "object":target_sq&~changed_sq,
        "background":~target_sq&~changed_sq,
        "clutter":changed_sq,
        "untouched_patch":~changed_patch,
        "object_patch":target_patch&~changed_patch,
        "background_patch":~target_patch&~changed_patch,
        "targets":per_target,
    }


def objects_of(row):
    """Parse the objects column of a metadata row"""
    return json.loads(row["objects"])
