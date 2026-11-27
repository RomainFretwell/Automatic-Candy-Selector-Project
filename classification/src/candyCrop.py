"""
Recadrage automatique d'une photo autour du bonbon (fond clair uni, un seul bonbon).

Principe, sur une version réduite de la photo :
    1. on modélise le fond par une surface douce (degré 2) pour suivre les dégradés d'éclairage ;
    2. un pixel appartient au bonbon s'il s'écarte nettement du fond (couleur ou luminosité) ;
    3. on garde la plus grosse tache proche du centre et on prend son rectangle englobant ;
    4. on découpe un carré autour, pour que le bonbon occupe ~FILL du côté.
Utilisé par importPhotos.py (pas à la prédiction pour l'instant).
"""
from collections import deque

import numpy as np
from PIL import Image, ImageFilter

WORK_WIDTH = 160        # largeur de la version réduite utilisée pour chercher le bonbon
THRESHOLD = 40          # écart minimal au fond (sur 255) pour être considéré comme bonbon
BACKGROUND_TOLERANCE = 20
MIN_PIXELS = 15         # taches plus petites ignorées (poussières, bruit)
FILL = 0.7              # part du côté du carré occupée par le bonbon


def backgroundModel(pixels, h, w):
    # ajuste une surface de degré 2 par canal, en excluant peu à peu les pixels du bonbon
    ys, xs = np.mgrid[0:h, 0:w]
    ys, xs = ys.ravel() / h - 0.5, xs.ravel() / w - 0.5
    basis = np.stack([np.ones_like(xs), xs, ys, xs * xs, ys * ys, xs * ys], axis=1)
    keep = np.ones(len(pixels), bool)
    for _ in range(3):
        coef, *_ = np.linalg.lstsq(basis[keep], pixels[keep], rcond=None)
        keep = np.abs(pixels - basis @ coef).max(axis=1) < BACKGROUND_TOLERANCE
    return (basis @ coef).reshape(h, w, 3)


def candyMask(small):
    # masque booléen des pixels qui s'écartent du fond (luminosité ou saturation)
    a = np.asarray(small, dtype=np.float32)
    h, w, _ = a.shape
    background = backgroundModel(a.reshape(-1, 3), h, w)
    distance = np.abs(a - background).max(axis=2)
    chroma = a.max(axis=2) - a.min(axis=2)
    backgroundChroma = background.max(axis=2) - background.min(axis=2)
    score = np.maximum(distance, 2 * (chroma - backgroundChroma))
    mask = Image.fromarray(((score > THRESHOLD) * 255).astype(np.uint8))
    mask = mask.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))
    return np.asarray(mask) > 0


def blobs(mask):
    # liste des taches (pixels connectés), chacune = liste de (y, x)
    h, w = mask.shape
    seen = np.zeros_like(mask)
    found = []
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        seen[y, x] = True
        queue, blob = deque([(y, x)]), []
        while queue:
            cy, cx = queue.popleft()
            blob.append((cy, cx))
            for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    queue.append((ny, nx))
        found.append(blob)
    return found


def blobScore(blob, h, w):
    # taille de la tache, pénalisée quand elle s'éloigne du centre de la photo
    ys, xs = zip(*blob)
    dy, dx = np.mean(ys) / h - 0.5, np.mean(xs) / w - 0.5
    return len(blob) * (1 - min(1.0, 2 * (dx * dx + dy * dy) ** 0.5))


def findCandy(img):
    # rectangle (x0, y0, x1, y1) du bonbon dans img, ou None si rien trouvé
    small = img.resize((WORK_WIDTH, round(WORK_WIDTH * img.height / img.width)))
    mask = candyMask(small)
    h, w = mask.shape
    candidates = [b for b in blobs(mask) if len(b) >= MIN_PIXELS]
    if not candidates:
        return None
    ys, xs = zip(*max(candidates, key=lambda b: blobScore(b, h, w)))
    k = img.width / w
    return min(xs) * k, min(ys) * k, (max(xs) + 1) * k, (max(ys) + 1) * k


def squareAround(img, box):
    # carré centré sur box (bonbon = FILL du côté), décalé si besoin pour rester dans la photo
    x0, y0, x1, y1 = box
    side = min(max(x1 - x0, y1 - y0) / FILL, img.width, img.height)
    left = min(max((x0 + x1 - side) / 2, 0), img.width - side)
    top = min(max((y0 + y1 - side) / 2, 0), img.height - side)
    return img.crop((round(left), round(top), round(left + side), round(top + side)))


def cropCandy(img):
    # renvoie (carré autour du bonbon, True) ou (carré central, False) si le bonbon n'est pas trouvé
    box = findCandy(img)
    if box is None:
        side = min(img.size)
        return squareAround(img, ((img.width - side) / 2, (img.height - side) / 2,
                                  (img.width + side) / 2, (img.height + side) / 2)), False
    return squareAround(img, box), True
