import cv2
import numpy as np
import os

INPUT_IMAGE = r"C:\Users\narut\Desktop\Diat\sem 1\Robot vision\Assignemnt\Q1\img1.tif"
OUT_DIR     = r"C:\Users\narut\Desktop\Diat\sem 1\Robot vision\Assignemnt\Q1"

BLUR_KSIZE = (7,7)
OPEN_KERNEL_SIZE = (3,3)
CLOSE_KERNEL_SIZE = (5,5)

KMEANS_CLUSTERS = 2
KMEANS_ATTEMPTS = 10

MIN_CONTOUR_AREA = 5

SMALL_MASK_FN   = os.path.join(OUT_DIR, "small_mask.png")
LARGE_MASK_FN   = os.path.join(OUT_DIR, "large_mask.png")
BORDER_MASK_FN  = os.path.join(OUT_DIR, "border_mask.png")

APPLIED_SMALL_FN  = os.path.join(OUT_DIR, "applied_small.png")
APPLIED_LARGE_FN  = os.path.join(OUT_DIR, "applied_large.png")
APPLIED_BORDER_FN = os.path.join(OUT_DIR, "applied_border.png")

def ensure_out_dir(p):
    if not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)

def apply_mask_keep_white(orig_color, mask):
    """Return image where mask==255 keeps original pixel, mask==0 -> white pixel."""
    # create white background
    result = np.full_like(orig_color, 255)  # white background
    result[mask == 255] = orig_color[mask == 255]
    return result
--
ensure_out_dir(OUT_DIR)

# 1) load original image (color + gray)
img_color = cv2.imread(INPUT_IMAGE)
if img_color is None:
    raise FileNotFoundError(f"Original image not found: {INPUT_IMAGE}")
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
H, W = img_gray.shape

# 2) preprocessing: blur -> otsu
blur = cv2.GaussianBlur(img_gray, BLUR_KSIZE, 0)
_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# ensure objects are white
if np.mean(thresh) > 127:
    thresh = cv2.bitwise_not(thresh)

kernel_open = np.ones(OPEN_KERNEL_SIZE, np.uint8)
kernel_close = np.ones(CLOSE_KERNEL_SIZE, np.uint8)
opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)
clean  = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

# 3) connected components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean, connectivity=8)

# 4) find internal blobs for kmeans
valid_areas = []
valid_indices = []
for i in range(1, num_labels):
    x, y, w, h, area = stats[i]
    border = (x==0 or y==0 or x+w>=W or y+h>=H)
    if not border:
        valid_areas.append(area)
        valid_indices.append(i)

# fallback if not enough internal blobs
if len(valid_areas) < 2:
    valid_areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
    valid_indices = list(range(1, num_labels))

if len(valid_areas) == 0:
    raise RuntimeError("No components found to classify.")

# 5) dynamic threshold via k-means
data_pts = np.float32(valid_areas).reshape(-1,1)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
_, _, centers = cv2.kmeans(
    data_pts, KMEANS_CLUSTERS, None, criteria,
    KMEANS_ATTEMPTS, cv2.KMEANS_RANDOM_CENTERS
)
dynamic_threshold = float(np.mean(centers))
print("Dynamic area threshold (pixels):", dynamic_threshold)

# 6) build masks
small_mask  = np.zeros((H, W), np.uint8)
large_mask  = np.zeros_like(small_mask)
border_mask = np.zeros_like(small_mask)

for i in range(1, num_labels):
    x, y, w, h, area = stats[i]
    comp_pixels = (labels == i)
    # border check
    if x==0 or y==0 or x+w>=W or y+h>=H:
        border_mask[comp_pixels] = 255
    else:
        # dynamic classification
        if area < dynamic_threshold:
            small_mask[comp_pixels] = 255
        else:
            large_mask[comp_pixels] = 255

# 7) save masks
cv2.imwrite(SMALL_MASK_FN,  small_mask)
cv2.imwrite(LARGE_MASK_FN,  large_mask)
cv2.imwrite(BORDER_MASK_FN, border_mask)
print("Masks saved:")
print(" -", SMALL_MASK_FN, "nonzero:", int(np.count_nonzero(small_mask)))
print(" -", LARGE_MASK_FN, "nonzero:", int(np.count_nonzero(large_mask)))
print(" -", BORDER_MASK_FN, "nonzero:", int(np.count_nonzero(border_mask)))

# 8) apply masks to original: keep pixel where mask==255, else white
applied_small = apply_mask_keep_white(img_color, small_mask)
applied_large = apply_mask_keep_white(img_color, large_mask)
applied_border = apply_mask_keep_white(img_color, border_mask)

cv2.imwrite(APPLIED_SMALL_FN, applied_small)
cv2.imwrite(APPLIED_LARGE_FN, applied_large)
cv2.imwrite(APPLIED_BORDER_FN, applied_border)

print("Applied-mask images saved:")
print(" -", APPLIED_SMALL_FN)
print(" -", APPLIED_LARGE_FN)
print(" -", APPLIED_BORDER_FN)
print("Done.")