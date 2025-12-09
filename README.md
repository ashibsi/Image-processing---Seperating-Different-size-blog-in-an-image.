Change the file name and location according to your need.

Methodology used:
 
1.	Load images
•	 Read the original color image (img1.tif) and also a grayscale copy (for processing).
•	Ensure file/paths exist and image was read correctly.

2.	Preprocessing (noise reduction)
•	Apply a Gaussian blur to the grayscale image (e.g. kernel (7,7)) to reduce sensor noise and small speckle that would fragment components.

3.	Global thresholding (binarization)
•	Use Otsu’s method to compute an automatic threshold and convert the blurred image to a binary image.
•	Check polarity: if the binary has mostly white pixels (background white), invert it so objects become white and background black. This makes contour/component detection straightforward.

4.	Morphological cleaning
•	Apply a morphological open (small kernel) to remove tiny isolated noise.
•	Apply a morphological close (slightly larger kernel) to fill small holes inside blobs and smooth object outlines.
•	Result: clean binary mask ready for connected component analysis.





5.	Connected component analysis
•	Run connected Components with Stats on the cleaned binary to get:
o	a labeled image (labels)
o	per-component stats (bounding box x,y,w,h and area)
o	component centroids
•	Skip label 0 (background).

6.	Border rejection
•	For each component, check if its bounding box touches any image edge (x==0, y==0, x+w>=W, y+h>=H).
•	If it touches, mark it as border (rejected from internal clustering) and assign it to border_mask.

7.	Collect areas for internal components
•	Collect area of every internal (non-border) component into a list valid areas.
•	If there are fewer than 2 internal components, fall back to using all component areas (so k-means still runs).

8.	Dynamic thresholding via k-means
•	Cluster valid areas into 2 clusters using k-means (k=2).
•	Compute the dynamic area threshold as the mean of the two cluster centers.

9.	Classify components
•	For each non-border component:
o	If area < dynamic threshold → mark pixel locations into small mask.
o	Else → mark into large mask.
•	Border components were already written into border mask.

10.	Save masks
•	Write three binary masks to disk:
o	small_mask.png (255 where small components are)
o	large_mask.png (255 where large components are)
o	border_mask.png (255 where border-touching components are)
•	
11.	Apply mask to original image (keep texture)
•	For each mask, create a color output that keeps original pixels where the mask is white and makes all other pixels white:
o	applied_small.png, applied_large.png, applied_border.png
•	Implementation: result = np full like (orig_color, 255); result[mask==255] = orig_color[mask==255]

To run 
```python 
python circle.py

