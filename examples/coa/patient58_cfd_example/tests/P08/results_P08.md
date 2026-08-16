# P08 — Results — Clustering KMeans/DBSCAN (section 8)

## Vessel axis
[-0.25771259 -0.90084273 -0.3493803 ]

## KMeans (n_clusters=3)
| label | n_faces | total_area | mean_dot_axis | std_dot_axis |
|-------|---------|------------|---------------|--------------|
| 0 | 3466 | 0.0075 | 0.3393 | 0.1740 |
| 1 | 5144 | 0.0134 | 0.3000 | 0.1343 |
| 2 | 1983 | 0.0027 | 0.7102 | 0.1643 |

Inferred opening label: 2
Inferred wall label: 0

## DBSCAN (eps=1.5, min_samples=5)
| label | n_faces | total_area | mean_dot_axis | std_dot_axis |
|-------|---------|------------|---------------|--------------|
| 0 | 10593 | 0.0236 | 0.3896 | 0.2184 |

Noise faces: 0
Inferred opening label: 0
Inferred wall label: 0

## Image
clustering_P08.png

## Status
completed
