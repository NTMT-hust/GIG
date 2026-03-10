import numpy as np
import cv2
import numpy as np
import csv
import pandas as pd

def calculate_mean(all_heatmaps):
    class_sum = {}
    class_count = {}

    for heatmap, cls in all_heatmaps:  # unpack correctly
        if cls not in class_sum:
            class_sum[cls] = heatmap.copy()
            class_count[cls] = 1
        else:
            class_sum[cls] += heatmap
            class_count[cls] += 1

    class_mean = {
        cls: class_sum[cls] / class_count[cls]
        for cls in class_sum
    }

    return class_mean


def visualize_mean_heatmap(mean_heatmap, save_path):
    heatmap_uint8 = np.uint8(255 * mean_heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    cv2.imwrite(save_path, heatmap_color)

def find_critical_pixel(heatmap, threshold):
    coords = np.argwhere(heatmap >= threshold)
    res = []
    for y,x in coords:
        res.append([x,y,float(heatmap[y,x])])
    return res

def find_critical_gene(pixels, filepath):
    df = pd.read_csv(filepath)
    genes = []
    for x,y,val in pixels:
        name = find_gene_by_pixel(df,x,y)
        if (name):
            genes.append([x,y,name])
    return genes
def find_gene_by_pixel(df, x, y):
    row = df[(df["pixel_x"] == x) & (df["pixel_y"] == y)]
    if row.empty:
        return None
    return row["gene_name"].values[0]

def save_to_csv(rows, csv_path):
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "value"])
        writer.writerows(rows)
