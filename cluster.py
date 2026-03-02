# -- coding: gbk --
import gc
import numpy as np
from collections import defaultdict, Counter
from scipy.signal import find_peaks, peak_widths
from scipy.spatial.distance import euclidean, cdist
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import kneighbors_graph
import sys
import heapq
from typing import List, Tuple, Dict, Set
import warnings
warnings.filterwarnings('ignore')

def find_anchors(pos, min_count=3, min_dis=20000, wlen=800000, res=10000, merge_threshold=50000):
    """Find anchor regions on positions"""
    min_dis = max(min_dis // res, 1)
    wlen = min(wlen // res, 20)
    merge_threshold = merge_threshold // res

    count = Counter(pos)
    if not count:
        return set()
    
    min_idx, max_idx = min(count), max(count)
    refidx = range(min_idx, max_idx + 1)
    signal = np.array([count[i] for i in refidx])
    
    summits = find_peaks(signal, height=min_count, distance=min_dis)[0]
    sorted_summits = sorted([(signal[i], i) for i in summits], key=lambda x: -x[0])

    peaks = set()
    records = {}
    for _, i in sorted_summits:
        widths, height, left_ips, right_ips = peak_widths(
            signal, [i], rel_height=1, wlen=wlen
        )
        if len(left_ips) == 0 or len(right_ips) == 0:  
            continue
        
        li = int(np.round(left_ips[0]))
        ri = int(np.round(right_ips[0]))
        lb = refidx[li]
        rb = refidx[ri]
        current_summit = refidx[i]
        merged = False

        for b in range(lb, rb + 1):
            if b in records:
                existing_summit, existing_lb, existing_rb = records[b]
                if abs(existing_summit - current_summit) > merge_threshold:
                    continue
                
                new_lb = min(lb, existing_lb)
                new_rb = max(rb, existing_rb)
                new_summit = existing_summit if signal[existing_summit-min_idx] >= signal[current_summit-min_idx] else current_summit
                
                peaks.remove((existing_summit, existing_lb, existing_rb))
                peaks.add((new_summit, new_lb, new_rb))
                
                for update_b in range(new_lb, new_rb + 1):
                    records[update_b] = (new_summit, new_lb, new_rb)
                merged = True
                break

        if not merged:
            peaks.add((current_summit, lb, rb))
            for b in range(lb, rb + 1):
                records[b] = (current_summit, lb, rb)

    return peaks

class HierarchicalMultiScaleClusterer:
    
    def __init__(self, res=10000, min_density=3, 
                 scale_levels=3, adaptive_radius=True,
                 use_graph_partition=True):
        self.res = res
        self.min_density = min_density
        self.scale_levels = scale_levels
        self.adaptive_radius = adaptive_radius
        self.use_graph_partition = use_graph_partition
        
    def _estimate_local_density(self, points, k=5):
        if len(points) < k:
            return np.ones(len(points))
        coords = np.array([list(p) for p in points])
        if len(coords) > 1000:
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(coords)
            distances, _ = nbrs.kneighbors(coords)
            densities = 1.0 / (distances[:, 1:].mean(axis=1) + 1e-10)
        else:
            distances = cdist(coords, coords, metric='euclidean')
            np.fill_diagonal(distances, np.inf)
            kth_distances = np.partition(distances, k-1, axis=1)[:, k-1]
            densities = 1.0 / (kth_distances + 1e-10)
        
        return densities / densities.max()  
    
    def _build_multi_scale_grid(self, coords, densities):
        
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        
        grid_clusters = []
        
        for level in range(self.scale_levels):
            
            grid_size = max(1, 2 ** (self.scale_levels - level - 1))
            
            
            x_bins = np.arange(x_min, x_max + grid_size, grid_size)
            y_bins = np.arange(y_min, y_max + grid_size, grid_size)
            
           
            x_indices = np.digitize(coords[:, 0], x_bins) - 1
            y_indices = np.digitize(coords[:, 1], y_bins) - 1
            
           
            for i in range(len(x_bins)-1):
                for j in range(len(y_bins)-1):
                    mask = (x_indices == i) & (y_indices == j)
                    if mask.sum() > 0:
                        grid_points = coords[mask]
                        grid_densities = densities[mask]
                        
                        if grid_densities.max() > self.min_density:
                            
                            center = grid_points.mean(axis=0)
                            weight = grid_densities.sum()
                            grid_clusters.append((center, weight, grid_size))
        
        return grid_clusters
    
    def _adaptive_spectral_clustering(self, coords, densities):
        
        n_points = len(coords)
        if n_points < 10:
            return [[i] for i in range(n_points)]
        
        
        if self.adaptive_radius:
            
            median_density = np.median(densities)
            radius = np.clip(2.0 / (densities + 1e-10), 0.5, 5.0)
            
            
            similarities = np.zeros((n_points, n_points))
            for i in range(n_points):
                for j in range(i+1, n_points):
                    dist = euclidean(coords[i], coords[j])
                    if dist < radius[i] and dist < radius[j]:
                        sim = np.exp(-dist**2 / (2 * min(radius[i], radius[j])**2))
                        sim *= np.sqrt(densities[i] * densities[j])
                        similarities[i, j] = sim
                        similarities[j, i] = sim
        else:
            
            radius = 2.0
            distances = cdist(coords, coords, metric='euclidean')
            similarities = np.exp(-distances**2 / (2 * radius**2))
            np.fill_diagonal(similarities, 0)
        
       
        from scipy.sparse.linalg import eigsh
        from scipy.linalg import eigh
        
        D = np.diag(similarities.sum(axis=1))
        L = D - similarities
        
       
        try:
            n_components, labels = connected_components(
                similarities > 0.1, directed=False
            )
            if n_components > 1:
                return [np.where(labels == i)[0] for i in range(n_components)]
        except:
            pass
        
       
        try:
            eigenvals, eigenvecs = eigsh(L, k=min(10, n_points-1), which='SM')
            eigen_gap = np.diff(eigenvals)
            n_clusters = np.argmax(eigen_gap) + 1 if len(eigen_gap) > 0 else 1
            
            if n_clusters > 1:
                from sklearn.cluster import KMeans
                features = eigenvecs[:, 1:n_clusters+1]
                kmeans = KMeans(n_clusters=n_clusters, n_init=10)
                cluster_ids = kmeans.fit_predict(features)
                
                clusters = []
                for i in range(n_clusters):
                    clusters.append(np.where(cluster_ids == i)[0])
                return clusters
        except:
            pass
        
        return [list(range(n_points))]
    
    def _density_based_hierarchical_clustering(self, coords, densities):
        
        n_points = len(coords)
        if n_points <= 1:
            return [[i] for i in range(n_points)]
        
        
        clusters = [[i] for i in range(n_points)]
        cluster_centers = coords.copy()
        cluster_densities = densities.copy()
        
        
        dist_matrix = cdist(coords, coords, metric='euclidean')
        np.fill_diagonal(dist_matrix, np.inf)
        
        while len(clusters) > 1:
            
            min_dist = np.inf
            merge_i, merge_j = -1, -1
            
            for i in range(len(clusters)):
                for j in range(i+1, len(clusters)):
                   
                    dist_ij = dist_matrix[clusters[i][0], clusters[j][0]]
                    density_factor = min(cluster_densities[i], cluster_densities[j])
                    
                    
                    threshold = 2.0 * np.sqrt(1.0 / (density_factor + 1e-10))
                    
                    if dist_ij < threshold and dist_ij < min_dist:
                        min_dist = dist_ij
                        merge_i, merge_j = i, j
            
            if merge_i == -1 or merge_j == -1:
                break
            
            
            merged_cluster = clusters[merge_i] + clusters[merge_j]
            new_center = np.mean(coords[merged_cluster], axis=0)
            new_density = np.mean(densities[merged_cluster])
            
            
            clusters[merge_i] = merged_cluster
            cluster_centers[merge_i] = new_center
            cluster_densities[merge_i] = new_density
            
            clusters.pop(merge_j)
            cluster_centers = np.delete(cluster_centers, merge_j, axis=0)
            cluster_densities = np.delete(cluster_densities, merge_j)
            
           
            if len(clusters) > 1:
                new_dist = cdist(cluster_centers, cluster_centers, metric='euclidean')
                np.fill_diagonal(new_dist, np.inf)
                dist_matrix = new_dist
        
        return clusters
    
    def cluster(self, points, scores=None):
        
        if not points:
            return []
        
        
        coords = np.array([list(p) for p in points])
        if scores is None:
            scores = np.ones(len(coords))
        
        
        densities = self._estimate_local_density(points)
        densities = densities * scores  
        
        
        if self.scale_levels > 1:
            grid_clusters = self._build_multi_scale_grid(coords, densities)
            
            
            grid_labels = -np.ones(len(coords), dtype=int)
            for idx, (center, weight, size) in enumerate(grid_clusters):
                distances = np.linalg.norm(coords - center, axis=1)
                close_points = distances < size * 2
                grid_labels[close_points] = idx
        else:
            grid_labels = np.zeros(len(coords), dtype=int)
        
        
        final_clusters = []
        unique_grids = np.unique(grid_labels)
        
        for grid_id in unique_grids:
            if grid_id == -1:  
                mask = grid_labels == -1
            else:
                mask = grid_labels == grid_id
            
            if mask.sum() == 0:
                continue
                
            grid_coords = coords[mask]
            grid_densities = densities[mask]
            
            if len(grid_coords) < 5:
                
                for i in range(len(grid_coords)):
                    point_idx = np.where(mask)[0][i]
                    final_clusters.append([point_idx])
            else:
                
                clusters = self._adaptive_spectral_clustering(grid_coords, grid_densities)
                
                for cluster in clusters:
                    original_indices = np.where(mask)[0][cluster]
                    if len(original_indices) > 0:
                        final_clusters.append(original_indices.tolist())
        
        
        merged_clusters = []
        for cluster in final_clusters:
            if len(cluster) < 3:  
               
                if len(merged_clusters) > 0:
                    cluster_center = coords[cluster].mean(axis=0)
                    min_dist = np.inf
                    best_merge_idx = -1
                    
                    for i, merged_cluster in enumerate(merged_clusters):
                        merged_center = coords[merged_cluster].mean(axis=0)
                        dist = euclidean(cluster_center, merged_center)
                        if dist < min_dist and dist < 5.0: 
                            min_dist = dist
                            best_merge_idx = i
                    
                    if best_merge_idx != -1:
                        merged_clusters[best_merge_idx].extend(cluster)
                    else:
                        merged_clusters.append(cluster)
                else:
                    merged_clusters.append(cluster)
            else:
                merged_clusters.append(cluster)
        
       
        result = []
        for cluster_indices in merged_clusters:
            if len(cluster_indices) == 0:
                continue
                
            cluster_points = [points[i] for i in cluster_indices]
            cluster_scores = [scores[i] for i in cluster_indices]
            
            
            points_array = np.array([list(p) for p in cluster_points])
            weights = np.array(cluster_scores)
            center = np.average(points_array, axis=0, weights=weights)
            center = tuple(center.round().astype(int))
            
            
            distances = [euclidean(points_array[i], center) for i in range(len(cluster_points))]
            radius = int(np.max(distances) + 1)
            
            
            rep_idx = np.argmax(cluster_scores)
            rep_point = cluster_points[rep_idx]
            
            result.append((rep_point, center, radius))
        
        return result

def local_clustering(Donuts, res, min_count=3, r=20000, max_keep=10):
    
    if not Donuts:
        return []
    
    
    points = list(Donuts.keys())
    scores = [Donuts[p] for p in points]
    
    
    clusterer = HierarchicalMultiScaleClusterer(
        res=res,
        min_density=min_count,
        scale_levels=3,  
        adaptive_radius=True,
        use_graph_partition=True
    )
    
    clusters = clusterer.cluster(points, scores)
    
    
    clusters_with_density = []
    for rep_point, center, radius in clusters:
        density = len([p for p in points 
                      if euclidean(p, center) <= radius])
        clusters_with_density.append((density, rep_point, center, radius))
    
    clusters_with_density.sort(reverse=True, key=lambda x: x[0])
    
    
    return [(rep, center, rad) for _, rep, center, rad in clusters_with_density[:max_keep]]


def process_chromosome(X, res, Athreshold):
    """Process chromosome data"""
    r = X[:, 0].astype(int) // res
    c = X[:, 1].astype(int) // res
    p = X[:, 2].astype(float)
    raw = X[:, 3].astype(float)
    
    d = c - r
    tmpr, tmpc, tmpp, tmpraw, tmpd = r, c, p, raw, d
    matrix = {(r[i], c[i]): p[i] for i in range(len(r))}
    
    count = 40001
    while count > 40000:
        D = defaultdict(float)
        P = defaultdict(float)
        unique_d = np.unique(tmpd)
        
        for distance in unique_d:
            dx = (tmpd == distance)
            dr, dc, dp, draw = tmpr[dx], tmpc[dx], tmpp[dx], tmpraw[dx]
            
            pct_10 = np.percentile(dp, 10)
            dx = dp > pct_10
            dr, dc, dp, draw = dr[dx], dc[dx], dp[dx], draw[dx]
            
            for i in range(dr.size):
                D[(dr[i], dc[i])] += draw[i]
                P[(dr[i], dc[i])] += dp[i]
        
        count = len(D)
        tmpr = np.array([k[0] for k in D])
        tmpc = np.array([k[1] for k in D])
        tmpp = np.array([P[k] for k in D])
        tmpraw = np.array([D[k] for k in D])
        tmpd = tmpc - tmpr
    
    return matrix, D

def write_output(Aoutfile, chrom, final_list, matrix, res):
    """Write filtered results to output file"""
    r = [coord[0] for coord in final_list]
    c = [coord[1] for coord in final_list]
    p = np.array([matrix.get((r[i], c[i]), 0) for i in range(len(r))])
    
    if len(r) > 7000:
        sorted_index = np.argsort(p)[-7000:]
        r = [r[i] for i in sorted_index]
        c = [c[i] for i in sorted_index]
    
    with open(Aoutfile, 'a') as f:
        for i in range(len(r)):
            P_value = matrix.get((r[i], c[i]), 0)
            line = [chrom, r[i] * res, r[i] * res + res,
                    chrom, c[i] * res, c[i] * res + res, P_value]
            f.write('\t'.join(map(str, line)) + '\n')

def main(Ainfile, Aoutfile, Athreshold):
    """Main function to process input file and generate output"""
    Aresolution = 10000
    
    x = {}
    
    try:
        with open(Ainfile, 'r') as source:
            for line in source:
                p = line.rstrip().split()
                if len(p) < 7:
                    continue
                    
                chrom = p[0]
                if float(p[6]) > Athreshold:
                    if chrom not in x:
                        x[chrom] = []
                    x[chrom].append([int(p[1]), int(p[4]), float(p[6]), float(p[7])])
    except Exception as e:
        print(f"File read error: {str(e)}")
        sys.exit(1)
    
    
    open(Aoutfile, 'w').close()
    
    for chrom in x:
        X = np.array(x[chrom])
        
        try:
            matrix, D = process_chromosome(X, Aresolution, Athreshold)
        except Exception as e:
            print(f"Error processing chromosome {chrom}: {str(e)}")
            continue
            
        del X
        gc.collect()
        
        try:
            final_list = [cluster[0] for cluster in local_clustering(D, res=Aresolution, max_keep=10000)]
            write_output(Aoutfile, chrom, final_list, matrix, Aresolution)
        except Exception as e:
            print(f"Clustering error for chromosome {chrom}: {str(e)}")
    
    if not x:
        print("Warning: No data found above threshold")

if __name__ == "__main__":
    infile = "/home/yanghao/TM-Loop/TM-Loop/candidate-loops/chr15.bed"
    outfile = "/home/yanghao/TM-Loop/TM-Loop/loops/chr15.bedpe"
    threshold = 0.985
    
    main(infile, outfile, threshold)