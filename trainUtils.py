# -- coding: gbk --
import pathlib
import numpy as np
import warnings
from collections import defaultdict,Counter
import joblib
import random
import keras
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pyBigWig
from sklearn.decomposition import KernelPCA
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import concurrent.futures
from multiprocessing import Manager
from sklearn.neighbors import KernelDensity
from scipy import stats
import threading  

Thread = threading.Thread

def getbigwig(file, chrome, start, end):
    
    try:
        bw = pyBigWig.open(file)
        sample = np.array(bw.values(chrome, start, end))
        sample[np.isnan(sample)] = 0
        bw.close()
        return sample
    except Exception as e:
        print(f"Error reading BigWig file: {str(e)}")
        return np.zeros(end - start)



def generate_multiomics(coords, chromname, atac_file, ctcf_file, resou, width=5):
    
    window_size = 2 * width + 1
    bin_length = 10000 
    expected_window_bp = window_size * bin_length  # 11 * 10000 = 110000 bp
    
    atac_results = []
    ctcf_results = []
    
    
    kpca = KernelPCA(n_components=50, kernel='rbf', gamma=0.01)
    
    pbar = tqdm(total=len(coords), desc='Processing Multiomics Data', leave=True)
    
    
    atac_bw = pyBigWig.open(atac_file)
    ctcf_bw = pyBigWig.open(ctcf_file)
    chrom_length = atac_bw.chroms()[chromname]
    
    for coord in coords:
        try:
            x, y = coord[0], coord[1]
            
            
            start_x = max(0, (x - width) * resou)
            end_x = min(chrom_length, (x + width + 1) * resou)
            start_y = max(0, (y - width) * resou)
            end_y = min(chrom_length, (y + width + 1) * resou)
            
            
            if start_x >= end_x or start_y >= end_y:
                continue
                
            
            window_x_atac = atac_bw.values(chromname, int(start_x), int(end_x))
            window_x_atac = np.array(window_x_atac)
            window_x_atac[np.isnan(window_x_atac)] = 0
            
            window_y_atac = atac_bw.values(chromname, int(start_y), int(end_y))
            window_y_atac = np.array(window_y_atac)
            window_y_atac[np.isnan(window_y_atac)] = 0
            
           
            if len(window_x_atac) < expected_window_bp:
                
                window_x_atac = np.pad(window_x_atac, (0, expected_window_bp - len(window_x_atac)), 'constant')
            elif len(window_x_atac) > expected_window_bp:
                
                window_x_atac = window_x_atac[:expected_window_bp]
                
            if len(window_y_atac) < expected_window_bp:
                window_y_atac = np.pad(window_y_atac, (0, expected_window_bp - len(window_y_atac)), 'constant')
            elif len(window_y_atac) > expected_window_bp:
                window_y_atac = window_y_atac[:expected_window_bp]
            
           
            window_x_atac = window_x_atac.reshape(window_size, bin_length)
            window_y_atac = window_y_atac.reshape(window_size, bin_length)
            
           
            kpca.fit(window_x_atac)
            window_x_reduced = kpca.transform(window_x_atac)
            kpca.fit(window_y_atac)
            window_y_reduced = kpca.transform(window_y_atac)
            
           
            window_atac = np.zeros((window_size, window_size))
            for i in range(window_size):
                for j in range(window_size):
                    corr = np.corrcoef(window_x_reduced[i], window_y_reduced[j])[0, 1]
                    window_atac[i, j] = corr
            
           
            min_val = window_atac.min()
            max_val = window_atac.max()
            if max_val - min_val > 1e-7:
                window_atac = (window_atac - min_val) / (max_val - min_val)
            else:
                window_atac = np.zeros((window_size, window_size))
            
           
            window_x_ctcf = ctcf_bw.values(chromname, int(start_x), int(end_x))
            window_x_ctcf = np.array(window_x_ctcf)
            window_x_ctcf[np.isnan(window_x_ctcf)] = 0
            
            window_y_ctcf = ctcf_bw.values(chromname, int(start_y), int(end_y))
            window_y_ctcf = np.array(window_y_ctcf)
            window_y_ctcf[np.isnan(window_y_ctcf)] = 0
            
            if len(window_x_ctcf) < expected_window_bp:
                window_x_ctcf = np.pad(window_x_ctcf, (0, expected_window_bp - len(window_x_ctcf)), 'constant')
            elif len(window_x_ctcf) > expected_window_bp:
                window_x_ctcf = window_x_ctcf[:expected_window_bp]
                
            if len(window_y_ctcf) < expected_window_bp:
                window_y_ctcf = np.pad(window_y_ctcf, (0, expected_window_bp - len(window_y_ctcf)), 'constant')
            elif len(window_y_ctcf) > expected_window_bp:
                window_y_ctcf = window_y_ctcf[:expected_window_bp]
            
            window_x_ctcf = window_x_ctcf.reshape(window_size, bin_length)
            window_y_ctcf = window_y_ctcf.reshape(window_size, bin_length)
            
            kpca.fit(window_x_ctcf)
            window_x_reduced = kpca.transform(window_x_ctcf)
            kpca.fit(window_y_ctcf)
            window_y_reduced = kpca.transform(window_y_ctcf)
            
            window_ctcf = np.zeros((window_size, window_size))
            for i in range(window_size):
                for j in range(window_size):
                    corr = np.corrcoef(window_x_reduced[i], window_y_reduced[j])[0, 1]
                    window_ctcf[i, j] = corr
            
            min_val = window_ctcf.min()
            max_val = window_ctcf.max()
            if max_val - min_val > 1e-7:
                window_ctcf = (window_ctcf - min_val) / (max_val - min_val)
            else:
                window_ctcf = np.zeros((window_size, window_size))
            
            atac_results.append(window_atac)
            ctcf_results.append(window_ctcf)
        except Exception as e:
            print(f"Error processing coordinates({x},{y}): {str(e)}")
            atac_results.append(np.zeros((window_size, window_size)))
            ctcf_results.append(np.zeros((window_size, window_size)))
        finally:
            pbar.update(1)
    
    atac_bw.close()
    ctcf_bw.close()
    pbar.close()
    
    return atac_results, ctcf_results
    


def create_enhanced_features(window,dist_corrected_features,ring_features, p2LL, sym_score, atac_features, ctcf_features, width, dist_matrix):
    
    key_features = np.array([
        p2LL,                          
        sym_score,                     
    ])
    
  
    structural_features = np.hstack([
       
        dist_corrected_features,
        
        ring_features
    ])
    
    
    epigenetic_features = np.hstack([atac_features.flatten(), ctcf_features.flatten()])
    
    
    raw_features = window.flatten()
    
  
    def standardize_group(features, weight=1.0):
        if len(features) == 0:
            return features
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)
        mean_val = np.mean(features)
        std_val = np.std(features)
        if std_val > 1e-7:
            return (features - mean_val) / std_val * weight
        else:
            return (features - mean_val) * weight
    
    
    enhanced_features = np.hstack([
        standardize_group(key_features, 2.0),       
        standardize_group(structural_features, 1.5),  
        standardize_group(epigenetic_features, 1.2),  
        standardize_group(raw_features, 1.0)        
    ])
    
    return enhanced_features

def build_vector_enhanced(Matrix, coords, width=5, lower=1, positive=True, resolution=10000, 
                         atac_features_list=None, ctcf_features_list=None):
    
    window_size = 2 * width + 1
    
   
    i, j = np.indices((window_size, window_size))
    genomic_dist = resolution * np.sqrt(np.maximum((i-width)**2 + (j-width)**2, 0.1)) / 1000
    dist_matrix = np.maximum(genomic_dist, 10)**0.75
    
       
    def extract_ring_features(window):
        
        ring_features = []
        center = (width, width)
        
       
        radii = range(0, width+1)
        
        for r in radii:
          
            mask = np.zeros_like(window, dtype=bool)
            
            if r == 0:
                mask[center] = True
            else:
               
                for angle in np.linspace(0, 2*np.pi, 8*r, endpoint=False):
                    x = int(round(center[0] + r * np.cos(angle)))
                    y = int(round(center[1] + r * np.sin(angle)))
                    
                   
                    if 0 <= x < window_size and 0 <= y < window_size:
                        mask[x, y] = True
            
           
            ring_values = window[mask]
            if len(ring_values) > 0:
                ring_features.extend([
                    np.mean(ring_values),      
                    np.max(ring_values),       
                    np.min(ring_values),       
                    np.median(ring_values),     
                    np.std(ring_values)        
                ])
            else:
                ring_features.extend([0, 0, 0, 0, 0])
        
        return np.array(ring_features)
    
    for idx, c in enumerate(coords):
        x, y = int(c[0]), int(c[1])
        
       
        valid_x = max(width, min(x, Matrix.shape[0]-width-1))
        valid_y = max(width, min(y, Matrix.shape[1]-width-1))
        if abs(valid_x - x) > 1 or abs(valid_y - y) > 1:
            continue
        
        try:
           
            x_start, x_end = valid_x-width, valid_x+width+1
            y_start, y_end = valid_y-width, valid_y+width+1
            
            window = Matrix[x_start:x_end, y_start:y_end].toarray()
            actual_size = window.shape[0] * window.shape[1]
            
           
            if np.count_nonzero(window) < actual_size * 0.1:
                continue
                
            center_val = window[width, width]
            
            
            try:
                corrected_window = window * dist_matrix[:window.shape[0], :window.shape[1]]
                dist_corrected_features = corrected_window.flatten()
            except:
                dist_corrected_features = np.zeros_like(window.flatten())
            
           
            ring_features = extract_ring_features(window)
            
            
            try:
                border_size = max(1, window.shape[0]//5)
                bg_rows = window[-border_size:, :]
                bg_cols = window[:, -border_size:]
                bg_region = np.concatenate((bg_rows.flatten(), bg_cols.flatten()))
                p2LL = center_val / (np.mean(bg_region) + 1e-7)
            except:
                p2LL = 0.0
            
            
            try:
                sym_matrix = np.abs(window - window.T)
                fro_window = np.linalg.norm(window, 'fro')
                fro_sym = np.linalg.norm(sym_matrix, 'fro')
                sym_score = 1 - fro_sym / (fro_window + 1e-7) if fro_window > 0 else 0
            except:
                sym_score = 0
            
            
            raw_window_features = window.flatten()
            
           
            if atac_features_list is not None and ctcf_features_list is not None and idx < len(atac_features_list):
                atac_features = atac_features_list[idx]
                ctcf_features = ctcf_features_list[idx]
            else:
               
                atac_features = np.zeros((window_size, window_size))
                ctcf_features = np.zeros((window_size, window_size))
            
           
            enhanced_features = create_enhanced_features(
                window,dist_corrected_features,ring_features, p2LL, sym_score, atac_features, ctcf_features, width, dist_matrix
            )
            
           
            expected_size = 516
            if enhanced_features.size != expected_size:
               
                if enhanced_features.size < expected_size:
                    enhanced_features = np.pad(enhanced_features, (0, expected_size - enhanced_features.size), 'constant')
                else:
                    enhanced_features = enhanced_features[:expected_size]
            
            if np.all(np.isfinite(enhanced_features)):
                yield enhanced_features
                
        except Exception as e:
            print(f"Error processing coordinates({x},{y}): {str(e)}")
            continue

def parsebed(chiafile, res=10000, lower=1, upper=5000000):
    # Parse BED file to extract genomic coordinates
    coords = defaultdict(set)
    upper = upper // res
    with open(chiafile) as o:
        for line in o:
            s = line.rstrip().split()
            a, b = float(s[1]), float(s[4])
            a, b = int(a), int(b)
            if a > b:
                a, b = b, a
            a //= res
            b //= res
            # Only include chromosome entries that don't contain 'M'
            if (b-a > lower) and (b-a < upper) and 'M' not in s[0]:
                chrom = 'chr' + s[0].lstrip('chr')
                coords[chrom].add((a, b))

    for c in coords:
        coords[c] = sorted(coords[c])

    return coords


def get_kde(coords):
    # Calculate Kernel Density Estimation for distance distribution
    dis = []
    for c in coords:
        for a, b in coords[c]:
            dis.append(b-a)

    lower = min(dis)

    kde = stats.gaussian_kde(dis)

    counts, bins = np.histogram(dis, bins=100)
    long_end = int(bins[-1])
    tp = np.where(np.diff(counts) >= 0)[0] + 2
    long_start = int(bins[tp[0]])

    return kde, lower, long_start, long_end

def negative_generating(M, kde, positives, lower, long_start, long_end):
    # Generate negative samples for training
    positives = set(positives)        
    N = 3 * len(positives)             
    # part 1: Sampling based on distance distribution
    part1 = kde.resample(N).astype(int).ravel()      
    part1 = part1[(part1 >= lower) & (part1 <= long_end)]

    # part 2: Uniform sampling for long-range interactions
    part2 = []
    pool = np.arange(long_start, long_end+1)   
    tmp = np.cumsum(M.shape[0]-pool)
    ref = tmp / tmp[-1]                        
    for i in range(N):                          
        r = np.random.random()                 
        ii = np.searchsorted(ref, r)            
        part2.append(pool[ii])                  

    sample_dis = Counter(list(part1) + part2)    

    neg_coords = []                        
    midx = np.arange(M.shape[0])          

    # Generate negative coordinates from distance pool
    for i in sorted(sample_dis):  
        n_d = sample_dis[i]                
        R, C = midx[:-i], midx[i:]           
        tmp = np.array(M[R, C]).ravel()   
        tmp[np.isnan(tmp)] = 0         
        mask = tmp > 0                 
        R, C = R[mask], C[mask]
        pool = set(zip(R, C)) - positives     
        sub = random.sample(pool, n_d)       
        neg_coords.extend(sub)     

    random.shuffle(neg_coords)    

    return neg_coords
