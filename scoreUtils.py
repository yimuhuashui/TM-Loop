# -- coding: gbk --
import pathlib
import numpy as np
from scipy import sparse
import time
from tqdm import tqdm  
import pyBigWig
from sklearn.decomposition import KernelPCA
import os
import pickle
import math

class Chromosome():
    def __init__(self, coomatrix, model, lower=1, upper=500, cname='chrm', res=10000, width=5, 
                 atac_file=None, ctcf_file=None):
        # Keep original initialization code unchanged
        R, C = coomatrix.nonzero()
        validmask = np.isfinite(coomatrix.data) & (C-R+1 > lower) & (C-R < upper)
        R, C, data = R[validmask], C[validmask], coomatrix.data[validmask]
        self.M = sparse.csr_matrix((data, (R, C)), shape=coomatrix.shape)
        self.ridx, self.cidx = R, C
        self.chromname = cname
        self.r = res
        self.w = width
        self.model = model
        self.atac_file = atac_file
        self.ctcf_file = ctcf_file
        
        # Precompute distance matrix
        window_size = 2 * width + 1
        i, j = np.indices((window_size, window_size))
        genomic_dist = res * np.sqrt(np.maximum((i-width)**2 + (j-width)**2, 0.1)) / 1000
        self.dist_matrix = np.maximum(genomic_dist, 10)**0.75
        
        # Feature storage settings
        self.feature_cache_dir = "/public_data/yanghao/K562/feature_cache"
        if not os.path.exists(self.feature_cache_dir):
            os.makedirs(self.feature_cache_dir)
    
    def extract_ring_features(self, window, width):
        """Extract ring features from center outward"""
        ring_features = []
        center = (width, width)
        window_size = 2 * width + 1
        
        for r in range(0, width+1):
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
    
    def create_enhanced_features(self, window, dist_corrected_features, ring_features, p2LL, sym_score, atac_features, ctcf_features, width, dist_matrix):

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
    
    def extract_multiomics_features(self, coord):
        
        if not self.atac_file or not self.ctcf_file:
            return np.zeros(121), np.zeros(121)
        
        window_size = 2 * self.w + 1
        bin_length = 10000  
        expected_window_bp = window_size * bin_length  # 11 * 10000 = 110000 bp
        
        try:
            
            atac_bw = pyBigWig.open(self.atac_file)
            ctcf_bw = pyBigWig.open(self.ctcf_file)
            chrom_length = atac_bw.chroms()[self.chromname]
            
            x, y = coord[0], coord[1]
            
            
            start_x = max(0, (x - self.w) * self.r)
            end_x = min(chrom_length, (x + self.w + 1) * self.r)
            start_y = max(0, (y - self.w) * self.r)
            end_y = min(chrom_length, (y + self.w + 1) * self.r)
            
            
            if start_x >= end_x or start_y >= end_y:
                return np.zeros(121), np.zeros(121)
                
            
            window_x_atac = atac_bw.values(self.chromname, int(start_x), int(end_x))
            window_x_atac = np.array(window_x_atac)
            window_x_atac[np.isnan(window_x_atac)] = 0
            
            window_y_atac = atac_bw.values(self.chromname, int(start_y), int(end_y))
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
            
           
            kpca = KernelPCA(n_components=50, kernel='rbf', gamma=0.01)
            
           
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
            
            
            window_x_ctcf = ctcf_bw.values(self.chromname, int(start_x), int(end_x))
            window_x_ctcf = np.array(window_x_ctcf)
            window_x_ctcf[np.isnan(window_x_ctcf)] = 0
            
            window_y_ctcf = ctcf_bw.values(self.chromname, int(start_y), int(end_y))
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
            
            atac_bw.close()
            ctcf_bw.close()
            
            return window_atac.flatten(), window_ctcf.flatten()
            
        except Exception as e:
            print(f"Error extracting multiomics features: {str(e)}")
            return np.zeros(121), np.zeros(121)
    
    def save_batch_to_cache(self, batch_id, features, coordinates):
        
        try:
            cache_file = os.path.join(
                self.feature_cache_dir, 
                f"{self.chromname}_batch_{batch_id}.pkl"
            )
            
            cache_data = {
                'features': features,
                'coordinates': coordinates
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            print(f"Saved batch {batch_id} for chromosome {self.chromname} to cache: {cache_file}")
        except Exception as e:
            print(f"Failed to save batch {batch_id} for chromosome {self.chromname} to cache: {str(e)}")
    
    def load_batch_from_cache(self, batch_id):
      
        try:
            cache_file = os.path.join(
                self.feature_cache_dir, 
                f"{self.chromname}_batch_{batch_id}.pkl"
            )
            
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                print(f"Loaded batch {batch_id} for chromosome {self.chromname} from cache: {cache_file}")
                return cache_data['features'], cache_data['coordinates']
        except Exception as e:
            print(f"Failed to load batch {batch_id} for chromosome {self.chromname} from cache: {str(e)}")
        return None, None
    
    def process_coordinate(self, c):
        
        width = self.w
        window_size = 2 * width + 1
        
        try:
            x, y = c[0], c[1]
            
            # Enhanced boundary check
            valid_x = max(width, min(x, self.M.shape[0]-width-1))
            valid_y = max(width, min(y, self.M.shape[1]-width-1))
            if abs(valid_x - x) > 1 or abs(valid_y - y) > 1:
                return None, None
            
            # Safe window extraction
            x_start, x_end = valid_x-width, valid_x+width+1
            y_start, y_end = valid_y-width, valid_y+width+1
            
            window = self.M[x_start:x_end, y_start:y_end].toarray()
            actual_size = window.shape[0] * window.shape[1]
            
            # Sparsity check
            if np.count_nonzero(window) < actual_size * 0.1:
                return None, None
            
            center_val = window[width, width]
            
            # Feature 1: Distance-corrected values
            try:
                corrected_window = window * self.dist_matrix[:window.shape[0], :window.shape[1]]
                dist_corrected_features = corrected_window.flatten()
            except:
                dist_corrected_features = np.zeros_like(window.flatten())
            
            # Feature 2: Ring features
            ring_features = self.extract_ring_features(window, width)
            
            # Feature 3: Center to background ratio
            try:
                border_size = max(1, window.shape[0]//5)
                bg_rows = window[-border_size:, :]
                bg_cols = window[:, -border_size:]
                bg_region = np.concatenate((bg_rows.flatten(), bg_cols.flatten()))
                p2LL = center_val / (np.mean(bg_region) + 1e-7)
            except:
                p2LL = 0.0
            
            # Feature 4: Ring symmetry
            try:
                sym_matrix = np.abs(window - window.T)
                fro_window = np.linalg.norm(window, 'fro')
                fro_sym = np.linalg.norm(sym_matrix, 'fro')
                sym_score = 1 - fro_sym / (fro_window + 1e-7) if fro_window > 0 else 0
            except:
                sym_score = 0
            
            # Feature 5: Raw window features
            raw_window_features = window.flatten()
            
            
            atac_features, ctcf_features = self.extract_multiomics_features(c)
            
            
            enhanced_features = self.create_enhanced_features(
                window, 
                dist_corrected_features, 
                ring_features, 
                p2LL, 
                sym_score, 
                atac_features.reshape(window_size, window_size) if atac_features.size == window_size**2 else np.zeros((window_size, window_size)),
                ctcf_features.reshape(window_size, window_size) if ctcf_features.size == window_size**2 else np.zeros((window_size, window_size)),
                width, 
                self.dist_matrix
            )
            
          
            expected_size = 516
            if enhanced_features.size != expected_size:
                
                if enhanced_features.size < expected_size:
                    enhanced_features = np.pad(enhanced_features, (0, expected_size - enhanced_features.size), 'constant')
                else:
                    enhanced_features = enhanced_features[:expected_size]
            
            if np.all(np.isfinite(enhanced_features)):
                return enhanced_features.reshape(1, -1), c
        except Exception as e:
            print(f"Error processing coordinate ({x},{y}) on chromosome {self.chromname}: {str(e)}")
        
        return None, None
    
    def getwindow(self, coords):
        
        fts, clist = [], []
        width = self.w
        
       
        valid_coords = [
            c for c in coords 
            if width <= c[0] < self.M.shape[0]-width-1 
            and width <= c[1] < self.M.shape[1]-width-1
        ]
        
        print(f"Chromosome {self.chromname}: {len(valid_coords)} valid coordinates")
        
        
        num_batches = 10
        batch_size = math.ceil(len(valid_coords) / num_batches)
        batches = [valid_coords[i:i+batch_size] for i in range(0, len(valid_coords), batch_size)]
        
        
        for batch_id, batch_coords in enumerate(batches):
            print(f"Processing batch {batch_id+1}/{len(batches)} for chromosome {self.chromname}")
            
            
            cached_features, cached_coords = self.load_batch_from_cache(batch_id)
            if cached_features is not None and cached_coords is not None:
                print(f"Using cached batch {batch_id} for chromosome {self.chromname}")
                fts.extend(cached_features)
                clist.extend(cached_coords)
                continue
            
            
            batch_fts, batch_clist = [], []
            
           
            for c in tqdm(batch_coords, desc=f"Processing batch {batch_id+1}"):
                features, coord = self.process_coordinate(c)
                if features is not None and coord is not None:
                    batch_fts.append(features)
                    batch_clist.append(coord)
            
            
            if batch_fts:
                self.save_batch_to_cache(batch_id, batch_fts, batch_clist)
                fts.extend(batch_fts)
                clist.extend(batch_clist)
        
        if fts:
            test_x = np.vstack(fts)
        else:
            test_x = np.empty((0, 516))  # Feature dimension is 516
        
        # Calculate actual feature dimension
        features = test_x.shape[1] if test_x.shape[0] > 0 else 516
        print(f"Chromosome {self.chromname}: Extracted {len(test_x)} samples, feature dimension: {features}")
        
        # Reshape to 3D tensor (samples, features, 1)
        test_x_r = test_x.reshape((len(test_x), features, 1))
        print(f'Starting model prediction for chromosome {self.chromname}...')
        
        # Initialize probability array
        probas = np.zeros(len(test_x))
        
        # Iterate through each model
        probas = self.model.predict(test_x_r, verbose=0, batch_size=8192)[:, 1]
        
        print(f'Prediction completed for chromosome {self.chromname}.')
        return probas, clist

    def score(self, thre=0.5):
        # Keep original scoring code unchanged
        print(f'Scoring matrix for chromosome {self.chromname}')
        print(f'Number of candidates: {self.M.data.size}')
        coords = [(r, c) for r, c in zip(self.ridx, self.cidx)]
        print(f'---------Coordinate loading finished for chromosome {self.chromname}--------')
        p, clist = self.getwindow(coords)
        print(f'---------Windows obtained for chromosome {self.chromname}----------------')
        clist = np.r_[clist]
        pfilter = p > thre
        ri = clist[:, 0][pfilter]
        ci = clist[:, 1][pfilter]
        result = sparse.csr_matrix((p[pfilter], (ri, ci)), shape=self.M.shape)
        data = np.array(self.M[ri, ci]).ravel()
        self.M = sparse.csr_matrix((data, (ri, ci)), shape=self.M.shape)

        return result, self.M

    def writeBed(self, out, prob_csr, raw_csr):
        # Keep original writing code unchanged
        print(f'---------Begin writing for chromosome {self.chromname}----------')
        pathlib.Path(out).mkdir(parents=True, exist_ok=True)
        with open(out + '/' + self.chromname + '.bed', 'w') as output_bed:
            r, c = prob_csr.nonzero()
            for i in range(r.size):
                line = [self.chromname, r[i]*self.r, (r[i]+1)*self.r,
                        self.chromname, c[i]*self.r, (c[i]+1)*self.r,
                        prob_csr[r[i],c[i]], raw_csr[r[i],c[i]]]

                output_bed.write('\t'.join(list(map(str, line)))+'\n')