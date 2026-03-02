# -- coding: gbk --
import pathlib
import numpy as np
import trainUtils, utils
import warnings
from collections import defaultdict
import joblib
import random
import keras
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pyBigWig
from sklearn.decomposition import KernelPCA
from tqdm import tqdm
import h5py
import os


def main():
    warnings.filterwarnings("ignore")
    learning_rate = 0.001
    epochs = 100
    kernel_size = 5
    Apath = "/public_data/yanghao/Rao2014-GM12878-MboI-allreps-filtered.10kb.cool"
    Aoutput = "/home/yanghao/TM-Loop/TM-Loop/model-chr15"
    Abedpe = "/home/yanghao//TM-Loop/TM-Loop/data/gm12878_ctcf_h3k27ac.bedpe"
    test_chrom = 'chr15'
    Aresolution = 10000
    Awidth = 5
    Abalance = 1
    atac_file = "/home/yanghao/TM-Loop/TM-Loop/data/ATAC-ENCFF816ZFB.bigWig"  
    ctcf_file = "/home/yanghao/TM-Loop/TM-Loop/data/CTCF-ENCFF797LSC.bigWig"     

    np.seterr(divide='ignore', invalid='ignore')
    pathlib.Path(Aoutput).mkdir(parents=True, exist_ok=True)
    hic_info = utils.read_hic_header(Apath)
    if hic_info is None:
        hic = False
    else:
        hic = True

    coords = trainUtils.parsebed(Abedpe, lower=2, res=Aresolution)
    kde, lower, long_start, long_end = trainUtils.get_kde(coords)

    if not hic:
        import cooler
        Lib = cooler.Cooler(Apath)
        chromosomes = Lib.chromnames[:]
    else:
        chromosomes = utils.get_hic_chromosomes(Apath, Aresolution)

    
    if len(chromosomes) > 2:
        chromosomes.pop()
        chromosomes.pop()
    
    
    features_dir = pathlib.Path(Aoutput) / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    
    positive_class = {}
    negative_class = {}
    
    for key in chromosomes:
        if key.startswith('chr'):
            chromname = key
        else:
            chromname = 'chr'+key
        
        if chromname not in coords:
            print(f'Skipping chromosome {chromname}, no coordinates')
            continue
            
        print(f"\n{'='*50}")
        print(f"Processing chromosome: {chromname}")
        print(f"{'='*50}")
            
        
        features_file = features_dir / f"{chromname}_features.npz"
        if features_file.exists():
            print(f"Loading precomputed features for {chromname}")
            data = np.load(features_file)
            positive_class[chromname] = data['pos_features']
            negative_class[chromname] = data['neg_features']
            print(f"Loaded features for {chromname}: "
                  f"Positive samples: {positive_class[chromname].shape[0]}, "
                  f"Negative samples: {negative_class[chromname].shape[0]}")
            continue
            
        print('Reading data: {}'.format(key))
        if not hic:
            X = Lib.matrix(balance=Abalance,sparse=True).fetch(key).tocsr()
        else:
            if Abalance:
                X = utils.csr_contact_matrix('KR', Apath, key, key, 'BP', Aresolution)
            else:
                X = utils.csr_contact_matrix('NONE', Apath, key, key, 'BP', Aresolution)
        
        clist = coords[chromname]
        
        try:
            print(f"\nBuilding features for {chromname}")
            print(f"Number of coordinates: {len(clist)}")
            
            
            print("Step 1/4: Generating multiomics features...")
            atac_features, ctcf_features = trainUtils.generate_multiomics(
                clist, chromname, atac_file, ctcf_file, Aresolution, width=Awidth
            )
            
            
            print("Step 2/4: Generating enhanced features...")
            enhanced_features_list = []
            valid_coords = []
            
            
            enhanced_features_generator = trainUtils.build_vector_enhanced(
                X, 
                clist, 
                width=Awidth,
                atac_features_list=atac_features,
                ctcf_features_list=ctcf_features
            )
            
            for features in tqdm(enhanced_features_generator, desc="Processing coordinates", total=len(clist)):
                enhanced_features_list.append(features)
            
            if not enhanced_features_list:
                print(f"No enhanced features generated for {chromname}")
                continue
                
            enhanced_features = np.vstack(enhanced_features_list)
            print(f"Enhanced features shape: {enhanced_features.shape}")
            
            
            pos_features = enhanced_features
            print(f"\nCombined positive features shape: {pos_features.shape}")
            
            print("\nStep 3/4: Generating negative samples...")
            neg_coords = trainUtils.negative_generating(X, kde, clist, lower, long_start, long_end)
            print(f"Number of negative coordinates: {len(neg_coords)}")
            
            
            print("\nStep 4/4: Generating negative multiomics features...")
            neg_atac_features, neg_ctcf_features = trainUtils.generate_multiomics(
                neg_coords, 
                chromname, 
                atac_file, 
                ctcf_file, 
                Aresolution, 
                width=Awidth
            )
            
           
            neg_enhanced_features_list = []
            
            neg_features_generator = trainUtils.build_vector_enhanced(
                X, 
                neg_coords, 
                width=Awidth,
                atac_features_list=neg_atac_features,
                ctcf_features_list=neg_ctcf_features,
                positive=False
            )
            
            for features in tqdm(neg_features_generator, desc="Processing negative coordinates", total=len(neg_coords)):
                neg_enhanced_features_list.append(features)
            
            if not neg_enhanced_features_list:
                print(f"No negative features generated for {chromname}")
                continue
                
            neg_features = np.vstack(neg_enhanced_features_list)
            print(f"Combined negative features shape: {neg_features.shape}")
            
            
            positive_class[chromname] = pos_features
            negative_class[chromname] = neg_features
            
            np.savez(
                features_file,
                pos_features=pos_features,
                neg_features=neg_features
            )
            print(f"\nSaved features for {chromname} to {features_file}")
            
            print(f"\nSummary for {chromname}:")
            print(f"Positive samples: {pos_features.shape[0]}")
            print(f"Negative samples: {neg_features.shape[0]}")
            print(f"Feature vector size: {pos_features.shape[1]} dimensions")
            
            
            print("\nFeature vector details:")
            print(f"First positive sample feature vector:")
            print(f"  Shape: {pos_features[0].shape}")
            print(f"  First 10 values: {pos_features[0, :10]}")
            print(f"  Last 10 values: {pos_features[0, -10:]}")
            print(f"  Min value: {np.min(pos_features[0])}")
            print(f"  Max value: {np.max(pos_features[0])}")
            print(f"  Mean value: {np.mean(pos_features[0])}")
            print(f"  Std value: {np.std(pos_features[0])}")
            
        except Exception as e:
            print(f"Error processing chromosome {chromname}: {str(e)}")
            continue

if __name__ == "__main__":
    main()