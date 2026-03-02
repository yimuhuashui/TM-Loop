# -- coding: gbk --
import numpy as np
import pathlib
import h5py
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from loopnet import loopnet  


def load_features(features_dir, chromosomes, test_chrom):
    
    train_pos, train_neg = [], []
    test_X, test_y = None, None
    
    print(f"Looking for feature files in: {features_dir}")
    
    
    if not features_dir.exists():
        print(f"Error: Feature directory does not exist: {features_dir}")
        return None, None, None, None
    
    
    print("Directory contents:")
    for f in features_dir.iterdir():
        print(f"  - {f.name}")
    
    for chrom in chromosomes:
       
        feature_file = features_dir / f"{chrom}_features.npz"
        print(f"Checking for feature file: {feature_file}")
        
        if not feature_file.exists():
            print(f"Feature file not found for {chrom}, skipping...")
            continue
            
        print(f"Found feature file for {chrom}")
        
        try:
            
            data = np.load(feature_file)
            pos_features = data['pos_features']
            neg_features = data['neg_features']
            
            print(f"Loaded features for {chrom}: "
                  f"Positive samples: {pos_features.shape[0]}, "
                  f"Negative samples: {neg_features.shape[0]}")
            
            
            n_neg = min(neg_features.shape[0], 3 * pos_features.shape[0])
            if neg_features.shape[0] > n_neg:
                idx = np.random.choice(neg_features.shape[0], n_neg, replace=False)
                neg_features = neg_features[idx]
            
            if chrom == test_chrom:
                
                test_X = np.vstack((pos_features, neg_features))
                test_y = np.hstack((np.ones(pos_features.shape[0]), np.zeros(neg_features.shape[0])))
                print(f"Test chromosome {test_chrom} - Positive samples: {pos_features.shape[0]}, Negative samples: {n_neg}")
            else:
                
                train_pos.append(pos_features)
                train_neg.append(neg_features)
                print(f"Training chromosome {chrom} - Positive samples: {pos_features.shape[0]}, Negative samples: {n_neg}")
        except Exception as e:
            print(f"Error loading features for {chrom}: {str(e)}")
    
    
    if not train_pos or not train_neg:
        print("Warning: No training features found!")
        return None, None, None, None
    
  
    train_X = np.vstack(train_pos)
    train_F = np.vstack(train_neg)
    
    return train_X, train_F, test_X, test_y

def train_model(train_X, train_F, test_X, test_y, test_chrom, learning_rate, epochs, kernel_size, output_dir):

    all_x = np.vstack((train_X, train_F))
    all_y = np.hstack((np.ones(train_X.shape[0]), np.zeros(train_F.shape[0])))
    
    
    unique, counts = np.unique(all_y, return_counts=True)
    print(f"{test_chrom} class distribution: Positive samples {counts[1]}, Negative samples {counts[0]}, Ratio 1:{counts[0]/counts[1]:.2f}")
    
    
    train_x, val_x, train_y, val_y = train_test_split(all_x, all_y, test_size=0.1, stratify=all_y, random_state=42)
    
    
    train_labels = to_categorical(train_y, num_classes=2)
    val_labels = to_categorical(val_y, num_classes=2)
    
    
    input_shape = (train_x.shape[1],)
    print(f"Input shape: {input_shape}")
    
    
    model = loopnet(
        learning_rate=learning_rate,
        epochs=epochs,
        train_x=train_x,
        train_y=train_labels,
        test_x=val_x,
        test_y=val_labels,
        chromname=test_chrom,
        kernel_size=kernel_size,
        save_dir=output_dir,
        input_shape=input_shape
    )
    model.train_model()

def main():
    
    learning_rate = 0.001
    epochs = 10
    kernel_size = 5
    output = "/home/yanghao/TM-Loop/TM-Loop/model-chr15/"
    test_chrom = 'chr15'
    
    
    chromosomes = [
        'chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10',
        'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20',
        'chr21', 'chr22', 'chrX'
    ]
    
    
    features_dir = pathlib.Path(output) / "features"
    train_X, train_F, test_X, test_y = load_features(features_dir, chromosomes, test_chrom)
    
    
    if train_X is None or train_F is None:
        print("Error: No training data available")
        return
    
    if test_X is None or test_y is None:
        print("Error: No test data available")
        return
    
    print(f"\nTraining set size: Positive {train_X.shape[0]}, Negative {train_F.shape[0]}")
    print(f"Test set size: {test_X.shape[0]}")
    
    
    train_model(train_X, train_F, test_X, test_y, test_chrom, learning_rate, epochs, kernel_size, output)

if __name__ == "__main__":
    main()