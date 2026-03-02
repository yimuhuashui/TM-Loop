# -- coding: gbk --
import pathlib
import os
import numpy as np
import scoreUtils, utils
import tensorflow as tf
from keras.models import load_model
from loopnete import PositionalEncoding,MultiHeadAttention
def main(chr_name, aoutput, amodel, apath, atac_file, ctcf_file):
    # Fixed parameters remain unchanged
    aresolution = 10000
    awidth = 5
    abalance = 1
    alower = 1
    aupper = 500
    
    np.seterr(divide='ignore', invalid='ignore')

    # Create output directory
    pathlib.Path(aoutput).mkdir(parents=True, exist_ok=True)
   
   
    model_path = os.path.join(amodel, f'{chr_name}_best_model.h5')
    try:
        
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'PositionalEncoding': PositionalEncoding,
                'MultiHeadAttention': MultiHeadAttention,
            }
        )
        print(f"Successfully loaded model: {model_path}")
    except Exception as e:
        print(f"Failed to load model {model_path}: {str(e)}")
        return

    # Determine file type
    hic_info = utils.read_hic_header(apath)
    hic = hic_info is not None

    # Get chromosome information
    if not hic:
        import cooler  # Imported only when needed
        Lib = cooler.Cooler(apath)
        chromosomes = Lib.chromnames[:]
    else:
        chromosomes = utils.get_hic_chromosomes(apath, aresolution)

    # Prepare chromosome names
    pre = utils.find_chrom_pre(chromosomes)
    ccname = pre + chr_name.lstrip('chr')  
    cikada = 'chr' + ccname.lstrip('chr')  # cikada always has prefix "chr"
    
    # Process based on file type
    if not hic:
        print("Processing with cooler library")
        data_matrix = Lib.matrix(balance=abalance, sparse=True).fetch(ccname).tocsr()
    else:
        print("Processing .hic file")
        norm_type = 'KR' if abalance else 'NONE'
        data_matrix = utils.csr_contact_matrix(norm_type, apath, ccname, ccname, 'BP', aresolution)
    
    # Create Chromosome instance and score
    print("Creating Chromosome instance...")
    X = scoreUtils.Chromosome(
        data_matrix,
        model=model,  
        cname=cikada, 
        lower=alower,
        upper=aupper, 
        res=aresolution,
        width=awidth,
        atac_file=atac_file,  
        ctcf_file=ctcf_file   
    )
    
    print("Running prediction...")
    result, R = X.score(thre=0.5)
    print("Prediction finished.")
    
    # Write results
    print("Saving results...")
    X.writeBed(aoutput, result, R)
    print(f"Results saved to {aoutput}")


if __name__ == "__main__":
    # Set parameters
    output = "/home/yanghao/TM-Loop/TM-Loop/candidate-loops/"
    model = "/home/yanghao/TM-Loop/TM-Loop/model-chr15/"
    path = "/public_data/yanghao/Rao2014-GM12878-MboI-allreps-filtered.10kb.cool"
    
    
    atac_file = "/home/yanghao/TM-Loop/TM-Loop/data/ATAC-ENCFF816ZFB.bigWig"  
    ctcf_file = "/home/yanghao/TM-Loop/TM-Loop/data/CTCF-ENCFF797LSC.bigWig"     
    
    # chromosome direct call
    chr_name = 'chr15'
    main(chr_name, output, model, path, atac_file, ctcf_file)