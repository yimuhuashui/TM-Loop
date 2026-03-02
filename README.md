# TM-Loop
# Introduction
TM-Loop: Transformer Multi-omics Hierarchical detection of chromatin Loop.
# Installation
TM-Loop is developed and tested on Linux machines and relies on the following packages:
```

tensorflow
keras
scikit-learn  
hic-straw
joblib  
numpy 
scipy   
pandas 
h5py   
cooler 
```
Create an environment.
```
git clone https://github.com/yimuhuashui/TM-Loop.git 
conda create -n TM-Loop python=3.6.13   
conda activate TM-Loop    
```
# Usage
## Feature extraction 
The datasets required by the model can be downloaded from auxiliary materials and other websites. Feature extraction of the datasets can be carried out in feature.py. Run: 
```
python feature.py
```
If you want to change the parameters, you can modify them by feature.py the parameters in the code file,Example:
```
    path = "/public_data/yanghao/Rao2014-GM12878-MboI-allreps-filtered.10kb.cool"
    output = "/home/yanghao/TM-Loop/TM-Loop/model-chr15"
    bedpe = "/home/yanghao//TM-Loop/TM-Loop/data/gm12878_ctcf_h3k27ac.bedpe"
    atac_file = "/home/yanghao/TM-Loop/TM-Loop/data/ATAC-ENCFF816ZFB.bigWig"
```
"path" is the path to the hic data file, "output" is the output path of the model, and "bedpe" is the path to the bedpe data file, "bigwig" is the path to the bigwig data file.
## Model training
The extracted features need to be trained in trainmodel.py: 
```
    python trainmodel.py
```
If you want to change the training chromosomes, you can modify them by trainmodel.py the parameters in the code file,Example:
```
    test_chrom = 'chr15'
```
"test_chrom" is the chromosome that needs training model.
## Model predicting
By predict.py prediction of the trained model, candidate chromatin loops are obtained, and run:  
```
python predict.py 
```
Example of modifiable parameters:
```
    output = "/home/yanghao/TM-Loop/TM-Loop/candidate-loop/"
    model = "/home/yanghao/TM-Loop/TM-Loop/model-chr15/"
    path = "/public_data/yanghao/Rao2014-GM12878-MboI-allreps-filtered.10kb.cool"
    atac_file = "/home/yanghao/TM-Loop/TM-Loop/data/ATAC-ENCFF816ZFB.bigWig"
```
"output" is the output path of the candidate chromatin loop, "model" is the save path of the trained model, and "path" is the path of the hic data file, "bigwig" is the path to the bigwig data file.
## Clustering
Run the cluster.py file to perform clustering and obtain the final chromatin loop file.
```
    python cluster.py 
```
Example of modifiable parameters:
```
    infile = "/home/yanghao/TM-Loop/TM-Loop/candidate-loop/chr15.bed"
    outfile = "/home/yanghao/TM-Loop/TM-Loop/loop/chr15.bedpe"
    threshold = 0.985
```
"infile" is the path for the candidate chromatin loop file, "outfile" is the path for the final chromatin loop, and "threshold" is the clustering threshold.
