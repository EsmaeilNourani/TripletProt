# TripletProt
Deep Representation Learning of Proteins based on Siamese Networks
+ In this work, we introduce TripletProt, a new approach for protein representation learning based on the Siamese neural networks trained over the protein-protein interaction (PPI) network. 

![TripletProt Architecture](/images/siamese.png)


+ We evaluate TripletProt in protein annotation tasks including  [`sub-cellular localization`](/Applications/subcellular_location)  and  [`gene ontology prediction`](/Applications/Function) , which are both multi-class multi-label classification machine learning problems.

## Data
* Function Prediction : http://deepgoplus.bio2vec.net/data/deepgo/data.tar.gz

  * We have used the following files:
    * test-bp.pkl, test-cc.pkl , test-mf.pkl
    * train-bp.pkl, train-cc.pkl, train-mf.pkl

* sub-cellular localization

  * Multi Kernel SVM [1]
  * REALoc [2] http://predictor.nchu.edu.tw/REALoc/S1_dataset.zip


## Execution

1. Unzip the files located in [`pickle files`](/data/pickles) to get the String IDs of proteins and Unirep embeddings  
2. Use [`sample visualization notebook`]( /Applications/subcellular_location/TripletProt_vs_REALoc/t-SNE_Visualization.ipynb) to generate TripletProt Embeddings

## References:

  + [1] Shen, Y., Tang, J. and Guo, F., 2019. Identification of protein subcellular localization via integrating evolutionary and physicochemical information into Chou’s general PseAAC. Journal of Theoretical Biology, 462, pp.230-239.

  + [2] Tung, C.H., Chen, C.W., Sun, H.H. and Chu, Y.W., 2017. Predicting human protein subcellular localization by heterogeneous and comprehensive approaches. PloS one, 12(6), p.e0178832.










