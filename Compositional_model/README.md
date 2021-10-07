This folder contains all the runnable notebooks Compositional Model. (made changes to notebooks in Compositional_model to make it work in "execute all cells" mode)


You can run the notebooks in the following order:

1) Synthetic Data Generation: 
    * generated files: 
        * gen1_33/fuzzing_data/10000_Composite_100001_integer.pt
        * gen1_33/fuzzing_data/20000_Composite_100001_integer.pt
        * gen1_33/fuzzing_data/30000_Composite_100001_integer.pt
        * gen1_33/fuzzing_data/40000_Composite_100001_integer.pt
        * gen1_33/fuzzing_data/50000_Composite_100001_integer.pt
        * gen1_33/fuzzing_data/60000_Composite_100001_integer.pt
        * gen1_33/fuzzing_data/70000_Composite_100001_integer.pt
        * gen1_33/fuzzing_data/80000_Composite_100001_integer.pt
        * gen1_33/fuzzing_data/90000_Composite_100001_integer.pt
    * You can use one of the above files as test data, and use different combinations of the remaining files as train data, and change the files used for training and testing in Model Training/Model Evaluation


2) Synthetic Data Encoding & Pre-processing
    * generated files:
        * gen1_33/training_embeddings/10000_training_embedding.pt
        * gen1_33/training_embeddings/20000_training_embedding.pt
        * gen1_33/training_embeddings/30000_training_embedding.pt
        * gen1_33/training_embeddings/40000_training_embedding.pt
        * gen1_33/training_embeddings/50000_training_embedding.pt
        * gen1_33/training_embeddings/60000_training_embedding.pt
        * gen1_33/training_embeddings/70000_training_embedding.pt
        * gen1_33/training_embeddings/80000_training_embedding.pt
        * gen1_33/90000_test_embedding.pt
3) Model Training: 
    * generated models:
        * 0_train_net_model.pt,  0_train_rnn_model.pt
        * 2_train_net_model.pt,  2_train_rnn_model.pt
        * 4_train_net_model.pt,  4_train_rnn_model.pt
        * ...
        * 18_train_net_model.pt,  18_train_rnn_model.pt
        * 20_train_net_model.pt,  20_train_rnn_model.pt
4) Model Evaluation
5) Compositionality Evaluation
6) Stack Overflow Benchmarks Evaluation: 
    * full
    * fos
7) Compositionality Visualization

