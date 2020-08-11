# AML4DHGermanVecs
This repository contains code to train and evaluate German medical embeddings.
Example code and usage examples are provided in Jupyter notebooks.

## Setup:
Define relevant paths in config.json (have a look at the default file).


To extend the processing pipeline do the following:

## Adding further preprocessing methods
Extend vectorization/embeddings.py->Embeddungs.sentence_data2vec() before calling Embeddings.calculate_vectors()

## Adding further vectorization algorithms
Extend vectorization/embeddings.py->Embeddings.calculate_vectors() by vectorization which accepts sentences as input

## Adding further resources
Add new ressources to resource/other_resources.py or resource/UMLS.py by inheriting the abstract class Evaluator and implementing its abstract methods

## Adding further benchmarks
Add new benchmarks to benchmarking/benchmarks by inheriting the abstract class Benchmark and implementing its abstract methods. 
Use the constructor to define relevant resource files such as knowledge bases, which will be passed by evaluate_embeddings.py as list.
