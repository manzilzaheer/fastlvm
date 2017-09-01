# Fast Sampling for Latent Variable Models

We present implementation of following latent variable models suitable for large scale deployment:

1. `CoverTree` - Fast nearest neighbour search
2. `KMeans` - Simple, fast, and distributed clustering with option of various initialization
3. `GMM` - Fast and distributed inference for Gaussian Mixture Models with diagonal covariance matrices
4. `LDA` - Fast and distributed inference for Latent Dirichlet Allocation
5. `GLDA` - Fast and distributed inference for Gaussian LDA with diagonal covariance matrices
6. `HDP` - Fast inference for Hierarchical Dirichlet Process
 
Under active development

## Organisation
1. All codes are under `src` within respective folder
2. Dependencies are provided under `lib` folder
3. Python wrapper classes reside in `fastlvm` folder
4. For running different models an example script is provided under `scripts`
5. `data` is a placeholder folder where to put the data
6. `build` and `dist` folder will be created to hold the executables

## Requirements
1. gcc >= 5.0 or Intel&reg; C++ Compiler 2017 for using C++14 features
2. Python 3.5+

## How to use
There are two ways to utilize the package: using Python wrapper or directly in C++

### Python
Just use `python setup.py install` and then in python you can `import fastlvm`. Example and test code is in `test.py`.
The python API details are provided in `API.pdf`, but all of the models utilise the following structure:

    class LVM:
        init(self, # hyperparameters)
            return model
        
        fit(self, X, ...):
            return validation score
            
        predict(self, X): 
            return prediction on each test example
            
        evaluate(self, X):             
            return test score

 If you do not have root priveledges, install with `python setup.py install --user` and make sure to have the folder in path. 
 
### C++
We will show how to compile our package and run, for example nearest neighbour search using cover trees, on a single machine using synthetic dataset

1. First of all compile by hitting make

   ```bash
     make
   ```

2. Generate synthetic dataset

   ```bash
     python data/generateData.py
   ```


3. Run Cover Tree

   ```bash
      dist/cover_tree data/train_100d_1000k_1000.dat data/test_100d_1000k_10.dat
   ```

The make file has some useful features:

- if you have Intel&reg; C++ Compiler, then you can instead

   ```bash
     make intel
   ```

- or if you want to use Intel&reg; C++ Compiler's cross-file optimization (ipo), then hit
   
   ```bash
     make inteltogether
   ```

- Also you can selectively compile individual modules by specifying

   ```bash
     make <module-name>
   ```

- or clean individually by

   ```bash
     make clean-<module-name>
   ```

## Performance


## Attributions
We use a distributed and parallel extension and implementation of Cover Tree data structure for nearest neighbour search. The data structure was originally presented in and improved in:

1. Alina Beygelzimer, Sham Kakade, and John Langford. "Cover trees for nearest neighbor." Proceedings of the 23rd international conference on Machine learning. ACM, 2006.
2. Mike Izbicki and Christian Shelton. "Faster cover trees." Proceedings of the 32nd International Conference on Machine Learning (ICML-15). 2015.

We implement a modified inference for Gaussian LDA. The original model was presented in:

1. Rajarshi Das, Manzil Zaheer, Chris Dyer. "Gaussian LDA for Topic Models with Word Embeddings." Proceedings of ACL (pp. 795-804) 2015.

We implement a modified inference for Hierarchical Dirichlet Process. The original model and inference methods were presented in:

1. Y. Teh, M. Jordan, M. Beal, and D. Blei. Hierarchical dirichlet processes. Journal of the American Statistical Association, 101(576):1566{1581, 2006.
2. C. Chen, L. Du, and W.L. Buntine. Sampling table configurations for the hierarchical poisson-dirichlet process. In  European Conference on Machine Learning, pages 296-311. Springer, 2011.

## Troubleshooting
If the build fails and throws error like "instruction not found", then most probably the system does not support AVX2 instruction sets. To solve this issue, in `setup.py` and `src/cover_tree/makefile` please change `march=core-avx2`to `march=corei7`.

