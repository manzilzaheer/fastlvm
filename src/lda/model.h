#ifndef _MODEL_H
#define _MODEL_H

#ifdef MULTIMACHINE
#include <mpi.h>
#endif

#include <algorithm>
#include <atomic> 
#include <fstream>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <vector>

#include "../commons/fast_rand.h"
#include "../commons/dataio.h"
#include "../commons/spvector.h"
#include "../commons/utils.h"

class model {

public:

    /****** constructor/destructor ******/
    model();
    virtual ~model();

    /****** interacting functions ******/
    static model* init(utils::ParsedArgs, const std::vector<std::string>&, int);// initialize the model randomly
    double fit(const DataIO::corpus&, const DataIO::corpus&);                   // train LDA using prescribed algorithm on training data
    double evaluate(const DataIO::corpus&) const;                               // test LDA according to specified method
    DataIO::corpus predict(const DataIO::corpus&) const;                 // test LDA according to specified method
    std::tuple<unsigned short, unsigned, double*> get_topic_matrix() const;     // test LDA according to specified method
    std::vector<std::vector<std::string>> get_top_words(unsigned num_words = 15) const;

protected:
    
    /****** Model Parameters ******/
    unsigned short K;                                   // Number of topics
    unsigned V;                                         // Number of words
    static constexpr double alpha = 50.0;                // per document topic proportions Dirichlet prior
    static constexpr double beta = 0.1;                 // Dirichlet language model prior
    std::vector<std::string> id2word;                   // word map [int => string]

    /****** Model variables ******/
    std::atomic<unsigned>* n_k;                         // number of tokens assigned to topic k
    std::atomic<unsigned>* n_wk;                        // number of times word w assigned to topic k
    std::vector<spvector> n_mks;                        // sparse representation of n_mk: number of words assigned to topic k in document m
    std::vector<spvector> p_mks;                        // sparse representation of n_mk: number of words assigned to topic k in document m
    double* p_wk;                                       // probability of word w assigned to topic k
    
    /****** Initialization ******/
    int init_train(const DataIO::corpus&);

    /****** Training aux ******/
    int rank;
    unsigned n_iters;                                   // Number of Gibbs sampling iterations
    unsigned n_save;                                    // Number of iters in between saving
    unsigned n_threads;                                 // Number of sampling threads
    unsigned n_top_words;                               // Number of top words to be printed per topic

    virtual int specific_init() { return 0; }           // if sampling algo need some specific inits
    virtual int sampling(const DataIO::corpus&, unsigned) { return 0; }      // sampling  outsourced to children
    virtual int updater();                              // updating sufficient statistics, can be outsourced to children
    void cleaner();
    void sharer();

    /****** Performance computations ******/
    std::vector<double> time_ellapsed;                  // time ellapsed after each iteration
    std::vector<double> likelihood;                     // likelihood after each iteration

    /****** File and Folder Paths ******/
    std::string name;                                   // dataset name
    std::string mdir;                                   // model directory

    /****** save LDA model to files ******/
    int save_model(unsigned iter) const;                // save model: call each of the following:      
    int save_model_time(std::string) const;             // model_name.time: time at which statistics calculated
    int save_model_llh(std::string) const;              // model_name.llh: lPer word likelihood on held out documents
    int save_model_top_words(std::string) const;        // model_name.twords: Top words in each top
    int save_model_phi(std::string) const;              // model_name.phi: topic-word distributions
    int save_model_params(std::string) const;
    
public:
    /****** serialize/deserialize the LDA model ******/
    char* serialize() const;                            // Serialize to a buffer
    size_t msg_size() const;
    void deserialize(char* buff);                       // Deserialize from a buffer
    
    void release() { p_wk = nullptr; }
};

#endif
