#ifndef _MODEL_H
#define	_MODEL_H

#include <algorithm>
#include <atomic> 
#include <fstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <vector>

#include "../commons/circqueue.h"
#include "../commons/fast_rand.h"
#include "../commons/dataio.h"
#include "../commons/minheap.h"
#include "../commons/taskqueue.h"
#include "../commons/spvector.h"
#include "../commons/utils.h"

class model {

public:

    /****** constructor/destructor ******/
    model();
    virtual ~model();

    /****** interacting functions ******/
    static model* init(utils::ParsedArgs, const std::vector<std::string>&, int);// initialize the model randomly
    double fit(const DataIO::corpus&, const DataIO::corpus&);                   // train HDP using prescribed algorithm on training data
    double evaluate(const DataIO::corpus&) const;                               // test HDP according to specified method
    DataIO::corpus predict(const DataIO::corpus&) const;                        // test HDP according to specified method
    std::tuple<unsigned short, unsigned, double*> get_topic_matrix() const;     // test HDP according to specified method
    std::vector<std::vector<std::string>> get_top_words(unsigned num_words = 15) const;

protected:
    
    /****** Model Parameters ******/
    unsigned short K;                                   // Number of topics
    unsigned V;                                         // Number of words in dictionary
    static constexpr unsigned short Kmax = 2000;            // Max number of topics
    static constexpr double alpha = 50.0;               // Per document topic proportions Dirichlet prior
    static constexpr double beta = 0.1;                 // Dirichlet language model prior
    double gamma;                                       // Root DP parameter
    double Vbeta;
    std::vector<std::string> id2word;                   // Word map [int => string]

    /****** Model variables ******/
    std::atomic<unsigned>* n_k;                         // Number of words assigned to topic k = sum_w n_wk = sum_m n_mk
    unsigned* n_wk;                                     // Number of times word w assigned to topic k
    double* p_wk;                                       // probabilit of word w given topic k
    unsigned short ** z;                                // Topic assignment for each word
    std::vector<spvector> n_mks;                        // Sparse representation of n_mk: number of words assigned to topic k in document m
    double* tau;                                        // Stick breaking for root
    double tau_left;
    
    /****** Nonparametric aux ******/
    MinHeap<unsigned short> kGaps;                      // Empty topic ids
    std::vector<unsigned short> kActive;                // Topic ids in use
    unsigned short spawn_topic();                       // Adds a topic to the list of active topics
    xorshift128plus global_rng;
    std::mutex mtx;
    
    /****** Initialization ******/
    virtual int init_train(const DataIO::corpus&);

    /****** Training aux ******/
    int rank;
    unsigned n_iters;                                   // Number of Gibbs sampling iterations
    unsigned n_save;                                    // Number of iters in between saving
    unsigned n_threads;                                 // Number of sampling threads
    unsigned n_top_words;                               // Number of top words to be printed per topic

    virtual int specific_init() { return 0; }           // if sampling algo need some specific inits
    virtual int sampling(const DataIO::corpus&, unsigned) { return 0; }	// sampling on document i outsourced to children
    virtual int updater() { return 0; }                 // updating sufficient statistics, can be outsourced to children
    virtual int writer(unsigned i);

    /****** Concurency parameters ******/
    virtual unsigned num_table_threads() const { return n_threads/16 > 0 ? n_threads/16 : 1; }
    volatile bool inf_stop;                         // flag for stopping inference
    task_queue doc_queue;
    struct delta                                    // sufficient statistic update message
    {
        unsigned word;
        unsigned short old_topic;
        unsigned short new_topic;
        bool table_destroy;
        bool table_create;

        delta()
        {   }

        delta(unsigned a, unsigned short b, unsigned short c) : word(a), old_topic(b), new_topic(c), table_destroy(false), table_create(false)
        {   }

        delta(unsigned a, unsigned short b, unsigned short c, bool d, bool e) : word(a), old_topic(b), new_topic(c), table_destroy(d), table_create(e)
        {   }
    };
    circular_queue<delta> *cbuff;                   // buffer for messages NST*NTT

    /****** Performance computations ******/
    std::vector<double> time_ellapsed;                  // time ellapsed after each iteration
    std::vector<double> likelihood;                     // likelihood after each iteration

    /****** File and Folder Paths ******/
    std::string name;                                   // dataset name
    std::string mdir;                                   // model directory

    /****** save HDP model to files ******/
    int save_model(unsigned iter) const;                // save model: call each of the following:		
    int save_model_time(std::string) const;	            // model_name.time: time at which statistics calculated
    int save_model_llh(std::string) const;	            // model_name.llh: Per word likelihood on held out documents
    int save_model_top_words(std::string) const;        // model_name.twords: Top words in each top
    int save_model_phi(std::string) const;              // model_name.phi: topic-word distributions
    int save_model_params(std::string) const;           // model_name.params: containing other parameters of the model (alpha, beta, M, V, K)

public:
    /****** serialize/deserialize the HDP model ******/
    char* serialize() const;                            // Serialize to a buffer
    size_t msg_size() const;
    void deserialize(char* buff);                       // Deserialize from a buffer
    
    void release() { p_wk = nullptr; }
};

#endif	/* MODEL_H */

