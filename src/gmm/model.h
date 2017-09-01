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
#include <vector>

#include "../commons/suff_stats.h"
#include "../commons/utils.h"

typedef Eigen::Map<Eigen::MatrixXd> pointList;

class model {

public:

    /****** constructor/destructor ******/
    model();
    virtual ~model();

    /****** interacting functions ******/
    static model* init(utils::ParsedArgs, const pointList&, const pointList&, int);// initialize the model randomly
    double fit(const pointList&, const pointList&);              // train GMM using prescribed algorithm on training data
    double evaluate(const pointList&) const;                     // test GMM according to specified method
    std::vector<unsigned> predict(const pointList&) const;       // test GMM according to specified method
    std::pair<pointList, pointList> get_centers() const;

protected:
    
    /****** Model Parameters ******/
    unsigned K;                                         // Number of clusters
    static constexpr double alpha = 1.0;                // mixture proportions dirichlet prior

    /****** Model variables ******/
    std::atomic<unsigned> * n_k;                        // number of points assigned to cluster k
    std::vector<double> logPi;
    std::vector<SuffStatsTwo> clusters;                 // Information about the clusters

    /****** Training aux ******/
    int rank;
    unsigned n_iters;                                   // Number of Gibbs sampling iterations
    unsigned n_save;                                    // Number of iters in between saving
    unsigned n_threads;                                 // number of sampling threads

    virtual int specific_init() { return 0; } ;         // if sampling algo need some specific inits
    virtual int sampling(const pointList&, unsigned) { return 0; }      // sampling on thread i outsourced to children
    virtual int updater();                              // updating sufficient statistics, can be outsourced to children
    void cleaner();
    void sharer();

    /****** Performance computations ******/
    std::vector<double> time_ellapsed;                  // time ellapsed after each iteration
    std::vector<double> likelihood;                     // likelihood after each iteration

    /****** File and Folder Paths ******/
    std::string name;                                   // dataset name
    std::string mdir;                                   // model directory

    /****** save FMM model to files ******/
    int save_model(int iter) const;                     // save model: call each of the following:      
    int save_model_time(std::string filename) const;    // model_name.time: time at which statistics calculated
    int save_model_llh(std::string filename) const;     // model_name.llh: likelihood on held out points
    int save_model_phi(std::string filename) const;     // model_name.phi: cluster-centres
    
public:
    /****** serialize/deserialize the FMM model ******/
    char* serialize() const;                            // Serialize to a buffer
    size_t msg_size() const;
    void deserialize(char* buff);                       // Deserialize from a buffer
};

#endif
