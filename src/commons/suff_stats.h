#ifndef _DATASET_H
#define _DATASET_H

#ifdef MULTIMACHINE
#include <mpi.h>
#endif

#include <algorithm>
#include <atomic>
#include <fstream>
#include <iostream>
#include <mutex>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Cholesky>
typedef Eigen::VectorXd pointType;


class SuffStatsOne
{
protected:
    std::mutex* mtx;
    pointType mean;         // mean of the cluster
    pointType meanAcc;      // sum of points associated to the cluster
    double weight;          // sum of weights of points assoiciated to the cluster (~= numPoints)

    //prior values
    static constexpr double kappa0 = 0.0001;
    static constexpr double mu0 = 0;

public:
    SuffStatsOne() : mtx(nullptr)
    {     }
    SuffStatsOne(unsigned);                    // initialize with Dimensions Only -> use init then
    SuffStatsOne(const pointType&);            // initialize with Mean
    ~SuffStatsOne()
    { if (mtx) delete mtx; }

    int init(const pointType&);                            // initialize mean
    double computeProbability(const pointType&) const;     // For a point compute probability of generation
    void updateParameters();                               // Updating the parameters without prior
    void addPoint(const pointType&);                       // Add a point to the parameter
    void addPoint(const pointType&, double);               // Add a point to the parameter, with a weight
    void resetParameters();                                // Reset the accumulated parameters with prior

    #ifdef MULTIMACHINE
    void allreduce();                                      // Use MPI to accumulate from other machines
    void synchronize();
    #endif

    int write_to_file(std::ofstream& fout) const;
    friend std::ostream& operator<<(std::ostream& os, const SuffStatsOne& ct);
    
    intptr_t get_dim() const {return mean.size();}
    pointType get_mean() const {return mean;}
};

class SuffStatsTwo
{
protected:
    std::mutex* mtx;
    pointType mean;         // mean of the cluster
    pointType varDiag;      // inverse diagonal of the covariance matrix = 1/(2*sigma^2)
    pointType meanAcc;      // sum of points associated to the cluster
    pointType varAcc;       // sum of element-wise square of points associated to the cluster
    double weight;          // sum of weights of points assoiciated to the cluster (~= numPoints)
    double sumLogVar;       // sum of log variances (caching to speed-up computation)

    //prior values
    static constexpr double kappa0 = 0.0001;
    static constexpr double nu0 = 3;
    static constexpr double mu0 = 0;
    static constexpr double sigma0 = 0.001;

public:
    SuffStatsTwo() : mtx(nullptr)
    {     }
    SuffStatsTwo(unsigned);                            // initialize with Dimensions Only -> use init then
    SuffStatsTwo(const pointType&, const pointType&);  // initialize with Mean and Variance
    ~SuffStatsTwo()
    { if (mtx) delete mtx; }

    int init(const pointType&, const pointType&);      // initialize mean and variance
    double computeProbability(const pointType&) const; // For a point compute probability of generation
    void updateParameters();                           // Updating the parameters without prior
    void addPoint(const pointType&);                   // Add a point to the parameter
    void addPoint(const pointType&, double);           // Add a point to the parameter, with a weight
    void resetParameters();                            // Reset the accumulated parameters with prior

    #ifdef MULTIMACHINE
    void allreduce();                                  // Use MPI to accumulate from other machines
    void synchronize();
    #endif

    int write_to_file(std::ofstream& fout) const;
    friend std::ostream& operator<<(std::ostream& os, const SuffStatsTwo& ct);

    intptr_t get_dim() const {return mean.size();}
    pointType get_mean() const { return mean; }
    pointType get_var() const { return varDiag; }

};

class SuffStatsThree
{
protected:
    std::mutex* mtx;
    pointType mean;                           // mean of the cluster
    Eigen::LLT<Eigen::MatrixXd> varChol;      // inverse diagonal of the covariance matrix = 1/(2*sigma^2)
    pointType meanAcc;                        // sum of points associated to the cluster
    Eigen::LLT<Eigen::MatrixXd> varAcc;       // sum of element-wise square of points associated to the cluster
    double weight;                            // sum of weights of points assoiciated to the cluster (~= numPoints)
    double logDet;                            // log determinant (caching to speed-up computation)

    //prior values
    static constexpr double kappa0 = 0.0001;
    static constexpr double nu0 = 3;
    static constexpr double mu0 = 0;
    static constexpr double sigma0 = 0.001;

public:
    SuffStatsThree() : mtx(nullptr)
    {     }
    SuffStatsThree(unsigned);                                  // initialize with Dimensions Only -> use init then
    SuffStatsThree(const pointType&, const Eigen::MatrixXd&);  // initialize with Mean and Variance
    ~SuffStatsThree()
    { if (mtx) delete mtx; }

    int init(const pointType&, const Eigen::MatrixXd&);         // initialize mean and variance
    double computeProbability(const pointType&) const;          // For a point compute probability of generation
    void updateParameters();                                    // Updating the parameters without prior
    void addPoint(const pointType&);                            // Add a point to the parameter
    void addPoint(const pointType&, double);                    // Add a point to the parameter, with a weight
    void resetParameters();                                     // Reset the accumulated parameters with prior

    #ifdef MULTIMACHINE
    void allreduce();                                           // Use MPI to accumulate from other machines
    void synchronize();
    #endif

    int write_to_file(std::ofstream& fout) const;
    friend std::ostream& operator<<(std::ostream& os, const SuffStatsTwo& ct);

    intptr_t get_dim() const {return mean.size();}
    pointType get_mean() const { return mean; }
    Eigen::MatrixXd get_var() const { return varChol.reconstructedMatrix(); }

};

#endif

