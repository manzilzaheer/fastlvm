#include "suff_stats.h"

SuffStatsOne::SuffStatsOne(unsigned numDims) : mtx(new std::mutex),
                                               mean(numDims),
                                               meanAcc(numDims),
                                               weight(0.0)
{   /*Only setup dimensions -> Inconsistent state*/    }

SuffStatsOne::SuffStatsOne(const pointType& m) : mtx(new std::mutex), 
                                          mean(m), meanAcc(m),
                                          weight(0.0)
{
    // Initialize sufficient statistics
    meanAcc.setZero();
}

int SuffStatsOne::init(const pointType& m)
{
    mtx->lock();
    // Initialize sufficient statistics
    weight = 0.0;
    meanAcc.setZero();
    
    // Initialize current mean
    mean = m;
    mtx->unlock();

    return 0;
}

double SuffStatsOne::computeProbability(const pointType& argPt) const
{
    double logProb = (mean - argPt).cwiseAbs2().sum();
    return -logProb;
}

void SuffStatsOne::updateParameters()
{
    double kappa = kappa0 + weight;
    
    // Update the mean
    mtx->lock();
    mean = (meanAcc.array() + kappa0*mu0) / kappa;

    mtx->unlock();
}

void SuffStatsOne::addPoint(const pointType& newPt)
{
    mtx->lock();
    weight += 1.0;
    meanAcc = meanAcc + newPt;
    mtx->unlock();
}

void SuffStatsOne::addPoint(const pointType& newPt, double weight)
{
    mtx->lock();
    this->weight += weight;
    meanAcc = meanAcc + weight*newPt;
    mtx->unlock();
}

void SuffStatsOne::resetParameters()
{
    mtx->lock();
    weight = 0.0;
    meanAcc.setZero();
    mtx->unlock();
}

#ifdef MULTIMACHINE
void SuffStatsOne::allreduce()
{
    unsigned D = mean.rows();
    mtx->lock();
    MPI_Allreduce(MPI_IN_PLACE, meanAcc.data(), D, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &weight, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    mtx->unlock();
}
void SuffStatsOne::synchronize()
{
    unsigned D = mean.rows();
    mtx->lock();
    MPI_Bcast(mean.data(), D, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    mtx->unlock();
}
#endif

int SuffStatsOne::write_to_file(std::ofstream& fout) const
{
    if (!fout)
        throw std::runtime_error("Cannot open to write!");
    size_t D = mean.rows();
    fout.write((char *)(mean.data()), sizeof(pointType::Scalar)*D);
    
    return 0;
}


std::ostream& operator<<(std::ostream& os, const SuffStatsOne& ct)
{
    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", "[", "]");
    os << "Cluster weight: " << ct.weight << std::endl;
    os << "\tM: " << ct.mean.format(CommaInitFmt) << std::endl;
    os << "\tSum: " << ct.meanAcc.format(CommaInitFmt) << std::endl;

    return os;
}

SuffStatsTwo::SuffStatsTwo(unsigned numDims) :
    mtx(new std::mutex),
    mean(numDims), varDiag(numDims),
    meanAcc(numDims), varAcc(numDims),
    weight(0.0), sumLogVar(0.0)
{   /*Only setup dimensions -> Inconsistent state*/    }

SuffStatsTwo::SuffStatsTwo(const pointType& m, const pointType& v) :
    mtx(new std::mutex), 
    mean(m), varDiag(v),
    meanAcc(m), varAcc(v),
    weight(0.0), sumLogVar((v.array().log()).sum())
{
    if(mean.rows() != varDiag.rows())
        throw std::runtime_error("Dimension of mean and variance do not match!");
    // Initialize sufficient statistics
    meanAcc.setZero();
    varAcc.setZero();
}

int SuffStatsTwo::init(const pointType& m, const pointType& v)
{
    mtx->lock();
    // Initialize sufficient statistics
    weight = 0.0;
    meanAcc.setZero();
    varAcc.setZero();
    
    // Initialize current mean/variance
    mean = m;
    varDiag = v;
    sumLogVar = (varDiag.array().log()).sum();
    mtx->unlock();

    return 0;
}

double SuffStatsTwo::computeProbability(const pointType& argPt) const
{
    double logProb = varDiag.dot((mean - argPt).cwiseAbs2());
    return 0.5 * sumLogVar - logProb;
}

void SuffStatsTwo::updateParameters()
{
    double kappa = kappa0 + weight;
    double nu = nu0 + weight;

    // Update the mean
    mtx->lock();
    mean = (meanAcc.array() + kappa0*mu0) / kappa;

    // Update the variance
    varDiag = (varAcc - kappa * mean.cwiseAbs2());
    varDiag = (varDiag.array() + nu0 * sigma0 + kappa0 * mu0 * mu0) / (3 + nu);

    // Store the inverse prior for the next iteration E step
    varDiag = 0.5 * varDiag.cwiseInverse();
    sumLogVar = (varDiag.array().log()).sum();
    mtx->unlock();
}

void SuffStatsTwo::addPoint(const pointType& newPt)
{
    mtx->lock();
    weight += 1.0;
    meanAcc = meanAcc + newPt;
    varAcc = varAcc + newPt.cwiseAbs2();
    mtx->unlock();
}

void SuffStatsTwo::addPoint(const pointType& newPt, double weight)
{
    mtx->lock();
    this->weight += weight;
    meanAcc = meanAcc + weight*newPt;
    varAcc = varAcc + weight*newPt.cwiseAbs2();
    mtx->unlock();
}

void SuffStatsTwo::resetParameters()
{
    mtx->lock();
    weight = 0.0;
    meanAcc.setZero();
    varAcc.setZero();
    mtx->unlock();
}

#ifdef MULTIMACHINE
void SuffStatsTwo::allreduce()
{
    unsigned D = mean.rows();
    mtx->lock();
    MPI_Allreduce(MPI_IN_PLACE, meanAcc.data(), D, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, varAcc.data(), D, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &weight, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    mtx->unlock();
}
void SuffStatsTwo::synchronize()
{
    unsigned D = mean.rows();
    mtx->lock();
    MPI_Bcast(mean.data(), D, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(varDiag.data(), D, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    mtx->unlock();
}
#endif

int SuffStatsTwo::write_to_file(std::ofstream& fout) const
{
    if (!fout)
        throw std::runtime_error("Cannot open to write!");
    unsigned D = mean.rows();
    fout.write((char *)(mean.data()), sizeof(pointType::Scalar)*D);
    pointType actualVariance = 0.5 * varDiag.cwiseInverse();
    fout.write((char *)(actualVariance.data()), sizeof(pointType::Scalar)*D);
    
    return 0;
}


std::ostream& operator<<(std::ostream& os, const SuffStatsTwo& ct)
{
    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", "[", "]");
    os << "Cluster weight: " << ct.weight << std::endl;
    os << "\tM: " << ct.mean.format(CommaInitFmt) << std::endl;
    os << "\tV: " << (0.5*ct.varDiag.cwiseInverse()).format(CommaInitFmt) << std::endl;
    os << "\tSum: " << ct.meanAcc.format(CommaInitFmt) << std::endl;
    os << "\tSum2: " << ct.varAcc.format(CommaInitFmt) << std::endl;

    return os;
}