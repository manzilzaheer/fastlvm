//# define EIGEN_USE_MKL_ALL        //uncomment if available

# include <chrono>
# include <iostream>
# include <exception>

#include <future>
# include <thread>

# include <Eigen/Core>
#define EIGEN_DONT_PARALLELIZE

// User header
# include "cover_tree.h"
# include "utils.h"
# include "dataio.h"

// Compute the nearest neighbor using brute force, O(n)
pointType bruteForceNeighbor(const Eigen::Map<Eigen::MatrixXd>& pointMatrix, const pointType queryPt)
{
    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", "[", "]");

    pointType ridiculous = 1e200 * queryPt;
    pointType minPoint = ridiculous;
    double minDist = 1e300, curDist; // ridiculously high number
    

    for(size_t i = 0; i < pointMatrix.cols(); ++i)
    {
        const pointType& p = pointMatrix.col(i);
        curDist = (queryPt-p).norm();
        // Re-assign minimum
        if (minDist > curDist)
        {
            minDist = curDist;
            minPoint = p;
        }
    }
    //std::cout << "Min Point: " << minPoint.format(CommaInitFmt) 
    //        << " for query " << queryPt.format(CommaInitFmt) << std::endl;

    if (minPoint == ridiculous)
    {
        throw std::runtime_error("Something is wrong! Brute force neighbor failed!\n");
    }
    
    return minPoint;
}

void rangeBruteForce(const Eigen::Map<Eigen::MatrixXd>& pointMatrix, const pointType queryPt, const double range, const std::vector<std::pair<CoverTree::Node*, double>>& nnList)
{
    // Check for correctness
    for (const auto& node: nnList){
        if ( (node.first->_p-queryPt).norm() > range)
            throw std::runtime_error( "Error in correctness point - range!\n" );
        if (node.second > range)
            throw std::runtime_error( "Error in correctness distance - range!\n" );
    }
    
    // Check for completeness
    int numPoints = 0;
    for(size_t i = 0; i < pointMatrix.cols(); ++i) 
    {
        const pointType& pt = pointMatrix.col(i);
        if ((queryPt - pt).norm() < range)
            numPoints++;
    }

    if (numPoints != nnList.size()){
        throw std::runtime_error( "Error in completeness - range!\n Brute force: " + std::to_string( numPoints ) + " Tree: " + std::to_string(  nnList.size() ) );
    }
}


void nearNeighborBruteForce(const Eigen::Map<Eigen::MatrixXd>& pointMatrix, const pointType queryPt, const int numNbrs, const std::vector<std::pair<CoverTree::Node*, double>>& nnList)
{

    double leastClose = (nnList.back().first->_p - queryPt).norm();

    // Check for correctness
    if (nnList.size() != numNbrs){
        std::cout << nnList.size() << " vs " << numNbrs << std::endl;
        throw std::runtime_error( "Error in correctness - knn (size)!" );
    }


    for (const auto& node: nnList)
    {
        if ( (node.first->_p - queryPt).norm() > leastClose + 1e-6)
        {
            std::cout << leastClose << " " << (node.first->_p-queryPt).norm() << std::endl;
            for (const auto& n: nnList) 
                std::cout << (n.first->_p-queryPt).norm() << std::endl;
            throw std::runtime_error( "Error in correctness - knn (dist)!" );
        }
        if ( node.second > leastClose + 1e-6)
        {
            std::cout << leastClose << " " << node.second << std::endl;
            for (const auto& n: nnList) 
                std::cout << n.second << std::endl;
            throw std::runtime_error( "Error in correctness - knn (dist)!" );
        }
    }
    
    // Check for completeness
    int numPoints = 0;
    std::vector<double> dists(pointMatrix.cols());
    for(size_t i = 0; i < pointMatrix.cols(); ++i)
    {
        const pointType& pt = pointMatrix.col(i);
        double dist = (queryPt - pt).norm();
        if (dist <= leastClose - 1e-6){
            numPoints++;
            dists[i] = dist;
        }
    }

    if (numPoints != nnList.size()-1){
        std::cout << "Error in completeness - k-nn!\n";
        std::cout << "Brute force: " << numPoints << " Tree: " << nnList.size();
        std::cout << std::endl;
        for (auto dist : dists) std::cout << dist << " ";
        std::cout << std::endl;
    }
}


int main(int argv, char** argc)
{
    if (argv < 2)
        throw std::runtime_error("Usage:\n./main <path_to_train_points> <path_to_test_points>");

    std::cout << argc[1] << std::endl;
    std::cout << argc[2] << std::endl;
    
    Eigen::setNbThreads(1);
    std::cout << "Number of OpenMP threads: " << Eigen::nbThreads();

    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", "[", "]");
    std::chrono::high_resolution_clock::time_point ts, tn;
    
    // Reading the file for points
    Eigen::Map<Eigen::MatrixXd> trngdata = DataIO::readPointFile(argc[1]);
    Eigen::Map<Eigen::MatrixXd> testdata = DataIO::readPointFile(argc[2]);

    CoverTree* cTree;
    // Parallel Cover tree construction
    ts = std::chrono::high_resolution_clock::now();
    cTree = CoverTree::from_matrix(trngdata, -1, false);  
    tn = std::chrono::high_resolution_clock::now();
    std::cout << "Build time: " << std::chrono::duration_cast<std::chrono::milliseconds>(tn - ts).count() << std::endl;
    std::cout << "Number of points in the tree: " << cTree->count_points() << std::endl;

    // find the nearest neighbor
    std::cout << "Quering Cover Tree NN" << std::endl;
    ts = std::chrono::high_resolution_clock::now();
    std::vector<pointType> ct_neighbors(testdata.cols());
    utils::parallel_for_progressbar(0, testdata.cols(), [&](size_t i)->void{
        const pointType& queryPt = testdata.col(i);
        std::pair<CoverTree::Node*, double> ct_nn = cTree->NearestNeighbour(queryPt);
        ct_neighbors[i] = ct_nn.first->_p;
    });
    tn = std::chrono::high_resolution_clock::now();
    std::cout << "Query time: " << std::chrono::duration_cast<std::chrono::milliseconds>(tn - ts).count() << std::endl;

    std::cout << "Quering Brute Force" << std::endl;
    ts = std::chrono::high_resolution_clock::now();
    std::vector<pointType> bt_neighbors(testdata.cols());
    utils::parallel_for_progressbar(0, testdata.cols(), [&](size_t i)->void{
        const pointType& queryPt = testdata.col(i);
        bt_neighbors[i] = bruteForceNeighbor(trngdata, queryPt);
    });
    tn = std::chrono::high_resolution_clock::now();
    std::cout << "Query time: " << std::chrono::duration_cast<std::chrono::milliseconds>(tn - ts).count() << std::endl;

    //Match answers
    int problems = 0;
    for(size_t i=0; i<testdata.cols(); ++i)
    {
        if (!ct_neighbors[i].isApprox(bt_neighbors[i]))
        {
            problems += 1;
            std::cout << "Something is wrong" << std::endl;
            std::cout << ct_neighbors[i].format(CommaInitFmt) << " " << bt_neighbors[i].format(CommaInitFmt) << " " << testdata.col(i).format(CommaInitFmt) << std::endl;
            std::cout << (ct_neighbors[i] - testdata.col(i)).norm() << " ";
            std::cout << (bt_neighbors[i] - testdata.col(i)).norm() << std::endl;
        }
    }
    if (problems)
        std::cout << "Nearest Neighbour test failed!" << std::endl;
    else
        std::cout << "Nearest Neighbour test passed!" << std::endl;
    
    std::cout << "Check k-NN" << std::endl;
    utils::parallel_for_progressbar(0, testdata.cols(), [&](size_t i)->void{
        const pointType& queryPt = testdata.col(i);
        std::vector<std::pair<CoverTree::Node*, double>> nnList = cTree->kNearestNeighbours(queryPt, 3);
        nearNeighborBruteForce(trngdata, queryPt, 3, nnList);
    });
    
    std::cout << "Check range" << std::endl;
    utils::parallel_for_progressbar(0, testdata.cols(), [&](size_t i)->void{
        const pointType& queryPt = testdata.col(i);
        std::vector<std::pair<CoverTree::Node*, double>> nnList = cTree->rangeNeighbours(queryPt, 10);
        rangeBruteForce(trngdata, queryPt, 10, nnList);
    });

    system("pause");

    // Clear memory
    if(trngdata.data()) delete[] trngdata.data();
    if(testdata.data()) delete[] testdata.data();
    
    // Success
    return 0;
}
