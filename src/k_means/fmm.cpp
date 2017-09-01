#include "fmm.h"

int simpleKM::sampling(const pointType& point)
{    
    // Compute probability for each cluster
    double maxProb = -1 * std::numeric_limits<double>::max();
    unsigned argMaxProb = -1;
    for (unsigned k = 0; k < K; ++k)
    {
        double p = clusters[k].computeProbability(point);
        if (maxProb < p)
        {
            maxProb = p;
            argMaxProb = k;
        }
    }

#ifdef DEBUG
    if (maxProb <= 0.0)
        throw std::runtime_error("MaxProb: " + std::to_string(maxProb) + "...Something wrong!");
#endif
    // Simple KM: Add point to nearest cluster
    clusters[argMaxProb].addPoint(point);
    n_k[argMaxProb] += 1;
    
    return 0;
}


int canopyKM::specific_init()
{
    // build cover-tree on cluster centres
    //std::cout << "Building mean tree at rank: " << rank << std::endl;
    ClusterTree = CoverTree::from_clusters(clusters);
    //std::cout << "Finished mean tree at rank: " << rank << std::endl;
    return 0;
}

int canopyKM::sampling(const pointType& point)
{
    
    std::pair<CoverTree::Node*, double> nn = ClusterTree->NearestNeighbour(point);
        
    clusters[nn.first->ID].addPoint(point);
    n_k[nn.first->ID] += 1;
    
    return 0;
}

int canopyKM::updater()
{
    // update mean, variance, pi
    utils::parallel_for_each(clusters.begin(), clusters.end(),
        [](SuffStatsOne& c){ c.updateParameters(); });

    // get rid of the old tree
    delete ClusterTree;
    
    // rebuild cover-tree on cluster centres
    //std::cout << "Building mean tree at rank: " << rank << std::endl;
    ClusterTree = CoverTree::from_clusters(clusters);
    //std::cout << "Finished mean tree at rank: " << rank << std::endl;    
    
    return 0;
}
