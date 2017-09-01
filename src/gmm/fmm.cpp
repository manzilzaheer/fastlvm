#include "fmm.h"

int simpleGMM::sampling(const pointList& trngdata, unsigned i)
{    
    xorshift128plus rng_;
    double * p = new double[K]; // temp variable for sampling

    // for each point of worker m
    size_t M = trngdata.cols();
    for (size_t m = i; m < M; m+=n_threads)
    {
        const pointType& point = trngdata.col(m);
        //std::cout << "Point " << m << ": " << point << "\n" << std::endl;

        // Compute probability for each cluster
        double maxLogProb = -1 * std::numeric_limits<double>::max();
        for (unsigned k = 0; k < K; ++k)
        {
            p[k] = clusters[k].computeProbability(point) + logPi[k];
            if (maxLogProb < p[k])
                maxLogProb = p[k];
        }

        // Now subtract the minimum, exponentiate, add and divide
        double psum = 0.;
        for (unsigned k = 0; k < K; ++k)
        {
            p[k] -= maxLogProb;
            psum += exp(p[k]);
            p[k] = psum;
        }

        #ifdef DEBUG
        // Normalize
        if (psum <= 0.0)
        {
            std::cout << "SumProb: " + std::to_string(psum) + "...Something wrong!" << std::endl;
            throw std::runtime_error("SumProb: " + std::to_string(psum) + "...Something wrong!");
        }
        #endif

        // Stochastic EM: Add point to sampled cluster
        double u = rng_.rand_double() * psum;
        unsigned sample = std::lower_bound(p, p + K, u) - p;
        clusters[sample].addPoint(point);
        n_k[sample] += 1;
    }

    delete[] p;
    return 0;
}


// int canopyKM::specific_init()
// {
//     // build cover-tree on cluster centres
//     //std::cout << "Building mean tree at rank: " << rank << std::endl;
//     ClusterTree = CoverTree::from_clusters(clusters);
//     //std::cout << "Finished mean tree at rank: " << rank << std::endl;
//     return 0;
// }

// int canopyKM::sampling(const pointType& point)
// {
    
//     std::pair<CoverTree::Node*, double> nn = ClusterTree->NearestNeighbour(point);
        
//     clusters[nn.first->ID].addPoint(point);
//     n_k[nn.first->ID] += 1;
    
//     return 0;
// }

// int canopyKM::updater()
// {
//     // update mean, variance, pi
//     utils::parallel_for_each(clusters.begin(), clusters.end(),
//         [](SuffStatsOne& c){ c.updateParameters(); });

//     // get rid of the old tree
//     delete ClusterTree;
    
//     // rebuild cover-tree on cluster centres
//     //std::cout << "Building mean tree at rank: " << rank << std::endl;
//     ClusterTree = CoverTree::from_clusters(clusters);
//     //std::cout << "Finished mean tree at rank: " << rank << std::endl;    
    
//     return 0;
// }
