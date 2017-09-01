#include "utils.h"

// Initializing using kmeans++
size_t* utils::KMeanspp(const Eigen::MatrixXd& points, unsigned K)
{
    size_t N = points.cols();     // number of points
    
    size_t * seeds = new size_t[K];    // Container for centres

    xorshift128plus rng_;           // random number

    // Pick any point as first seed uniformly at random
    seeds[0] = rng_.rand_k(N);

    // Store the distances
    double* dist = new double[N];
    std::fill(dist, dist + N, 1e300);
    double* cumdist = new double[N];

    for (unsigned k = 1; k < K; ++k)
    {
        // Compute distance to previous selected mean
        utils::parallel_for_progressbar(0, N, [&](size_t i)->void{
            double temp = (points.col(i) - points.col(seeds[k-1])).squaredNorm();
            if(temp<dist[i]) dist[i] = temp;
        });

        double dsum = 0;
        for(size_t i = 0; i<N; ++i)
        {
            dsum += dist[i];
            cumdist[i] = dsum;
        }

        double u = rng_.rand_double() * dsum;
        seeds[k] = std::lower_bound(cumdist, cumdist + N, u) - cumdist;
        //std::cout << "select: " << seeds[k] << ", " << u << ", " << dsum << std::endl;
        //seeds.col(k) = points.col(choice);
    }

    delete[] cumdist;
    delete[] dist;
    return seeds;
}
