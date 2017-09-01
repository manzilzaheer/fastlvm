#include "glda.h"

int adGLDA::sampling(const DataIO::corpus& trngdata, unsigned i)
{
    xorshift128plus rng_;
    
    double alphaK = alpha/K;

    double * p = new double[K]; // temp variable for sampling
    unsigned *nd_m = new unsigned[K];
    std::fill(nd_m, nd_m + K, 0);
    unsigned short *rev_mapper = new unsigned short[K];
    std::fill(rev_mapper, rev_mapper + K, K);
    
    //std::cout << "I reached 2 " << std::endl;
    
    // for each document of worker i
    size_t M = trngdata.size();
    for (size_t m = i; m < M; m+=n_threads)
    {
        //std::cout<<i << ", " << m<< std::endl;
        auto & read_n_mks = p_mks[m];
        auto & write_n_mks = n_mks[m];
        const auto& doc = trngdata[m];
        
        //std::cout << "I reached 3 " << std::endl;

        for (const auto& k : read_n_mks)
            nd_m[k.idx] = k.val;
        
        //std::cout << "I reached 4 " << std::endl;
        
        for (const auto& w : doc)
        {
            //std::cout << "I reached 5 " << std::endl;

            // do multinomial sampling via cumulative method
            const pointType& wvec = id2vec->col(w);

            //std::cout << "I reached 6 " << std::endl;

            // Compute probability for each topic
            double maxLogProb = -1 * std::numeric_limits<double>::max();
            for (unsigned short k = 0; k < K; ++k)
            {
                p[k] = log(nd_m[k] + alphaK) + topics[k].computeProbability(wvec);
                if (maxLogProb < p[k])
                    maxLogProb = p[k];
            }

            //std::cout << "I reached 7 " << std::endl;

            // Now subtract the minimum, exponentiate, add and divide
            double psum = 0.;
            for (unsigned short k = 0; k < K; ++k)
            {
                p[k] -= maxLogProb;
                psum += exp(p[k]);
                p[k] = psum;
            }

            #ifdef DEBUG
            // Normalize
            if (sumProb <= 0.0)
                throw std::runtime_error("SumProb: " + std::to_string(sumProb) + "...Something wrong!");
            #endif
                            
            // scaled sample because of unnormalized p[]
            double u = rng_.rand_double() * psum;
            
            //std::cout << "I reached 8 " << std::endl;

            // Do a binary search instead!
            unsigned short topic = std::lower_bound(p, p + K, u) - p;
            
            //std::cout << "I reached 8 " << std::endl;

            //if ( (0 > topic) || (topic >= K) )
            //  std::cout<<w<<", "<<m<<", "<<n<<", "<<topic<<std::endl;

            // number of instances of word i assigned to topic j
            topics[topic].addPoint(wvec);
            
            //std::cout << "I reached 9 " << std::endl;

            // number of words in document i assigned to topic j
            if (rev_mapper[topic] == K) 
                rev_mapper[topic] = write_n_mks.push_back(topic, 1);
            else
                write_n_mks.increment(rev_mapper[topic]);

        }
        for (const auto& k : write_n_mks)
            rev_mapper[k.idx] = K;
        for (const auto& k : read_n_mks)
            nd_m[k.idx] = 0;
    }

    delete[] p;
    delete[] nd_m;
    delete[] rev_mapper;
    return 0;
}

int scaGLDA::specific_init()
{
    q.resize(V, voseAlias(K));
    return 0;
}

int scaGLDA::sampling(const DataIO::corpus& trngdata, unsigned i)
{
    xorshift128plus rng_;
    
    double alphaK = alpha/K;

    double * p = new double[K]; // temp variable for sampling
    unsigned short *rev_mapper = new unsigned short[K];
    std::fill(rev_mapper, rev_mapper + K, K);
    
    // for each document of worker i
    size_t M = trngdata.size();
    for (size_t m = i; m < M; m += n_threads)
    {
        const auto& read_n_mks = p_mks[m];
        auto& write_n_mks = n_mks[m];
        const auto& doc = trngdata[m];

        for (const auto& w : doc)
        {
            // do multinomial sampling via cumulative method
            const pointType& wvec = id2vec->col(w);           
            
            /* Travese all non-zero document-topic distribution */
            // Compute probability for each topic
            unsigned short ii = 0;
            double maxLogProb = -1 * std::numeric_limits<double>::max();
            for (const auto& k : read_n_mks)
            {
                p[ii] = log(k.val) + topics[k.idx].computeProbability(wvec);
                if (maxLogProb < p[ii])
                    maxLogProb = p[ii];
                ++ii;
            }

            //std::cout << "I reached 7 " << std::endl;

            // Now subtract the minimum, exponentiate, add and divide
            double psum = 0.;
            for (unsigned short k = 0; k < ii; k++)
            {
                p[k] -= maxLogProb;
                psum += exp(p[k]);
                p[k] = psum;
            }

            // scaled sample because of unnormalized p[]
            unsigned short topic;
            double u = rng_.rand_double() * (psum + alphaK*q[w].getsum());
            if (u < psum)
            {
                topic = std::lower_bound(p, p + ii, u) - p;
                topic = read_n_mks.idx_in(topic);
            }
            else
            {
                topic = q[w].sample(rng_.rand_k(K), rng_.rand_double());
            }

            //if ( (0 > topic) || (topic >= K) )
            //  std::cout<<w<<", "<<m<<", "<<n<<", "<<topic<<", "<<u<<", "<<psum<<std::endl;

            // number of instances of word i assigned to topic j
            topics[topic].addPoint(wvec);
            
            // number of words in document i assigned to topic j
            if (rev_mapper[topic] == K)
                rev_mapper[topic] = write_n_mks.push_back(topic, 1);
            else
                write_n_mks.increment(rev_mapper[topic]);

        }
        for (const auto& k : write_n_mks)
            rev_mapper[k.idx] = K;
    }

    delete[] p;
    delete[] rev_mapper;
    return 0;
}

int scaGLDA::updater()
{
    //if (rank == 0) std::cout << "Creating alias table for side: " << side << std::endl;
    std::swap(n_mks, p_mks);
    utils::parallel_for_each(topics.begin(), topics.end(),
        [](SuffStatsTwo& t){ t.updateParameters();
    });
    utils::parallel_for(0, V, [&](size_t w)->void{
        double *p = new double[K];
        const pointType& wvec = id2vec->col(w);

        // Compute probability for each topic
        double maxLogProb = -1 * std::numeric_limits<double>::max();
        for (unsigned k = 0; k < K; ++k)
        {
            p[k] = topics[k].computeProbability(wvec);
            if (maxLogProb < p[k])
                maxLogProb = p[k];
        }

        // Now subtract the minimum, exponentiate, add and divide
        double psum = 0.;
        for (unsigned short k = 0; k < K; k++)
        {
            p[k] -= maxLogProb;
            psum += exp(p[k]);
            p[k] = psum;
        }
        q[w].recompute(p, psum);
        delete[] p;
    });
    return 0;
}
