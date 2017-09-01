#include "lda.h"

int adLDA::sampling(const DataIO::corpus& trngdata, unsigned i)
{
    xorshift128plus rng_;
    
    double alphaK = alpha/K;

    double * p = new double[K]; // temp variable for sampling
    unsigned *nd_m = new unsigned[K];
    std::fill(nd_m, nd_m + K, 0);
    unsigned *nlocal_k = new unsigned[K];
    std::fill(nlocal_k, nlocal_k + K, 0);
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
            size_t ptr = w*K;
            
            //std::cout << "I reached 5 " << std::endl;

            // do multinomial sampling via cumulative method
            double psum = 0;
            for (unsigned short k = 0; k < K; k++)
            {
                psum += (nd_m[k] + alphaK) * p_wk[ptr + k];
                p[k] = psum;
            }
            
            //std::cout << "I reached 6 " << std::endl;
                
            // scaled sample because of unnormalized p[]
            double u = rng_.rand_double() * psum;
            
            //std::cout << "I reached 7 " << std::endl;

            // Do a binary search instead!
            unsigned short topic = std::lower_bound(p, p + K, u) - p;
            
            //std::cout << "I reached 8 " << std::endl;

            //if ( (0 > topic) || (topic >= K) )
            //  std::cout<<w<<", "<<m<<", "<<n<<", "<<topic<<std::endl;

            // total number of words assigned to topic j
            nlocal_k[topic] += 1;
            
            //std::cout << "I reached 9 " << std::endl;
            
            // number of instances of word i assigned to topic j
            n_wk[ptr + topic].fetch_add(1, std::memory_order_relaxed);
            
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

    for (unsigned short k = 0; k < K; ++k)
        n_k[k].fetch_add(nlocal_k[k], std::memory_order_relaxed);

    delete[] p;
    delete[] nd_m;
    delete[] rev_mapper;
    delete[] nlocal_k;
    return 0;
}

int scaLDA::specific_init()
{
    q.resize(V, voseAlias(K));
    return 0;
}

int scaLDA::sampling(const DataIO::corpus& trngdata, unsigned i)
{
    xorshift128plus rng_;
    
    double alphaK = alpha/K;

    double * p = new double[K]; // temp variable for sampling
    unsigned *nlocal_k = new unsigned[K];
    std::fill(nlocal_k, nlocal_k + K, 0);
    unsigned short *rev_mapper = new unsigned short[K];
    std::fill(rev_mapper, rev_mapper + K, K);
    //DataIO::corpus trngdata;
    // for each document of worker i
    size_t M = trngdata.size();
    for (size_t m = i; m < M; m += n_threads)
    {
        const auto& read_n_mks = p_mks[m];
        auto& write_n_mks = n_mks[m];
        const auto& doc = trngdata[m];

        for (const auto& w : doc)
        {
            size_t ptr = w*K;

            double psum = 0;            
            unsigned short ii = 0;
            /* Travese all non-zero document-topic distribution */
            for (const auto& k : read_n_mks)
            {
                psum += k.val * p_wk[ptr + k.idx];
                p[ii++] = psum;
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

            // total number of words assigned to topic j
            nlocal_k[topic] += 1;
            
            // number of instances of word i assigned to topic j
            n_wk[ptr + topic].fetch_add(1, std::memory_order_relaxed);
            
            // number of words in document i assigned to topic j
            if (rev_mapper[topic] == K)
                rev_mapper[topic] = write_n_mks.push_back(topic, 1);
            else
                write_n_mks.increment(rev_mapper[topic]);

        }
        for (const auto& k : write_n_mks)
            rev_mapper[k.idx] = K;
    }

    for (unsigned short k = 0; k < K; ++k)
        n_k[k].fetch_add(nlocal_k[k], std::memory_order_relaxed);

    delete[] p;
    delete[] rev_mapper;
    delete[] nlocal_k;
    return 0;
}

int scaLDA::updater()
{
    //if (rank == 0) std::cout << "Creating alias table for side: " << side << std::endl;
    std::swap(n_mks, p_mks);
    utils::parallel_for(0, V, [&](size_t w)->void{
        double Vbeta = V*beta;
        double sum = 0.0;
        size_t ptr = w*K;
        for (unsigned short k = 0; k<K; ++k)
        {
            p_wk[ptr + k] = (n_wk[ptr + k] + beta) / (n_k[k] + Vbeta);
            sum += p_wk[ptr + k];
            n_wk[ptr + k] = 0;
        }
        q[w].recompute(p_wk+ptr, sum);
    });
    return 0;
}
