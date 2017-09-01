#include "hdp.h"

#define MH_STEPS 2

int simpleHDP::sampling(const DataIO::corpus& trngdata, unsigned i)
{
    xorshift128plus rng_;

    double * p = new double[Kmax]; // temp variable for sampling
    unsigned *nd_m = new unsigned[Kmax];
    std::fill(nd_m, nd_m + Kmax, 0);
    unsigned short *rev_mapper = new unsigned short[Kmax];
    std::fill(rev_mapper, rev_mapper + Kmax, Kmax);
    
    //std::cout << "I reached 1" << std::endl;
    unsigned ntt = num_table_threads();
    unsigned nst = n_threads - ntt;
    
    // for each document of worker i
    size_t m;
    while ((m = doc_queue.pop()))
    {
    // size_t M = trngdata.size();
    // for (size_t m = i; m < M; m+=nst)
    // {
        //std::cout << "I reached 2 with m=" << m << std::endl;
        //std::cout << "The circular buffer load is: " << cbuff[0].size() << std::endl;
        --m;
        auto& ns_m = n_mks[m];
        unsigned short kc = 0;
        for (const auto& k : ns_m)
        {
            nd_m[k.idx] = k.val;
            rev_mapper[k.idx] = kc++;
        }

        const auto& doc = trngdata[m];
        size_t N = doc.size();
        for (size_t n = 0; n < N; ++n)
        {
            unsigned w = doc[n];
            size_t ptr = w*Kmax;

            // remove z_ij from the count variables
            unsigned short topic = z[m][n]; unsigned short old_topic = topic;
            --nd_m[topic];
            ns_m.decrement(rev_mapper[topic]);

            // do multinomial sampling via cumulative method
            double psum = 0;
            for (unsigned short kk = 0; kk < K; ++kk)
            {
                unsigned short k = kActive[kk];
                psum += (nd_m[k] + alpha*tau[k]) * (n_wk[ptr + k] + beta) / (n_k[k].load(std::memory_order_relaxed) + Vbeta);
                p[kk] = psum;
            }
            // likelihood of new component
            psum += alpha * tau_left / V;
            p[K] = psum;

            // scaled sample because of unnormalized p[]
            double u = rng_.rand_double() * psum;

            // Do a binary search instead!
            topic = std::lower_bound(p, p + K, u) - p;
            topic = topic < K ? kActive[topic] : spawn_topic();

            // add newly estimated z_i to count variables
            if (topic!=old_topic)
            {
                if(nd_m[topic] == 0)
                    rev_mapper[topic] = ns_m.push_back(topic, 1);
                else
                    ns_m.increment(rev_mapper[topic]);

                nd_m[topic] += 1;
                if (nd_m[old_topic] == 0)
                {
                    unsigned short pos = ns_m.erase_pos(rev_mapper[old_topic]);
                    rev_mapper[pos] = rev_mapper[old_topic];
                    rev_mapper[old_topic] = Kmax;                        
                }

                cbuff[nst*(w%ntt)+i].push(delta(w, old_topic, topic));
            }
            else
            {
                ns_m.increment(rev_mapper[topic]);
                ++nd_m[topic];
            }
            z[m][n] = topic;
        }
        for (const auto& k : ns_m)
        {
                nd_m[k.idx] = 0;
                rev_mapper[k.idx] = Kmax;
        }
    }
    //std::cout << "I reached 3" << std::endl;

    delete[] p;
    delete[] nd_m;
    delete[] rev_mapper;
    
    return 0;
}

int simpleHDP::updater()
{    
    std::atomic<unsigned>* t_k = new std::atomic<unsigned>[Kmax];
    std::fill(t_k, t_k + Kmax, 0);
    
    size_t M = n_mks.size();
    utils::parallel_block_for(0, M, [&](size_t start, size_t end)->void{
        // thread local random number generator
        xorshift128plus rng_;
        
        for(size_t m = start; m < end; ++m)
        {
            const auto& ns_m = n_mks[m];
            for (const auto& k : ns_m)
            {
                if(k.val > 1)
                {           
                    double a = alpha*tau[k.idx];
                    for (unsigned short t = 0; t < k.val; ++t)
                        t_k[k.idx].fetch_add(rng_.rand_double() < a/(a+t), std::memory_order_relaxed);
                }
                else
                    t_k[k.idx].fetch_add(k.val, std::memory_order_relaxed);
            }
        }
    });
 
    double total = 0;
    for (const auto& k : kActive)
    {
        tau[k] = global_rng.rand_gamma(t_k[k]);
        total += tau[k];
    }
    tau_left = global_rng.rand_gamma(gamma);
    total += tau_left;

    for (const auto& k : kActive)
    {
        tau[k] /= total;
        //std::cout << k << ", " << tau[k] << std::endl;
    }
    tau_left /= total;


    unsigned tsum = 0;
    for (const auto& k : kActive)   tsum += t_k[k];
    //std::cout << "pnew = " << alpha*gamma/V/(tsum + gamma);

    //sample gamma
    if(1)
    {
        const double gamma_prior_a = 1;
        const double gamma_prior_b = 1;

        double gamma_sum = 0;
        for(int rep = 0; rep < 20; ++rep)
        {
            double logeta = log( global_rng.rand_beta(gamma+1, tsum) );
            double pipi = tsum*( gamma_prior_b - logeta )/(gamma_prior_a + K - 1);
            int s = global_rng.rand_double()*(pipi + 1) < 1;
            gamma = global_rng.rand_gamma( gamma_prior_a + K - (1-s), gamma_prior_b - logeta);
            if(rep >= 10) gamma_sum += gamma;
        }
        gamma = gamma_sum/10;
        //std::cout << "Gamma = " << gamma << std::endl;
    }

    delete[] t_k;

    // Copy the word|topic distribution
    delete[] p_wk;
    p_wk = new double[V*K];
    utils::parallel_for(0, V, [&](size_t w)->void{
        size_t ptr1 = w*K;
        size_t ptr2 = w*Kmax;
        for (unsigned short kk = 0; kk < K; ++kk)
        {
            unsigned short k = kActive[kk];
            p_wk[ptr1 + kk] = tau[k] * (n_wk[ptr2 + k] + beta) / (n_k[k].load(std::memory_order_relaxed) + Vbeta);
        }
    });
    
    return 0;
}

int aliasHDP::specific_init()
{
    std::cout << "Initializing the alias tables ..." << std::endl;
    q.resize(V, voseAlias(K));
    sample_count.resize(V, 0);
    std::vector<std::mutex> temp(V);
    qmtx.swap(temp);
    revK.resize(Kmax);
    Kold = K;
    return 0;
}

int aliasHDP::sampling(const DataIO::corpus& trngdata, unsigned i)
{
    xorshift128plus rng_;

    double * p = new double[Kmax]; // temp variable for sampling
    double * r = new double[Kmax]; // temp variable for sampling
    unsigned *nd_m = new unsigned[Kmax];
    std::fill(nd_m, nd_m + Kmax, 0);
    unsigned short *rev_mapper = new unsigned short[Kmax];
    std::fill(rev_mapper, rev_mapper + Kmax, Kmax);
    
    //std::cout << "I reached 1" << std::endl;
    unsigned ntt = num_table_threads();
    unsigned nst = n_threads - ntt;
    
    // for each document of worker i
    size_t m;
    while ((m = doc_queue.pop()))
    {
    // size_t M = trngdata.size();
    // for (size_t m = i; m < M; m+=nst)
    // {
        //std::cout << "I reached 2 with m=" << m << std::endl;
        //std::cout << "The circular buffer load is: " << cbuff[0].size() << std::endl;
        --m;
        auto& ns_m = n_mks[m];
        unsigned short kc = 0;
        for (const auto& k : ns_m)
        {
            nd_m[k.idx] = k.val;
            rev_mapper[k.idx] = kc++;
        }

        const auto& doc = trngdata[m];
        size_t N = doc.size();
        for (size_t n = 0; n < N; ++n)
        {
            unsigned w = doc[n];
            size_t ptr = w*Kmax;
            size_t dtr = w*Kold;

            // remove z_ij from the count variables
            unsigned short topic = z[m][n]; unsigned short new_topic; unsigned short old_topic = topic;
            --nd_m[topic];
            ns_m.decrement(rev_mapper[topic]);

            // do multinomial sampling via cumulative method
            double psum = 0;
            unsigned short ii = 0;
            /* Travese all non-zero document-topic distribution */
            for (const auto& k : ns_m)
            {
                psum += k.val * (n_wk[ptr + k.idx] + beta) / (n_k[k.idx].load(std::memory_order_relaxed) + Vbeta);
                p[ii++] = psum;
            }
            double rsum = 0;
            unsigned short jj = 0;
            for (const auto& k : kRecent)
            {
                rsum += alpha * tau[k] * (n_wk[ptr + k] + beta) / (n_k[k].load(std::memory_order_relaxed) + Vbeta);
                r[jj++] = rsum;
            }
            // likelihood of new component
            double pnew = alpha * tau_left / V;

            // scaled sample because of unnormalized p[]
            double p_tot = psum + rsum + pnew + alpha*q[w].getsum();

            //std::cout << "I reached 3 " << m << std::endl;

            //MHV to draw new topic
            bool flag = true;
            double fkw_old = (n_wk[ptr + topic] + beta) / (n_k[topic].load(std::memory_order_relaxed) + Vbeta);
            double ratio_old = (nd_m[topic] * fkw_old + alpha*p_wk[dtr + revK[topic]]) / ((nd_m[topic] + alpha*tau[topic])*fkw_old);
            for (unsigned rep = 0; rep < MH_STEPS; ++rep)
            {
                double u = rng_.rand_double() * p_tot;
                //1. Flip a coin
                if ( u < psum )
                {
                    new_topic = std::lower_bound(p,p+ii,u) - p;
                    new_topic = ns_m.idx_in(new_topic);
                    flag = true;
                    //choice = 1;
                }
                else if ( u < psum + rsum )
                {
                    u -= psum;
                    new_topic = std::lower_bound(r,r+jj,u) - r;
                    new_topic = kRecent[new_topic];
                    flag = false;
                    //choice = 2;
                }
                else if ( u < psum + rsum + pnew )
                {
                    new_topic = Kmax;
                    flag = false;
                    //choice = 3;
                }
                else
                {
                    ++sample_count[w];
                    if(sample_count[w] > Kold)
                    {   
                        if(qmtx[w].try_lock())
                        {
                            generateQtable(w);
                            sample_count[w] = 0;
                            qmtx[w].unlock();
                        }
                    }
                    new_topic = q[w].sample(rng_.rand_k(Kold), rng_.rand_double());
                    new_topic = kActive[new_topic];
                    flag = true;
                    //choice = 4;
                }

                if (topic != new_topic)
                {
                    //2. Find acceptance probability
                    double ratio_new;
                    if(flag)
                    {
                        double fkw_new = (n_wk[ptr + new_topic] + beta) / (n_k[new_topic].load(std::memory_order_relaxed) + Vbeta);
                        ratio_new = ((nd_m[new_topic] + alpha*tau[new_topic])*fkw_new) / (nd_m[new_topic] * fkw_new + alpha*p_wk[dtr + revK[new_topic]]);
                    }
                    else
                    {
                        ratio_new = 1;
                    }
                    double acceptance = ratio_old * ratio_new;

                    //3. Compare against uniform[0,1]
                    if (rng_.rand_double() < acceptance)
                    {
                        topic = new_topic;
                        ratio_old = 1/ratio_new;
                    }
                }
            }
            if(topic == Kmax)
                topic = spawn_topic();

            //std::cout << "I reached 4 " << m << std::endl;

            // add newly estimated z_i to count variables
            if (topic!=old_topic)
            {
                if(nd_m[topic] == 0)
                    rev_mapper[topic] = ns_m.push_back(topic, 1);
                else
                    ns_m.increment(rev_mapper[topic]);

                nd_m[topic] += 1;
                if (nd_m[old_topic] == 0)
                {
                    unsigned short pos = ns_m.erase_pos(rev_mapper[old_topic]);
                    rev_mapper[pos] = rev_mapper[old_topic];
                    rev_mapper[old_topic] = Kmax;                        
                }

                cbuff[nst*(w%ntt)+i].push(delta(w, old_topic, topic));
            }
            else
            {
                ns_m.increment(rev_mapper[topic]);
                ++nd_m[topic];
            }
            z[m][n] = topic;
        }
        for (const auto& k : ns_m)
        {
                nd_m[k.idx] = 0;
                rev_mapper[k.idx] = Kmax;
        }
    }
    //std::cout << "I reached 3" << std::endl;

    delete[] p;
    delete[] r;
    delete[] nd_m;
    delete[] rev_mapper;
    
    return 0;
}

int aliasHDP::updater()
{    
    std::atomic<unsigned>* t_k = new std::atomic<unsigned>[Kmax];
    std::fill(t_k, t_k + Kmax, 0);
    
    size_t M = n_mks.size();
    utils::parallel_block_for(0, M, [&](size_t start, size_t end)->void{
        // thread local random number generator
        xorshift128plus rng_;
        
        for(size_t m = start; m < end; ++m)
        {
            const auto& ns_m = n_mks[m];
            for (const auto& k : ns_m)
            {
                if(k.val > 1)
                {           
                    double a = alpha*tau[k.idx];
                    for (unsigned short t = 0; t < k.val; ++t)
                        t_k[k.idx].fetch_add(rng_.rand_double() < a/(a+t), std::memory_order_relaxed);
                }
                else
                    t_k[k.idx].fetch_add(k.val, std::memory_order_relaxed);
            }
        }
    });
 
    double total = 0;
    for (const auto& k : kActive)
    {
        tau[k] = global_rng.rand_gamma(t_k[k]);
        total += tau[k];
    }
    tau_left = global_rng.rand_gamma(gamma);
    total += tau_left;

    for (const auto& k : kActive)
    {
        tau[k] /= total;
        //std::cout << k << ", " << tau[k] << std::endl;
    }
    tau_left /= total;

    unsigned tsum = 0;
    for (const auto& k : kActive)   tsum += t_k[k];
    //std::cout << "pnew = " << alpha*gamma/V/(tsum + gamma);

    //sample gamma
    if(1)
    {
        const double gamma_prior_a = 1;
        const double gamma_prior_b = 1;

        double gamma_sum = 0;
        for(int rep = 0; rep < 20; ++rep)
        {
            double logeta = log( global_rng.rand_beta(gamma+1, tsum) );
            double pipi = tsum*( gamma_prior_b - logeta )/(gamma_prior_a + K - 1);
            int s = global_rng.rand_double()*(pipi + 1) < 1;
            gamma = global_rng.rand_gamma( gamma_prior_a + K - (1-s), gamma_prior_b - logeta);
            if(rep >= 10) gamma_sum += gamma;
        }
        gamma = gamma_sum/10;
        //std::cout << "Gamma = " << gamma << std::endl;
    }

    delete[] t_k;
    kRecent.clear();    

    for (unsigned short kk = 0; kk < K; ++kk)
    {
        unsigned short k = kActive[kk];
        revK[k] = kk;
    }

    // Copy the word|topic distribution
    delete[] p_wk;
    p_wk = new double[V*K];
    utils::parallel_for(0, V, [&](size_t w)->void{
        double wsum = 0.0;
        size_t ptr1 = w*K;
        size_t ptr2 = w*Kmax;
        for (unsigned short kk = 0; kk < K; ++kk)
        {
            unsigned short k = kActive[kk];
            p_wk[ptr1 + kk] = tau[k] * (n_wk[ptr2 + k] + beta) / (n_k[k].load(std::memory_order_relaxed) + Vbeta);
            wsum += p_wk[ptr1 + kk];
        }
        q[w].resize_and_recompute(K, p_wk+ptr1, wsum);
    });
    Kold = K;
    
    return 0;
}

void aliasHDP::generateQtable(unsigned w)
{
    double wsum = 0.0;
    size_t ptr1 = w*Kold;
    size_t ptr2 = w*Kmax;
    for (unsigned short kk = 0; kk < Kold; ++kk)
    {
        unsigned short k = kActive[kk];
        p_wk[ptr1 + kk] = tau[k] * (n_wk[ptr2 + k] + beta) / (n_k[k].load(std::memory_order_relaxed) + Vbeta);
        wsum += p_wk[ptr1 + kk];
    }
    q[w].recompute(p_wk+ptr1, wsum);
}

int stcHDP::specific_init()
{
    //std::cout << "Initializing the alias tables ..." << std::endl;
    //q.resize(V, voseAlias(K));
    //sample_count.resize(V, 0);
    //std::vector<std::mutex> temp(V);
    //qmtx.swap(temp);
    //revK.resize(Kmax);
    //Kold = K;

    tsum = 0;
    t_k = new std::atomic<unsigned>[Kmax];
    std::fill(t_k, t_k + Kmax, 0);
    return 0;
}

int stcHDP::init_train(const DataIO::corpus& trngdata)
{
    size_t M = trngdata.size();
    
    // allocate temporary buffers
    n_k = new std::atomic<unsigned>[Kmax];
    std::fill(n_k, n_k + Kmax, 0);

    n_wk = new unsigned[Kmax*V];
    utils::parallel_for(0, Kmax, [&](size_t k)->void{
        std::fill(n_wk + V*k, n_wk + V*(k+1), 0);
    });

    // locks for parallel initialization of data-structures involving n_wk
    std::vector<std::mutex> mtx(V);

    // allocate heap memory for per document variables
    nt_mks.resize(M);
    doc_queue.resize(M);

    // random consistent assignment for model variables
    z = new unsigned short*[M];
    utils::parallel_for_progressbar(0, M, [&](size_t m)->void{
        // random number generator
        xorshift128plus rng_;
        
        unsigned* nlocal_k = new unsigned[K];
        std::fill(nlocal_k, nlocal_k + K, 0);
        std::map<unsigned short, unsigned > map_nd_m;
        const auto& doc_w = trngdata[m];

        size_t N = doc_w.size();
        z[m] = new unsigned short[N];
        auto& doc_z = z[m];

        // initialize for z
        for (size_t n = 0; n < N; ++n)
        {
            unsigned short topic = rng_.rand_k(K);
            //unsigned short topic = (m + rng_.rand_k((unsigned short) 15))%K;
            doc_z[n] = topic;
            unsigned w = doc_w[n];

            // number of instances of word i assigned to topic j
            mtx[w].lock();
            n_wk[w*Kmax + topic] += 1;
            mtx[w].unlock();
            // number of words in document i assigned to topic j
            map_nd_m[topic] += 1;
            // total number of words assigned to topic j
            nlocal_k[topic] += 1;
        }
        // transfer to sparse representation
        for (auto myc : map_nd_m)
        {
            nt_mks[m].push_back(myc.first, myc.second, 1);
            t_k[myc.first].fetch_add(1, std::memory_order_relaxed);
        }
        // transfer to global counts
        for (unsigned short k = 0; k < K; ++k)
            n_k[k] += nlocal_k[k];

        delete[] nlocal_k;
    });

    tsum = 0;
    for (unsigned short k = 0; k<K; ++k)
        tsum += t_k[k].load(std::memory_order_relaxed);

    for(unsigned short k = 0; k<K; ++k)
    {
        if (n_k[k] > 0)
            kActive.push_back(k);
        else
            kGaps.push(k);
    }
    for(unsigned short k = K; k<Kmax; ++k)
        kGaps.push(k);

    if(K!=kActive.size())
    {
        K = kActive.size();
        //std::cout << "Initial number of topics resized to " << K << std::endl;
    }

    // algorithm specific data structures
    updater();

    return 0;
}

int stcHDP::updater()
{    
    size_t M = nt_mks.size();

    double total = 0;
    for (const auto& k : kActive)
    {
        tau[k] = global_rng.rand_gamma(t_k[k].load(std::memory_order_relaxed));
        total += tau[k];
    }
    tau_left = global_rng.rand_gamma(gamma);
    total += tau_left;

    for (const auto& k : kActive)
    {
        tau[k] /= total;
        //std::cout << k << ", " << tau[k] << std::endl;
    }
    tau_left /= total;

    //sample gamma
    if(1)
    {
        const double gamma_prior_a = 1;
        const double gamma_prior_b = 1;

        double gamma_sum = 0;
        for(int rep = 0; rep < 20; ++rep)
        {
            double logeta = log( global_rng.rand_beta(gamma+1, tsum) );
            double pipi = tsum*( gamma_prior_b - logeta )/(gamma_prior_a + K - 1);
            int s = global_rng.rand_double()*(pipi + 1) < 1;
            gamma = global_rng.rand_gamma( gamma_prior_a + K - (1-s), gamma_prior_b - logeta);
            if(rep >= 10) gamma_sum += gamma;
        }
        gamma = gamma_sum/10;
        //std::cout << "Gamma = " << gamma << std::endl;
    }
    
    // Copy the word|topic distribution
    delete[] p_wk;
    p_wk = new double[V*K];
    utils::parallel_for(0, V, [&](size_t w)->void{
        size_t ptr1 = w*K;
        size_t ptr2 = w*Kmax;
        for (unsigned short kk = 0; kk < K; ++kk)
        {
            unsigned short k = kActive[kk];
            p_wk[ptr1 + kk] = tau[k] * (n_wk[ptr2 + k] + beta) / (n_k[k].load(std::memory_order_relaxed) + Vbeta);
        }
    });
    
    return 0;
}

int stcHDP::sampling(const DataIO::corpus& trngdata, unsigned i)
{
    xorshift128plus rng_;

    double * pu = new double[Kmax]; // temp variable for sampling
    double * pd = new double[Kmax]; // temp variable for sampling
    unsigned *nd_m = new unsigned[Kmax];
    std::fill(nd_m, nd_m + Kmax, 0);
    unsigned *td_m = new unsigned[Kmax];
    std::fill(td_m, td_m + Kmax, 0);
    unsigned short *rev_mapper = new unsigned short[Kmax];
    std::fill(rev_mapper, rev_mapper + Kmax, Kmax);
    
    //std::cout << "I reached 1" << std::endl;
    unsigned ntt = num_table_threads();
    unsigned nst = n_threads - ntt;
    
    // for each document of worker i
    size_t m;
    while ((m = doc_queue.pop()))
    {
    // size_t M = trngdata.size();
    // for (size_t m = i; m < M; m+=nst)
    // {
        //std::cout << "I reached 2 with m=" << m << " on thread " << i << std::endl;
        //std::cout << "The circular buffer load is: " << cbuff[0].size() << std::endl;
        --m;
        auto& nts_m = nt_mks[m];
        unsigned short kc = 0;
        for (const auto& k : nts_m)
        {
            nd_m[k.idx] = k.customers;
            td_m[k.idx] = k.tables;
            rev_mapper[k.idx] = kc++;
        }

        const auto& doc = trngdata[m];
        size_t N = doc.size();
        for (size_t n = 0; n < N; ++n)
        {
            unsigned w = doc[n];
            size_t ptr = w*Kmax;

            // remove z_ij from the count variables
            unsigned short topic = z[m][n]; unsigned short old_topic = topic;
            bool table_create = false, table_destroy = false;
            --nd_m[topic];
            nts_m.decrement_customers(rev_mapper[topic]);

            // Remove customer mn: sample to decide whether to decrement table count
            //if (nd_m[topic] * rng_.rand_double() <= td_m[topic] - 1)
            if ((nd_m[topic]==0) || ((td_m[topic]>1) && (nd_m[topic] * rng_.rand_double() <= td_m[topic])))
            {
                // Sitting at alone at a table
                //tsum.fetch_add(-1, std::memory_order_relaxed);
                //t_k[topic].fetch_add(-1, std::memory_order_relaxed);
                table_destroy = true;
                nts_m.decrement_tables(rev_mapper[topic]);
                td_m[topic] -= 1;
            }

            // do multinomial sampling via cumulative method
            double pusum = 0., pdsum = 0.;
            /* Travese all document-topic distribution */
            for (unsigned short kk = 0; kk < K; ++kk)
            {
                unsigned short k = kActive[kk];
                double fkw = (n_wk[ptr + k] + beta) / (n_k[k].load(std::memory_order_relaxed) + Vbeta);    // language model
                unsigned local_t_k = t_k[k].load(std::memory_order_relaxed);
                double tau_k = alpha * local_t_k * local_t_k / (local_t_k + 1) / (tsum.load(std::memory_order_relaxed) + gamma);
                if(nd_m[k] > 0)
                {
                    pusum += stirling_.uratio(nd_m[k], td_m[k]) * (nd_m[k] + 1 - td_m[k]) * fkw / (nd_m[k] + 1);
                    pdsum += tau_k * fkw * stirling_.wratio(nd_m[k], td_m[k]) * (td_m[k] + 1) / (nd_m[k] + 1);
                }
                else
                {
                    pdsum += tau_k * fkw;
                }
                pu[kk] = pusum;
                pd[kk] = pdsum;
            }
            // Totally new dish
            double pnew = alpha * gamma / V / (tsum.load(std::memory_order_relaxed) + gamma);

            //std::cout << "I reached 2.1 pusum=" << pusum << " pdsum=" << pdsum << " pgsum=" << pgsum << " pnew=" << pnew << std::endl;

            // scaled sample because of unnormalized p[]
            double r = rng_.rand_double() * (pusum + pdsum + pnew);
            // Case 1: Sit at an existing table in the restaurant
            if (r < pusum)
            {
                size_t new_idx = std::lower_bound(pu, pu + K, r) - pu;
                topic = kActive[new_idx];
            }
            // Case 2: Sit at a new table
            else if (r < pusum + pdsum)
            {
                r -= pusum;
                table_create = true;
                size_t new_idx = std::lower_bound(pd, pd + K, r) - pd;
                topic = kActive[new_idx];
            }
            // Case 3: Start a completely new topic
            else
            {
                table_create = true;
                topic = spawn_topic();
            }

            // add newly estimated z_i to count variables
            if (topic!=old_topic)
            {
                if(nd_m[topic] == 0)
                {
                    rev_mapper[topic] = nts_m.push_back(topic, 1, 1);
                    td_m[topic] = 1;
                }
                else
                {
                    nts_m.increment_customers(rev_mapper[topic]);
                    if (table_create) 
                    {
                        nts_m.increment_tables(rev_mapper[topic]);
                        td_m[topic] += 1;
                    }
                }

                nd_m[topic] += 1;
                if (nd_m[old_topic] == 0)
                {
                    unsigned short pos = nts_m.erase_pos(rev_mapper[old_topic]);
                    rev_mapper[pos] = rev_mapper[old_topic];
                    rev_mapper[old_topic] = Kmax;                        
                }

                cbuff[nst*(w%ntt)+i].push(delta(w, old_topic, topic, table_destroy, table_create));
            }
            else
            {
                nts_m.increment_customers(rev_mapper[topic]);
                ++nd_m[topic];
                if (table_create)
                {
                    nts_m.increment_tables(rev_mapper[topic]);
                    td_m[topic] += 1;
                }
                if (table_create != table_destroy)
                {
                    int diff = table_create - table_destroy;
                    tsum.fetch_add(diff, std::memory_order_relaxed);
                    t_k[topic].fetch_add(diff, std::memory_order_relaxed);
                }
            }
            z[m][n] = topic;
        }
        for (const auto& k : nts_m)
        {
                nd_m[k.idx] = 0;
                td_m[k.idx] = 0;
                rev_mapper[k.idx] = Kmax;
        }
    }
    //std::cout << "I reached 3" << std::endl;

    delete[] pu;
    delete[] pd;
    delete[] nd_m;
    delete[] td_m;
    delete[] rev_mapper;
    
    return 0;
}

int stcHDP::writer(unsigned i)
{
    unsigned ntt = num_table_threads();
    unsigned nst = n_threads - ntt;
    do
    {
        for (unsigned tn = 0; tn<nst; ++tn)
        {
            if (!(cbuff[i*nst + tn].empty()))
            {
                delta temp = cbuff[i*nst + tn].front();
                cbuff[i*nst + tn].pop();
                size_t ptr = temp.word*Kmax;
                --n_wk[ptr + temp.old_topic];
                ++n_wk[ptr + temp.new_topic];
                n_k[temp.old_topic].fetch_add(-1, std::memory_order_relaxed);
                n_k[temp.new_topic].fetch_add(+1, std::memory_order_relaxed);
                if(temp.table_destroy)
                {
                    t_k[temp.old_topic].fetch_add(-1, std::memory_order_relaxed);
                    tsum.fetch_add(-1, std::memory_order_relaxed);
                }
                if(temp.table_create)
                {
                    t_k[temp.new_topic].fetch_add(+1, std::memory_order_relaxed);
                    tsum.fetch_add(+1, std::memory_order_relaxed);
                }
            }
        }
    } while (!inf_stop); //(!done[i]);

    for (unsigned tn = 0; tn<nst; ++tn)
    {
        while(!(cbuff[i*nst + tn].empty()))
        {
            delta temp = cbuff[i*nst + tn].front();
            cbuff[i*nst + tn].pop();
            size_t ptr = temp.word*Kmax;
            --n_wk[ptr + temp.old_topic];
            ++n_wk[ptr + temp.new_topic];
            n_k[temp.old_topic].fetch_add(-1, std::memory_order_relaxed);
            n_k[temp.new_topic].fetch_add(+1, std::memory_order_relaxed);
            if(temp.table_destroy)
            {
                t_k[temp.old_topic].fetch_add(-1, std::memory_order_relaxed);
                tsum.fetch_add(-1, std::memory_order_relaxed);
            }
            if(temp.table_create)
            {
                t_k[temp.new_topic].fetch_add(+1, std::memory_order_relaxed);
                tsum.fetch_add(+1, std::memory_order_relaxed);
            }
        }
    }

    return 0;
}

int stcAliasHDP::specific_init()
{
    std::cout << "Initializing the alias tables ..." << std::endl;

    tsum = 0;
    t_k = new std::atomic<unsigned>[Kmax];
    std::fill(t_k, t_k + Kmax, 0);

    q.resize(V, voseAlias(K));
    sample_count.resize(V, 0);
    std::vector<std::mutex> temp(V);
    qmtx.swap(temp);
    revK.resize(Kmax);
    Kold = K;

    return 0;
}

int stcAliasHDP::init_train(const DataIO::corpus& trngdata)
{
    size_t M = trngdata.size();
    
    // allocate temporary buffers
    n_k = new std::atomic<unsigned>[Kmax];
    std::fill(n_k, n_k + Kmax, 0);

    n_wk = new unsigned[Kmax*V];
    utils::parallel_for(0, Kmax, [&](size_t k)->void{
        std::fill(n_wk + V*k, n_wk + V*(k+1), 0);
    });

    // locks for parallel initialization of data-structures involving n_wk
    std::vector<std::mutex> mtx(V);

    // allocate heap memory for per document variables
    nt_mks.resize(M);
    doc_queue.resize(M);

    // random consistent assignment for model variables
    z = new unsigned short*[M];
    utils::parallel_for_progressbar(0, M, [&](size_t m)->void{
        // random number generator
        xorshift128plus rng_;
        
        unsigned* nlocal_k = new unsigned[K];
        std::fill(nlocal_k, nlocal_k + K, 0);
        std::map<unsigned short, unsigned > map_nd_m;
        const auto& doc_w = trngdata[m];

        size_t N = doc_w.size();
        z[m] = new unsigned short[N];
        auto& doc_z = z[m];

        // initialize for z
        for (size_t n = 0; n < N; ++n)
        {
            unsigned short topic = rng_.rand_k(K);
            //unsigned short topic = (m + rng_.rand_k((unsigned short) 15))%K;
            doc_z[n] = topic;
            unsigned w = doc_w[n];

            // number of instances of word i assigned to topic j
            mtx[w].lock();
            n_wk[w*Kmax + topic] += 1;
            mtx[w].unlock();
            // number of words in document i assigned to topic j
            map_nd_m[topic] += 1;
            // total number of words assigned to topic j
            nlocal_k[topic] += 1;
        }
        // transfer to sparse representation
        for (auto myc : map_nd_m)
        {
            nt_mks[m].push_back(myc.first, myc.second, 1);
            t_k[myc.first].fetch_add(1, std::memory_order_relaxed);
        }
        // transfer to global counts
        for (unsigned short k = 0; k < K; ++k)
            n_k[k] += nlocal_k[k];

        delete[] nlocal_k;
    });

    tsum = 0;
    for (unsigned short k = 0; k<K; ++k)
        tsum += t_k[k].load(std::memory_order_relaxed);

    for(unsigned short k = 0; k<K; ++k)
    {
        if (n_k[k] > 0)
            kActive.push_back(k);
        else
            kGaps.push(k);
    }
    for(unsigned short k = K; k<Kmax; ++k)
        kGaps.push(k);

    if(K!=kActive.size())
    {
        K = kActive.size();
        //std::cout << "Initial number of topics resized to " << K << std::endl;
    }

    // algorithm specific data structures
    updater();

    return 0;
}

int stcAliasHDP::updater()
{    
    size_t M = nt_mks.size();

    //sample gamma
    if(1)
    {
        const double gamma_prior_a = 1;
        const double gamma_prior_b = 1;

        double gamma_sum = 0;
        for(int rep = 0; rep < 20; ++rep)
        {
            double logeta = log( global_rng.rand_beta(gamma+1, tsum) );
            double pipi = tsum*( gamma_prior_b - logeta )/(gamma_prior_a + K - 1);
            int s = global_rng.rand_double()*(pipi + 1) < 1;
            gamma = global_rng.rand_gamma( gamma_prior_a + K - (1-s), gamma_prior_b - logeta);
            if(rep >= 10) gamma_sum += gamma;
        }
        gamma = gamma_sum/10;
        //std::cout << "Gamma = " << gamma << std::endl;
    }

    for (const auto& k : kActive)
    {
        unsigned tlocal_k = t_k[k].load(std::memory_order_relaxed);
        tau[k] = 1.0*tlocal_k*tlocal_k/(tlocal_k + 1)/(tsum.load(std::memory_order_relaxed) + gamma);
    }
    tau_left = gamma/(tsum.load(std::memory_order_relaxed) + gamma);

    kRecent.clear();    

    std::fill(revK.begin(), revK.end(), Kmax);
    for (unsigned short kk = 0; kk < K; ++kk)
    {
        unsigned short k = kActive[kk];
        revK[k] = kk;
    }

    // Copy the word|topic distribution
    delete[] p_wk;
    p_wk = new double[V*K];
    utils::parallel_for(0, V, [&](size_t w)->void{
        double wsum = 0.0;
        size_t ptr1 = w*K;
        size_t ptr2 = w*Kmax;
        for (unsigned short kk = 0; kk < K; ++kk)
        {
            unsigned short k = kActive[kk];
            p_wk[ptr1 + kk] = tau[k] * (n_wk[ptr2 + k] + beta) / (n_k[k].load(std::memory_order_relaxed) + Vbeta);
            wsum += p_wk[ptr1 + kk];
        }
        q[w].resize_and_recompute(K, p_wk+ptr1, wsum);
    });
    Kold = K;
    
    return 0;
}

int stcAliasHDP::sampling(const DataIO::corpus& trngdata, unsigned i)
{
    xorshift128plus rng_;

    double * pu = new double[Kmax]; // temp variable for sampling
    double * pd = new double[Kmax]; // temp variable for sampling
    double * pg = new double[Kmax]; // temp variable for sampling
    unsigned *nd_m = new unsigned[Kmax];
    std::fill(nd_m, nd_m + Kmax, 0);
    unsigned *td_m = new unsigned[Kmax];
    std::fill(td_m, td_m + Kmax, 0);
    unsigned short *rev_mapper = new unsigned short[Kmax];
    std::fill(rev_mapper, rev_mapper + Kmax, Kmax);
    
    //std::cout << "I reached 1" << std::endl;
    unsigned ntt = num_table_threads();
    unsigned nst = n_threads - ntt;
    
    // for each document of worker i
    size_t m;
    while ((m = doc_queue.pop()))
    {
    // size_t M = trngdata.size();
    // for (size_t m = i; m < M; m+=nst)
    // {
        //std::cout << "I reached 2 with m=" << m << " on thread " << i << std::endl;
        //std::cout << "The circular buffer load is: " << cbuff[0].size() << std::endl;
        --m;
        auto& nts_m = nt_mks[m];
        unsigned short kc = 0;
        for (const auto& k : nts_m)
        {
            nd_m[k.idx] = k.customers;
            td_m[k.idx] = k.tables;
            rev_mapper[k.idx] = kc++;
        }

        const auto& doc = trngdata[m];
        size_t N = doc.size();
        for (size_t n = 0; n < N; ++n)
        {
            unsigned w = doc[n];
            size_t ptr = w*Kmax;
            size_t dtr = w*Kold;

            // remove z_ij from the count variables
            unsigned short topic = z[m][n]; unsigned short new_topic; unsigned short old_topic = topic;
            bool table_create = false, table_destroy = false;
            --nd_m[topic];
            nts_m.decrement_customers(rev_mapper[topic]);

            // Remove customer mn: sample to decide whether to decrement table count
            //if (nd_m[topic] * rng_.rand_double() <= td_m[topic] - 1)
            if ((nd_m[topic]==0) || ((td_m[topic]>1) && (nd_m[topic] * rng_.rand_double() <= td_m[topic])))
            {
                // Sitting at alone at a table
                table_destroy = true;
                table_create = true;
                nts_m.decrement_tables(rev_mapper[topic]);
                td_m[topic] -= 1;
            }

            //std::cout << "I reached 3" << std::endl;

            // do multinomial sampling via cumulative method
            double pusum = 0., pdsum = 0.;
            unsigned short ii = 0;
            /* Travese all non-zero document-topic distribution */
            for (const auto& k : nts_m)
            {
                double fkw = (n_wk[ptr + k.idx] + beta) / (n_k[k.idx].load(std::memory_order_relaxed) + Vbeta);    // language model
                pusum += stirling_.uratio(k.customers, k.tables) * (k.customers + 1 - k.tables) * fkw / (k.customers + 1);
                unsigned local_t_k = t_k[k.idx].load(std::memory_order_relaxed);
                double k_exists = alpha * local_t_k * local_t_k * fkw / (local_t_k + 1) / (tsum.load(std::memory_order_relaxed) + gamma);
                pdsum += k_exists * (stirling_.wratio(k.customers, k.tables) * (k.tables + 1) / (k.customers + 1) - 1);
                pu[ii] = pusum;
                pd[ii++] = pdsum;
            }

            double pgsum = 0.;
            unsigned short jj = 0;
            for (const auto& k : kRecent)
            {
                double fkw = (n_wk[ptr + k] + beta) / (n_k[k].load(std::memory_order_relaxed) + Vbeta);    // language model
                unsigned local_t_k = t_k[k].load(std::memory_order_relaxed);
                pgsum += alpha * local_t_k * local_t_k * fkw / (local_t_k + 1) / (tsum.load(std::memory_order_relaxed) + gamma);
                pg[jj++] = pgsum;
            }

            // Totally new dish
            double pnew = alpha * gamma / V / (tsum.load(std::memory_order_relaxed) + gamma);

            //std::cout << "I reached 6 pusum=" << pusum << " pdsum=" << pdsum << " pgsum=" << pgsum << " pnew=" << pnew << std::endl;

            // scaled sample because of unnormalized p[]
            double p_tot = pusum + pdsum + pgsum + pnew + alpha*q[w].getsum();

            //MHV to draw new topic
            bool flag = true;
            double ratio_old = 1;
            if(table_destroy)
            {
                double fkw_old = (n_wk[ptr + topic] + beta) / (n_k[topic].load(std::memory_order_relaxed) + Vbeta);
                unsigned local_t_k = t_k[topic].load(std::memory_order_relaxed);
                double tau_k = fkw_old * local_t_k * local_t_k / (local_t_k + 1) / (tsum.load(std::memory_order_relaxed) + gamma);
                double multiplier = stirling_.wratio(nd_m[topic], td_m[topic]) * (td_m[topic] + 1) / (nd_m[topic] + 1);
                double p_old = tau_k * multiplier;
                double q_old = p_old - tau_k + p_wk[dtr + revK[topic]];
                ratio_old = q_old/p_old;
            }
            for (unsigned rep = 0; rep < MH_STEPS; ++rep)
            {
                // scaled sample because of unnormalized p[]
                bool new_table;
                double r = rng_.rand_double() * p_tot;
                // Case 1: Sit at an existing table in the restaurant
                if (r < pusum)
                {
                    new_table = false;
                    new_topic = std::lower_bound(pu, pu + ii, r) - pu;
                    new_topic = nts_m.idx_in(new_topic);
                    flag = false;
                }
                // Case 2: Sit at a new table
                else if (r < pusum + pdsum)
                {
                    r -= pusum;
                    new_table = true;
                    new_topic = std::lower_bound(pd, pd + ii, r) - pd;
                    new_topic = nts_m.idx_in(new_topic);
                    flag = revK[new_topic] != Kmax;
                }
                // Case 3: Explore recent topics
                else if (r < pusum + pdsum + pgsum)
                {
                    r -= pusum + pdsum;
                    new_table = true;
                    new_topic = std::lower_bound(pg, pg + K, r) - pg;
                    new_topic = kActive[new_topic];
                    flag = false;
                }
                // Case 3: Start a completely new topic
                else if (r < pusum + pdsum + pgsum + pnew)
                {
                    new_table = true;
                    new_topic = Kmax;
                    flag = false;
                }
                else
                {
                    new_table = true;
                    ++sample_count[w];
                    if(sample_count[w] > Kold)
                    {   
                        if(qmtx[w].try_lock())
                        {
                            generateQtable(w);
                            sample_count[w] = 0;
                            qmtx[w].unlock();
                        }
                    }
                    new_topic = q[w].sample(rng_.rand_k(Kold), rng_.rand_double());
                    new_topic = kActive[new_topic];
                    flag = true;
                }

                if (topic != new_topic || table_create != new_table)
                {
                    //2. Find acceptance probability
                    double ratio_new;
                    if(flag)
                    {
                        double fkw_new = (n_wk[ptr + new_topic] + beta) / (n_k[new_topic].load(std::memory_order_relaxed) + Vbeta);
                        unsigned local_t_k = t_k[new_topic].load(std::memory_order_relaxed);
                        double tau_k = fkw_new * local_t_k * local_t_k / (local_t_k + 1) / (tsum.load(std::memory_order_relaxed) + gamma);
                        double multiplier = stirling_.wratio(nd_m[new_topic], td_m[new_topic]) * (td_m[new_topic] + 1) / (nd_m[new_topic] + 1);
                        double p_new = tau_k * multiplier;
                        double q_new = p_new - tau_k + p_wk[dtr + revK[new_topic]];
                        ratio_new = q_new/p_new;
                    }
                    else
                    {
                        ratio_new = 1;
                    }
                    double acceptance = ratio_old * ratio_new;

                    //3. Compare against uniform[0,1]
                    if (rng_.rand_double() < acceptance)
                    {
                        topic = new_topic;
                        table_create = new_table;
                        ratio_old = 1/ratio_new;
                    }
                }
            }
            if(topic == Kmax)
                topic = spawn_topic();

            // add newly estimated z_i to count variables
            if (topic!=old_topic)
            {
                if(nd_m[topic] == 0)
                {
                    rev_mapper[topic] = nts_m.push_back(topic, 1, 1);
                    td_m[topic] = 1;
                }
                else
                {
                    nts_m.increment_customers(rev_mapper[topic]);
                    if (table_create) 
                    {
                        nts_m.increment_tables(rev_mapper[topic]);
                        td_m[topic] += 1;
                    }
                }

                nd_m[topic] += 1;
                if (nd_m[old_topic] == 0)
                {
                    unsigned short pos = nts_m.erase_pos(rev_mapper[old_topic]);
                    rev_mapper[pos] = rev_mapper[old_topic];
                    rev_mapper[old_topic] = Kmax;                        
                }

                cbuff[nst*(w%ntt)+i].push(delta(w, old_topic, topic, table_destroy, table_create));
            }
            else
            {
                nts_m.increment_customers(rev_mapper[topic]);
                ++nd_m[topic];
                if (table_create)
                {
                    nts_m.increment_tables(rev_mapper[topic]);
                    td_m[topic] += 1;
                    tsum.fetch_add(1, std::memory_order_relaxed);
                    t_k[topic].fetch_add(1, std::memory_order_relaxed);
                }
                if(table_destroy)
                {
                    tsum.fetch_add(-1, std::memory_order_relaxed);
                    t_k[topic].fetch_add(-1, std::memory_order_relaxed);
                }
            }
            z[m][n] = topic;
        }
        for (const auto& k : nts_m)
        {
                nd_m[k.idx] = 0;
                td_m[k.idx] = 0;
                rev_mapper[k.idx] = Kmax;
        }
    }
    //std::cout << "I reached 3" << std::endl;

    delete[] pu;
    delete[] pd;
    delete[] pg;
    delete[] nd_m;
    delete[] td_m;
    delete[] rev_mapper;
    
    return 0;
}

int stcAliasHDP::writer(unsigned i)
{
    unsigned ntt = num_table_threads();
    unsigned nst = n_threads - ntt;
    do
    {
        for (unsigned tn = 0; tn<nst; ++tn)
        {
            if (!(cbuff[i*nst + tn].empty()))
            {
                delta temp = cbuff[i*nst + tn].front();
                cbuff[i*nst + tn].pop();
                size_t ptr = temp.word*Kmax;
                --n_wk[ptr + temp.old_topic];
                ++n_wk[ptr + temp.new_topic];
                n_k[temp.old_topic].fetch_add(-1, std::memory_order_relaxed);
                n_k[temp.new_topic].fetch_add(+1, std::memory_order_relaxed);
                if(temp.table_destroy)
                {
                    t_k[temp.old_topic].fetch_add(-1, std::memory_order_relaxed);
                    tsum.fetch_add(-1, std::memory_order_relaxed);
                }
                if(temp.table_create)
                {
                    t_k[temp.new_topic].fetch_add(+1, std::memory_order_relaxed);
                    tsum.fetch_add(+1, std::memory_order_relaxed);
                }
            }
        }
    } while (!inf_stop); //(!done[i]);

    for (unsigned tn = 0; tn<nst; ++tn)
    {
        while(!(cbuff[i*nst + tn].empty()))
        {
            delta temp = cbuff[i*nst + tn].front();
            cbuff[i*nst + tn].pop();
            size_t ptr = temp.word*Kmax;
            --n_wk[ptr + temp.old_topic];
            ++n_wk[ptr + temp.new_topic];
            n_k[temp.old_topic].fetch_add(-1, std::memory_order_relaxed);
            n_k[temp.new_topic].fetch_add(+1, std::memory_order_relaxed);
            if(temp.table_destroy)
            {
                t_k[temp.old_topic].fetch_add(-1, std::memory_order_relaxed);
                tsum.fetch_add(-1, std::memory_order_relaxed);
            }
            if(temp.table_create)
            {
                t_k[temp.new_topic].fetch_add(+1, std::memory_order_relaxed);
                tsum.fetch_add(+1, std::memory_order_relaxed);
            }
        }
    }

    return 0;
}

void stcAliasHDP::generateQtable(unsigned w)
{
    double wsum = 0.0;
    size_t ptr1 = w*Kold;
    size_t ptr2 = w*Kmax;
    for (unsigned short kk = 0; kk < Kold; ++kk)
    {
        unsigned short k = kActive[kk];
        double fkw = (n_wk[ptr2 + k] + beta) / (n_k[k].load(std::memory_order_relaxed) + Vbeta);    // language model
        unsigned local_t_k = t_k[k].load(std::memory_order_relaxed);
        p_wk[ptr1 + kk] = fkw * local_t_k * local_t_k / (local_t_k + 1) / (tsum.load(std::memory_order_relaxed) + gamma);
        wsum += p_wk[ptr1 + kk];
    }
    q[w].recompute(p_wk+ptr1, wsum);
}
