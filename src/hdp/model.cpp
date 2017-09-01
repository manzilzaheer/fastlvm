#include "model.h"
#include "hdp.h"
#include <sstream>

model::model() :
    K(100),
    V(0),

    n_k(nullptr),
    n_wk(nullptr),
    p_wk(nullptr),
    z(nullptr),
    tau(nullptr),

    rank(0),
    n_iters(1000),
    n_save(200),
    n_threads(8),
    n_top_words(15),

    inf_stop(false),
    cbuff(nullptr),

    name("default"),
    mdir("./")

{    }

model::~model()
{
    if (z)      delete[] z;
    if (n_k)    delete[] n_k;
    if (n_wk)   delete[] n_wk;
    if (p_wk)   delete[] p_wk;
    if (tau)    delete[] tau;
    if (cbuff)  delete[] cbuff;
}

model* model::init(utils::ParsedArgs args, const std::vector<std::string>& word_map, int world_rank)
{
    model *hdp = NULL;

    if(args.K > std::numeric_limits<unsigned short>::max() )
            throw std::runtime_error("Error: Number of topics must be less than 65535! Supplied: " + args.K);

    if (args.algo == "simple")
    {
        hdp = new simpleHDP;
        if (world_rank == 0)    std::cout << "Running HDP inference using SDA (Sampling by Direct Assignment)" << std::endl;
    }
    else if (args.algo == "aliasHDP")
    {
        hdp = new aliasHDP;
        if (world_rank == 0)    std::cout << "Running HDP inference using SDA and FTree" << std::endl;
    }
    else if (args.algo == "stcHDP")
    {
        hdp = new stcHDP;
        if (world_rank == 0)    std::cout << "Running HDP inference using STC (Sampling Table Configuration)" << std::endl;
    }
    else if (args.algo == "stcAliasHDP")
    {
        hdp = new stcAliasHDP;
        std::cout << "Running HDP inference using Compact Configuration" << std::endl;
    }
    else
    {
        throw std::runtime_error("Error: Invalid inference algorithm!");
    }

    if (hdp != NULL)
    {
        // Set parameters

        hdp->K = args.K;
        hdp->V = (unsigned) word_map.size();
        hdp->rank = world_rank;
        hdp->n_iters = args.n_iters;
        hdp->n_save = args.n_save;
        hdp->n_threads = args.n_threads;
        hdp->n_top_words = args.n_top_words;
        hdp->name = args.name;
        hdp->mdir = args.out_path;

        hdp->Vbeta = hdp->V*hdp->beta;
        hdp->gamma = 1.;
        
        unsigned ntt = hdp->num_table_threads();
        unsigned nst = hdp->n_threads - ntt;

        // Copy wordmap
        hdp->id2word = word_map;
        
        // memory assignment
        hdp->p_wk = new double[hdp->K*hdp->V];
        utils::parallel_for(0, hdp->K, [&](size_t k)->void{
            std::fill(hdp->p_wk + hdp->V*k, hdp->p_wk + hdp->V*(k+1), 1.0/hdp->V);
        });
        hdp->tau = new double[hdp->Kmax];
        std::fill(hdp->tau, hdp->tau + hdp->Kmax, 1.0/(hdp->K+1));
        hdp->tau_left = 1.0/(hdp->K+1);

        // setup concurrency control variables
        hdp->cbuff = new circular_queue<delta>[nst*ntt];
        hdp->inf_stop = false;

        // reserve memory for training statistics
        hdp->time_ellapsed.reserve(hdp->n_iters);
        hdp->likelihood.reserve(hdp->n_iters);

        if (world_rank == 0) std::cout << "Initialising inference method specific data structures" << std::endl;
        hdp->specific_init();
        
        //display all configurations
        if (world_rank == 0)
        {
            std::cout << "model dir = " << hdp->mdir << std::endl;
            std::cout << "K = " << hdp->K << std::endl;
            std::cout << "V = " << hdp->V << std::endl;
            std::cout << "alpha = " << hdp->alpha << std::endl;
            std::cout << "beta = " << hdp->beta << std::endl;
            std::cout << "gamma = " << hdp->gamma << std::endl;
            std::cout << "n_iters = " << hdp->n_iters << std::endl;
            std::cout << "n_save = " << hdp->n_save << std::endl;
            std::cout << "num_threads = " << hdp->n_threads << std::endl;
            std::cout << "num_top_words = " << hdp->n_top_words << std::endl;
            std::cout << "NST = " << nst << std::endl;
            std::cout << "NTT = " << ntt << std::endl;
        }

    }
    else
    {
        throw std::runtime_error("Error: Inference algorithm not specified!");
    }

    return hdp;
}

double model::fit(const DataIO::corpus& trngdata, const DataIO::corpus& testdata)
{
    //initialize
    init_train(trngdata);
    
    std::chrono::high_resolution_clock::time_point ts, te;
    if (rank == 0)
    {
        std::cout << "Running " << n_iters << " iterations!" << std::endl;
        time_ellapsed.push_back(0);
        likelihood.push_back(evaluate(testdata));
        std::cout << "Likelihood on held out points: " << likelihood.back() << " at time " << time_ellapsed.back() << std::endl;
    }
    
    #ifdef MULTIMACHINE
    MPI_Barrier(MPI_COMM_WORLD);
    #endif

    unsigned ntt = num_table_threads();
    unsigned nst = n_threads - ntt;

    for (unsigned iter = 0; iter < n_iters; ++iter)
    {
        if (rank == 0)
        {
            std::cout << "Iteration " << iter << " ..." << std::endl;
            std::cout << "Num topics " << K << std::endl;
            if (iter % n_save == 0)
            {
                // saving the model
                std::cout << "Saving the model at iteration " << iter << "..." << std::endl;
                save_model(iter);
            }
            ts = std::chrono::high_resolution_clock::now();
        }

        //sample each document
        doc_queue.reset();
        inf_stop = false;
        std::vector<std::future<int>>  writing_threads;
        for (unsigned tn = 0; tn < ntt; ++tn)
            writing_threads.emplace_back(std::async(std::launch::async, &model::writer, this, tn));

        std::vector<std::future<int>>  sampling_threads;
        for (unsigned tn = 0; tn < nst; ++tn)
            sampling_threads.emplace_back(std::async(std::launch::async, &model::sampling, this, std::cref(trngdata), tn));

        for (unsigned tn = 0; tn < nst; ++tn)
            sampling_threads[tn].get();

        inf_stop = true;
        for (unsigned tn = 0; tn < ntt; ++tn)
            writing_threads[tn].get();

        // gather, clean and update from all workers
        //std::cout << "I reached 06" << std::endl;
        // Remove unused topics
        for(unsigned short kk = K; kk > 0; --kk)
        {
            unsigned short k = kActive[kk-1];
            if(n_k[k] == 0)
            {
                //for(unsigned short kkk = 0; kkk < kActive.size(); ++kkk)
                //     std::cout << "(" << kActive[kkk] << ", " << n_k[kActive[kkk]] <<"), ";
                //std::cout<<std::endl;
                kActive[kk-1] = kActive.back();
                kActive.pop_back();
                --K;
                kGaps.push(k);
                //for(unsigned short kkk = 0; kkk < kActive.size(); ++kkk)
                //     std::cout << "(" << kActive[kkk] << ", " << n_k[kActive[kkk]] <<"), ";
                //std::cout<<std::endl;
            }
        }
        // Recompute table statistics, hyper parameters,specific data structures
        updater();
    
        if (rank == 0)
        {
            te = std::chrono::high_resolution_clock::now();
            time_ellapsed.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count());
            likelihood.push_back(evaluate(testdata));
            std::cout << "Likelihood on held out points: " << likelihood.back() << " at time " << time_ellapsed.back() << std::endl;
        }

        //unsigned tsum = 0;
        //for (unsigned short kk = 0; kk < K; ++kk) tsum += n_k[kActive[kk]];
        //std::cout << "Total number of tokens: " << tsum << std::endl;

        //for(unsigned short kk = 0; kk < K; ++kk)
	//{
        //    tsum = 0;
        //    unsigned short k = kActive[kk];
        //    std::cout << k << ", ";
        //    for(unsigned w = 0; w < V; ++w)
        //        tsum += n_wk[w*Kmax + k];
        //    if(tsum != n_k[k])
        //        std::cout << "Counts do not match! k=" << k << ", sum n_wk=" << tsum << ", n_k=" << n_k[k] << std::endl;
	//}
        //std::cout << std::endl;
    }

    if (rank == 0)
    {
        std::cout << "LDA completed!" << std::endl;
        std::cout << "Saving the final model!" << std::endl;
        save_model(n_iters);
    }

    // free up buffer memory
    if (n_k)    delete[] n_k;
    n_k = nullptr;
    if (n_wk)    delete[] n_wk;
    n_wk = nullptr;
    n_mks.resize(0);
    if (z)
    {
        size_t M = trngdata.size();
        for (size_t m = 0; m < M; ++m)
        {
            if (z[m])
            {
                delete[] z[m];
            }
        }
        delete[] z;
    }
    z = nullptr;
    //kActive.clear();
    kGaps.clear();

    return likelihood.back();
}

double model::evaluate(const DataIO::corpus& testdata) const
{
    size_t M = testdata.size();
    std::atomic<double> sum{0};
    std::atomic<size_t> tokens{0};
    
    utils::parallel_block_for(0, M, [&](size_t start, size_t end)->void{
        // thread local random number generator
        xorshift128plus rng_;
        // thread sum
        double tsum = 0;
        size_t ttokens = 0;
        
        unsigned* nd_m = new unsigned[K];
        std::vector<unsigned short> z;
        double* p = new double[K];
        
        for(size_t i = start; i < end; ++i)
        {
            const auto& doc = testdata[i];
            size_t N = doc.size();
            ttokens += N;
            z.resize(N);
        
            //Initialize
            std::fill(nd_m, nd_m + K, 0);
            for (size_t j = 0; j < N; ++j)
            {
                //unsigned short topic = rng_.rand_k(K);
                unsigned short topic = (i + rng_.rand_k((unsigned short) 15))%K;
                z[j] = topic;
                
                // number of words in document i assigned to topic j
                nd_m[topic] += 1;
            }
        
            for(unsigned iter = 0; iter < 50; ++iter)
            {
                for(size_t j = 0; j < N; ++j)
                {
                    // remove z_i from the count variables
                    unsigned short topic = z[j];
                    nd_m[topic] -= 1;
                
                    // do multinomial sampling via cumulative method
                    unsigned w = doc[j];
                    unsigned ptr = w*K;
                    double psum = 0;
                    for (unsigned short k = 0; k < K; ++k)
                    {
                        psum += (nd_m[k]/tau[kActive[k]] + alpha) * p_wk[ptr+k];
                        p[k] = psum;
                    }

                    // scaled sample because of unnormalized p[]
                    double u = rng_.rand_double() * psum;
                    topic = std::lower_bound(p, p + K, u) - p;

                    // add newly estimated z_i to count variables
                    nd_m[topic] += 1;
                    z[j] = topic;
                }
            }
            
            
            double dsum = 0;
            for(const auto& w : doc)
            {
                unsigned ptr = w*K;
                double wsum = alpha*tau_left/V;
                for (unsigned short k = 0; k<K; ++k)
                    wsum += (nd_m[k]/tau[kActive[k]] + alpha) * p_wk[ptr+k];
                dsum += log(wsum);
            }
            tsum += dsum - N*log(N + alpha);
        }
        utils::add_to_atomic(sum, tsum);
        tokens.fetch_add(ttokens, std::memory_order_relaxed);
        delete[] nd_m;
        delete[] p;
    });
    return sum.load(std::memory_order_relaxed)/tokens.load(std::memory_order_relaxed);
}

/*std::vector<unsigned> model::predict(const dataset& testdata) const
{
    return labels;
}*/

std::tuple<unsigned short, unsigned, double*> model::get_topic_matrix() const
{
    return std::make_tuple(K, V, p_wk);
}

std::vector<std::vector<std::string>> model::get_top_words(unsigned num_words /*= 15*/) const
{
    if (num_words > V)
        num_words = V;
    
    std::vector<std::vector<std::string>> result(K, std::vector<std::string>(num_words));
    utils::parallel_for(0, K, [&](size_t k)->void{
        std::vector<unsigned> idx(V);
        std::iota(std::begin(idx), std::end(idx), 0);
        auto comp_x = [&](unsigned a, unsigned b) { return p_wk[a*K+k] > p_wk[b*K+k]; };
        std::sort(std::begin(idx), std::end(idx), comp_x);

        for (unsigned i = 0; i < num_words; ++i)
            result[k][i] = id2word[idx[i]];
    });
    return result;
}

int model::init_train(const DataIO::corpus& trngdata)
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
    n_mks.resize(M);
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
            n_mks[m].push_back(myc.first, myc.second);
        // transfer to global counts
        for (unsigned short k = 0; k < K; ++k)
            n_k[k] += nlocal_k[k];

        delete[] nlocal_k;
    });

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

int model::writer(unsigned i)
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
            }
        }
    } while (!inf_stop); //(!done[i]);
    for (unsigned tn = 0; tn<nst; ++tn)
    {   
        while (!(cbuff[i*nst + tn].empty()))
        {   
            delta temp = cbuff[i*nst + tn].front();
            cbuff[i*nst + tn].pop();
            size_t ptr = temp.word*Kmax;
            --n_wk[ptr + temp.old_topic];
            ++n_wk[ptr + temp.new_topic];
            n_k[temp.old_topic].fetch_add(-1, std::memory_order_relaxed);
            n_k[temp.new_topic].fetch_add(+1, std::memory_order_relaxed);
        }
    }
    return 0;
}

unsigned short model::spawn_topic()
{
    //std::cout << "Insert topic!" << std::endl;
    if (kGaps.empty())
        throw std::runtime_error("Ran out of topic");
    
    unsigned short k;
    
    // reuse gap
    mtx.lock();
    k = kGaps.top();
    kGaps.pop();
    kActive.push_back(k);
    //std::cout << "Inserting topic!" << k << std::endl;
    ++K;
    double new_stick = global_rng.rand_beta(1,gamma) * tau_left;
    tau_left -= new_stick;
    tau[k] = new_stick;
    mtx.unlock();
    //std::cout << "Inserted topic!" << std::endl;
    
    return k;
}

int model::save_model(unsigned iter) const
{
    std::string model_name = name + "-" + std::to_string(iter);
    
    save_model_params(mdir + model_name + ".params");
    save_model_phi(mdir + model_name + ".phi");
    save_model_time(mdir + model_name + ".time");
    save_model_llh(mdir + model_name + ".llh");
    save_model_top_words(mdir + model_name + ".twords");
    
    return 0;
}

int model::save_model_time(std::string filename) const
{
    std::ofstream fout(filename);
    if (!fout)
        throw std::runtime_error( "Error: Cannot open file to save: " + filename);
		
    for (size_t r = 0; r < time_ellapsed.size(); ++r)
        fout << time_ellapsed[r] << std::endl;

    fout.close();
    std::cout << "time done" << std::endl;

    return 0;
}

int model::save_model_llh(std::string filename) const
{
    std::ofstream fout(filename);
    if (!fout)
        throw std::runtime_error( "Error: Cannot open file to save: " + filename );
    
    for (size_t r = 0; r < likelihood.size(); ++r)
        fout << likelihood[r] << std::endl;
	
    fout.close();
    std::cout << "llh done" << std::endl;

    return 0;
}

int model::save_model_params(std::string filename) const
{
    std::ofstream fout(filename);
    if (!fout)
        throw std::runtime_error( "Error: Cannot open file to save: " + filename );

    fout << "alpha=" << alpha << std::endl;
    fout << "beta=" << beta << std::endl;
    fout << "num-topics=" << K << std::endl;
    fout << "num-words=" << V << std::endl;
    fout << "num-iters=" << n_iters << std::endl;
    fout << "num-threads = " << n_threads << std::endl;
    fout << "num-top-words = " << n_top_words << std::endl;
    fout << "output-state-interval = " << n_save << std::endl;
    fout << "num-sampling-threads=" << n_threads - num_table_threads() << std::endl;
    fout << "num-table-threads=" << num_table_threads() << std::endl;
    
    fout.close();
    std::cout << "others done" << std::endl;
    
    return 0;
}

int model::save_model_phi(std::string filename) const
{
    std::ofstream fout(filename, std::ios::binary);
    if (!fout)
        throw std::runtime_error( "Error: Cannot open file to save: " + filename );

    unsigned version = 1;
    unsigned Ku = K;
    //write (version, numDims, numClusters)
    fout.write((char *)&(version), sizeof(unsigned));
    fout.write((char *)&V, sizeof(unsigned));
    fout.write((char *)&Ku, sizeof(unsigned));
    
    fout.write((char *)(p_wk), sizeof(double)*K*V);

    fout.close();
    std::cout << "phi done" << std::endl;

    return 0;
}


int model::save_model_top_words(std::string filename) const
{
    std::ofstream fout(filename);
    if (!fout)
        throw std::runtime_error( "Error: Cannot open file to save: " + filename );

    unsigned _n_top_words = n_top_words;
    if (_n_top_words > V)   _n_top_words = V;

    std::vector<unsigned> idx(V);
    for (unsigned short k = 0; k < K; k++)
    {
        std::iota(std::begin(idx), std::end(idx), 0);
        auto comp_x = [&](unsigned a, unsigned b) { return p_wk[a*K+k] > p_wk[b*K+k]; };
        std::sort(std::begin(idx), std::end(idx), comp_x);
        
        fout << "Topic " << k << "th:" << std::endl;
        for (unsigned i = 0; i < _n_top_words; i++)
            fout << "\t" << id2word[idx[i]] << "   " << p_wk[idx[i]*K+k] << std::endl;
    }

    fout.close();
    std::cout << "twords done" << std::endl;

    return 0;
}

size_t model::msg_size() const
{
    size_t vocab_len = 0;
    for(const auto& s : id2word )
        vocab_len += s.size() + 1;
    return sizeof(unsigned short)
        + sizeof(unsigned)*6
        + sizeof(double)*K*V
        + sizeof(char)*vocab_len
        + sizeof(double)*time_ellapsed.size()
        + sizeof(double)*likelihood.size();
}

// Serialize to a buffer
char* model::serialize() const
{
    //Covert following to char* buff with following order
    // K | V | n_iters | n_save | n_top_words | p_wk | id2word | time_ellapsed.size() | time_ellapsed | likelihood.size() | likelihood
    char* buff = new char[msg_size()];

    char* pos = buff;
    unsigned temp = 0;

    // insert K
    size_t shift = sizeof(unsigned short);
    const char* start = (char*)&(K);
    const char* end = start + shift;
    std::copy(start, end, pos);
    pos += shift;

    // insert V
    shift = sizeof(unsigned);
    start = (char*)&(V);
    end = start + shift;
    std::copy(start, end, pos);
    pos += shift;
    
    // insert n_iters
    shift = sizeof(unsigned);
    start = (char*)&(n_iters);
    end = start + shift;
    std::copy(start, end, pos);
    pos += shift;
    
    // insert n_save
    shift = sizeof(unsigned);
    start = (char*)&(n_save);
    end = start + shift;
    std::copy(start, end, pos);
    pos += shift;
    
    // insert n_top_words
    shift = sizeof(unsigned);
    start = (char*)&(n_top_words);
    end = start + shift;
    std::copy(start, end, pos);
    pos += shift;

    // insert word|topic distribution
    shift = sizeof(double)*K*V;
    start = (char*)(p_wk);
    end = start + shift;
    std::copy(start, end, pos);
    pos += shift;
    
    // insert id2word
    for(const auto& w : id2word )
    {
        shift = w.size() + 1;
        start = w.c_str();
        end = start + shift;
        std::copy(start, end, pos);
        pos += shift;
    }
    
    // insert time_size
    temp = time_ellapsed.size();
    shift = sizeof(unsigned);
    start = (char*)&(temp);
    end = start + shift;
    std::copy(start, end, pos);
    pos += shift;
    shift = sizeof(double)*temp;
    start = (char *)(time_ellapsed.data());
    end = start + shift;
    std::copy(start, end, pos);
    pos += shift;
    
    // insert likelihood
    temp = likelihood.size();
    shift = sizeof(unsigned);
    start = (char*)&(temp);
    end = start + shift;
    std::copy(start, end, pos);
    pos += shift;
    shift = sizeof(double)*temp;
    start = (char *)(likelihood.data());
    end = start + shift;
    std::copy(start, end, pos);
    pos += shift;

    //std::cout<<"Message size: " << msg_size() << ", " << pos - buff << std::endl;
    return buff;
}

// Deserialize from a buffer
void model::deserialize(char* buff)
{
    /** Convert char* buff into following buff
    K | V | n_iters | n_save | n_top_words | p_wk | id2word | time_ellapsed.size() | time_ellapsed | likelihood.size() | likelihood **/
    //char* save = buff;

    // extract K and V
    K = *((unsigned short *)buff);
    //std::cout << "K: " << K << std::endl;
    buff += sizeof(unsigned short);
    V = *((unsigned *)buff);
    //std::cout << "V: " << V << std::endl;
    buff += sizeof(unsigned);
    //std::cout << "K: " << K << ", V: " << V << std::endl;
    Vbeta = V*beta;

    // extract n_iters, n_save, and n_top_words
    n_iters = *((unsigned *)buff);
    //std::cout << "n_iters: " << n_iters << std::endl;
    buff += sizeof(unsigned);
    n_save = *((unsigned *)buff);
    //std::cout << "n_save: " << n_save << std::endl;
    buff += sizeof(unsigned);
    n_top_words = *((unsigned *)buff);
    //std::cout << "n_top_words: " << n_top_words << std::endl;
    buff += sizeof(unsigned);
    
    // extract p_wk
    if (p_wk)    delete[] p_wk;
    p_wk = new double[V*K];
    std::copy(buff, buff + sizeof(double)*V*K, (char *) p_wk);
    buff += sizeof(double)*V*K;
    
    // extract id2word
    id2word.reserve(V);
    //std::cout << "Extracting words: " << std::endl;
    for(unsigned w = 0; w < V; ++w)
    {
        id2word.emplace_back(buff);
        buff += id2word.back().size() + 1;
        //std::cout << id2word.back() << std::endl;
    }
    
    // extract time_ellapsed
    unsigned temp = *((unsigned *)buff);
    //std::cout << "time_ellapsed.size: " << temp << std::endl;
    buff += sizeof(unsigned);
    time_ellapsed.reserve(temp);
    for(unsigned r = 0; r < temp; ++r)
    {
        time_ellapsed.push_back( *((double *)buff) );
        //std::cout << "time[" << r << "]: " << time_ellapsed[r] << std::endl;
        buff += sizeof(double);
    }
    
    // extract likelihood
    temp = *((unsigned *)buff);
    //std::cout << "likelihood.size: " << temp << std::endl;
    buff += sizeof(unsigned);
    likelihood.reserve(temp);
    for(unsigned r = 0; r < temp; ++r)
    {
        likelihood.push_back( *((double *)buff) );
        //std::cout << "llh[" << r << "]: " << likelihood[r] << std::endl;
        buff += sizeof(double);
    }

    // initialize concurency control variables
    unsigned ntt = num_table_threads();
    unsigned nst = n_threads - ntt;
    cbuff = new circular_queue<delta>[nst*ntt];
    inf_stop = false;
    
    // rebuild any auxiliary data structures
    specific_init();
    
    //delete[] save;
}
