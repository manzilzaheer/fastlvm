#include "model.h"
#include "glda.h"

model::model() :
    K(100),
    V(0),

    id2vec(nullptr),
    
    rank(0),
    n_iters(1000),
    n_save(200),
    n_threads(8),
    n_top_words(15),
    
    name("default"),
    mdir("./")
{    }

model::model(utils::ParsedArgs args, const std::vector<std::string>& word_map,
             const Eigen::Map<Eigen::MatrixXd>& word_vec, int world_rank) :
    K(args.K),
    V(word_map.size()),
    
    // Copy wordmap
    id2word(word_map),
    id2vec(new Eigen::Map<Eigen::MatrixXd>(const_cast<double*>(word_vec.data()), word_vec.rows(), word_vec.cols())),

    // Set parameters
    rank(world_rank),
    n_iters(args.n_iters),
    n_save(args.n_save),
    n_threads(args.n_threads),
    n_top_words(args.n_top_words),
    name(args.name),
    mdir(args.out_path)
{
    // topics
    topics.reserve(K);
    for(unsigned short k=0; k<K; ++k)
    {
        topics.emplace_back(word_vec.col(k), word_vec.col(k).cwiseAbs2());
        #ifdef MULTIMACHINE
        topics[k].synchronize();
        #endif
    }

    // reserve memory for training statistics
    time_ellapsed.reserve(n_iters);
    likelihood.reserve(n_iters);

    //display all configurations
    if (world_rank == 0)
    {
        std::cout << "model dir = " << mdir << std::endl;
        std::cout << "K = " << K << std::endl;
        std::cout << "V = " << V << std::endl;
        std::cout << "alpha = " << alpha << std::endl;
        std::cout << "n_iters = " << n_iters << std::endl;
        std::cout << "n_save = " << n_save << std::endl;
        std::cout << "num_threads = " << n_threads << std::endl;
        std::cout << "num_top_words = " << n_top_words << std::endl;
    }
}
    
model::~model()
{ 
    if (id2vec)     delete id2vec;
}

model* model::init(utils::ParsedArgs args, const std::vector<std::string>& word_map,
                   const Eigen::Map<Eigen::MatrixXd>& word_vec, int world_rank)
{
    model *glda = nullptr;
    
    if(args.K > std::numeric_limits<unsigned short>::max() )
            throw std::runtime_error("Error: Number of topics must be less than 65535! Supplied: " + args.K);

    if (args.algo == "simple")
    {
        glda = new adGLDA;
        if (world_rank == 0)    std::cout << "Running simple AD-GLDA" << std::endl;
    }
    else if (args.algo == "scaGLDA")
    {
        glda = new scaGLDA;
        if (world_rank == 0)    std::cout << "Running GLDA using ESCA" << std::endl;
    }
    else if (args.algo == "canopyGLDA")
    {
        glda = new canopyGLDA;
        if (world_rank == 0)    std::cout << "Running GLDA using canopy" << std::endl;
    }
    else
    {
        std::cout << "Error: Invalid inference algorithm! " << args.algo << std::endl;
        throw std::runtime_error("Error: Invalid inference algorithm! " + args.algo);
    }

    if (glda != nullptr)
    {
        // Set parameters
        glda->K = args.K;
        glda->V = (unsigned) word_map.size();
        glda->rank = world_rank;
        glda->n_iters = args.n_iters;
        glda->n_save = args.n_save;
        glda->n_threads = args.n_threads;
        glda->n_top_words = args.n_top_words;
        glda->name = args.name;
        glda->mdir = args.out_path;
        
        // Copy wordmap
        glda->id2word = word_map;
        glda->id2vec = new Eigen::Map<Eigen::MatrixXd>(const_cast<double*>(word_vec.data()), word_vec.rows(), word_vec.cols());

        // topics
        glda->topics.reserve(glda->K);
        for(unsigned short k=0; k<glda->K; ++k)
        {
            glda->topics.emplace_back(word_vec.col(k), word_vec.col(k));
            #ifdef MULTIMACHINE
            glda->topics[k].synchronize();
            #endif
        }

        // reserve memory for training statistics
        glda->time_ellapsed.reserve(glda->n_iters);
        glda->likelihood.reserve(glda->n_iters);

        if (world_rank == 0) std::cout << "Initialising inference method specific data structures" << std::endl;
        glda->specific_init();
        
        //display all configurations
        if (world_rank == 0)
        {
            std::cout << "model dir = " << glda->mdir << std::endl;
            std::cout << "K = " << glda->K << std::endl;
            std::cout << "V = " << glda->V << std::endl;
            std::cout << "alpha = " << glda->alpha << std::endl;
            std::cout << "n_iters = " << glda->n_iters << std::endl;
            std::cout << "n_save = " << glda->n_save << std::endl;
            std::cout << "num_threads = " << glda->n_threads << std::endl;
            std::cout << "num_top_words = " << glda->n_top_words << std::endl;
        }
    }
    else
    {
        throw std::runtime_error("Error: Inference algorithm not specified!");
    }

    return glda;
}

double model::fit(const DataIO::corpus& trngdata, const DataIO::corpus& testdata)
{
    //initialize
    init_train(trngdata);
    
    std::chrono::high_resolution_clock::time_point ts, tn;
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
    
    //size_t M = trngdata.size();
    for (unsigned iter = 0; iter < n_iters; ++iter)
    {
        if (rank == 0)
        {
            std::cout << "Iteration " << iter << " ..." << std::endl;
            if (iter % n_save == 0)
            {
                // saving the model
                std::cout << "Saving the model at iteration " << iter << "..." << std::endl;
                save_model(iter);
            }
            ts = std::chrono::high_resolution_clock::now();
        }

        //sample each document
        std::vector<std::future<int>>  sampling_threads;
        for (unsigned tsn = 0; tsn < n_threads; ++tsn)
            sampling_threads.emplace_back(std::async(std::launch::async, &model::sampling, this, std::cref(trngdata), tsn));
        for (auto& thread : sampling_threads)
            if(thread.get())
                std::cout << "Sampling failed!!!" << std::endl;

        // gather, clean and update from all workers
        sharer();
        updater();
        cleaner();
    
        if (rank == 0)
        {
            tn = std::chrono::high_resolution_clock::now();
            time_ellapsed.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(tn - ts).count());
            likelihood.push_back(evaluate(testdata));
            std::cout << "Likelihood on held out points: " << likelihood.back() << " at time " << time_ellapsed.back() << std::endl;
        }
    }

    if (rank == 0)
    {
        std::cout << "GLDA completed!" << std::endl;
        std::cout << "Saving the final model!" << std::endl;
        save_model(n_iters);
    }
    
    // free up buffer memory
    n_mks.resize(0);
    p_mks.resize(0);

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
        // per-topic prior
        double alphaK = alpha/K;
        
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
                
                    
                    // Compute probability for each topic
                    const pointType& wvec = id2vec->col(doc[j]);
                    double maxLogProb = -1 * std::numeric_limits<double>::max();
                    for (unsigned short k = 0; k < K; ++k)
                    {
                        p[k] = log(nd_m[k] + alphaK) + topics[k].computeProbability(wvec);
                        if (maxLogProb < p[k])
                            maxLogProb = p[k];
                    }
                    
                    double psum = 0;
                    for (unsigned short k = 0; k < K; ++k)
                    {
                        p[k] -= maxLogProb;
                        psum += exp(p[k]);
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
            for(size_t j = 0; j < N; ++j)
            {
                const pointType& wvec = id2vec->col(doc[j]);
                double wsum = 0;
                double maxLogProb = -1*std::numeric_limits<double>::max();
                for (unsigned short k = 0; k < K; ++k)
                {
                    p[k] = log(nd_m[k] + alphaK) + topics[k].computeProbability(wvec);
                    if (maxLogProb < p[k])
                        maxLogProb = p[k];
                }
                for (unsigned short k = 0; k < K; ++k)
                    wsum += exp(p[k] - maxLogProb);
                dsum += log(wsum) + maxLogProb;
            }
            tsum += dsum - N*log(N + alpha);
        }
        utils::add_to_atomic(sum, tsum);
        tokens.fetch_add(ttokens, std::memory_order_relaxed);
        delete[] nd_m;
        delete[] p;
    });
    unsigned D = static_cast<unsigned>(topics[0].get_dim());
    return sum.load(std::memory_order_relaxed)/tokens.load(std::memory_order_relaxed) - 0.5*D*log(M_PI);
}

/*std::vector<unsigned> model::predict(const dataset& testdata) const
{
    return labels;
}*/

std::pair<Eigen::Map<Eigen::MatrixXd>, Eigen::Map<Eigen::MatrixXd>> model::get_topic_matrix() const
{
    intptr_t D = topics[0].get_dim();
    //std::cout << "I am inside model get topic matrix" << std::endl;
    //std::cout << D << ", " << K << std::endl;
    double *data = new double[D*K];
    Eigen::Map<Eigen::MatrixXd> means(data, D, K);
    for(unsigned short k = 0; k < K; ++k)
        means.col(k) =  topics[k].get_mean();

    data = new double[D*K];
    Eigen::Map<Eigen::MatrixXd> vars(data, D, K);
    for(unsigned short k = 0; k < K; ++k)
        vars.col(k) =  0.5 * topics[k].get_var().cwiseInverse();

    return std::make_pair(means, vars);
}

std::vector<std::vector<std::string>> model::get_top_words(unsigned num_words /*= 15*/) const
{
    if (num_words > V)
        num_words = V;
    
    std::vector<std::vector<std::string>> result(K, std::vector<std::string>(num_words));
    utils::parallel_for(0, K, [&](size_t k)->void{
        std::vector<unsigned> idx(V);
        std::iota(std::begin(idx), std::end(idx), 0);
        auto comp_x = [&](unsigned a, unsigned b) { return topics[k].computeProbability(id2vec->col(a)) > topics[k].computeProbability(id2vec->col(b)); };
        std::sort(std::begin(idx), std::end(idx), comp_x);

        for (unsigned i = 0; i < num_words; ++i)
            result[k][i] = id2word[idx[i]];
    });
    return result;
}

int model::init_train(const DataIO::corpus& trngdata)
{   
    size_t M = trngdata.size();
    
    // allocate heap memory for per document variables
    n_mks.resize(M);
    p_mks.resize(M);
    
    // random consistent assignment for model variables
    utils::parallel_for_progressbar(0, M, [&](size_t m)->void{
        // random number generator
        xorshift128plus rng_;
        
        std::map<unsigned short, unsigned> map_nd_m;
        const auto& doc = trngdata[m];

        //std::cout << "I reached 01 " << std::endl;

        for (const auto& w : doc)
        {
            //unsigned short topic = rng_.rand_k(K);
            unsigned short topic = (m + rng_.rand_k((unsigned short) 15))%K;

            // number of words in document i assigned to topic j
            map_nd_m[topic] += 1;

            // number of instances of word i assigned to topic j
            topics[topic].addPoint(id2vec->col(w));
        }
        // transfer to sparse representation
        for (auto myc : map_nd_m)
                n_mks[m].push_back(myc.first, myc.second);
    });
    
    sharer();
    updater();
    cleaner();
    
    return 0;
}

void model::sharer()
{
    #ifdef MULTIMACHINE
    for (auto& topic : topics)
        topic.allreduce();
    #endif
}

int model::updater()
{
    std::swap(n_mks, p_mks);
    utils::parallel_for_each(topics.begin(), topics.end(),
        [](SuffStatsTwo& t){ t.updateParameters();
    });
    return 0;
}

void model::cleaner()
{
    utils::parallel_for(0, n_mks.size(), [&](size_t m)->void{
        n_mks[m].clear_to_capacity(p_mks[m].size());
    });
    utils::parallel_for_each(topics.begin(), topics.end(),
        [](SuffStatsTwo& t){ t.resetParameters();
    });
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
        throw std::runtime_error("Error: Cannot open file to save: " + filename);

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
        throw std::runtime_error("Error: Cannot open file to save: " + filename);

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

    fout << "alpha = " << alpha << std::endl;
    fout << "num-topics = " << K << std::endl;
    fout << "num-words = " << V << std::endl;
    fout << "num-iters = " << n_iters << std::endl;
    fout << "num-threads = " << n_threads << std::endl;
    fout << "num-top-words = " << n_top_words << std::endl;
    fout << "output-state-interval = " << n_save << std::endl;
    
    fout.close();
    std::cout << "others done" << std::endl;
    
    return 0;
}

int model::save_model_phi(std::string filename) const
{
    std::ofstream fout(filename, std::ios::binary);
    if (!fout)
        throw std::runtime_error("Error: Cannot open file to save: " + filename);

    unsigned version = 1;
    unsigned Ku = 2*K;
    unsigned D = unsigned(topics[0].get_dim());
    //write (version, numDims, 2xnumTopics)
    fout.write((char *)&(version), sizeof(unsigned));
    fout.write((char *)&(D), sizeof(unsigned));
    fout.write((char *)&(Ku), sizeof(unsigned));
    
    //write mean+variance
    for (const auto& topic : topics)
        topic.write_to_file(fout);

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
        auto comp_x = [&](unsigned a, unsigned b) { return topics[k].computeProbability(id2vec->col(a)) > topics[k].computeProbability(id2vec->col(b)); };
        std::sort(std::begin(idx), std::end(idx), comp_x);
        
        fout << "Topic " << k << "th:" << std::endl;
        for (unsigned i = 0; i < _n_top_words; i++)
            fout << "\t" << id2word[idx[i]] << "   " << topics[k].computeProbability(id2vec->col(idx[i])) << std::endl;
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
    unsigned D = unsigned(topics[0].get_dim());
    return sizeof(unsigned short)
        + sizeof(unsigned)*7
        + sizeof(char)*vocab_len
        + sizeof(double)*V*D
        + sizeof(double)*K*D*2
        + sizeof(double)*time_ellapsed.size()
        + sizeof(double)*likelihood.size();
}

// Serialize to a buffer
char* model::serialize() const
{
    //Covert following to char* buff with following order
    // K | V | D | n_iters | n_save | n_top_words | id2word | id2vec | topics | time_ellapsed.size() | time_ellapsed | likelihood.size() | likelihood
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

    // insert D
    unsigned D = unsigned(topics[0].get_dim());
    shift = sizeof(unsigned);
    start = (char*)&(D);
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
    
    // insert id2word
    for(const auto& w : id2word )
    {
        shift = w.size() + 1;
        start = w.c_str();
        end = start + shift;
        std::copy(start, end, pos);
        pos += shift;
    }

    // insert id2vec
    shift = sizeof(double)*V*D;
    start = (char*)(id2vec->data());
    end = start + shift;
    std::copy(start, end, pos);
    pos += shift;

    // insert topics
    shift = sizeof(pointType::Scalar)*D;
    for (const auto& topic : topics)
    {
        // mean
        pointType mean = topic.get_mean();
        start = (char *)(mean.data());
        end = start + shift;
        std::copy(start, end, pos);
        pos += shift;
        // variance
        pointType var = topic.get_var();
        start = (char *)(var.data());
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
    K | V | D | n_iters | n_save | n_top_words | id2word | id2vec | topics | time_ellapsed.size() | time_ellapsed | likelihood.size() | likelihood **/
    //char* save = buff;

    // extract K and V
    K = *((unsigned short *)buff);
    //std::cout << "K: " << K << std::endl;
    buff += sizeof(unsigned short);
    V = *((unsigned *)buff);
    //std::cout << "V: " << V << std::endl;
    buff += sizeof(unsigned);
    unsigned D = *((unsigned *)buff);
    //std::cout << "D: " << D << std::endl;
    buff += sizeof(unsigned);
    //std::cout << "K: " << K << ", V: " << V << std::endl;

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

    // extract id2word
    id2word.reserve(V);
    //std::cout << "Extracting words: " << std::endl;
    for(unsigned w = 0; w < V; ++w)
    {
        id2word.emplace_back(buff);
        buff += id2word.back().size() + 1;
        //std::cout << id2word.back() << std::endl;
    }
    //std::cout << "Got words: " << id2word.size() << std::endl;
    
    // extract id2vec
    double* p_wk = new double[V*D];
    std::copy(buff, buff + sizeof(double)*V*D, (char*)p_wk);
    buff += sizeof(double)*V*D;
    id2vec = new Eigen::Map<Eigen::MatrixXd>(p_wk, D, V);
    //std::cout << "Word vectors extracted" << std::endl;

    // extract topics
    topics.reserve(K);
    for(unsigned short k = 0; k < K; ++k)
    {
        pointType m(D), v(D);
        std::copy(buff, buff + sizeof(double)*D, (char*)m.data());
        buff += sizeof(double)*D;
        std::copy(buff, buff + sizeof(double)*D, (char*)v.data());
        buff += sizeof(double)*D;
        /*for (unsigned i = 0; i < D; ++i)
        {
            m[i] = *((pointType::Scalar *)buff);
            std::cout << "m_" << k << "[" << i << "]: " << m[i] << std::endl;
            buff += sizeof(pointType::Scalar);
        }
        for (unsigned i = 0; i < D; ++i)
        {
            v[i] = *((pointType::Scalar *)buff);
            std::cout << "v_" << k << "[" << i << "]: " << v[i] << std::endl;
            buff += sizeof(pointType::Scalar);
        }*/
        topics.emplace_back(m, v);
        #ifdef MULTIMACHINE
        topics[k].synchronize();
        #endif
    }
    //std::cout << "Topics read " << std::endl;
    
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
    
    // rebuild any auxiliary data structures
    specific_init();
    
    //delete[] save;
}
