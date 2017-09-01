#include "model.h"
#include "fmm.h"

model::model() : 
    K(100),
    
    n_k(nullptr),
    
    rank(0),
    n_iters(1000),
    n_save(200),
    n_threads(8),
    
    name("default"),
    mdir("./")
{    }

model::~model()
{
    if (n_k)    delete[] n_k;
}

model* model::init(utils::ParsedArgs args, const pointList& mean_inits, const pointList& var_inits, int world_rank)
{
    model *fmm = nullptr;

    if (args.algo == "simple")
    {
        fmm = new simpleGMM;
        if (world_rank == 0)    std::cout << "Running simple GMM using ESCA" << std::endl;
    }
    else if (args.algo == "canopy1GMM")
    {
        fmm = new canopy1GMM;
        if (world_rank == 0)    std::cout << "Running GMM using canopy I" << std::endl;
    }
    else if (args.algo == "canopy2GMM")
    {
        fmm = new canopy2GMM;
        if (world_rank == 0)    std::cout << "Running GMM using canopy II" << std::endl;
    }
    else
    {
        std::cout << "Error: Invalid inference algorithm! " << args.algo << std::endl;
        throw std::runtime_error("Error: Invalid inference algorithm! " + args.init_type);
    }

    if (fmm != nullptr)
    {
        fmm->rank = world_rank;
        fmm->K = args.K;
        fmm->n_iters = args.n_iters;
        fmm->n_save = args.n_save;
        fmm->n_threads = args.n_threads;
        fmm->name = args.name;
        fmm->mdir = args.out_path;
        
        fmm->n_k = new std::atomic<unsigned>[fmm->K];
        std::fill(fmm->n_k, fmm->n_k + fmm->K, 0);
        fmm->logPi.resize(fmm->K);
        std::fill(fmm->logPi.begin(), fmm->logPi.end(), -log(fmm->K));

        fmm->clusters.reserve(fmm->K);
        for(unsigned k=0; k<fmm->K; ++k)
        {
            if (var_inits.cols()==1)
                fmm->clusters.emplace_back(mean_inits.col(k), var_inits.col(0));
            else
                fmm->clusters.emplace_back(mean_inits.col(k), var_inits.col(k));
            #ifdef MULTIMACHINE
            fmm->clusters[k].synchronize();
            #endif
        }
        fmm->time_ellapsed.reserve(fmm->n_iters);
        fmm->likelihood.reserve(fmm->n_iters);

        if (world_rank == 0) std::cout << "Initialising inference method specific data structures" << std::endl;
        fmm->specific_init();
        
        //display all configurations
        if (world_rank == 0){
            std::cout << "model dir = " << fmm->mdir << std::endl;
            std::cout << "K = " << fmm->K << std::endl;
            std::cout << "n_iters = " << fmm->n_iters << std::endl;
            std::cout << "n_save = " << fmm->n_save << std::endl;
            std::cout << "num_threads = " << fmm->n_threads << std::endl;
        }
        
    }
    else
    {
        throw std::runtime_error("Error: Inference algorithm not specified!");
    }

    return fmm;
}

double model::fit(const pointList& trngdata, const pointList& testdata)
{
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
    
    size_t N = trngdata.cols();
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

        //sample each point
        std::vector<std::future<int>>  sampling_threads;
        for (unsigned tsn = 0; tsn < n_threads; ++tsn)
            sampling_threads.emplace_back(std::async(std::launch::async, &model::sampling, this, std::cref(trngdata), tsn));
        for (auto& thread : sampling_threads)
            if(thread.get())
                std::cout << "Sampling failed!!!" << std::endl;
        // n_threads=1;
        // sampling(std::cref(trngdata), 0);

        // for (const auto& cluster : clusters)
        //  std::cout << cluster;

        // gather, clean and update from all workers
        sharer();
        updater();
        cleaner();
    
        // for (const auto& cluster : clusters)
        //   std::cout << cluster;
    
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
        std::cout << "GMM completed!" << std::endl;
        std::cout << "Saving the final model!" << std::endl;
        save_model(n_iters);
    }

    return likelihood.back();
}

double model::evaluate(const pointList& testdata) const
{
    size_t N = testdata.cols();
    size_t D = testdata.rows();
    std::atomic<double> sum{0};
    utils::parallel_for_progressbar(0, N, [&](size_t i)->void{
        double clusP = 0;
        double *p = new double[K];
        const pointType& point = testdata.col(i);
        double maxP = -1*std::numeric_limits<double>::max();
        for (unsigned k = 0; k < K; ++k)
        {
            double p = clusters[k].computeProbability(point) + logPi[k];
            maxP = (p > maxP) ? p : maxP;
        }
        for (unsigned k = 0; k < K; ++k)
            clusP += exp(p[k] - maxP);
        clusP = log(clusP) + maxP;
        utils::add_to_atomic(sum, maxP);
        delete[] p;
    });
    return sum.load(std::memory_order_relaxed)/N - 0.5*D*log(M_PI);
}

std::vector<unsigned> model::predict(const pointList& testdata) const
{
    size_t N = testdata.cols();
    std::vector<unsigned> labels(N);
    utils::parallel_for_progressbar(0, N, [&](size_t i)->void{
        const pointType& point = testdata.col(i);
        unsigned argMaxP = -1;
        double maxP = -1*std::numeric_limits<double>::max();
        for (unsigned k = 0; k < K; ++k)
        {
            double p = clusters[k].computeProbability(point) + logPi[k];
            if (p > maxP)
            {
                maxP = p;
                argMaxP = k;
            }
        }
        labels[i] = argMaxP;
    });
    return labels;
}

std::pair<pointList, pointList> model::get_centers() const
{
    intptr_t D = clusters[0].get_dim();
    //std::cout << "I am inside model get centers" << std::endl;
    //std::cout << D << ", " << K << std::endl;
    double *data = new double[D*K];
    Eigen::Map<Eigen::MatrixXd> means(data, D, K);
    for(unsigned short k = 0; k < K; ++k)
        means.col(k) =  clusters[k].get_mean();

    data = new double[D*K];
    Eigen::Map<Eigen::MatrixXd> vars(data, D, K);
    for(unsigned short k = 0; k < K; ++k)
        vars.col(k) =  0.5 * clusters[k].get_var().cwiseInverse();

    return std::make_pair(means, vars);
}

int model::updater()
{
    utils::parallel_for_each(clusters.begin(), clusters.end(),
        [](SuffStatsTwo& c){ c.updateParameters(); });
    double denominator = log(std::accumulate(n_k, n_k+K, 0) + K*alpha);
    for (unsigned k = 0; k < K; ++k)
        logPi[k] = log(n_k[k] + alpha) - denominator;
    return 0;
}

void model::sharer()
{
    #ifdef MULTIMACHINE
    MPI_Allreduce(MPI_IN_PLACE, n_k, K, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
    for (auto& cluster : clusters)
        cluster.allreduce();
    #endif
}

void model::cleaner()
{
    std::fill(n_k, n_k + K, 0);
    utils::parallel_for_each(clusters.begin(), clusters.end(), [](SuffStatsTwo& c){ c.resetParameters(); });
}

int model::save_model(int iter) const
{
    std::string model_name = name + "-" + std::to_string(iter);
    
    save_model_phi(mdir + model_name + ".phi");
    save_model_time(mdir + model_name + ".time");
    save_model_llh(mdir + model_name + ".llh");
    
    return 0;
}

int model::save_model_time(std::string filename) const
{
    std::ofstream fout(filename);
    if (!fout)
        throw std::runtime_error("Error: Cannot open file to save: " + filename);

    for (unsigned r = 0; r < time_ellapsed.size(); ++r)
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

    for (unsigned r = 0; r < likelihood.size(); ++r)
        fout << likelihood[r] << std::endl;

    fout.close();
    std::cout << "llh done" << std::endl;

    return 0;
}

int model::save_model_phi(std::string filename) const
{
    std::ofstream fout(filename, std::ios::binary);
    if (!fout)
        throw std::runtime_error("Error: Cannot open file to save: " + filename);

    unsigned version = 1;
    unsigned Ku = 2*K;
    unsigned D = unsigned(clusters[0].get_dim());
    //write (version, numDims, numClusters)
    fout.write((char *)&(version), sizeof(unsigned));
    fout.write((char *)&(D), sizeof(unsigned));
    fout.write((char *)&(Ku), sizeof(unsigned));
    
    //write mean+variance
    for (const auto& cluster : clusters)
        cluster.write_to_file(fout);

    fout.close();
    std::cout << "phi done" << std::endl;

    return 0;
}

size_t model::msg_size() const
{
    unsigned D = unsigned(clusters[0].get_dim());
    return 6 * sizeof(unsigned)
        + sizeof(double)*K
        + sizeof(pointType::Scalar)*D*K*2
        + sizeof(double)*time_ellapsed.size()
        + sizeof(double)*likelihood.size();
}

// Serialize to a buffer
char* model::serialize() const
{
    //Covert following to char* buff with following order
    // K | D | n_iters | n_save | logPi | centres | time_ellapsed.size() | time_ellapsed | likelihood.size() | likelihood
    char* buff = new char[msg_size()];

    char* pos = buff;
    unsigned temp = 0;

    // insert K
    size_t shift = sizeof(unsigned);
    char* start = (char*)&(K);
    char* end = start + shift;
    std::copy(start, end, pos);
    pos += shift;

    // insert D
    unsigned D = unsigned(clusters[0].get_dim());
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

    // insert logPi
    shift = sizeof(double)*K;
    start = (char*)(logPi.data());
    end = start + shift;
    std::copy(start, end, pos);
    pos += shift;

    // insert centers
    shift = sizeof(pointType::Scalar)*D;
    for (const auto& cluster : clusters)
    {
        // mean
        pointType mean = cluster.get_mean();
        start = (char *)(mean.data());
        end = start + shift;
        std::copy(start, end, pos);
        pos += shift;
        // variance
        pointType var = cluster.get_var();
        start = (char *)(var.data());
        end = start + shift;
        std::copy(start, end, pos);
        pos += shift;
    }
    
    // insert time_size
    temp = (unsigned)time_ellapsed.size();
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
    temp = (unsigned)likelihood.size();
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
    K | D | n_iters | n_save | centres | time_ellapsed.size() | time_ellapsed | likelihood.size() | likelihood **/
    //char* save = buff;

    // extract K and D
    K = *((unsigned *)buff);
    //std::cout << "K: " << K << std::endl;
    buff += sizeof(unsigned);
    unsigned D = *((unsigned *)buff);
    //std::cout << "D: " << D << std::endl;
    buff += sizeof(unsigned);
    
    //std::cout << "K: " << K << ", D: " << D << std::endl;

    // extract n_iters and n_save
    n_iters = *((unsigned *)buff);
    //std::cout << "n_iters: " << n_iters << std::endl;
    buff += sizeof(unsigned);
    n_save = *((unsigned *)buff);
    //std::cout << "n_save: " << n_save << std::endl;
    buff += sizeof(unsigned);
    
    // extract n_k
    if (n_k)    delete[] n_k;
    n_k = new std::atomic<unsigned>[K];
    std::fill(n_k, n_k + K, 0);

    // extract logPi
    logPi.resize(K);
    std::copy(buff, buff + sizeof(double)*K, (char*)logPi.data());
    buff += sizeof(double)*K;
    
    // extract centers
    clusters.reserve(K);
    for(unsigned k = 0; k < K; ++k)
    {
        pointType m(D), v(D);
        std::copy(buff, buff + sizeof(double)*D, (char*)m.data());
        buff += sizeof(double)*D;
        std::copy(buff, buff + sizeof(double)*D, (char*)v.data());
        buff += sizeof(double)*D;
        clusters.emplace_back(m, v);
        #ifdef MULTIMACHINE
        clusters[k].synchronize();
        #endif
    }
    //std::cout << " Clusters read " << std::endl; 
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
