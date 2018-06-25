#ifndef _UTILS_H
#define _UTILS_H

#include <map>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <random>
#include <string>
#include <atomic>
#include <thread>
#include <future>

#include <Eigen/Core>

#include "suff_stats.h"
#include "fast_rand.h"

#ifdef _MSC_VER

typedef __int32 int32_t;
typedef unsigned __int32 uint32_t;
typedef __int64 int64_t;
typedef unsigned __int64 uint64_t;
typedef intptr_t ssize_t;

#endif

namespace utils
{
        
    inline void progressbar(size_t x, size_t n, unsigned int w = 50)
    {
        static std::mutex mtx;
        if ( (x != n) && (x % (n/10+1) != 0) ) return;

        float ratio =  x/(float)n;
        unsigned c = unsigned(ratio * w);

        if(mtx.try_lock())
        {
            std::cerr << std::setw(3) << (int)(ratio*100) << "% [";
            for (unsigned x=0; x<c; x++) std::cerr << "=";
            for (unsigned x=c; x<w; x++) std::cerr << " ";
            std::cerr << "]\r" << std::flush;
            mtx.unlock();
        }
    }
    
    template<class InputIt, class UnaryFunction>
    UnaryFunction parallel_for_each(InputIt first, InputIt last, UnaryFunction f)
    {
        unsigned cores = std::thread::hardware_concurrency();

        auto task = [&f](InputIt start, InputIt end)->void{
            for (; start < end; ++start)
                f(*start);
        };

        const size_t total_length = std::distance(first, last);
        const size_t chunk_length = total_length / cores;
        InputIt chunk_start = first;
        std::vector<std::future<void>>  for_threads;
        for (unsigned i = 0; i < cores - 1; ++i)
        {
            const auto chunk_stop = std::next(chunk_start, chunk_length);
            for_threads.push_back(std::async(std::launch::async, task, chunk_start, chunk_stop));
            chunk_start = chunk_stop;
        }
        for_threads.push_back(std::async(std::launch::async, task, chunk_start, last));

        for (auto& thread : for_threads)
            thread.get();
        return f;
    }

    template<class UnaryFunction>
    UnaryFunction parallel_for(size_t first, size_t last, UnaryFunction f)
    {
        unsigned cores = std::thread::hardware_concurrency();

        auto task = [&f](size_t start, size_t end)->void{
            for (; start < end; ++start)
                f(start);
        };

        const size_t total_length = last - first;
        const size_t chunk_length = total_length / cores;
        size_t chunk_start = first;
        std::vector<std::future<void>>  for_threads;
        for (unsigned i = 0; i < cores - 1; ++i)
        {
            const auto chunk_stop = chunk_start + chunk_length;
            for_threads.push_back(std::async(std::launch::async, task, chunk_start, chunk_stop));
            chunk_start = chunk_stop;
        }
        for_threads.push_back(std::async(std::launch::async, task, chunk_start, last));

        for (auto& thread : for_threads)
            thread.get();
        return f;
    }
    
    template<class UnaryFunction>
    UnaryFunction parallel_block_for(size_t first, size_t last, UnaryFunction f)
    {
        unsigned cores = std::thread::hardware_concurrency();

        const size_t total_length = last - first;
        const size_t chunk_length = total_length / cores;
        size_t chunk_start = first;
        std::vector<std::future<void>>  for_threads;
        for (unsigned i = 0; i < cores - 1; ++i)
        {
            const auto chunk_stop = chunk_start + chunk_length;
            for_threads.push_back(std::async(std::launch::async, f, chunk_start, chunk_stop));
            chunk_start = chunk_stop;
        }
        for_threads.push_back(std::async(std::launch::async, f, chunk_start, last));

        for (auto& thread : for_threads)
            thread.get();
        return f;
    }

    template<class UnaryFunction>
    UnaryFunction parallel_for_progressbar(size_t first, size_t last, UnaryFunction f)
    {
        unsigned cores = std::thread::hardware_concurrency();
        const size_t total_length = last - first;
        const size_t chunk_length = total_length / cores;
        
        if(total_length <= 10000)
        {
            auto task = [&f,&chunk_length](size_t start, size_t end)->void{
                for (; start < end; ++start){
                    f(start);
                }
            };
            
            size_t chunk_start = first;
            std::vector<std::future<void>>  for_threads;
            for (unsigned i = 0; i < cores - 1; ++i)
            {
                const auto chunk_stop = chunk_start + chunk_length;
                for_threads.push_back(std::async(std::launch::async, task, chunk_start, chunk_stop));
                chunk_start = chunk_stop;
            }
            for_threads.push_back(std::async(std::launch::async, task, chunk_start, last));

            for (auto& thread : for_threads)
                thread.get();
        }
        else
        {
            auto task = [&f,&chunk_length](size_t start, size_t end)->void{
                for (; start < end; ++start){
                    progressbar(start%chunk_length, chunk_length);
                    f(start);
                }
            };
            
            size_t chunk_start = first;
            std::vector<std::future<void>>  for_threads;
            for (unsigned i = 0; i < cores - 1; ++i)
            {
                const auto chunk_stop = chunk_start + chunk_length;
                for_threads.push_back(std::async(std::launch::async, task, chunk_start, chunk_stop));
                chunk_start = chunk_stop;
            }
            for_threads.push_back(std::async(std::launch::async, task, chunk_start, last));

            for (auto& thread : for_threads)
                thread.get();
            progressbar(chunk_length, chunk_length);
            std::cerr << std::endl;
        }
        
        return f;
    }

    template<typename T>
    void add_to_atomic(std::atomic<T>& foo, T& bar)
    {
        auto current = foo.load(std::memory_order_relaxed);
        while (!foo.compare_exchange_weak(current, current + bar, std::memory_order_relaxed, std::memory_order_relaxed));
    }
    
    class ParallelAddList
    {
        size_t left;
        size_t right;
        Eigen::VectorXd res;
        const std::vector<Eigen::VectorXd>& pList;
        static std::atomic<int> objectCount;

        void run()
        {
            res = Eigen::VectorXd::Zero(pList[0].size());
            for(size_t i = left; i<right; ++i)
                res += pList[i];
        }

    public:
        ParallelAddList(const std::vector<Eigen::VectorXd>& pL) : pList(pL)
        {
	    objectCount++;
            this->left = 0;
            this->right = pL.size();
            compute();
        }
        ParallelAddList(size_t left, size_t right, const std::vector<Eigen::VectorXd>& pL) : pList(pL)
        {
	    objectCount++;
            this->left = left;
            this->right = right;
        }

        ~ParallelAddList()
        {
	    objectCount--;
	}

        int compute()
        {
            if ((right - left < 500000) || (objectCount.load() > 128))
            {
                run();
                return 0;
            }

            size_t split = (right - left) / 2;

            ParallelAddList* t1 = new ParallelAddList(left, left + split, pList);
            ParallelAddList* t2 = new ParallelAddList(left + split, right, pList);

            std::future<int> f1 = std::async(std::launch::async, &ParallelAddList::compute, t1);
            std::future<int> f2 = std::async(std::launch::async, &ParallelAddList::compute, t2);

            f1.get();
            f2.get();

            res = t1->res + t2->res;

            delete t1;
            delete t2;

            return 0;
        }
        
        Eigen::VectorXd get_result()
        {
            return res;
        }
    };
    
    class ParallelAddMatrix
    {
        size_t left;
        size_t right;
        Eigen::VectorXd res;
        const Eigen::MatrixXd& pMatrix;
        static std::atomic<int> objectCount;

        void run()
        {
            res = Eigen::VectorXd::Zero(pMatrix.rows());
            for(size_t i = left; i<right; ++i)
                res += pMatrix.col(i);
        }

    public:
        ParallelAddMatrix(const Eigen::MatrixXd& pM) : pMatrix(pM)
        {
	    objectCount++;
            this->left = 0;
            this->right = pM.cols();
            compute();
        }
        ParallelAddMatrix(size_t left, size_t right, const Eigen::MatrixXd& pM) : pMatrix(pM)
        {
	    objectCount++;
            this->left = left;
            this->right = right;
        }

        ~ParallelAddMatrix()
        {
	    objectCount--;
	}

        int compute()
        {
            if ((right - left < 500000) || (objectCount.load() > 128))
            {
                run();
                return 0;
            }

            size_t split = (right - left) / 2;

            ParallelAddMatrix* t1 = new ParallelAddMatrix(left, left + split, pMatrix);
            ParallelAddMatrix* t2 = new ParallelAddMatrix(left + split, right, pMatrix);

            std::future<int> f1 = std::async(std::launch::async, &ParallelAddMatrix::compute, t1);
            std::future<int> f2 = std::async(std::launch::async, &ParallelAddMatrix::compute, t2);

            f1.get();
            f2.get();

            res = t1->res + t2->res;

            delete t1;
            delete t2;

            return 0;
        }
        
        Eigen::VectorXd get_result()
        {
            return res;
        }
    };
    
    class ParallelAddMatrixNP
    {
        size_t left;
        size_t right;
        Eigen::VectorXd res;
        const Eigen::Map<Eigen::MatrixXd>& pMatrix;
        static std::atomic<int> objectCount;

        void run()
        {
            res = Eigen::VectorXd::Zero(pMatrix.rows());
            for(size_t i = left; i<right; ++i)
                res += pMatrix.col(i);
        }

    public:
        ParallelAddMatrixNP(const Eigen::Map<Eigen::MatrixXd>& pM) : pMatrix(pM)
        {
	    objectCount++;
            this->left = 0;
            this->right = pM.cols();
            compute();
        }
        ParallelAddMatrixNP(size_t left, size_t right, const Eigen::Map<Eigen::MatrixXd>& pM) : pMatrix(pM)
        {
	    objectCount++;
            this->left = left;
            this->right = right;
        }

        ~ParallelAddMatrixNP()
        {
	    objectCount--;
	}

        int compute()
        {
            if ((right - left < 500000) || (objectCount.load() > 128))
            {
                run();
                return 0;
            }

            size_t split = (right - left) / 2;

            ParallelAddMatrixNP* t1 = new ParallelAddMatrixNP(left, left + split, pMatrix);
            ParallelAddMatrixNP* t2 = new ParallelAddMatrixNP(left + split, right, pMatrix);

            std::future<int> f1 = std::async(std::launch::async, &ParallelAddMatrixNP::compute, t1);
            std::future<int> f2 = std::async(std::launch::async, &ParallelAddMatrixNP::compute, t2);

            f1.get();
            f2.get();

            res = t1->res + t2->res;

            delete t1;
            delete t2;

            return 0;
        }
        
        Eigen::VectorXd get_result()
        {
            return res;
        }
    };
    
    class ParallelAddClusters
    {
        size_t left;
        size_t right;
        Eigen::VectorXd res;
        const std::vector<SuffStatsOne>& clusters;
        static std::atomic<int> objectCount;

        void run()
        {
            res = Eigen::VectorXd::Zero(clusters[0].get_dim());
            for(size_t i = left; i<right; ++i)
                res += clusters[i].get_mean();
        }

    public:
        ParallelAddClusters(const std::vector<SuffStatsOne>& cL) : clusters(cL)
        {
	    objectCount++;
            this->left = 0;
            this->right = cL.size();
            compute();
        }
        ParallelAddClusters(size_t left, size_t right, const std::vector<SuffStatsOne>& cL) : clusters(cL)
        {
	    objectCount++;
            this->left = left;
            this->right = right;
        }

        ~ParallelAddClusters()
        {
	    objectCount--;
	}

        int compute()
        {
            if ((right - left < 500000) || (objectCount.load() > 128))
            {
                run();
                return 0;
            }

            size_t split = (right - left) / 2;

            ParallelAddClusters* t1 = new ParallelAddClusters(left, left + split, clusters);
            ParallelAddClusters* t2 = new ParallelAddClusters(left + split, right, clusters);

            std::future<int> f1 = std::async(std::launch::async, &ParallelAddClusters::compute, t1);
            std::future<int> f2 = std::async(std::launch::async, &ParallelAddClusters::compute, t2);

            f1.get();
            f2.get();

            res = t1->res + t2->res;

            delete t1;
            delete t2;

            return 0;
        }
        
        Eigen::VectorXd get_result()
        {
            return res;
        }
    };
    
    class ParallelDistanceComputeList
    {
        size_t left;
        size_t right;
        Eigen::VectorXd res;
        Eigen::VectorXd& vec;
        const std::vector<Eigen::VectorXd>& pList;
        static std::atomic<int> objectCount;

        void run()
        {
            res = Eigen::VectorXd::Zero(pList.size());
            for(size_t i = left; i<right; ++i)
                res[i] = (pList[i]-vec).norm();
        }

    public:
        ParallelDistanceComputeList(const std::vector<Eigen::VectorXd>& pL, Eigen::VectorXd& v) : vec(v), pList(pL)
        {
	    objectCount++;
            this->left = 0;
            this->right = pL.size();
            compute();
        }
        ParallelDistanceComputeList(size_t left, size_t right, const std::vector<Eigen::VectorXd>& pL, Eigen::VectorXd& v) : vec(v), pList(pL)
        {
	    objectCount++;
            this->left = left;
            this->right = right;
        }

        ~ParallelDistanceComputeList()
        {
	    objectCount--;
	}

        int compute()
        {
            if ((right - left < 10000) || (objectCount.load() > 128))
            {
                run();
                return 0;
            }

            size_t split = (right - left) / 2;

            ParallelDistanceComputeList* t1 = new ParallelDistanceComputeList(left, left + split, pList, vec);
            ParallelDistanceComputeList* t2 = new ParallelDistanceComputeList(left + split, right, pList, vec);

            std::future<int> f1 = std::async(std::launch::async, &ParallelDistanceComputeList::compute, t1);
            std::future<int> f2 = std::async(std::launch::async, &ParallelDistanceComputeList::compute, t2);

            f1.get();
            f2.get();

            res = t1->res + t2->res;

            delete t1;
            delete t2;

            return 0;
        }
        
        Eigen::VectorXd get_result()
        {
            return res;
        }
    };
    
    class ParallelDistanceCompute
    {
        size_t left;
        size_t right;
        Eigen::VectorXd res;
        Eigen::VectorXd& vec;
        const Eigen::MatrixXd& pMatrix;
        static std::atomic<int> objectCount;

        void run()
        {
            res = Eigen::VectorXd::Zero(pMatrix.cols());
            for(size_t i = left; i<right; ++i)
                res[i] = (pMatrix.col(i)-vec).norm();
        }

    public:
        ParallelDistanceCompute(const Eigen::MatrixXd& pM, Eigen::VectorXd& v) : vec(v), pMatrix(pM)
        {
            objectCount++;
            this->left = 0;
            this->right = pM.cols();
            compute();
        }
        ParallelDistanceCompute(size_t left, size_t right, const Eigen::MatrixXd& pM, Eigen::VectorXd& v) : vec(v), pMatrix(pM)
        {
	    objectCount++;
            this->left = left;
            this->right = right;
        }

        ~ParallelDistanceCompute()
        {
	    objectCount--;
	}

        int compute()
        {
            if ((right - left < 10000) || (objectCount.load() > 128))
            {
                run();
                return 0;
            }

            size_t split = (right - left) / 2;

            ParallelDistanceCompute* t1 = new ParallelDistanceCompute(left, left + split, pMatrix, vec);
            ParallelDistanceCompute* t2 = new ParallelDistanceCompute(left + split, right, pMatrix, vec);

            std::future<int> f1 = std::async(std::launch::async, &ParallelDistanceCompute::compute, t1);
            std::future<int> f2 = std::async(std::launch::async, &ParallelDistanceCompute::compute, t2);

            f1.get();
            f2.get();

            res = t1->res + t2->res;

            delete t1;
            delete t2;

            return 0;
        }
        
        Eigen::VectorXd get_result()
        {
            return res;
        }
    };
    
    class ParallelDistanceComputeNP
    {
        size_t left;
        size_t right;
        Eigen::VectorXd res;
        Eigen::VectorXd& vec;
        const Eigen::Map<Eigen::MatrixXd>& pMatrix;
        static std::atomic<int> objectCount;

        void run()
        {
            res = Eigen::VectorXd::Zero(pMatrix.cols());
            for(size_t i = left; i<right; ++i)
                res[i] = (pMatrix.col(i)-vec).norm();
        }

    public:
        ParallelDistanceComputeNP(const Eigen::Map<Eigen::MatrixXd>& pM, Eigen::VectorXd& v) : vec(v), pMatrix(pM)
        {
	    objectCount++;
            this->left = 0;
            this->right = pM.cols();
            compute();
        }
        ParallelDistanceComputeNP(size_t left, size_t right, const Eigen::Map<Eigen::MatrixXd>& pM, Eigen::VectorXd& v) : vec(v), pMatrix(pM)
        {
	    objectCount++;
            this->left = left;
            this->right = right;
        }

        ~ParallelDistanceComputeNP()
        {
	    objectCount--;
	}

        int compute()
        {
            if ((right - left < 10000) || (objectCount.load() > 128))
            {
                run();
                return 0;
            }

            size_t split = (right - left) / 2;

            ParallelDistanceComputeNP* t1 = new ParallelDistanceComputeNP(left, left + split, pMatrix, vec);
            ParallelDistanceComputeNP* t2 = new ParallelDistanceComputeNP(left + split, right, pMatrix, vec);

            std::future<int> f1 = std::async(std::launch::async, &ParallelDistanceComputeNP::compute, t1);
            std::future<int> f2 = std::async(std::launch::async, &ParallelDistanceComputeNP::compute, t2);

            f1.get();
            f2.get();

            res = t1->res + t2->res;

            delete t1;
            delete t2;

            return 0;
        }
        
        Eigen::VectorXd get_result()
        {
            return res;
        }
    };
    
    class ParallelDistanceComputeCluster
    {
        size_t left;
        size_t right;
        Eigen::VectorXd res;
        Eigen::VectorXd& vec;
        const std::vector<SuffStatsOne>& clusters;
        static std::atomic<int> objectCount;

        void run()
        {
            res = Eigen::VectorXd::Zero(clusters.size());
            for(size_t i = left; i<right; ++i)
                res[i] = (clusters[i].get_mean()-vec).norm();
        }

    public:
        ParallelDistanceComputeCluster(const std::vector<SuffStatsOne>& cL, Eigen::VectorXd& v) : vec(v), clusters(cL)
        {
	    objectCount++;
            this->left = 0;
            this->right = cL.size();
            compute();
        }
        ParallelDistanceComputeCluster(size_t left, size_t right, const std::vector<SuffStatsOne>& cL, Eigen::VectorXd& v) : vec(v), clusters(cL)
        {
	    objectCount++;
            this->left = left;
            this->right = right;
        }

        ~ParallelDistanceComputeCluster()
        {
            objectCount--;
	}

        int compute()
        {
            if ((right - left < 10000) || (objectCount.load() > 128))
            {
                run();
                return 0;
            }

            size_t split = (right - left) / 2;

            ParallelDistanceComputeCluster* t1 = new ParallelDistanceComputeCluster(left, left + split, clusters, vec);
            ParallelDistanceComputeCluster* t2 = new ParallelDistanceComputeCluster(left + split, right, clusters, vec);

            std::future<int> f1 = std::async(std::launch::async, &ParallelDistanceComputeCluster::compute, t1);
            std::future<int> f2 = std::async(std::launch::async, &ParallelDistanceComputeCluster::compute, t2);

            f1.get();
            f2.get();

            res = t1->res + t2->res;

            delete t1;
            delete t2;

            return 0;
        }
        
        Eigen::VectorXd get_result()
        {
            return res;
        }
    };

    struct ParsedArgs
    {
        unsigned K;
        unsigned n_iters;
        unsigned n_save;
        unsigned n_threads;
        unsigned n_top_words;
        std::string algo;
        std::string init_type;
        std::string data_path;
        std::string name;
        std::string out_path;
        
        ParsedArgs(int argc, char ** argv)
        {
            // set default values
            K = 100;
            n_iters = 1000;
            n_save = 200;
            n_threads = std::thread::hardware_concurrency();
            n_top_words = 15;
            algo = "simple";
            data_path = "./";
            name = "";
            out_path = "./";
            
            // iterate
            std::vector<std::string> arguments(argv, argv + argc);
            for (auto arg = arguments.begin(); arg != arguments.end(); ++arg)
            {
                if (*arg == "--method")
                {
                    algo = *(++arg);
                    if (algo == "")
                        algo = "simple";
                }
                else if (*arg == "--output-model")
                {
                    out_path = *(++arg);
                    if (out_path == "") 
                        out_path = "./";
                }
                else if (*arg == "--init-type")
                {
                    init_type = *(++arg);
                    if (init_type == "") 
                        init_type = "random";
                }
                else if (*arg == "--dataset")
                {
                    name = *(++arg);
                    if (name == "")
                        throw std::runtime_error("Error: Invalid file path to training corpus.");
                    std::string::size_type idx = name.find_last_of("/");
                    if (idx == std::string::npos)
                    {
                        data_path = "./";
                    }
                    else
                    {
                        data_path = name.substr(0, idx + 1);
                        name = name.substr(idx + 1, name.size() - data_path.size());
                    }
                }            
                else if (*arg == "--num-clusters")
                {
                    int _K = std::stoi(*(++arg));
                    if (_K > 0)
                        K = _K;
                    else
                        throw std::runtime_error("Error: Invalid number of clusters.");
                }
                else if (*arg == "--num-topics")
                {
                    int _K = std::stoi(*(++arg));
                    if (_K > 0)
                        K = _K;
                    else
                        throw std::runtime_error("Error: Invalid number of topics.");
                }
                else if (*arg == "--num-iterations")
                {
                    int _n_iters = std::stoi(*(++arg));
                    if (_n_iters > 0)
                        n_iters = _n_iters;
                    else
                        throw std::runtime_error("Error: Invalid number of iterations.");
                }
                else if (*arg == "--num-top-words")
                {
                    int _n_top_words = std::stoi(*(++arg));
                    if (_n_top_words > 0)
                        n_top_words = _n_top_words;
                    else
                        throw std::runtime_error("Error: Invalid number of top words.");
                }
                else if (*arg == "--num-threads")
                {
                    int _n_threads = std::stoi(*(++arg));
                    if(_n_threads > 0)
                        n_threads = _n_threads;
                    else
                        throw std::runtime_error("Error: Invalid number of threads.");
                }
                else if (*arg == "--output-state-interval")
                {
                    int _n_save = std::stoi(*(++arg));
                    if (_n_save > 0)
                        n_save = _n_save;
                    else
                        throw std::runtime_error("Error: Invalid output state interval.");
                }
            }
        }
        
        ParsedArgs(unsigned num_atoms=100, unsigned num_iters=1000, std::string algorithm="simple",
                   unsigned output_interval=200, unsigned top_words=15,
                   unsigned num_threads=std::thread::hardware_concurrency(),
                   std::string init_scheme="random", std::string output_path="./")
        {
            K = num_atoms;
            n_iters = num_iters;
            n_save = output_interval;
            n_threads = num_threads;
            n_top_words = top_words;
            algo = algorithm;
            init_type = init_scheme;
            data_path = "./";
            name = "custom";
            out_path = output_path;
        }
    };
    
    size_t* KMeanspp(const Eigen::MatrixXd& points, unsigned K);
}


#endif

