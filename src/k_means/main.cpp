//#define EIGEN_USE_MKL_ALL        //uncomment if available
#include "fmm.h"
#include "dataio.h"

void show_help()
{
    std::cout << "Command line usage:" << std::endl;
    std::cout << "\t./kmeans --method xxx --testing-mode yyy [--num-sampling-threads <int>] [--num-table-threads <int>] [--alpha <real+>] [--beta <real+>] [--num-topics <int>] [--num-iterations <int>] [-odir <string>] [-savestep <int>] [-twords <int>] --training-file <string> [--testing-file <string>]" << std::endl;
    std::cout << "--method xxx :\n"
        << "  xxx can be{ simpleKM, canopyKM}.\n"
        << "--num-clusters <int> :\n"
        << "  The number of clusters. The default value is 100.\n"
        << "--num-iterations <int> :\n"
        << "  The number of iterations. The default value is 2000.\n"
        << "--output-model <string> :\n"
        << "  The location where statistics and trained topics will be stored. The default location is the current working directory.\n"
        << "--output-state-interval <int> :\n"
        << "  The step(counted by the number of iterations) at which the K-Means model is saved to hard disk. The default value is 200.\n"
        << "--init-type zzz :\n"
        << "  zz cab be{ random, firstk, kmeanspp, covertree}. The default value is random.\n"
        << "--dataset <string> :\n"
        << "  The location of dataset. Data format described in readme under data folder." << std::endl;
}

pointList get_init_points(unsigned K, std::string init_type, const pointList& trngdata)
{
    unsigned D = trngdata.rows();
    unsigned N = trngdata.cols();
    double *data = new double[K*D];
    pointList initial_centres(data, D, K);
    
    if (init_type == "firstk")
    {
        std::copy ( trngdata.data(), trngdata.data()+K*D, data );
    }
    else if (init_type == "random")
    {
        xorshift128plus rng_;
        for(unsigned k=0; k<K; ++k)
        {
            unsigned r = rng_.rand_k(N);
            initial_centres.col(k) = trngdata.col(r);
        }
    }
    else if (init_type == "kmeanspp")
    {
        size_t* init_idx = utils::KMeanspp(trngdata, K);
        for(unsigned k=0; k<K; ++k)
        {
            initial_centres.col(k) = trngdata.col(init_idx[k]);
        }
        delete[] init_idx;
    }
    else if (init_type == "covertree")
    {
        CoverTree* ct = CoverTree::from_multimachine(trngdata, 3); 
        initial_centres = ct->getBestInitialPoints(K, data);
        delete ct;
    }
    
    return initial_centres;
}

int main(int argc, char ** argv) 
{
    int world_size = 1;
    int world_rank = 0;
    #ifdef MULTIMACHINE
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    #endif
    
    model *fmm = NULL;

    // read options and data
    utils::ParsedArgs args(argc, argv);    
    pointList trngdata = DataIO::readPointFile(args.data_path + args.name + "-" + std::to_string(world_rank) + ".dat");
    pointList testdata = DataIO::readPointFile(args.data_path + args.name + "-test.dat");
    if (testdata.rows() != trngdata.rows())
        throw std::runtime_error("Train and test dimensions do not match!");
    pointList inits = get_init_points(args.K, args.init_type, trngdata);
    
    // initialize the model
    fmm = model::init(args, inits, world_rank);
    
    // Train the model
    fmm->fit(trngdata, testdata);

    // Finally test the model   
    std::cout << "Final log likelihood: " << fmm->evaluate(testdata) << std::endl;

    system("pause");
    
    #ifdef MULTIMACHINE
    // Finalize the MPI environment.
    MPI_Finalize();
    #endif
    
    // Clear memory
    if(trngdata.data()) delete[] trngdata.data();
    if(testdata.data()) delete[] testdata.data();
    if(inits.data()) delete[] inits.data();
    if(fmm) delete fmm;
    
    return 0;
}
