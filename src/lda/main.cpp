//#define EIGEN_USE_MKL_ALL        //uncomment if available
#include "lda.h"
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
    
    model *lda = NULL;

    // read options and data
    utils::ParsedArgs args(argc, argv);

    // read vocabulary
    std::vector<std::string> word_map = DataIO::read_wordmap(args.data_path + args.name + ".vocab");

    // read data
    DataIO::corpus trngdata(args.data_path + args.name + "-" + std::to_string(world_rank) + ".dat");
    DataIO::corpus testdata(args.data_path + args.name + "-test.dat");
    
    // initialize the model
    lda = model::init(args, word_map, world_rank);
    
    // Train the model
    lda->fit(trngdata, testdata);

    // Finally test the model   
    std::cout << "Final log likelihood: " << lda->evaluate(testdata) << std::endl;

    system("pause");
    
    #ifdef MULTIMACHINE
    // Finalize the MPI environment.
    MPI_Finalize();
    #endif
    
    // Clear memory
    if(lda) delete lda;
    
    return 0;
}
