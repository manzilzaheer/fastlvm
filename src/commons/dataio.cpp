#include "dataio.h"

Eigen::Map<Eigen::MatrixXd> DataIO::readPointFile(std::string fileName, double *data /* = nullptr */)
{
    std::ifstream fin(fileName, std::ios::in|std::ios::binary);

    // Check for existance for file
    if (!fin)
        throw std::runtime_error("File not found : " + fileName);

    // Read the header for number of points, dimensions
    unsigned temp = 0;
    unsigned numPoints = 0;
    unsigned numDims = 0;
    fin.read((char *)&temp, sizeof(unsigned));
    //if (temp != version)
    //    throw std::runtime_error("Dataset version incorrect!");
    fin.read((char *)&numDims, sizeof(unsigned));
    fin.read((char *)&numPoints, sizeof(unsigned));

    // Printing for debugging
    std::cout << "\nNumber of points: " << numPoints << "\nNumber of dims : " << numDims << std::endl;
    
    // allocate memory
    if (data == nullptr)
        data = new double[numDims*numPoints];

    // Read the points
    fin.read((char *)(data), sizeof(double)*numPoints*numDims);
    
    // Close the file
    fin.close();
    
    // Matrix of points
    Eigen::Map<Eigen::MatrixXd> pointMatrix(data, numDims, numPoints);
    std::cout<<"IsRowMajor?: "<<pointMatrix.IsRowMajor << std::endl;

    std::cout<<pointMatrix.rows() << " " << pointMatrix.cols() << std::endl;
    std::cout<<pointMatrix(0,0) << " " << pointMatrix(0,1) << " " << pointMatrix(1,0) << std::endl;

    return pointMatrix;
}

std::vector<std::string> DataIO::read_wordmap(std::string wordmapfile)
{
    std::string temp;

    std::ifstream fin(wordmapfile);
    if (!fin)
        throw std::runtime_error( "Error: Unable to read the file: " + wordmapfile );

    std::vector<std::string> word_map;
    while (std::getline(fin, temp))
        word_map.push_back(temp);
    fin.close();

    return word_map;
}

int DataIO::document::write(std::ofstream& fout) const
{
    if (!fout)
        throw std::runtime_error("Cannot open to write!");
    unsigned M = (unsigned)_size;
    fout.write((char *)&M, sizeof(unsigned));
    fout.write((char *)(words), sizeof(unsigned)*_size);
    
    return 0;
}

int DataIO::document::read(std::ifstream& fin)
{
    if (!fin)
        throw std::runtime_error("Dataset not found!");
    unsigned M;
    fin.read((char *)&M, sizeof(unsigned));
    _size = M;
    
    // free existing memory
    if(words) delete[] words;
    
    words = new unsigned[_size];
    fin.read((char *)(words), sizeof(unsigned)*_size);
    
    return 0;
}

int DataIO::corpus::read_data(std::string dfile, std::map<std::string, unsigned> * pword2id, std::set<std::string> * stopwords)
{       
    std::ifstream fin(dfile);
    if (!fin)
    {
        std::cout << "Error: Invalid data file" << std::endl;
        throw std::runtime_error("Error: Invalid data file: " + dfile);
    }
    
    // if (pword2id == nullptr)
        // pword2id = new std::map<std::string, unsigned>;
    
    if (stopwords == nullptr)
        stopwords = new std::set<std::string>;
    std::cout << "Size of stopwords: " << stopwords->size() << std::endl;
    
    // free existing memory
    if(docs)
        delete[] docs;
    
    std::string line;
    // retrieve the number of documents in dataset
    std::getline(fin, line);
    try{
        _size = std::stoi(line);
    }catch (std::exception& e)
    {
        //std::cout << "While trying to read number of lines, encountered exception: ";
        //std::cout << e.what() << std::endl;
        //std::cout << "Trying to estimate number of lines via brute force ..." << std::endl;
        _size = 1;
        while ( std::getline(fin, line) )
            ++_size;
        fin.close();
        fin.open(dfile);
    }
    //std::cout << "Num documents found: " << _size << std::endl;
    if (_size <= 0)
        throw std::runtime_error( "Error: Empty corpus!" );
        
    // allocate memory for corpus
    docs = new DataIO::document[_size];
    
    unsigned temp_words[100000];
    std::map<std::string, unsigned>::iterator it;
    for (size_t i = 0; i < _size; ++i)
    {
        //progressbar
        if ( (i % (_size/10+1) == 0) && _size>10000 )
        {
            const unsigned w = 50;
            double ratio =  i/(double)_size;
            unsigned c = unsigned(ratio * w);

            std::cerr << std::setw(3) << (int)(ratio*100) << "% [";
            for (unsigned x=0; x<c; x++) std::cerr << "=";
            for (unsigned x=c; x<w; x++) std::cerr << " ";
            std::cerr << "]\r" << std::flush;
        }
        
        std::getline(fin, line);
        StringTokenizer strtok(line);

        unsigned length = strtok.count_tokens();
        if (length <= 0)
            std::runtime_error("Error: Invalid document object! " + i);
        
        unsigned js = 0;
        for (unsigned j = 0; j < length; ++j)
        {
            std::string key = strtok.nextToken();
            //std::cout << j << ", " << key << std::endl;

            if(stopwords->count(key))
                continue;
            
            it = pword2id->lower_bound(key);
            if (it == pword2id->end() || key < it->first)
            {
                // word not found, i.e., new word
                temp_words[js] = (unsigned) pword2id->size();
                pword2id->insert(it, std::map<std::string, unsigned>::value_type(key, (unsigned)pword2id->size()));
            }
            else
            {
                // Give the word current id
                temp_words[js] = it->second;
            }
            ++js;
        }

        // add new doc to the corpus
        docs[i].reassign(temp_words, temp_words + js);
    }
    if ( _size>10000 )
    {
        std::cerr << std::setw(3) << (int)(100) << "% [";
        for (unsigned x=0; x<50; x++) std::cerr << "=";
        std::cerr << "]" << std::endl;
    }
    
    fin.close();
    
    return 0;
}

int DataIO::corpus::write(std::ofstream& fout) const
{
    if (!fout)
        throw std::runtime_error("Cannot open to write!");
    unsigned M = (unsigned)_size;
    fout.write((char *)&version, sizeof(int));
    fout.write((char *)&M, sizeof(unsigned));
    for(const auto& d : *this)
        d.write(fout);
    
    return 0;
}

int DataIO::corpus::read(std::ifstream& fin)
{
    int temp;
    if (!fin)
        throw std::runtime_error("Dataset not found!");
    fin.read((char *)&temp, sizeof(int));
    if (temp != version)
        throw std::runtime_error("Dataset version incorrect!");
    
    // free existing memory
    if(docs)
        delete[] docs;

    unsigned M;
    fin.read((char *)&M, sizeof(unsigned));
    _size = M;
    
    // allocate memory for corpus
    docs = new DataIO::document[_size];
   
    for (auto& d : *this)
        d.read(fin);

    return 0;
}
