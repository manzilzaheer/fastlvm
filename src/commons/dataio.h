#ifndef _DATAIO_H
#define _DATAIO_H

#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <string>

#include <Eigen/Core>

#include "stringtokenizer.h"

#ifndef NPY_NO_DEPRECATED_API
struct PyObject;
#endif

namespace DataIO
{
    Eigen::Map<Eigen::MatrixXd> readPointFile(std::string fileName, double *data = nullptr);
    std::vector<std::string> read_wordmap(std::string fileName);
    
    class document
    {
        size_t _size;
        unsigned *words;
    
    public:
        document() : _size(0), words(nullptr)
        {     }

        document(size_t length) : _size(length), words(new unsigned[length])
        {     }
        
        document(size_t length, unsigned* arr) : _size(length), words(arr)
        {     }
        
        template <class InputIterator>
        document(InputIterator first, InputIterator last) : _size(std::distance(first, last)), words(new unsigned[std::distance(first, last)])
        {
            std::copy(first, last, words);
        }

        document(const document& d) : _size(d._size), words(new unsigned[d._size])
        {
            std::copy(d.words, d.words + _size, words); 
        }

        document(document&& d) : _size(0), words(nullptr)
        {
            // Copy the data pointer and its length
            _size = d._size;
            words = d.words;
  
            // Release the data pointer from the source object
            d._size = 0;  
            d.words = nullptr;
        }
        
        // document(initializer_list<unsigned> il) : _size(il.size()), words(new unsigned[il.size()])
        // {
            // std::copy(il.begin(), il.end(), words);
        // }

        ~document()
        {
            if (words) delete[] words;
        }
        
        
        /*** Copy/Move assignment operator ***/
        document& operator=(const document& d)  
        {  
            if (this != &d)
            {  
                // Free the existing resource
                if(words) delete[] words;
  
                _size = d._size;
                words = new unsigned[_size];
                std::copy(d.words, d.words + _size, words);
            }
            return *this;
        }

        document& operator=(document&& d)  
        {  
            if (this != &d)
            {  
                // Free the existing resource
                if(words) delete[] words;
  
                _size = d._size;
                words = d.words;
                
                // Release the data pointer from the source 
                d._size = 0;  
                d.words = nullptr;
            }
            return *this;
        }

        template <class InputIterator>
        void reassign(InputIterator first, InputIterator last)
        {
            // Free the existing resource
            if(words) delete[] words;
  
            _size = std::distance(first, last);
            words = new unsigned[_size];
            std::copy(first, last, words);
        }

        void reassign(size_t length, unsigned* arr)
        {
            // Free the existing resource
            if(words) delete[] words;
  
            _size = length;
            words = arr;
        }
        
        
        /*** Iterator access ***/
        inline unsigned* begin()
        {
            return _size>0 ? words : nullptr; 
        }

        inline unsigned* end()
        {
            return _size>0 ? &words[_size] : nullptr; 
        }

        inline const unsigned* begin() const
        {
            return _size>0 ? words : nullptr;
        }

        inline const unsigned* end() const
        {
            return _size>0 ? &words[_size] : nullptr;
        }


        /*** Element access ***/
        inline unsigned*& data()
        {
            return words;
        }
        
        inline const unsigned* data() const
        {
            return words;
        }

        inline unsigned& at(size_t idx)
        {
            return words[idx];
        }
        
        inline const unsigned at(size_t idx) const
        {
            return words[idx];
        }
        
        inline unsigned& operator [](size_t idx)
        {
            return words[idx];
        }
        
        inline const unsigned operator [](size_t idx) const
        {
            return words[idx];
        }
        
        /*** Capacity ***/
        inline size_t size() const
        {
            return _size;
        }
        
        int write(std::ofstream& fout) const;
        int read(std::ifstream& fin);
    };
    
    class corpus
    {
        const int version = 2;
        size_t _size; // number of documents
        document * docs;
    
    public:
        corpus() : _size(0), docs(nullptr)
        {     }
        
        corpus(document * d, size_t N) : _size(N), docs(d)
        {     }

        corpus(std::string fname) : _size(0), docs(nullptr)
        {
            std::ifstream fin(fname, std::ios::binary); 
            read(fin);
            fin.close();
        }

        corpus(const corpus& c) : _size(c._size), docs(new document[c._size])
	{
	    std::copy(c.docs, c.docs + _size, docs);
	}

        corpus(corpus&& c) : _size(0), docs(nullptr)
	{
	    // Copy the data pointer and its length
	    _size = c._size;
	    docs = c.docs;

	    // Release the data pointer from the source object
	    c._size = 0;
	    c.docs = nullptr;
	}
        
        ~corpus()
        {
            if (docs)
                delete[] docs;
        }

	/*** Copy/Move assignment operator ***/
	corpus& operator=(const corpus& c)
	{
	    if (this != &c)
	    {
		// Free the existing resource
		if(docs) delete[] docs;

		_size = c._size;
		docs = new document[_size];
		std::copy(c.docs, c.docs + _size, docs);
	    }
	    return *this;
	}

	corpus& operator=(corpus&& c)
	{
	    if (this != &c)
	    {
		// Free the existing resource
		if(docs) delete[] docs;

		_size = c._size;
		docs = c.docs;

		// Release the data pointer from the source
		c._size = 0;
		c.docs = nullptr;
	    }
	    return *this;
        }
        
        /*** Iterator access ***/
        inline document* begin()
        {
            return _size>0 ? docs : nullptr; 
        }

        inline document* end()
        {
            return _size>0 ? &docs[_size] : nullptr; 
        }

        inline const document* begin() const
        {
            return _size>0 ? docs : nullptr;
        }

        inline const document* end() const
        {
            return _size>0 ? &docs[_size] : nullptr;
        }


        /*** Element access ***/
        inline document* data()
        {
            return docs;
        }
        
        inline const document* data() const
        {
            return docs;
        }

        inline document& at(size_t idx)
        {
            return docs[idx];
        }
        
        inline const document& at(size_t idx) const
        {
            return docs[idx];
        }
        
        inline document& operator [](size_t idx)
        {
            return docs[idx];
        }
        
        inline document& operator [](size_t idx) const
        {
            return docs[idx];
        }
        
        /*** Capacity ***/
        inline size_t size() const
        {
            return _size;
        }
        
        void release()
        {
            for(auto& d : *this)
                d.data() = nullptr;
        }
        

        int read_data(std::string dfile, std::map<std::string, unsigned> * wordmap, std::set<std::string> * stopwords = nullptr);
        int from_python(PyObject* collection);

        int write(std::ofstream& fout) const;
        int read(std::ifstream& fin);
    };
}


#endif

