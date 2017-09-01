#ifndef _FTREE_H
#define _FTREE_H

#include "utils.h"

class fTree
{
    unsigned short T;       //Dimension
    double *w;	            //tree structure

public:
    fTree(unsigned short num) : T(num), w(new double[2*num])
    {    }

    fTree(const fTree& f) : T(f.T), w(new double[2*f.T])
    {
        std::copy(f.w, f.w + 2*T, w); 
    }

    ~fTree()
    {
        if (w)  delete[] w;
    }

    inline void recompute(double* weights)
    {
        // Reversely initialize elements
        unsigned short i = 2 * T - 1;
        while (i >= T)
        {
            w[i] = weights[i - T];
            --i;
        }

        while(i > 0)
        {
            w[i] = w[2 * i] + w[2 * i + 1];
            --i;
        }
    }

    inline void update(unsigned short t, double new_w)
    {
        unsigned short i = t + T;
        double delta = new_w - w[i];
        while (i > 0)
        {
            w[i] += delta;
            i = i >> 1;
        }
    }

    inline double getComponent(unsigned short t) const
    {
        return w[t + T];
    }

    inline double getsum() const
    {
        return w[1];
    }

    inline unsigned short sample(double u) const
    {
        unsigned short i = 1;
        u = u * w[i];
        while (i < T)
        {
            i <<= 1;
            if (u >=  w[i])
            {
                u -= w[i];
                ++i;
            }
        }
        return i - T;
    }

};

#endif
