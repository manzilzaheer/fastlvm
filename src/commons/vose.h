#ifndef _VOSE_H
#define _VOSE_H

#include <atomic>
#include <queue>
#include <vector>
#include <random>
#include <algorithm>
#include "utils.h"

class voseAlias
{
    unsigned short n;                               //Dimension
    double wsum;                                    //Sum of proportions
    std::vector<std::pair<double, unsigned>> table; //Alias probabilities and indices
    
public:

    voseAlias(unsigned short num);        
    void recompute(const double* p, double T);
    inline double getsum() const { return wsum; }
    inline unsigned short size() const { return n; }
    inline void resize_and_recompute(unsigned short num, const double* p, double T)
    {
        if (n!=num)
        {
            n = num;
            table.resize(n);
        }
        recompute(p, T);
    }
    inline unsigned short sample(unsigned short fair_die , double u) const
    {
        //1. Generate a fair die roll from an n-sided die; call the side i.
        //2. Flip a biased coin that comes up heads with probability Prob[i].
        bool res = u < table[fair_die].first;
        //3. If the coin comes up "heads," return i. Otherwise, return Alias[i].
        return res ? fair_die : table[fair_die].second;
    }
};

#endif
