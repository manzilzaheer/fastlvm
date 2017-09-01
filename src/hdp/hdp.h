/* 
 * File:   hdp.h
 * Author: Manzil
 *
 * Created on February 29, 2016, 2:50 AM
 */

#ifndef _HDP_H
#define _HDP_H

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

#include "model.h"
#include "../commons/stirling.h"
#include "../commons/utils.h"
#include "../commons/vose.h"

class simpleHDP : public model
{
    int sampling(const DataIO::corpus&, unsigned);
    int updater();
};

class aliasHDP : public model
{
    std::vector<voseAlias> q;
    std::vector<unsigned short> sample_count;
    std::vector<std::mutex> qmtx;                                // lock for data-structures involving n_wk
    std::vector<unsigned short> kRecent;
    std::vector<size_t> revK;

    unsigned short Kold;

    int specific_init();
    int sampling(const DataIO::corpus&, unsigned);
    int updater();
    void generateQtable(unsigned w);
};


class stcHDP : public model
{
    Stirling stirling_;
    
    std::vector<spvector2> nt_mks;
    std::atomic<unsigned>* t_k;
    std::atomic<unsigned> tsum;

    int specific_init();
    int init_train(const DataIO::corpus&);
    int sampling(const DataIO::corpus&, unsigned);
    int updater();
    int writer(unsigned);
};

class stcAliasHDP : public model
{
    Stirling stirling_;
    
    std::vector<spvector2> nt_mks;
    std::atomic<unsigned>* t_k;
    std::atomic<unsigned> tsum;

    std::vector<voseAlias> q;
    std::vector<unsigned short> sample_count;
    std::vector<std::mutex> qmtx;                                // lock for data-structures involving n_wk
    std::vector<unsigned short> kRecent;
    std::vector<size_t> revK;

    unsigned short Kold;

    int specific_init();
    int init_train(const DataIO::corpus&);
    int sampling(const DataIO::corpus&, unsigned);
    int updater();
    int writer(unsigned);
    void generateQtable(unsigned w);
};

#endif
