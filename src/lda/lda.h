#ifndef _LDA_H
#define _LDA_H

#include <iostream>
#include <vector>

#include "model.h"
#include "../commons/vose.h"

class adLDA : public model
{
protected:
    // estimate LDA model using Gibbs sampling
    int sampling(const DataIO::corpus&, unsigned);
};

class scaLDA : public model
{
protected:
    std::vector<voseAlias> q;

    // estimate LDA model using alias sampling
    int specific_init();
    int sampling(const DataIO::corpus&, unsigned);
    int updater();
};

#endif
