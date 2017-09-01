#ifndef _LDA_H
#define _LDA_H

#include <iostream>
#include <vector>

#include "model.h"
#include "../commons/vose.h"

class adGLDA : public model
{
public:
    // estimate GLDA model using Gibbs sampling
    int sampling(const DataIO::corpus&, unsigned);
};

class scaGLDA : public model
{
public:
    std::vector<voseAlias> q;

    // estimate GLDA model using alias sampling
    int specific_init();
    int sampling(const DataIO::corpus&, unsigned);
    int updater();
};

class canopyGLDA : public model
{
public:
    std::vector<voseAlias> q;

    // estimate GLDA model using alias sampling
    //int specific_init();
    //int sampling(const DataIO::corpus&, unsigned);
    //int updater();
};

#endif
