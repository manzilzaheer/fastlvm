# ifndef FMM_H
# define FMM_H
// Header containing datastructures and methods for gmm

# include <Eigen/Core>
# include <iostream>
# include <vector>

#include "model.h"
#include "../cover_tree/cover_tree.h"
#include "../commons/vose.h"

class simpleGMM : public model
{
protected:
    int sampling(const pointList&, unsigned);   // sampling on machine i outsourced to children
};

class canopy1GMM : public model
{
    std::vector<unsigned> surrogateMAP;
	std::vector<pointType> surrogate;
	std::vector<voseAlias> q;
protected:
    //int specific_init();    // if sampling algo need some specific inits
    //int sampling(const pointType& p);   // sampling on machine i outsourced to children
    //int updater();
};

class canopy2GMM : public model
{
    std::vector<unsigned> surrogateMAP;
	std::vector<pointType> surrogate;
	std::vector<voseAlias> q;
	std::vector<std::vector<unsigned>> closeMeanID;
	std::vector<double> residualProb;
protected:
    //int specific_init();    // if sampling algo need some specific inits
    //int sampling(const pointType& p);   // sampling on machine i outsourced to children
    //int updater();
};

#endif
