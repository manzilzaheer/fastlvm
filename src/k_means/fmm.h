# ifndef FMM_H
# define FMM_H
// Header containing datastructures and methods for gmm

# include <Eigen/Core>
# include <iostream>
# include <vector>

#include "model.h"
#include "../cover_tree/cover_tree.h"

class simpleKM : public model
{
protected:
    int sampling(const pointType& p);   // sampling on machine i outsourced to children
};

class canopyKM : public model
{
    CoverTree* ClusterTree;
protected:
    int specific_init();    // if sampling algo need some specific inits
    int sampling(const pointType& p);   // sampling on machine i outsourced to children
    int updater();
public:
    canopyKM()
    { ClusterTree = NULL; }
    ~canopyKM()
    { if (ClusterTree) delete ClusterTree; }
};

#endif
