﻿# ifndef _COVER_TREE_H
# define _COVER_TREE_H

#include <atomic>
#include <fstream>
#include <iostream>
#include <stack>
#include <map>
#include <numeric>
#include <vector>
#include <shared_mutex>

#include "../commons/suff_stats.h"
#include "../commons/utils.h"

#include <Eigen/Core>

class CoverTree
{
/************************* Internal Functions ***********************************************/
protected:
    /*** Base to use for the calculations ***/
    static constexpr double base = 1.3;
    static double* compute_pow_table();
    static double* powdict;

public:
    /*** structure for each node ***/
    struct Node
    {
        pointType _p;                       // point associated with the node
        std::vector<Node*> children;        // list of children
        int level;                          // current level of the node
        double maxdistUB;                   // upper bound of distance to any of descendants
        unsigned ID;                        // mutable ID of current node
        unsigned UID;                       // external unique ID for current node

        mutable std::shared_timed_mutex mut;// lock for current node
        
        /*** Node modifiers ***/
        double covdist() const                   // covering distance of subtree at current node
        {
            return powdict[level + 1024];
        }
        double sepdist() const                   // separating distance between nodes at current level
        {
            return powdict[level + 1023];
        }
        double dist(const pointType& pp) const   // L2 distance between current node and point pp
        {
            return (_p - pp).norm();
        }
        double dist(const Node* n) const         // L2 distance between current node and node n
        {
            return (_p - n->_p).norm();
        }
        Node* setChild(const pointType& pIns,    // insert a new child of current node with point pIns
                       unsigned UID = 0, 
                       int new_id=-1)   
        {
            Node* temp = new Node;
            temp->_p = pIns;
            temp->level = level - 1;
            temp->maxdistUB = 0; // powdict[level + 1024];
            temp->ID = new_id;
            temp->UID = UID;
            children.push_back(temp);
            return temp;
        }

        /*** erase child ***/
        void erase(size_t pos)
        {
            children[pos] = children.back();
            children.pop_back();
        }

        void erase(std::vector<Node*>::iterator pos)
        {
            *pos = children.back();
            children.pop_back();
        }

        /*** Iterator access ***/
        inline std::vector<Node*>::iterator begin()
        {
            return children.begin();
        }
        inline std::vector<Node*>::iterator end()
        {
            return children.end();
        }
        inline std::vector<Node*>::const_iterator begin() const
        {
            return children.begin();
        }
        inline std::vector<Node*>::const_iterator end() const
        {
            return children.end();
        }

        /*** Pretty print ***/
        friend std::ostream& operator<<(std::ostream& os, const Node& ct)
        {
            if (ct._p.rows()<6)
            {
                Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", "[", "]");
                os << "(" << ct._p.format(CommaInitFmt) << ":" << ct.level << ":" << ct.maxdistUB <<  ":" << ct.ID << ")";
            }
            else
            {
                Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", "", "");
                os << "([" << ct._p.head<3>().format(CommaInitFmt) << ", ..., " << ct._p.tail<3>().format(CommaInitFmt) << "]:" << ct.level << ":" << ct.maxdistUB <<  ":" << ct.ID << ")";
            }
            return os;
        }
    };
    // mutable std::map<int,std::atomic<unsigned>> dist_count;
    std::map<int,unsigned> level_count;

protected:
    Node* root;                         // Root of the tree
    std::atomic<int> min_scale;         // Minimum scale
    std::atomic<int> max_scale;         // Minimum scale
    int truncate_level;                 // Relative level below which the tree is truncated
    bool id_valid;

    std::atomic<unsigned> N;            // Number of points in the cover tree
    unsigned D;                         // Dimension of the points

    std::shared_timed_mutex global_mut;	// lock for changing the root

    /*** Insert point or node at current node ***/
    bool insert(Node* current, const pointType& p, unsigned UID);

    /*** Nearest Neighbour search ***/
    void NearestNeighbour(Node* current, double dist_current, const pointType &p, std::pair<CoverTree::Node*, double>& nn) const;

    /*** k-Nearest Neighbour search ***/
    void kNearestNeighbours(Node* current, double dist_current, const pointType& p, std::vector<std::pair<CoverTree::Node*, double>>& nnList) const;

    /*** Range search ***/
    void rangeNeighbours(Node* current, double dist_current, const pointType &p, double range, std::vector<std::pair<CoverTree::Node*, double>>& nnList) const;
    
    /*** Furthest Neighbour search ***/
    void FurthestNeighbour(Node* current, double dist_current, const pointType &p, std::pair<CoverTree::Node*, double>& nn) const;

    /*** Serialize/Desrialize helper function ***/
    char* preorder_pack(char* buff, Node* current) const;       // Pre-order traversal
    char* postorder_pack(char* buff, Node* current) const;      // Post-order traversal
    void PrePost(Node*& current, char*& pre, char*& post);

    /*** debug functions ***/
    void calc_maxdist();                            //find true maxdist
    void generate_id(Node* current);                //Generate IDs for each node from root as 0
    
public:
    /*** Internal Contructors ***/
    /*** Constructor: needs atleast 1 point to make a valid covertree ***/
    // NULL tree
    CoverTree(int truncate = -1);   
    // cover tree with one point as root
    CoverTree(const pointType& p, int truncate = -1);
    // cover tree using points in the list between begin and end
    CoverTree(const std::vector<pointType>& pList, int truncate = -1, bool use_multi_core = true);
    // cover tree using points in the list between begin and end
    CoverTree(const Eigen::MatrixXd& pMatrix, int truncate = -1, bool use_multi_core = true);
    // cover tree using points in the list between begin and end
    CoverTree(const Eigen::Map<Eigen::MatrixXd>& pMatrix, int truncate = -1, bool use_multi_core = true);
    // cover tree using points in the clusters between begin and end
    CoverTree(const std::vector<SuffStatsOne>& clusters, int truncate = -1, bool use_multi_core = true);

    /*** Destructor ***/
    /*** Destructor: deallocating all memories by a post order traversal ***/
    ~CoverTree();

/************************* Public API ***********************************************/
public:
    /*** construct cover tree using all points in the list ***/
    static CoverTree* from_points(const std::vector<pointType>& pList, int truncate = -1, bool use_multi_core = true);

    /*** construct cover tree using all points in the matrix in row-major form ***/
    static CoverTree* from_matrix(const Eigen::MatrixXd& pMatrix, int truncate = -1, bool use_multi_core = true);
    
    /*** construct cover tree using all points in the matrix in row-major form ***/
    static CoverTree* from_matrix(const Eigen::Map<Eigen::MatrixXd>& pMatrix, int truncate = -1, bool use_multi_core = true);
    
    /*** construct cover tree using all points in the list of clusters ***/
    static CoverTree* from_clusters(const std::vector<SuffStatsOne>& clusters, int truncate = -1, bool use_multi_core = true);
        
    /*** construct cover tree from data in multiple machines ***/
    static CoverTree* from_multimachine(const Eigen::Map<Eigen::MatrixXd>& pMatrix_i, int truncate = -1);

    /*** Insert point p into the cover tree ***/
    bool insert(const pointType& p, unsigned UID);

    /*** Remove point p into the cover tree ***/
    bool remove(const pointType& p) {return false;}

    /*** Nearest Neighbour search ***/
    std::pair<CoverTree::Node*, double> NearestNeighbour(const pointType &p) const;
    
    /*** k-Nearest Neighbour search ***/
    std::vector<std::pair<CoverTree::Node*, double>> kNearestNeighbours(const pointType &p, unsigned k = 10) const;
    
    /*** Range search ***/
    std::vector<std::pair<CoverTree::Node*, double>> rangeNeighbours(const pointType &queryPt, double range = 1.0) const;
    
    /*** Furthest Neighbour search ***/
    std::pair<CoverTree::Node*, double> FurthestNeighbour(const pointType &p) const;

    /*** Serialize/Desrialize: useful for MPI ***/
    char* serialize() const;                                    // Serialize to a buffer
    size_t msg_size() const;
    void deserialize(char* buff);                               // Deserialize from a buffer
    
    /*** Unit Tests ***/
    bool check_covering() const;

    /*** Return the level of root in the cover tree (== max_level) ***/
    int get_level();
    void print_levels();

    /*** Return all points in the tree ***/
    Eigen::MatrixXd get_points();

    /*** Count the points in the tree ***/
    unsigned get_count();
    unsigned count_points();
    
    /*** Some spread out points in the space ***/
    Eigen::Map<Eigen::MatrixXd> getBestInitialPoints(unsigned numBest, double* data = NULL) const;

    /*** Pretty print ***/
    friend std::ostream& operator<<(std::ostream& os, const CoverTree& ct);
};

#endif //_COVER_TREE_H
