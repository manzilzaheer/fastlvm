#include "cover_tree.h"

double* CoverTree::compute_pow_table()
{
    double* powdict = new double[2048];
    for (int i = 0; i<2048; ++i)
        powdict[i] = pow(CoverTree::base, i - 1024);
    return powdict;
}

double* CoverTree::powdict = compute_pow_table();

/******************************* Insert ***********************************************/
bool CoverTree::insert(CoverTree::Node* current, const pointType& p)
{
    bool result = false;
#ifdef DEBUG
    if (current->dist(p) > current->covdist())
        throw std::runtime_error("Internal insert got wrong input!");
    if (truncateLevel > 0 && current->level < maxScale - truncateLevel)
    {
        std::cout << maxScale;
        std::cout << " skipped" << std::endl;
        return false;
    }
#endif
    if (truncate_level > 0 && current->level < max_scale-truncate_level)
        return true;
    
    //acquire read lock
    current->mut.lock_shared();

    // Sort the children
    unsigned num_children = unsigned(current->children.size());
    std::vector<int> idx(num_children);
    std::iota(std::begin(idx), std::end(idx), 0);
    std::vector<double> dists(num_children);
    for (unsigned i = 0; i < num_children; ++i)
        dists[i] = current->children[i]->dist(p);
    auto comp_x = [&dists](int a, int b) { return dists[a] < dists[b]; };
    std::sort(std::begin(idx), std::end(idx), comp_x);

    bool flag = true;
    for (const auto& child_idx : idx)
    {
        Node* child = current->children[child_idx];
        double dist_child = dists[child_idx];
        if (dist_child <= 0.0)
        {
            //release read lock then enter child
            current->mut.unlock_shared();
            flag = false;
            std::cout << "Duplicate entry!!!" << std::endl;
            break;
        }
        else if (dist_child <= child->covdist())
        {
            //release read lock then enter child
            if (child->maxdistUB < dist_child)
                child->maxdistUB = dist_child;
            current->mut.unlock_shared();
            result = insert(child, p);
            flag = false;
            break;
        }
    }

    if (flag)
    {
        //release read lock then acquire write lock
        current->mut.unlock_shared();
        current->mut.lock();
        // check if insert is still valid, i.e. no other point was inserted else restart
        if (num_children==current->children.size())
        {
            int new_id = N++;
            current->setChild(p, new_id);
            result = true;
            current->mut.unlock();
            
            int local_min = min_scale.load();
            while( local_min > current->level - 1){
                min_scale.compare_exchange_weak(local_min, current->level - 1, std::memory_order_relaxed, std::memory_order_relaxed);
                local_min = min_scale.load();
            }
        }
        else
        {
            current->mut.unlock();
            result = insert(current, p);
        }
        // if (min_scale > current->level - 1)
        // {
            // min_scale = current->level - 1;
            // //std::cout << minScale << " " << maxScale << std::endl;
        // }
    }
    return result;
}

// bool CoverTree::insert(CoverTree::Node* current, CoverTree::Node* p)
// {
    // bool result = false;
    // std::cout << "Node insert called!";
// #ifdef DEBUG
    // if (current->dist(p) > current->covdist())
        // throw std::runtime_error("Internal insert got wrong input!");
    // if (truncateLevel > 0 && current->level < maxScale - truncateLevel)
    // {
        // std::cout << maxScale;
        // std::cout << " skipped" << std::endl;
        // return false;
    // }
// #endif
    // if (truncate_level > 0 && current->level < max_scale-truncate_level)
        // return false;
    
    // //acquire read lock
    // current->mut.lock_shared();

    // // Sort the children
    // unsigned num_children = unsigned(current->children.size());
    // std::vector<int> idx(num_children);
    // std::iota(std::begin(idx), std::end(idx), 0);
    // std::vector<double> dists(num_children);
    // for (unsigned i = 0; i < num_children; ++i)
        // dists[i] = current->children[i]->dist(p);
    // auto comp_x = [&dists](int a, int b) { return dists[a] < dists[b]; };
    // std::sort(std::begin(idx), std::end(idx), comp_x);

    // bool flag = true;
    // for (const auto& child_idx : idx)
    // {
        // Node* child = current->children[child_idx];
        // double dist_child = dists[child_idx];
        // if (dist_child <= 0.0)
        // {
            // //release read lock then enter child
            // current->mut.unlock_shared();
            // flag = false;
            // break;
        // }
        // else if (dist_child <= child->covdist())
        // {
            // //release read lock then enter child
            // current->mut.unlock_shared();
            // result = insert(child, p);
            // flag = false;
            // break;
        // }
    // }

    // if (flag)
    // {
        // //release read lock then acquire write lock
        // current->mut.unlock_shared();
        // current->mut.lock();
        // // check if insert is still valid, i.e. no other point was inserted else restart
        // if (num_children==current->children.size())
        // {
            // ++N;
            // current->setChild(p);
            // result = true;
            // current->mut.unlock();
            
            // int local_min = min_scale.load();
            // while( local_min > current->level - 1){
                // min_scale.compare_exchange_weak(local_min, current->level - 1, std::memory_order_relaxed, std::memory_order_relaxed);
                // local_min = min_scale.load();
            // }
        // }
        // else
        // {
            // current->mut.unlock();
            // result = insert(current, p);
        // }
        // // if (min_scale > current->level - 1)
        // // {
            // // min_scale = current->level - 1;
            // // //std::cout << minScale << " " << maxScale << std::endl;
        // // }
    // }
    // return result;
// }

bool CoverTree::insert(const pointType& p)
{
    bool result = false;
    id_valid = false;
    global_mut.lock_shared();
    double curr_root_dist = root->dist(p);
    if (curr_root_dist <= 0.0)
    {
        std::cout << "Duplicate entry!!!" << std::endl;
    }
    else if (curr_root_dist > root->covdist())
    {
        std::pair<CoverTree::Node*, double> fn = FurthestNeighbour(p);
        global_mut.unlock_shared();
        std::cout<<"Entered case 1: " << root->dist(p) << " " << root->covdist() << " " << root->level <<std::endl;
        std::cout<<"Requesting global lock!" <<std::endl;
        global_mut.lock();
        while (root->dist(p) > base * root->covdist()/(base-1))
        {
            CoverTree::Node* current = root;
            CoverTree::Node* parent = NULL;
            while (current->children.size()>0)
            {
                parent = current;
                current = current->children.back();
            }
            if (parent != NULL)
            {
                parent->children.pop_back();
                std::pair<CoverTree::Node*, double> fni = FurthestNeighbour(current->_p);
                current->level = root->level + 1;
                //current->parent = NULL;
                current->maxdistUB = fni.second; // powdict[current->level + 1025];
                current->children.push_back(root);
                root = current;
            }
            else
            {
                root->level += 1;
                //root->maxdistUB = powdict[root->level + 1025];
            }
        }
        CoverTree::Node* temp = new CoverTree::Node;
        temp->_p = p;
        temp->level = root->level + 1;
        temp->ID = N++;
        temp->maxdistUB = fn.second; //powdict[temp->level+1025];
        //temp->parent = NULL;
        temp->children.push_back(root);
        //root->parent = temp;
        root = temp;
        max_scale = root->level;
        result = true;
        //std::cout << "Upward: " << minScale << " " << maxScale << std::endl;
        global_mut.unlock();
        global_mut.lock_shared();
    }
    else
    {
        //root->tempDist = root->dist(p);
        result = insert(root, p);
    }
    global_mut.unlock_shared();
    return result;
}

/******************************* Remove ***********************************************/
// void CoverTree::remove(CoverTree::Node* current, double dist_current, const pointType &p, std::pair<CoverTree::Node*, double>& nn) const
// {
    // // If the current node is the nearest neighbour
    // if (dist_current < nn.second)
    // {
        // nn.first = current;
        // nn.second = dist_current;
    // }
    
    // // Sort the children
    // unsigned num_children = current->children.size();
    // std::vector<int> idx(num_children);
    // std::iota(std::begin(idx), std::end(idx), 0);
    // std::vector<double> dists(num_children);
    // //dist_count[current->level].fetch_add(num_children, std::memory_order_relaxed);
    // for (unsigned i = 0; i < num_children; ++i)
        // dists[i] = current->children[i]->dist(p);
    // auto comp_x = [&dists](int a, int b) { return dists[a] < dists[b]; };
    // std::sort(std::begin(idx), std::end(idx), comp_x);

    // for (const auto& child_idx : idx)
    // {
        // Node* child = current->children[child_idx];
        // double dist_child = dists[child_idx];
        // if (nn.second > dist_child - child->maxdistUB)
            // remove(child, dist_child, p, nn);
    // }
    
    // if (nn.second <= 0.0)
    // {
        // for(const auto& orphan : nn.first)
        // {
            // //check if can be inserted as a child of sibling
            // bool inserted = false;
            // for (const auto& child : current)
            // {
                // double dist_child = child->dist(orphan);
                // if (dist_child <= child->covdist())
                // {
                    // child->setChild(orphan);
                    // //remove from nn.first
                    // inserted = true;
                    // break;
                // }
            // }
        
            // //else else promote the subtree
            // if(!inserted)
            // {
                // CoverTree::Node* current = orphan;
                // CoverTree::Node* parent = NULL;
                // while (current->children.size()>0)
                // {
                    // parent = current;
                    // current = current->children.back();
                // }
                // if (parent != NULL)
                // {
                    // parent->children.pop_back();
                    // current->level = orphan->level + 1;
                    // //current->maxdistUB = powdict[current->level + 1025];
                    // current->children.push_back(orphan);
                    // orphan = current;
                // }
                // else
                // {
                    // orphan->level += 1;
                // }
            // }
        // }
    // }
// }

// bool CoverTree::remove(const pointType &p)
// {
    // bool ret_val = false;
    // //First find the point
    // std::pair<CoverTree::Node*, double> result(root, root->dist(p));
    // NearestNeighbour(root, result.second, p, result);

    // if (result.second<=0.0)
    // {   // point found
        // CoverTree::Node* node_p = result.first;
        // CoverTree::Node* parent_p = node_p->parent;
        // if (node_p == root)
        // {
            // std::cout << "Sorry can not delete root efficiently!" << std::endl;
        // }
        // else
        // {
            // //1. Remove p from parent's list of child
            // unsigned num_children = parent_p->children.size();
            // for (unsigned i = 0; i < num_children; ++i)
            // {
                // if (parent_p->children[i]==node_p)
                // {
                    // parent_p->children[i] =  parent_p->children.back();
                    // parent_p->children.pop_back();
                    // break;
                // }
            // }

            // //2. For each child q of p:
            // for(CoverTree::Node* q : *node_p)
            // {
                // //1. check if can be inserted as a child of sibling
                // //2. else promote the subtree height
            // }
            
            // //3. delete
            // delete node_p;

            // ret_val = true;
        // }
    // }
    // //calc_maxdist();
    // return ret_val;
// }


/****************************** Nearest Neighbour *************************************/
void CoverTree::NearestNeighbour(CoverTree::Node* current, double dist_current, const pointType &p, std::pair<CoverTree::Node*, double>& nn) const
{
    // If the current node is the nearest neighbour
    if (dist_current < nn.second)
    {
        nn.first = current;
        nn.second = dist_current;
    }
    
    // Sort the children
    unsigned num_children = unsigned(current->children.size());
    std::vector<int> idx(num_children);
    std::iota(std::begin(idx), std::end(idx), 0);
    std::vector<double> dists(num_children);
    //dist_count[current->level].fetch_add(num_children, std::memory_order_relaxed);
    for (unsigned i = 0; i < num_children; ++i)
        dists[i] = current->children[i]->dist(p);
    auto comp_x = [&dists](int a, int b) { return dists[a] < dists[b]; };
    std::sort(std::begin(idx), std::end(idx), comp_x);

    for (const auto& child_idx : idx)
    {
        Node* child = current->children[child_idx];
        double dist_child = dists[child_idx];
        if (child->maxdistUB > current->covdist()/(base-1))
            std::cout << "I am crazy because max upper bound is bigger than 2**i " << child->maxdistUB << " " << current->covdist()/(base-1) << std::endl;
        if (nn.second > dist_child - child->maxdistUB)
            NearestNeighbour(child, dist_child, p, nn);
    }
}

// First the number of nearest neighbor
std::pair<CoverTree::Node*, double> CoverTree::NearestNeighbour(const pointType &p) const
{
    std::pair<CoverTree::Node*, double> result(root, root->dist(p));
    NearestNeighbour(root, result.second, p, result);
    return result;
}

/****************************** k-Nearest Neighbours *************************************/

void CoverTree::kNearestNeighbours(CoverTree::Node* current, double dist_current, const pointType& p, std::vector<std::pair<CoverTree::Node*, double>>& nnList) const
{   
    // TODO(manzilz): An efficient implementation ?
    
    // If the current node is eligible to get into the list
    if(dist_current < nnList.back().second)
    {
        auto comp_x = [](std::pair<CoverTree::Node*, double> a, std::pair<CoverTree::Node*, double> b) { return a.second < b.second; };
        std::pair<CoverTree::Node*, double> temp(current, dist_current);
        nnList.insert( 
            std::upper_bound( nnList.begin(), nnList.end(), temp, comp_x ),
            temp 
        );
        nnList.pop_back();
    }

    // Sort the children
    unsigned num_children = unsigned(current->children.size());
    std::vector<int> idx(num_children);
    std::iota(std::begin(idx), std::end(idx), 0);
    std::vector<double> dists(num_children);
    //dist_count[current->level].fetch_add(num_children, std::memory_order_relaxed);
    for (unsigned i = 0; i < num_children; ++i)
        dists[i] = current->children[i]->dist(p);
    auto comp_x = [&dists](int a, int b) { return dists[a] < dists[b]; };
    std::sort(std::begin(idx), std::end(idx), comp_x);

    for (const auto& child_idx : idx)
    {
        Node* child = current->children[child_idx];
        double dist_child = dists[child_idx];
        if ( nnList.back().second > dist_child - child->maxdistUB)
            kNearestNeighbours(child, dist_child, p, nnList);
    }
}
    
std::vector<std::pair<CoverTree::Node*, double>> CoverTree::kNearestNeighbours(const pointType &queryPt, unsigned numNbrs) const
{
    // Do the worst initialization
    std::pair<CoverTree::Node*, double> dummy(NULL, std::numeric_limits<double>::max());
    // List of k-nearest points till now
    std::vector<std::pair<CoverTree::Node*, double>> nnList(numNbrs, dummy);

    // Call with root
    double dist_root = root->dist(queryPt);
    kNearestNeighbours(root, dist_root, queryPt, nnList);
    
    return nnList;
}
    
/****************************** Range Neighbours Search *************************************/

void CoverTree::rangeNeighbours(CoverTree::Node* current, double dist_current, const pointType &p, double range, std::vector<std::pair<CoverTree::Node*, double>>& nnList) const
{   
    // If the current node is eligible to get into the list
    if (dist_current < range)
    {
        std::pair<CoverTree::Node*, double> temp(current, dist_current);
        nnList.push_back(temp);
    }

    // Sort the children
    unsigned num_children = unsigned(current->children.size());
    std::vector<int> idx(num_children);
    std::iota(std::begin(idx), std::end(idx), 0);
    std::vector<double> dists(num_children);
    for (unsigned i = 0; i < num_children; ++i)
        dists[i] = current->children[i]->dist(p);
    auto comp_x = [&dists](int a, int b) { return dists[a] < dists[b]; };
    std::sort(std::begin(idx), std::end(idx), comp_x);

    for (const auto& child_idx : idx)
    {
        Node* child = current->children[child_idx];
        double dist_child = dists[child_idx];
        if (range > dist_child - child->maxdistUB)
            rangeNeighbours(child, dist_child, p, range, nnList);
    }
}

std::vector<std::pair<CoverTree::Node*, double>> CoverTree::rangeNeighbours(const pointType &queryPt, double range) const
{
    // List of nearest neighbors in the range
    std::vector<std::pair<CoverTree::Node*, double>> nnList;

    // Call with root
    double dist_root = root->dist(queryPt);
    rangeNeighbours(root, dist_root, queryPt, range, nnList);

    return nnList;
}

/****************************** Furthest Neighbour *************************************/
void CoverTree::FurthestNeighbour(CoverTree::Node* current, double dist_current, const pointType &p, std::pair<CoverTree::Node*, double>& fn) const
{
    // If the current node is the furthest neighbour
    if (dist_current > fn.second)
    {
        fn.first = current;
        fn.second = dist_current;
    }
    
    // Sort the children
    unsigned num_children = unsigned(current->children.size());
    std::vector<int> idx(num_children);
    std::iota(std::begin(idx), std::end(idx), 0);
    std::vector<double> dists(num_children);
    //dist_count[current->level].fetch_add(num_children, std::memory_order_relaxed);
    for (unsigned i = 0; i < num_children; ++i)
        dists[i] = current->children[i]->dist(p);
    auto comp_x = [&dists](int a, int b) { return dists[a] > dists[b]; };
    std::sort(std::begin(idx), std::end(idx), comp_x);

    for (const auto& child_idx : idx)
    {
        Node* child = current->children[child_idx];
        double dist_child = dists[child_idx];
        if (child->maxdistUB > current->covdist()/(base-1))
            std::cout << "I am crazy because max upper bound is bigger than 2**i " << child->maxdistUB << " " << current->covdist()/(base-1) << std::endl;
        if (fn.second < dist_child + child->maxdistUB)
            FurthestNeighbour(child, dist_child, p, fn);
    }
}

// First the number of nearest neighbor
std::pair<CoverTree::Node*, double> CoverTree::FurthestNeighbour(const pointType &p) const
{
    std::pair<CoverTree::Node*, double> result(root, root->dist(p));
    FurthestNeighbour(root, result.second, p, result);
    return result;
}

/****************************** Cover Trees Properties *************************************/

void CoverTree::generate_id(CoverTree::Node* current)
{
    // assign current node
    current->ID = N++;
#ifdef DEBUG
    std::cout << "Pre: " << current->ID << std::endl;
#endif

    // travrse children
    for (const auto& child : *current)
        generate_id(child);
}

//find true maxdist
void CoverTree::calc_maxdist()
{
    std::vector<CoverTree::Node*> travel;
    std::vector<CoverTree::Node*> active;

    CoverTree::Node* current = root;

    root->maxdistUB = 0.0;
    travel.push_back(root);
    while (travel.size() > 0)
    {
        current = travel.back();
        if (current->maxdistUB <= 0) {
            while (current->children.size()>0)
            {
                active.push_back(current);
                // push the children
                for (int i = int(current->children.size()) - 1; i >= 0; --i)
                {
                    current->children[i]->maxdistUB = 0.0;
                    travel.push_back(current->children[i]);
                }
                current = current->children[0];
            }
        }
        else
            active.pop_back();

        // find distance with current node
        for (const auto& n : active)
            n->maxdistUB = std::max(n->maxdistUB, n->dist(current));

        // Pop
        travel.pop_back();
    }
}

/****************************** Serialization of Cover Trees *************************************/

// Pre-order traversal
char* CoverTree::preorder_pack(char* buff, CoverTree::Node* current) const
{
    // copy current node
    size_t shift = current->_p.rows() * sizeof(pointType::Scalar);
    char* start = (char*)current->_p.data();
    char* end = start + shift;
    std::copy(start, end, buff);
    buff += shift;

    shift = sizeof(int);
    start = (char*)&(current->level);
    end = start + shift;
    std::copy(start, end, buff);
    buff += shift;
    
    shift = sizeof(unsigned);
    start = (char*)&(current->ID);
    end = start + shift;
    std::copy(start, end, buff);
    buff += shift;
    
    shift = sizeof(double);
    start = (char*)&(current->maxdistUB);
    end = start + shift;
    std::copy(start, end, buff);
    buff += shift;

#ifdef DEBUG
    std::cout << "Pre: " << current->ID << std::endl;
#endif

    // travrse children
    for (const auto& child : *current)
        buff = preorder_pack(buff, child);

    return buff;
}

// Post-order traversal
char* CoverTree::postorder_pack(char* buff, CoverTree::Node* current) const
{
    // travrse children
    for (const auto& child : *current)
        buff = postorder_pack(buff, child);

    // save current node ID
#ifdef DEBUG
    std::cout << "Post: " << current->ID << std::endl;
#endif
    size_t shift = sizeof(unsigned);
    char* start = (char*)&(current->ID);
    char* end = start + shift;
    std::copy(start, end, buff);
    buff += shift;

    return buff;
}

// reconstruct tree from Pre&Post traversals
void CoverTree::PrePost(CoverTree::Node*& current, char*& pre, char*& post)
{   
    // The top element in preorder list PRE is the root of T
    current = new CoverTree::Node();
    current->_p = pointType(D);
    for (unsigned i = 0; i < D; ++i)
    {
        current->_p[i] = *((pointType::Scalar *)pre);
        pre += sizeof(pointType::Scalar);
    }
    current->level = *((int *)pre);
    pre += sizeof(int);
    current->ID = *((unsigned *)pre);
    pre += sizeof(unsigned);
    current->maxdistUB = *((double *)pre);
    pre += sizeof(double);
    
    // std::cout << current->_p << std::endl;
    // std::cout << current->ID << std::endl;
    // std::cout << *((unsigned*)post) << std::endl;

    // Construct subtrees until the root is found in the postorder list
    while (*((unsigned*)post) != current->ID)
    {
        // std::cout << "I am in loop for " << current->ID << std::endl;
        CoverTree::Node* temp = NULL;
        PrePost(temp, pre, post);
        current->children.push_back(temp);
    }

    //All subtrees of T are constructed
    post += sizeof(unsigned);       //Delete top element of POST
}

size_t CoverTree::msg_size() const
{
    return 2 * sizeof(unsigned)
        + sizeof(pointType::Scalar)*D*N
        + sizeof(int)*N
        + sizeof(unsigned)*N
        + sizeof(double)*N
        + sizeof(unsigned)*N;
}

// Serialize to a buffer
char* CoverTree::serialize() const
{
    //// check if valid id present
    //if (!id_valid)
    //{
    // N = 0;
    // generate_id(root);
    // id_valid = true;
    //}
    //count_points();

    //Covert following to char* buff with following order
    // N | D | (points, levels) | List
    char* buff = new char[msg_size()];

    char* pos = buff;

    // insert N
    unsigned shift = sizeof(unsigned);
    char* start = (char*)&(N);
    char* end = start + shift;
    std::copy(start, end, pos);
    pos += shift;

    // insert D
    shift = sizeof(unsigned);
    start = (char*)&(D);
    end = start + shift;
    std::copy(start, end, pos);
    pos += shift;

    // insert points and level
    pos = preorder_pack(pos, root);
    pos = postorder_pack(pos, root);

    //std::cout<<"Message size: " << msg_size() << ", " << pos - buff << std::endl;

    return buff;
}

// Deserialize from a buffer
void CoverTree::deserialize(char* buff)
{
    /** Convert char* buff into following buff = N | D | (points, levels) | List **/
    //char* save = buff;

    // extract N and D
    N = *((unsigned *)buff);
    buff += sizeof(unsigned);
    D = *((unsigned *)buff);
    buff += sizeof(unsigned);
    
    //std::cout << "N: " << N << ", D: " << D << std::endl;

    // pointer to postorder list
    char* post = buff + sizeof(pointType::Scalar)*D*N
        + sizeof(int)*N + sizeof(unsigned)*N + sizeof(double)*N;
        
    // for(char *tmp = post; tmp < post + 4*N; tmp+=4)
        // std::cout << *((unsigned*)tmp) << ", ";
    // std::cout << std::endl;

    //reconstruction
    PrePost(root, buff, post);

    //delete[] save;
}

/****************************** Unit Tests for Cover Trees *************************************/
bool CoverTree::check_covering() const
{
    bool result = true;
    std::stack<CoverTree::Node*> travel;
    CoverTree::Node* curNode;

    // Initialize with root
    travel.push(root);

    // Pop, check and then push the children
    while (travel.size() > 0)
    {
        // Pop
        curNode = travel.top();
        travel.pop();

        // Check covering for the current -> children pair
        for (const auto& child : *curNode)
        {
            travel.push(child);
            if( curNode->dist(child) > curNode->covdist() )
                result = false;
            //std::cout << *curNode << " -> " << *child << " @ " << curNode->dist(child) << " | " << curNode->covdist() << std::endl;
        }
    }

    return result;
}

/****************************** Internal Constructors of Cover Trees *************************************/

//constructor: NULL tree
CoverTree::CoverTree(int truncate /*=-1*/ )
{
    root = NULL;
    min_scale = 1000;
    max_scale = 0;
    truncate_level = truncate;
    N = 0;
    D = 0;
}

//constructor: needs atleast 1 point to make a valid covertree
CoverTree::CoverTree(const pointType& p, int truncateArg /*=-1*/)
{
    min_scale = 1000;
    max_scale = 0;
    truncate_level = truncateArg;
    N = 1;
    D = unsigned(p.rows());

    root = new CoverTree::Node;
    root->_p = p;
    root->ID = 0;
    root->level = 0;
    root->maxdistUB = 0;
}

//constructor: cover tree using points in the list between begin and end
CoverTree::CoverTree(const std::vector<pointType>& pList, int truncateArg /*= 0*/)
{
    size_t numPoints = pList.size();

    //1. Compute the mean of entire data
    pointType mx = utils::ParallelAddList(pList).get_result()/(1.0*numPoints);
    
    //2. Compute distance of every point from the mean || Variance
    pointType dists = utils::ParallelDistanceComputeList(pList, mx).get_result();
    
    //3. argort the distance to find approximate mediod
    std::vector<size_t> idx(numPoints);
    std::iota(std::begin(idx), std::end(idx), 0);
    auto comp_x = [&dists](size_t a, size_t b) { return dists[a] > dists[b]; };
    std::sort(std::begin(idx), std::end(idx), comp_x);
    //std::cout<<"Max distance: " << dists[idx[0]] << std::endl;
    
    //4. Compute distance of every point from the mediod
    mx = pList[idx[numPoints-1]];
    dists = utils::ParallelDistanceComputeList(pList, mx).get_result();
    double max_dist = dists.maxCoeff();
    
    int scale_val = int(std::ceil(std::log(max_dist)/std::log(base)));
    //std::cout<<"Scale chosen: " << scale_val << std::endl;
    min_scale = scale_val; //-1000;
    max_scale = scale_val; //-1000;
    truncate_level = truncateArg;
    N = 1;
    D = unsigned(mx.rows());
    
    root = new CoverTree::Node;
    root->_p = mx;
    root->level = scale_val; //-1000;
    root->maxdistUB = max_dist; // powdict[scale_val+1024];
    root->ID = 0;

    //std::cout << "(" << pList[0].rows() << ", " << pList.size() << ")" << std::endl;
    if (50000 >= numPoints)
    {
        for (size_t i = 0; i < numPoints-1; ++i){
            if(!insert(pList[idx[i]]))
                std::cout << "Insert failed!!!" << std::endl;
        }
    }
    else
    {
        for (size_t i = 0; i < 50000; ++i){
            utils::progressbar(i, 50000);
            if(!insert(pList[idx[i]]))
                std::cout << "Insert failed!!!" << std::endl;
        }
        utils::progressbar(50000, 50000);
        std::cerr<<std::endl;
        utils::parallel_for_progressbar(50000, numPoints-1, [&](size_t i)->void{
            if(!insert(pList[idx[i]]))
                std::cout << "Insert failed!!!" << std::endl;
        });
    }
}

//constructor: cover tree using points in the list between begin and end
CoverTree::CoverTree(const Eigen::MatrixXd& pMatrix, int truncateArg /*= 0*/)
{
    size_t numPoints = pMatrix.cols();

    //1. Compute the mean of entire data
    pointType mx = utils::ParallelAddMatrix(pMatrix).get_result()/(1.0*numPoints);
    
    //2. Compute distance of every point from the mean || Variance
    pointType dists = utils::ParallelDistanceCompute(pMatrix, mx).get_result();
    
    //3. argort the distance to find approximate mediod
    std::vector<size_t> idx(numPoints);
    std::iota(std::begin(idx), std::end(idx), 0);
    auto comp_x = [&dists](size_t a, size_t b) { return dists[a] > dists[b]; };
    std::sort(std::begin(idx), std::end(idx), comp_x);
    //std::cout<<"Max distance: " << dists[idx[0]] << std::endl;
    
    //4. Compute distance of every point from the mediod
    mx = pMatrix.col(idx[numPoints-1]);
    dists = utils::ParallelDistanceCompute(pMatrix, mx).get_result();
    double max_dist = dists.maxCoeff();
    
    int scale_val = int(std::ceil(std::log(max_dist)/std::log(base)));
    //std::cout<<"Scale chosen: " << scale_val << std::endl;
    min_scale = scale_val; //-1000;
    max_scale = scale_val; //-1000;
    truncate_level = truncateArg;
    N = 1;
    D = unsigned(mx.rows());

    root = new CoverTree::Node;
    root->_p = mx;
    root->level = scale_val; //-1000;
    root->maxdistUB = max_dist; // powdict[scale_val+1024];
    root->ID = 0;
    
    //std::cout << "(" << pMatrix.rows() << ", " << pMatrix.cols() << ")" << std::endl;
    if (50000 >= numPoints)
    {
        for (size_t i = 0; i < numPoints-1; ++i){
            if(!insert(pMatrix.col(idx[i])))
                std::cout << "Insert failed!!!" << std::endl;
        }
    }
    else
    {
        for (size_t i = 0; i < 50000; ++i){
            utils::progressbar(i, 50000);
            if(!insert(pMatrix.col(idx[i])))
                std::cout << "Insert failed!!!" << std::endl;
        }
        utils::progressbar(50000, 50000);
        std::cerr<<std::endl;
        utils::parallel_for_progressbar(50000, numPoints-1, [&](size_t i)->void{
            if(!insert(pMatrix.col(idx[i])))
                std::cout << "Insert failed!!!" << std::endl;
        });
    }
}

//constructor: cover tree using points in the list between begin and end
CoverTree::CoverTree(const Eigen::Map<Eigen::MatrixXd>& pMatrix, int truncateArg /*= 0*/)
{
    size_t numPoints = pMatrix.cols();
    
    //1. Compute the mean of entire data
    pointType mx = utils::ParallelAddMatrixNP(pMatrix).get_result()/(1.0*numPoints);
    
    //2. Compute distance of every point from the mean || Variance
    pointType dists = utils::ParallelDistanceComputeNP(pMatrix, mx).get_result();
    
    //3. argort the distance to find approximate mediod
    std::vector<size_t> idx(numPoints);
    std::iota(std::begin(idx), std::end(idx), 0);
    auto comp_x = [&dists](size_t a, size_t b) { return dists[a] > dists[b]; };
    std::sort(std::begin(idx), std::end(idx), comp_x);
    ///std::cout<<"Max distance: " << dists[idx[0]] << std::endl;
    
    //4. Compute distance of every point from the mediod
    mx = pMatrix.col(idx[numPoints-1]);
    dists = utils::ParallelDistanceComputeNP(pMatrix, mx).get_result();
    double max_dist = dists.maxCoeff();
    
    int scale_val = int(std::ceil(std::log(max_dist)/std::log(base)));
    //std::cout<<"Scale chosen: " << scale_val << std::endl;
    min_scale = scale_val; //-1000;
    max_scale = scale_val; //-1000;
    truncate_level = truncateArg;
    N = 1;
    D = unsigned(mx.rows());

    root = new CoverTree::Node;
    root->_p = mx;
    root->level = scale_val; //-1000;
    root->maxdistUB = max_dist; // powdict[scale_val+1024];
    root->ID = 0;
   
    //std::cout << "(" << pMatrix.rows() << ", " << pMatrix.cols() << ")" << std::endl;
    if (50000 >= numPoints)
    {
        for (size_t i = 0; i < numPoints-1; ++i){
            if(!insert(pMatrix.col(idx[i])))
                std::cout << "Insert failed!!!" << std::endl;
        }
    }
    else
    {
        for (size_t i = 0; i < 50000; ++i){
            utils::progressbar(i, 50000);
            if(!insert(pMatrix.col(idx[i])))
                std::cout << "Insert failed!!!" << std::endl;
        }
        utils::progressbar(50000, 50000);
        std::cerr<<std::endl;
        utils::parallel_for_progressbar(50000, numPoints-1, [&](size_t i)->void{
            if(!insert(pMatrix.col(idx[i])))
                std::cout << "Insert failed!!!" << std::endl;
        });
    }
}

//constructor: cover tree using clusters in the list between begin and end
CoverTree::CoverTree(const std::vector<SuffStatsOne>& clusters, int truncateArg /*= 0*/)
{
    size_t numPoints = clusters.size();

    //1. Compute the mean of entire data
    pointType mx = utils::ParallelAddClusters(clusters).get_result()/(1.0*numPoints);
    
    //2. Compute distance of every point from the mean || Variance
    pointType dists = utils::ParallelDistanceComputeCluster(clusters, mx).get_result();
    
    //3. argort the distance to find approximate mediod
    std::vector<size_t> idx(numPoints);
    std::iota(std::begin(idx), std::end(idx), 0);
    auto comp_x = [&dists](size_t a, size_t b) { return dists[a] > dists[b]; };
    std::sort(std::begin(idx), std::end(idx), comp_x);
    //std::cout<<"Max distance: " << dists[idx[0]] << std::endl;
    
    //4. Compute distance of every point from the mediod
    mx = clusters[idx[numPoints-1]].get_mean();
    dists = utils::ParallelDistanceComputeCluster(clusters, mx).get_result();
    double max_dist = dists.maxCoeff();
    
    int scale_val = int(std::ceil(std::log(max_dist)/std::log(base)));
    //std::cout<<"Scale chosen: " << scale_val << std::endl;
    min_scale = scale_val; //-1000;
    max_scale = scale_val; //-1000;
    truncate_level = truncateArg;
    N = 1;
    D = unsigned(mx.rows());
    
    root = new CoverTree::Node;
    root->_p = mx;
    root->level = scale_val; //-1000;
    root->maxdistUB = max_dist; //powdict[scale_val+1024];
    root->ID = 0;

    //std::cout << "(" << clusters[0].get_dim() << ", " << clusters.size() << ")" << std::endl;
    if (50000 >= numPoints)
    {
        for (size_t i = 0; i < numPoints-1; ++i){
            if(!insert(clusters[idx[i]].get_mean()))
                std::cout << "Insert failed!!!" << std::endl;
        }
    }
    else
    {
        for (size_t i = 0; i < 50000; ++i){
            utils::progressbar(i, 50000);
            if(!insert(clusters[idx[i]].get_mean()))
                std::cout << "Insert failed!!!" << std::endl;
        }
        utils::progressbar(50000, 50000);
        std::cerr<<std::endl;
        utils::parallel_for_progressbar(50000, numPoints-1, [&](size_t i)->void{
            if(!insert(clusters[idx[i]].get_mean()))
                std::cout << "Insert failed!!!" << std::endl;
        });
    }
}

//destructor: deallocating all memories by a post order traversal
CoverTree::~CoverTree()
{
    std::stack<CoverTree::Node*> travel;

    if (root != NULL)
        travel.push(root);
    while (travel.size() > 0)
    {
        CoverTree::Node* current = travel.top();
        travel.pop();

        for (const auto& child : *current)
        {
            if (child != NULL)
                travel.push(child);
        }

        delete current;
    }
}


/****************************** Public API for creation of Cover Trees *************************************/

//contructor: using point list
CoverTree* CoverTree::from_points(const std::vector<pointType>& pList, int truncate /*=-1*/, bool use_multi_core /*=true*/)
{
    std::cout << "Faster Cover Tree with base " << CoverTree::base << std::endl;
    CoverTree* cTree = NULL;
    if (use_multi_core)
    {
        cTree = new CoverTree(pList, truncate);
    }
    else
    {
        cTree = new CoverTree(pList, truncate);
    }

    //cTree->calc_maxdist();
    //cTree->print_levels();

    return cTree;
}

//contructor: using matrix in row-major form!
CoverTree* CoverTree::from_matrix(const Eigen::MatrixXd& pMatrix, int truncate /*=-1*/, bool use_multi_core /*=true*/)
{
    std::cout << "Faster Cover Tree with base " << CoverTree::base << std::endl;
    CoverTree* cTree = NULL;
    if (use_multi_core)
    {
        cTree = new CoverTree(pMatrix, truncate);
    }
    else
    {
        cTree = new CoverTree(pMatrix, truncate);
    }

    //cTree->calc_maxdist();
    //cTree->print_levels();

    return cTree;
}

//contructor: using matrix in col-major form!
CoverTree* CoverTree::from_matrix(const Eigen::Map<Eigen::MatrixXd>& pMatrix, int truncate /*=-1*/, bool use_multi_core /*=true*/)
{
    std::cout << "Faster Cover Tree with base " << CoverTree::base << std::endl;
    CoverTree* cTree = NULL;
    if (use_multi_core)
    {
        cTree = new CoverTree(pMatrix, truncate);
    }
    else
    {
        cTree = new CoverTree(pMatrix, truncate);
    }

    //cTree->calc_maxdist();
    //cTree->print_levels();

    return cTree;
}

//contructor: using cluster list
CoverTree* CoverTree::from_clusters(const std::vector<SuffStatsOne>& clusters, int truncate /*=-1*/, bool use_multi_core /*=true*/)
{
    CoverTree* cTree = NULL;
    if (use_multi_core)
    {
        cTree = new CoverTree(clusters, truncate);
    }
    else
    {
        cTree = new CoverTree(clusters, truncate);
    }

    //cTree->calc_maxdist();
    //cTree->print_levels();

    return cTree;
}

CoverTree* CoverTree::from_multimachine(const Eigen::Map<Eigen::MatrixXd>& pMatrix_i, int truncate)
{
    CoverTree* ct = CoverTree::from_matrix(pMatrix_i, truncate);

    #ifdef MULTIMACHINE
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    // Get the rank of the process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    #ifdef DEBUG
    std::ofstream fout("log_"+std::to_string(rank)+".txt");
    fout << "Cover tree before transmission: \n"
         << *ct << std::endl;

    std::chrono::seconds dura(5);
    MPI_Barrier(MPI_COMM_WORLD);
    std::this_thread::sleep_for( dura );
    if(rank==0)
      std::cout << "Cover tree before transmission at rank 0: \n"
                << *ct << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    std::this_thread::sleep_for( dura );
    if(rank==1)
      std::cout << "Cover tree before transmission at rank 1: \n"
                << *ct << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    std::this_thread::sleep_for( dura );
    if(rank==2)
      std::cout << "Cover tree before transmission at rank 2: \n"
                << *ct << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    std::this_thread::sleep_for( dura );
    if(rank==3)
      std::cout << "Cover tree before transmission at rank 3: \n"
                << *ct << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    std::this_thread::sleep_for( dura );
    #endif

    int level = 1;
    while (level < world_size)
    {
        int divisor = 2 * level;
        if(rank % divisor == 0 && rank + level < world_size)
        {
            #ifdef DEBUG
            std::cout << "Probing at rank " << rank <<std::endl;
            fout << "Probing at rank " << rank <<std::endl;
            #endif
            int length;
            MPI_Status status;
            // Probe for an incoming message from process zero
            MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            #ifdef DEBUG
            std::cout << "Get count at rank " << rank <<std::endl;
            fout << "Get count at rank " << rank <<std::endl;
            #endif
            // Get the message size
            MPI_Get_count(&status, MPI_BYTE, &length);
            
            // Allocate a buffer to hold the incoming numbers
            char* buff = new char[length];

            #ifdef DEBUG
            std::cout << "Receiving at rank " << rank <<std::endl;
            fout << "Receiving at rank " << rank <<std::endl;
            #endif
            // Now receive the message with the allocated buffer
            MPI_Recv(buff, length, MPI_BYTE, MPI_ANY_SOURCE, MPI_ANY_TAG,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            #ifdef DEBUG
            std::cout << "Unpacking at rank " << rank <<std::endl;
            fout << "Unpacking at rank " << rank <<std::endl;
            #endif
            // Process received message as cover tree
            size_t numPoints = length/(ct->D*sizeof(double));
            double * fnp = reinterpret_cast< double * >( buff );
            Eigen::Map<Eigen::MatrixXd> pMatrix_r(fnp, ct->D, numPoints);

            #ifdef DEBUG
            std::cout << "Remote points received: " << std::endl;
            std::cout << numPoints << "\n" << pMatrix_r << std::endl;
            fout << "Remote points received: " << std::endl;
            fout << numPoints << "\n" << pMatrix_r << std::endl;
            #endif

            #ifdef DEBUG
            std::cout << "Merging at rank " << rank <<std::endl;
            fout << "Merging at rank " << rank <<std::endl;
            #endif
            // Merge received tree with local tree
            utils::parallel_for_progressbar(0, numPoints, [&](size_t i)->void{
                //for (int i = begin + 1; i < end; ++i){
                //utils::progressbar(i, end-50000);
                if(!ct->insert(pMatrix_r.col(i))){
                    std::cout << "Insert failed!!!" << std::endl;
                    #ifdef DEBUG
                    fout << "Insert failed!!!" << std::endl;
                    #endif
                }
            });

            #ifdef DEBUG
            std::cout << "Cover tree after merging with remote: " << std::endl;
            std::cout << *ct << std::endl;
            fout << "Cover tree after merging with remote: " << std::endl;
            fout << *ct << std::endl;
            #endif
            
        }
        else if (rank % divisor == level) 
        {
            #ifdef DEBUG
            std::cout << "Transmitting at rank " << rank <<std::endl;
            fout << "Transmitting at rank " << rank <<std::endl;
            #endif
            // Pack local tree as byte stream
            //char* buff = ct->serialize();
            //unsigned msg_size = ct->msg_size();
            
            Eigen::MatrixXd pMatrix_ir = ct->get_points();
            char* buff = reinterpret_cast< char * >(pMatrix_ir.data());
            int msg_size = pMatrix_ir.rows()*pMatrix_ir.cols()*sizeof(double);

            // Transmit to parent
            MPI_Send(buff, msg_size, MPI_BYTE, rank - level, 0, MPI_COMM_WORLD);

            // delete packing buffer
            //delete buff;
        }

        level = divisor;
    }

    // Broadcast to all other
    if (rank == 0)
    {
        #ifdef DEBUG
        std::cout << "Broadcasting size at rank 0" <<std::endl;
        fout << "Broadcasting size at rank 0" <<std::endl;
        #endif

        // Pack merged tree as byte stream
        char* buff = ct->serialize();

        // Broadcast size of message
        unsigned msg_size = ct->msg_size();
        MPI_Bcast(&msg_size, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

        #ifdef DEBUG
        std::cout << "Message size at rank 0: " << msg_size << std::endl;
        fout << "Message size at rank 0: " << msg_size << std::endl;
        #endif

        #ifdef DEBUG
        std::cout << "Broadcasting tree at rank 0" <<std::endl;
        fout << "Broadcasting tree at rank 0" <<std::endl;
        #endif
        // Broadcast the tree
        MPI_Bcast(buff, msg_size, MPI_BYTE, 0, MPI_COMM_WORLD);
        #ifdef DEBUG
        std::cout << "Broadcasting done at rank 0" <<std::endl;
        fout << "Broadcasting done at rank 0" <<std::endl;
        #endif

        // delete packing buffer
        delete buff;
    }
    else
    {
        #ifdef DEBUG
        std::cout << "Receiving size at rank " << rank <<std::endl;
        fout << "Receiving size at rank " << rank <<std::endl;
        #endif

        // Broadcast size of message
        unsigned msg_size;
        MPI_Bcast(&msg_size, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

        // Allocate a buffer to hold the incoming numbers
        char* buff = new char[msg_size];

        #ifdef DEBUG
        std::cout << "Message size at rank " << rank << " " << msg_size << std::endl;
        fout << "Message size at rank " << rank << " " << msg_size << std::endl;
        #endif
        
        #ifdef DEBUG
        std::cout << "Receiving tree at rank " << rank <<std::endl;
        fout << "Receiving tree at rank " << rank <<std::endl;
        #endif
        
        // Broadcast the tree
        MPI_Bcast(buff, msg_size, MPI_BYTE, 0, MPI_COMM_WORLD);
        #ifdef DEBUG
        std::cout << "Receiving done at rank " << rank <<std::endl;
        fout << "Receiving done at rank " << rank <<std::endl;
        #endif

        // Process received message as cover tree
        delete ct;
        ct = new CoverTree();
        ct->deserialize(buff);
    }
    #ifdef DEBUG
    std::cout << "0 has been reached by " << rank << std::endl;
    fout << "0 has been reached by " << rank << std::endl;
    #endif

    //if(rank==1)
    //std::cout << "Finally merged tree is: \n"
    //          << *ct << std::endl;
    if(rank==0)
        std::cout << "Cover tree dispersed" << std::endl;

    #ifdef DEBUG
    std::cout << "Tree done at rank: " << rank << std::endl;
    fout << "Tree done at rank: " << rank << std::endl;
    #endif

    #ifdef DEBUG
    fout << "Cover tree finally: \n"
        << *ct << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    std::this_thread::sleep_for( dura );
    if(rank==0)
      std::cout << "Cover tree finally at rank 0: \n"
                << *ct << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    std::this_thread::sleep_for( dura );
    if(rank==1)
      std::cout << "Cover tree finally at rank 1: \n"
                << *ct << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    std::this_thread::sleep_for( dura );
    if(rank==2)
      std::cout << "Cover tree finally at rank 2: \n"
                << *ct << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    std::this_thread::sleep_for( dura );
    if(rank==3)
      std::cout << "Cover tree finally at rank 3: \n"
                << *ct << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    std::this_thread::sleep_for( dura );
    #endif

    #endif

    return ct;
}

/******************************************* Auxiliary Functions ***************************************************/

//get root level == max_level
int CoverTree::get_level()
{
    return root->level;
}

void CoverTree::print_levels()
{
    std::stack<CoverTree::Node*> travel;
    CoverTree::Node* curNode;

    // Initialize with root
    travel.push(root);

    // Pop, print and then push the children
    while (travel.size() > 0)
    {
        // Pop
        curNode = travel.top();
        travel.pop();

        // Count the level
        level_count[curNode->level]++;

        // Now push the children
        for (const auto& child : *curNode)
            travel.push(child);
    }

    for(auto const& qc : level_count)
    {
        std::cout << "Number of nodes at level " << qc.first << " = " << qc.second << std::endl;
        //dist_count[qc.first].store(0);
    }
}

// Pretty print
std::ostream& operator<<(std::ostream& os, const CoverTree& ct)
{
    std::stack<CoverTree::Node*> travel;
    CoverTree::Node* curNode;

    // Initialize with root
    travel.push(ct.root);

    // Qualititively keep track of number of prints
    int numPrints = 0;
    // Pop, print and then push the children
    while (travel.size() > 0)
    {
        if (numPrints > 5000)
            throw std::runtime_error("Printing stopped prematurely, something wrong!");
        numPrints++;

        // Pop
        curNode = travel.top();
        travel.pop();

        // Print the current -> children pair
        for (const auto& child : *curNode)
            os << *curNode << " -> " << *child << std::endl;

        // Now push the children
        for (int i = int(curNode->children.size()) - 1; i >= 0; --i)
            travel.push(curNode->children[i]);
    }

    return os;
}


Eigen::MatrixXd CoverTree::get_points()
{
    Eigen::MatrixXd points(D, N.load());
    
    std::stack<CoverTree::Node*> travel;
    CoverTree::Node* current;

    // Initialize with root
    travel.push(root);

    unsigned counter = 0;
    // Pop, print and then push the children
    while (travel.size() > 0)
    {
        // Pop
        current = travel.top();
        travel.pop();

        // Add to dataset
        points.col(counter++) = current->_p;

        // Now push the children
        for (const auto& child : *current)
            travel.push(child);
    }
    //std::cout << counter << ", " << N << std::endl;

    return points;
}


// Getting the best K members
Eigen::Map<Eigen::MatrixXd> CoverTree::getBestInitialPoints(unsigned numBest, double* data /* = NULL */) const
{
    if(!data) data = new double[D*numBest];
    Eigen::Map<Eigen::MatrixXd> points(data, D, numBest);
    unsigned i = 0;

    // Current level
    std::vector<CoverTree::Node*> curLevel;
    curLevel.push_back(root);
    // Next level
    std::vector<CoverTree::Node*> nextLevel;

    // Keep going down each level
    while (curLevel.size() < numBest)
    {
        //std::cout << "Size: " << curLevel.size() << std::endl;

        bool childLeft = false;
        for (const auto& member : curLevel)
        {
            if (member->children.size()>0)
            {
                for (const auto& child : *member)
                {
                    childLeft = true;
                    nextLevel.push_back(child);
                }
            }
            else
            {
                points.col(i++) = member->_p;
                numBest--;
            }
        }

        // End of tree, but K members not found (rare)
        // TODO: Fix this!
        if (!childLeft)
            throw std::runtime_error("Insufficient members for initialization");

        // Exit if nextLevel has > K members
        if (nextLevel.size() > numBest) break;

        curLevel.swap(nextLevel);
        std::vector<Node*>().swap(nextLevel);
    }

    // Number of children needed
    intptr_t numChild = numBest - curLevel.size();
    for (const auto& member : curLevel)
    {
        if (numChild > 0)
        {
            // Parent will not be included, increase child count by 1
            numChild++;
            for (const auto& child : *member){
                points.col(i++) = child->_p;
                numChild--;

                // Stop pushing children
                if (numChild == 0) break;
            }
        }
        else
        {
            points.col(i++) = member->_p;
            //std::cout << member->_p << std::endl;
        }
    }
    //std::cout << "Found points: " << i << std::endl;

    return points;
}

/******************************************* Functions to remove ***************************************************/

unsigned CoverTree::count_points()
{
    std::stack<CoverTree::Node*> travel;
    CoverTree::Node* current;

    // Initialize with root
    travel.push(root);

    unsigned result = 0;
    // Pop, print and then push the children
    while (travel.size() > 0)
    {
        // Pop
        current = travel.top();
        travel.pop();

        // Add to dataset
        ++result;

        // Now push the children
        for (const auto& child : *current)
            travel.push(child);
    }
    return result;
}

