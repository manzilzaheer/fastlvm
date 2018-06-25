//#define EIGEN_USE_MKL_ALL        //uncomment if available
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN 

#include <Python.h>
#include "numpy/arrayobject.h"
#include "cover_tree.h"

#include <future>
#include <thread>

#include <iostream>
#include <iomanip>

static PyObject *CovertreecError;

static PyObject *new_covertreec(PyObject *self, PyObject *args)
{
  int trunc;
  int use_multi_core;
  PyArrayObject *in_array;
  std::cout << "I am here" << std::endl;
  if (!PyArg_ParseTuple(args,"O!ip:new_covertreec", &PyArray_Type, &in_array, &trunc, &use_multi_core))
    return NULL;

  npy_intp numPoints = PyArray_DIM(in_array, 0);
  npy_intp numDims = PyArray_DIM(in_array, 1);
  //std::cout<<numPoints<<", "<<numDims<<std::endl;
  npy_intp idx[2] = {0, 0};
  double * fnp = reinterpret_cast< double * >( PyArray_GetPtr(in_array, idx) );
  Eigen::Map<Eigen::MatrixXd> pointMatrix(fnp, numDims, numPoints);

  CoverTree* cTree = CoverTree::from_matrix(pointMatrix, trunc, use_multi_core!=0);
  size_t int_ptr = reinterpret_cast< size_t >(cTree);

  return Py_BuildValue("n", int_ptr);
}

static PyObject *delete_covertreec(PyObject *self, PyObject *args)
{
  CoverTree *obj;
  size_t int_ptr;

  if (!PyArg_ParseTuple(args,"n:delete_covertreec", &int_ptr))
    return NULL;

  obj = reinterpret_cast< CoverTree * >(int_ptr);
  delete obj;

  return Py_BuildValue("n", int_ptr);
}


static PyObject *covertreec_insert(PyObject *self, PyObject *args) {

  CoverTree *obj;
  size_t int_ptr;
  PyArrayObject *in_array;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "nO!:covertreec_insert", &int_ptr, &PyArray_Type, &in_array))
    return NULL;

  int d = PyArray_NDIM(in_array);
  npy_intp *idx = new npy_intp[d];
  for(int i=0; i<d; ++i)
    idx[i] = 0;
  double * fnp = reinterpret_cast< double * >( PyArray_GetPtr(in_array, idx) );
  Eigen::Map<pointType> value(fnp, PyArray_SIZE(in_array));

  obj = reinterpret_cast< CoverTree * >(int_ptr);
  obj->insert(value, -1);

  Py_RETURN_NONE;
}

static PyObject *covertreec_remove(PyObject *self, PyObject *args) {

  CoverTree *obj;
  size_t int_ptr;
  PyArrayObject *in_array;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "nO!:covertreec_remove", &int_ptr, &PyArray_Type, &in_array))
    return NULL;

  int d = PyArray_NDIM(in_array);
  npy_intp *idx = new npy_intp[d];
  for(int i=0; i<d; ++i)
    idx[i] = 0;
  double * fnp = reinterpret_cast< double * >( PyArray_GetPtr(in_array, idx) );
  Eigen::Map<pointType> value(fnp, PyArray_SIZE(in_array));

  obj = reinterpret_cast< CoverTree * >(int_ptr);
  bool val = obj->remove(value);

  if (val)
  	Py_RETURN_TRUE;

  Py_RETURN_FALSE;
}

static PyObject *covertreec_nn(PyObject *self, PyObject *args) {

  CoverTree *obj;
  size_t int_ptr;
  int use_multi_core;
  int return_points;
  PyArrayObject *in_array;
  PyObject *return_value; 

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "nO!pp:covertreec_nn", &int_ptr, &PyArray_Type, &in_array, &use_multi_core, &return_points))
    return NULL;

  npy_intp idx[2] = {0,0};
  npy_intp numPoints = PyArray_DIM(in_array, 0);
  npy_intp numDims = PyArray_DIM(in_array, 1);
  //std::cout<<numPoints<<", "<<numDims<<std::endl;
  double * fnp = reinterpret_cast< double * >( PyArray_GetPtr(in_array, idx) );
  Eigen::Map<Eigen::MatrixXd> queryPts(fnp, numDims, numPoints);

  obj = reinterpret_cast< CoverTree * >(int_ptr);

  double *dist = new double[numPoints];
  long *indices = new long[numPoints];
  double *results = nullptr;
  if(return_points!=0)
  {
    results = new double[numDims*numPoints];
    if(use_multi_core!=0)
    {
        utils::parallel_for_progressbar(0, numPoints, [&](npy_intp i)->void{
            std::pair<CoverTree::Node*, double> ct_nn = obj->NearestNeighbour(queryPts.col(i));
            npy_intp offset = i;
            dist[offset] = ct_nn.second;
            indices[offset] = ct_nn.first->UID;
            double *data = ct_nn.first->_p.data();
            offset = i*numDims;
            for(npy_intp j=0; j<numDims; ++j)
                results[offset++] = data[j];
        });
    }
    else
    {
        for(npy_intp i = 0; i < numPoints; ++i) {
            std::pair<CoverTree::Node*, double> ct_nn = obj->NearestNeighbour(queryPts.col(i));
            npy_intp offset = i;
            dist[offset] = ct_nn.second;
            indices[offset] = ct_nn.first->UID;
            double *data = ct_nn.first->_p.data();
            offset = i*numDims;
            for(npy_intp j=0; j<numDims; ++j)
                results[offset++] = data[j];
        }
    }
    npy_intp dims[1] = {numPoints};
    PyObject *out_dist = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT64, dist);
    PyArray_ENABLEFLAGS((PyArrayObject *)out_dist, NPY_ARRAY_OWNDATA);
    PyObject *out_indices = PyArray_SimpleNewFromData(1, dims, NPY_LONG, indices);
    PyArray_ENABLEFLAGS((PyArrayObject *)out_indices, NPY_ARRAY_OWNDATA);
    npy_intp odims[2] = {numPoints, numDims};
    PyObject *out_array = PyArray_SimpleNewFromData(2, odims, NPY_FLOAT64, results);
    PyArray_ENABLEFLAGS((PyArrayObject *)out_array, NPY_ARRAY_OWNDATA);
    
    return_value = Py_BuildValue("NNN", out_indices, out_dist, out_array);
  }
  else
  {
    if(use_multi_core!=0)
    {
        utils::parallel_for_progressbar(0, numPoints, [&](npy_intp i)->void{
            std::pair<CoverTree::Node*, double> ct_nn = obj->NearestNeighbour(queryPts.col(i));
            npy_intp offset = i;
            dist[offset] = ct_nn.second;
            indices[offset] = ct_nn.first->UID;
        });
    }
    else
    {
        for(npy_intp i = 0; i < numPoints; ++i) {
            std::pair<CoverTree::Node*, double> ct_nn = obj->NearestNeighbour(queryPts.col(i));
            npy_intp offset = i;
            dist[offset] = ct_nn.second;
            indices[offset] = ct_nn.first->UID;
        }
    }
    npy_intp dims[1] = {numPoints};
    PyObject *out_dist = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT64, dist);
    PyArray_ENABLEFLAGS((PyArrayObject *)out_dist, NPY_ARRAY_OWNDATA);
    PyObject *out_indices = PyArray_SimpleNewFromData(1, dims, NPY_LONG, indices);
    PyArray_ENABLEFLAGS((PyArrayObject *)out_indices, NPY_ARRAY_OWNDATA);

    return_value = Py_BuildValue("NN", out_indices, out_dist);
  }
  
  return return_value;
}

static PyObject *covertreec_knn(PyObject *self, PyObject *args) {

  long k=2L;
  CoverTree *obj;
  size_t int_ptr;
  int use_multi_core;
  int return_points;
  PyArrayObject *in_array;
  PyObject *return_value; 

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "nO!lpp:covertreec_knn", &int_ptr, &PyArray_Type, &in_array, &k, &use_multi_core, &return_points))
    return NULL;

  npy_intp idx[2] = {0,0};
  npy_intp numPoints = PyArray_DIM(in_array, 0);
  npy_intp numDims = PyArray_DIM(in_array, 1);
  //std::cout<<numPoints<<", "<<numDims<<std::endl;
  double * fnp = reinterpret_cast< double * >( PyArray_GetPtr(in_array, idx) );
  Eigen::Map<Eigen::MatrixXd> queryPts(fnp, numDims, numPoints);

  obj = reinterpret_cast< CoverTree * >(int_ptr);

  long *indices = new long[k*numPoints];
  double *dist = new double[k*numPoints];
  double *results = nullptr;
  if(return_points!=0)
  {
    results = new double[k*numDims*numPoints];
    if(use_multi_core!=0)
    {
        utils::parallel_for_progressbar(0, numPoints, [&](npy_intp i)->void{
            std::vector<std::pair<CoverTree::Node*, double>> ct_nn = obj->kNearestNeighbours(queryPts.col(i), k);
            npy_intp offset = k*i;
            for(long t=0; t<k; ++t)
            {
                indices[offset] = ct_nn[t].first->UID;
                dist[offset] = ct_nn[t].second;
                double *data = ct_nn[t].first->_p.data();
                npy_intp inner_offset = (offset++)*numDims;
                for(long j=0; j<numDims; ++j)
                    results[inner_offset++] = data[j];
            }
        });
    }
    else
    {
        for(npy_intp i = 0; i < numPoints; ++i) {
            std::vector<std::pair<CoverTree::Node*, double>> ct_nn = obj->kNearestNeighbours(queryPts.col(i), k);
            npy_intp offset = k*i;
            for(long t=0; t<k; ++t)
            {
                indices[offset] = ct_nn[t].first->UID;
                dist[offset] = ct_nn[t].second;
                double *data = ct_nn[t].first->_p.data();
                npy_intp inner_offset = (offset++)*numDims;
                for(long j=0; j<numDims; ++j)
                    results[inner_offset++] = data[j];
            }
        }
    }
    npy_intp dims[2] = {numPoints, k};
    PyObject *out_indices = PyArray_SimpleNewFromData(2, dims, NPY_LONG, indices);
    PyArray_ENABLEFLAGS((PyArrayObject *)out_indices, NPY_ARRAY_OWNDATA);
    PyObject *out_dist = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT64, dist);
    PyArray_ENABLEFLAGS((PyArrayObject *)out_dist, NPY_ARRAY_OWNDATA);
    npy_intp odims[3] = {numPoints, k, numDims};
    PyObject *out_array = PyArray_SimpleNewFromData(3, odims, NPY_FLOAT64, results);
    PyArray_ENABLEFLAGS((PyArrayObject *)out_array, NPY_ARRAY_OWNDATA);

    return_value = Py_BuildValue("NNN", out_indices, out_dist, out_array);
  }
  else
  {
    if(use_multi_core!=0)
    {
        utils::parallel_for_progressbar(0, numPoints, [&](npy_intp i)->void{
            std::vector<std::pair<CoverTree::Node*, double>> ct_nn = obj->kNearestNeighbours(queryPts.col(i), k);
            npy_intp offset = k*i;
            for(long t=0; t<k; ++t)
            {
                indices[offset] = ct_nn[t].first->UID;
                dist[offset++] = ct_nn[t].second;
            }
        });
    }
    else
    {
        for(npy_intp i = 0; i < numPoints; ++i) {
            std::vector<std::pair<CoverTree::Node*, double>> ct_nn = obj->kNearestNeighbours(queryPts.col(i), k);
            npy_intp offset = k*i;
            for(long t=0; t<k; ++t)
            {
                indices[offset] = ct_nn[t].first->UID;
                dist[offset++] = ct_nn[t].second;
            }
        }
    }
    npy_intp dims[2] = {numPoints, k};
    PyObject *out_indices = PyArray_SimpleNewFromData(2, dims, NPY_LONG, indices);
    PyArray_ENABLEFLAGS((PyArrayObject *)out_indices, NPY_ARRAY_OWNDATA);
    PyObject *out_dist = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT64, dist);
    PyArray_ENABLEFLAGS((PyArrayObject *)out_dist, NPY_ARRAY_OWNDATA);

    return_value = Py_BuildValue("NN", out_indices, out_dist);
  }
  
  return return_value;
}

static PyObject *covertreec_range(PyObject *self, PyObject *args) {

  double r=0.0;
  CoverTree *obj;
  size_t int_ptr;
  int use_multi_core;
  int return_points;
  PyArrayObject *in_array;
  PyObject *return_value; 

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "nO!dpp:covertreec_knn", &int_ptr, &PyArray_Type, &in_array, &r, &use_multi_core, &return_points))
    return NULL;

  npy_intp idx[2] = {0,0};
  npy_intp numPoints = PyArray_DIM(in_array, 0);
  npy_intp numDims = PyArray_DIM(in_array, 1);
  //std::cout<<numPoints<<", "<<numDims<<std::endl;
  double * fnp = reinterpret_cast< double * >( PyArray_GetPtr(in_array, idx) );
  Eigen::Map<Eigen::MatrixXd> queryPts(fnp, numDims, numPoints);

  obj = reinterpret_cast< CoverTree * >(int_ptr);

  PyObject *indices = PyList_New(numPoints);
  PyObject *dist = PyList_New(numPoints);
  if(return_points!=0)
  {
    PyObject *results = PyList_New(numPoints);
    if(use_multi_core!=0)
    {
        PyObject **array_indices = new PyObject*[numPoints];
        PyObject **array_dist = new PyObject*[numPoints];
        PyObject **array_point = new PyObject*[numPoints];
        utils::parallel_for_progressbar(0, numPoints, [&](npy_intp i)->void{
            std::vector<std::pair<CoverTree::Node*, double>> ct_nn = obj->rangeNeighbours(queryPts.col(i), r);
            size_t num_neighbours = ct_nn.size();
            long *point_indices = new long[num_neighbours];
            double *point_dist = new double[num_neighbours];
            double *point_point = new double[num_neighbours*numDims];
            for(size_t t=0; t<num_neighbours; ++t)
            {
                point_indices[t] = ct_nn[t].first->UID;
                point_dist[t] = ct_nn[t].second;
                double *data = ct_nn[t].first->_p.data();
                npy_intp offset = t*numDims;
                for(long j=0; j<numDims; ++j)
                    point_point[offset++] = data[j];
            }
            npy_intp dims[1] = {(npy_intp)num_neighbours};
            PyObject *neighbour_indices = PyArray_SimpleNewFromData(1, dims, NPY_LONG, point_indices);
            PyObject *neighbour_dist = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, point_dist);
            PyArray_ENABLEFLAGS((PyArrayObject *)neighbour_indices, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject *)neighbour_dist, NPY_ARRAY_OWNDATA);
            npy_intp odims[2] = {(npy_intp)num_neighbours, numDims};
            PyObject *neighbour_point = PyArray_SimpleNewFromData(2, odims, NPY_DOUBLE, point_point);
            PyArray_ENABLEFLAGS((PyArrayObject *)neighbour_point, NPY_ARRAY_OWNDATA);
            array_indices[i] = neighbour_indices;
            array_dist[i] = neighbour_dist;
            array_point[i] = neighbour_point;
        });
        for(npy_intp i = 0; i < numPoints; ++i) {
            PyList_SET_ITEM(indices, i, array_indices[i]);
            PyList_SET_ITEM(dist, i, array_dist[i]);
            PyList_SET_ITEM(results, i, array_point[i]);
        }
    }
    else
    {
        for(npy_intp i = 0; i < numPoints; ++i) {
            std::vector<std::pair<CoverTree::Node*, double>> ct_nn = obj->rangeNeighbours(queryPts.col(i), r);
            size_t num_neighbours = ct_nn.size();
            long *point_indices = new long[num_neighbours];
            double *point_dist = new double[num_neighbours];
            double *point_point = new double[num_neighbours*numDims];
            for(size_t t=0; t<num_neighbours; ++t)
            {
                point_indices[t] = ct_nn[t].first->UID;
                point_dist[t] = ct_nn[t].second;
                double *data = ct_nn[t].first->_p.data();
                npy_intp offset = t*numDims;
                for(long j=0; j<numDims; ++j)
                    point_point[offset++] = data[j];
            }
            npy_intp dims[1] = {(npy_intp)num_neighbours};
            PyObject *neighbour_indices = PyArray_SimpleNewFromData(1, dims, NPY_LONG, point_indices);
            PyObject *neighbour_dist = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, point_dist);
            PyArray_ENABLEFLAGS((PyArrayObject *)neighbour_indices, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject *)neighbour_dist, NPY_ARRAY_OWNDATA);
            npy_intp odims[2] = {(npy_intp)num_neighbours, numDims};
            PyObject *neighbour_point = PyArray_SimpleNewFromData(2, odims, NPY_DOUBLE, point_point);
            PyArray_ENABLEFLAGS((PyArrayObject *)neighbour_point, NPY_ARRAY_OWNDATA);
            PyList_SET_ITEM(indices, i, neighbour_indices);
            PyList_SET_ITEM(dist, i, neighbour_dist);
            PyList_SET_ITEM(results, i, neighbour_point);
        }
    }
    return_value = Py_BuildValue("NNN", indices, dist, results);
  }
  else
  {
    if(use_multi_core!=0)
    {
        PyObject **array_indices = new PyObject*[numPoints];
        PyObject **array_dist = new PyObject*[numPoints];
        utils::parallel_for_progressbar(0, numPoints, [&](npy_intp i)->void{
            std::vector<std::pair<CoverTree::Node*, double>> ct_nn = obj->rangeNeighbours(queryPts.col(i), r);
            size_t num_neighbours = ct_nn.size();
            long *point_indices = new long[num_neighbours];
            double *point_dist = new double[num_neighbours];
            for(size_t t=0; t<num_neighbours; ++t)
            {
                point_indices[t] = ct_nn[t].first->UID;
                point_dist[t] = ct_nn[t].second;
            }
            npy_intp dims[1] = {(npy_intp)num_neighbours};
            PyObject *neighbour_indices = PyArray_SimpleNewFromData(1, dims, NPY_LONG, point_indices);
            PyObject *neighbour_dist = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, point_dist);
            PyArray_ENABLEFLAGS((PyArrayObject *)neighbour_indices, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject *)neighbour_dist, NPY_ARRAY_OWNDATA);
            array_indices[i] = neighbour_indices;
            array_dist[i] = neighbour_dist;
        });
        for(npy_intp i = 0; i < numPoints; ++i) {
            PyList_SET_ITEM(indices, i, array_indices[i]);
            PyList_SET_ITEM(dist, i, array_dist[i]);
        }
    }
    else
    {
        for(npy_intp i = 0; i < numPoints; ++i) {
            std::vector<std::pair<CoverTree::Node*, double>> ct_nn = obj->rangeNeighbours(queryPts.col(i), r);
            size_t num_neighbours = ct_nn.size();
            long *point_indices = new long[num_neighbours];
            double *point_dist = new double[num_neighbours];
            for(size_t t=0; t<num_neighbours; ++t)
            {
                point_indices[t] = ct_nn[t].first->UID;
                point_dist[t] = ct_nn[t].second;
            }
            npy_intp dims[1] = {(npy_intp)num_neighbours};
            PyObject *neighbour_indices = PyArray_SimpleNewFromData(1, dims, NPY_LONG, point_indices);
            PyObject *neighbour_dist = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, point_dist);
            PyArray_ENABLEFLAGS((PyArrayObject *)neighbour_indices, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject *)neighbour_dist, NPY_ARRAY_OWNDATA);
            PyList_SET_ITEM(indices, i, neighbour_indices);
            PyList_SET_ITEM(dist, i, neighbour_dist);
        }
    }
    return_value = Py_BuildValue("NN", indices, dist);
  }
  
  return return_value;
}

static PyObject *covertreec_serialize(PyObject *self, PyObject *args)
{
  CoverTree *obj;
  size_t int_ptr;

  if (!PyArg_ParseTuple(args,"n:covertreec_serialize", &int_ptr))
    return NULL;
  
  obj = reinterpret_cast< CoverTree * >(int_ptr);
  char* buff = obj->serialize();
  size_t len = obj->msg_size();
  
  return Py_BuildValue("y#", buff, len);
}

static PyObject *covertreec_deserialize(PyObject *self, PyObject *args)
{
  char* buff;
  size_t len;
  if (!PyArg_ParseTuple(args,"y#:covertreec_deserialize", &buff, &len))
    return NULL;

  CoverTree* cTree = new CoverTree();
  cTree->deserialize(buff);
  size_t int_ptr = reinterpret_cast< size_t >(cTree);

  return Py_BuildValue("n", int_ptr);
}


static PyObject *covertreec_display(PyObject *self, PyObject *args) {

  CoverTree *obj;
  size_t int_ptr;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "n:covertreec_display", &int_ptr))
    return NULL;
  
  obj = reinterpret_cast< CoverTree * >(int_ptr);
  std::cout << *obj;

  Py_RETURN_NONE;
}

static PyObject *covertreec_spreadout(PyObject *self, PyObject *args) {

  CoverTree *obj;
  size_t int_ptr;
  unsigned K;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "nI:covertreec_display", &int_ptr, &K))
    return NULL;
  
  obj = reinterpret_cast< CoverTree * >(int_ptr);
  Eigen::Map<Eigen::MatrixXd> results = obj->getBestInitialPoints(K);
  
  npy_intp numPoints = results.cols();
  npy_intp numDims = results.rows();
  //std::cout<<numPoints<<", "<<numDims<<std::endl;
  npy_intp dims[2] = {numPoints, numDims};
  PyObject *out_array = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT64, results.data());
  PyArray_ENABLEFLAGS((PyArrayObject *)out_array, NPY_ARRAY_OWNDATA);
  
  return out_array;
}

static PyObject *covertreec_test_covering(PyObject *self, PyObject *args) {

  CoverTree *obj;
  size_t int_ptr;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "n:covertreec_test_covering", &int_ptr))
    return NULL;
  
  obj = reinterpret_cast< CoverTree * >(int_ptr);
  if(obj->check_covering())
    Py_RETURN_TRUE;

  Py_RETURN_FALSE;
}

PyMODINIT_FUNC PyInit_covertreec(void)
{
  PyObject *m;
  static PyMethodDef CovertreecMethods[] = {
    {"new", new_covertreec, METH_VARARGS, "Initialize a new Cover Tree."},
    {"delete", delete_covertreec, METH_VARARGS, "Delete the Cover Tree."},
    {"insert", covertreec_insert, METH_VARARGS, "Insert a point to the Cover Tree."},
    {"remove", covertreec_remove, METH_VARARGS, "Remove a point from the Cover Tree."},
    {"NearestNeighbour", covertreec_nn, METH_VARARGS, "Find the nearest neighbour."},
    {"kNearestNeighbours", covertreec_knn, METH_VARARGS, "Find the k nearest neighbours."},
    {"RangeSearch", covertreec_range, METH_VARARGS, "Find all the neighbours in range."},
    {"serialize", covertreec_serialize, METH_VARARGS, "Serialize the current Cover Tree."},
    {"deserialize", covertreec_deserialize, METH_VARARGS, "Construct a Cover Tree from deserializing."},
    {"display", covertreec_display, METH_VARARGS, "Display the Cover Tree."},
    {"spreadout", covertreec_spreadout, METH_VARARGS, "Find well spreadout k points."},
    {"test_covering", covertreec_test_covering, METH_VARARGS, "Check if covering property is satisfied."},
    {NULL, NULL, 0, NULL}
  };
  static struct PyModuleDef mdef = {PyModuleDef_HEAD_INIT,
                                 "covertreec",
                                 "Example module that creates an extension type.",
                                 -1,
                                 CovertreecMethods};
  m = PyModule_Create(&mdef);
  if ( m == NULL )
    return NULL;

  /* IMPORTANT: this must be called */
  import_array();

  CovertreecError = PyErr_NewException("covertreec.error", NULL, NULL);
  Py_INCREF(CovertreecError);
  PyModule_AddObject(m, "error", CovertreecError);
  
  return m;
}

int main(int argc, char *argv[])
{
  /* Convert to wchar */
  wchar_t *program = Py_DecodeLocale(argv[0], NULL);
  if (program == NULL) {
     std::cerr << "Fatal error: cannot decode argv[0]" << std::endl;
     return 1;
  }
  
  /* Add a built-in module, before Py_Initialize */
  //PyImport_AppendInittab("covertreec", PyInit_covertreec);
    
  /* Pass argv[0] to the Python interpreter */
  Py_SetProgramName(program);

  /* Initialize the Python interpreter.  Required. */
  Py_Initialize();

  /* Add a static module */
  //PyInit_covertreec();
}

