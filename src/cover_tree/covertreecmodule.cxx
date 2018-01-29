//#define EIGEN_USE_MKL_ALL        //uncomment if available
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN 

#include <Python.h>
#include "numpy/arrayobject.h"
#include "cover_tree.h"

#include <future>
#include <thread>

#include <iostream>
# include <iomanip>

// template<class UnaryFunction>
// UnaryFunction parallel_for_each(size_t first, size_t last, UnaryFunction f)
// {
  // unsigned cores = std::thread::hardware_concurrency();
  // //std::cout << "Number of cores: " << cores << std::endl;

  // auto task = [&f](size_t start, size_t end)->void{
    // for (; start < end; ++start)
      // f(start);
  // };

  // const size_t total_length = last - first;
  // const size_t chunk_length = total_length / cores;
  // size_t chunk_start = first;
  // std::vector<std::future<void>>  for_threads;
  // for (unsigned i = 0; i < cores - 1; ++i)
  // {
    // const auto chunk_stop = chunk_start + chunk_length;
    // for_threads.push_back(std::async(std::launch::async, task, chunk_start, chunk_stop));
    // chunk_start = chunk_stop;
  // }
  // for_threads.push_back(std::async(std::launch::async, task, chunk_start, last));

  // for (auto& thread : for_threads)
    // thread.get();
  // return f;
// }

// static inline void progressbar(size_t x, size_t n, size_t w = 50){
    // if ( (x != n) && (x % (n/10+1) != 0) ) return;

    // float ratio =  x/(float)n;
    // unsigned c = unsigned(ratio * w);

    // std::cout << std::setw(3) << (int)(ratio*100) << "% [";
    // for (unsigned x=0; x<c; x++) std::cout << "=";
    // for (unsigned x=c; x<w; x++) std::cout << " ";
    // std::cout << "]\r" << std::flush;
// }

// template<class UnaryFunction>
// UnaryFunction parallel_for_progressbar(size_t first, size_t last, UnaryFunction f)
// {
    // unsigned cores = std::thread::hardware_concurrency();
    // const size_t total_length = last - first;
    // const size_t chunk_length = total_length / cores;

    // auto task = [&f,&chunk_length](size_t start, size_t end)->void{
        // for (; start < end; ++start){
            // progressbar(start%chunk_length, chunk_length);
            // f(start);
        // }
    // };

    // size_t chunk_start = first;
    // std::vector<std::future<void>>  for_threads;
    // for (unsigned i = 0; i < cores - 1; ++i)
    // {
        // const auto chunk_stop = chunk_start + chunk_length;
        // for_threads.push_back(std::async(std::launch::async, task, chunk_start, chunk_stop));
        // chunk_start = chunk_stop;
    // }
    // for_threads.push_back(std::async(std::launch::async, task, chunk_start, last));

    // for (auto& thread : for_threads)
        // thread.get();
    // progressbar(chunk_length, chunk_length);
    // std::cout << std::endl;
    // return f;
// }


static PyObject *CovertreecError;

static PyObject *new_covertreec(PyObject *self, PyObject *args)
{
  int trunc;
  PyArrayObject *in_array;

  if (!PyArg_ParseTuple(args,"O!i:new_covertreec", &PyArray_Type, &in_array, &trunc))
    return NULL;

  std::cout<<"Hi reached: " << in_array << ", " << trunc <<std::endl;
  npy_intp numPoints = PyArray_DIM(in_array, 0);
  npy_intp numDims = PyArray_DIM(in_array, 1);
  std::cout<<numPoints<<", "<<numDims<<std::endl;
  npy_intp idx[2] = {0, 0};
  double * fnp = reinterpret_cast< double * >( PyArray_GetPtr(in_array, idx) );
  Eigen::Map<Eigen::MatrixXd> pointMatrix(fnp, numDims, numPoints);

  CoverTree* cTree = CoverTree::from_matrix(pointMatrix, trunc, false);
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
  obj->insert(value, 0);

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
  PyArrayObject *in_array;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "nO!:covertreec_nn", &int_ptr, &PyArray_Type, &in_array))
    return NULL;

  npy_intp idx[2] = {0,0};
  npy_intp numPoints = PyArray_DIM(in_array, 0);
  npy_intp numDims = PyArray_DIM(in_array, 1);
  //std::cout<<numPoints<<", "<<numDims<<std::endl;
  double * fnp = reinterpret_cast< double * >( PyArray_GetPtr(in_array, idx) );
  Eigen::Map<Eigen::MatrixXd> queryPts(fnp, numDims, numPoints);

  obj = reinterpret_cast< CoverTree * >(int_ptr);

  //obj->dist_count.clear();

  //double *results = new double[numDims*numPoints];
  double *dist = new double[numPoints];
  long *indices = new long[numPoints];
  utils::parallel_for_progressbar(0, numPoints, [&](npy_intp i)->void{
  //for(npy_intp i = 0; i < numPoints; ++i) {
    std::pair<CoverTree::Node*, double> ct_nn = obj->NearestNeighbour(queryPts.col(i));
    //double *data = ct_nn.first->_p.data();
    npy_intp offset = i;
    dist[offset] = ct_nn.second;
    indices[offset++] = ct_nn.first->UID;
    //for(npy_intp j=0; j<numDims; ++j)
    //  results[offset++] = data[j];
  });
  //std::pair<CoverTree::Node*, double> cnn = obj->NearestNeighbour(value);

  // unsigned tot_comp = 0;
  // for(auto const& qc : obj->dist_count)
  // {
  //   std::cout << "Average number of distance computations at level: " << qc.first << " = " << 1.0 * (qc.second.load())/numPoints << std::endl;
  //   tot_comp += qc.second.load();
  // }
  // std::cout << "Average number of distance computations: " << 1.0*tot_comp/numPoints << std::endl;
  // std::cout << cnn.first->_p << std::endl;
  //Py_RETURN_NONE;

  npy_intp dims[1] = {numPoints};
  PyObject *out_dist = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT64, dist);
  PyArray_ENABLEFLAGS((PyArrayObject *)out_dist, NPY_ARRAY_OWNDATA);
  PyObject *out_indices = PyArray_SimpleNewFromData(1, dims, NPY_LONG, indices);
  PyArray_ENABLEFLAGS((PyArrayObject *)out_indices, NPY_ARRAY_OWNDATA);

  //Py_INCREF(out_array);
  return Py_BuildValue("NN", out_indices, out_dist);
}

static PyObject *covertreec_knn(PyObject *self, PyObject *args) {

  CoverTree *obj;
  size_t int_ptr;
  PyArrayObject *in_array;
  long k=2L;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "nO!l:covertreec_knn", &int_ptr, &PyArray_Type, &in_array, &k))
    return NULL;

  npy_intp idx[2] = {0,0};
  npy_intp numPoints = PyArray_DIM(in_array, 0);
  npy_intp numDims = PyArray_DIM(in_array, 1);
  //std::cout<<numPoints<<", "<<numDims<<std::endl;
  double * fnp = reinterpret_cast< double * >( PyArray_GetPtr(in_array, idx) );
  Eigen::Map<Eigen::MatrixXd> queryPts(fnp, numDims, numPoints);
  
  //std::cout << queryPts.col(0) << std::endl;

  obj = reinterpret_cast< CoverTree * >(int_ptr);

  //double *results = new double[k*numDims*numPoints];
  long *indices = new long[k*numPoints];
  double *dist = new double[k*numPoints];
  utils::parallel_for_progressbar(0, numPoints, [&](npy_intp i)->void{
    std::vector<std::pair<CoverTree::Node*, double>> ct_nn = obj->kNearestNeighbours(queryPts.col(i), k);
    npy_intp offset = k*i;
    for(long t=0; t<k; ++t)
    {
      indices[offset] = ct_nn[t].first->UID;
      dist[offset++] = ct_nn[t].second;
      //double *data = ct_nn[t].first->_p.data();
      //for(long j=0; j<numDims; ++j)
      //  results[offset++] = data[j];
    }
  });
  //std::pair<CoverTree::Node*, double> cnn = obj->NearestNeighbour(value);

  // std::cout << cnn.first->_p << std::endl;
  //Py_RETURN_NONE;

  npy_intp dims[2] = {numPoints, k};
  PyObject *out_indices = PyArray_SimpleNewFromData(2, dims, NPY_LONG, indices);
  PyArray_ENABLEFLAGS((PyArrayObject *)out_indices, NPY_ARRAY_OWNDATA);

  PyObject *out_dist = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT64, dist);
  PyArray_ENABLEFLAGS((PyArrayObject *)out_dist, NPY_ARRAY_OWNDATA);

  //Py_INCREF(out_array);
  //return out_array;

  return Py_BuildValue("NN", out_indices, out_dist);
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

  // std::cout << len << std::endl;
  // std::cout << "[";
  // for(int i=0; i<10; ++i)
      // std::cout << int(buff[i]) << ", ";
  // std::cout << "]" << std::endl;
      
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
  
  //std::cout << results << std::endl;

  //Py_INCREF(out_array);
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

