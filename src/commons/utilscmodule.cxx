#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include "numpy/arrayobject.h"
#include "dataio.h"
#include "stirling.h"
#include "utils.h"

#include <map>
#include <set>
#include <string>

static PyObject *UtilscError;

static PyObject *utilsc_read_corpus(PyObject *self, PyObject *args)
{
  char* filename;
  PyObject *in_vocab;
  PyObject *out_vocab;
  PyObject *in_stopwords;
  PyObject *out_data;

  if (!PyArg_ParseTuple(args,"sO!O!:utilsc_read_corpus", &filename, &PyList_Type, &in_vocab, &PyList_Type, &in_stopwords))
    return NULL;

  /* Convert existing vocab into a map for fast search and insertion */
  std::map<std::string, unsigned> word2id;
  unsigned V = (unsigned)PyList_GET_SIZE(in_vocab);
  for(unsigned w = 0; w < V; ++w)
      word2id.emplace((char*)PyUnicode_DATA(PyList_GET_ITEM(in_vocab, w)), w);
  
  /* Convert stopword list into a set for fast search */
  std::set<std::string> stopwords;
  unsigned SWV = (unsigned)PyList_GET_SIZE(in_stopwords);
  for(unsigned w = 0; w < SWV; ++w)
      stopwords.emplace((char*)PyUnicode_DATA(PyList_GET_ITEM(in_stopwords, w)));
  
  std::string fname(filename);
  DataIO::corpus c;
  c.read_data(fname, &word2id, &stopwords);
  
  /* Transfer vocabulary */
  out_vocab = PyList_New(word2id.size());
  for(const auto& kv : word2id)
      PyList_SET_ITEM(out_vocab, kv.second, PyUnicode_FromString(kv.first.c_str()));
  
  size_t M = c.size();
  out_data = PyList_New(M);
  for(size_t i = 0; i < M; ++i)
  {
    auto& d = c[i];
    npy_intp dims[1] = {(npy_intp)d.size()};
    PyObject *out_array = PyArray_SimpleNewFromData(1, dims, NPY_UINT32, d.data());
    d.data() = nullptr;
    PyArray_ENABLEFLAGS((PyArrayObject *)out_array, NPY_ARRAY_OWNDATA);
    PyList_SET_ITEM(out_data, i, out_array);
  }
  
  // for(const auto& kv : stopwords)
      // std::cout << kv << std::endl;
  
  //obj = reinterpret_cast< model * >(int_ptr);
  //char* buff = obj->serialize();
  //size_t len = obj->msg_size();
  
  //Py_DECREF(in_vocab);
  return Py_BuildValue("NN", out_data, out_vocab);
  //return out_vocab;
}

static PyObject *utilsc_ref_count(PyObject *self, PyObject *args) {

  PyObject* in_data;
  unsigned count;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "O:utilsc_ref_count", &in_data))
    return NULL;

  if(in_data == Py_None)
  {
      std::cout << "None/1" << std::endl;
      count = 1;
  }
  else
  {
      count = Py_REFCNT(in_data);
      std::cout << "Not None/" << count << std::endl;
  }
  
  return Py_BuildValue("I", count);
}


static PyObject *utilsc_kmeanspp(PyObject *self, PyObject *args)
{
  unsigned K;
  PyArrayObject *in_array;
  
  if (!PyArg_ParseTuple(args,"IO!:utilsc_kmeanspp", &K, &PyArray_Type, &in_array))
    return NULL;

  npy_intp idx[2] = {0,0};
  npy_intp numPoints_in = PyArray_DIM(in_array, 0);
  npy_intp numDims_in = PyArray_DIM(in_array, 1);
  //std::cout<<numPoints_in<<", "<<numDims_in<<std::endl;
  double * fnp = reinterpret_cast< double * >( PyArray_GetPtr(in_array, idx) );
  Eigen::Map<Eigen::MatrixXd> points(fnp, numDims_in, numPoints_in);
      
  size_t* results = utils::KMeanspp(points, K);

  npy_intp dims[1] = {K};
  PyObject *out_array = PyArray_SimpleNewFromData(1, dims, NPY_INT64, results);
  PyArray_ENABLEFLAGS((PyArrayObject *)out_array, NPY_ARRAY_OWNDATA);

  return out_array;
}

static PyObject *utilsc_new_stirling(PyObject *self, PyObject *args)
{
  if (!PyArg_ParseTuple(args,":utilsc_new_stirling"))
    return NULL;

  Stirling* stir = new Stirling();
  size_t int_ptr = reinterpret_cast< size_t >(stir);

  return Py_BuildValue("n", int_ptr);
}

static PyObject *utilsc_log_stirling_num(PyObject *self, PyObject *args)
{
  Stirling *obj;
  size_t int_ptr;
  unsigned n;
  unsigned m;

  if (!PyArg_ParseTuple(args,"nII:utilsc_log_stirling_num", &int_ptr, &n, &m))
    return NULL;

  obj = reinterpret_cast< Stirling * >(int_ptr);
  double result = obj->get_log_stirling_num(n, m);

  return Py_BuildValue("d", result);
}

static PyObject *utilsc_uratio(PyObject *self, PyObject *args)
{
  Stirling *obj;
  size_t int_ptr;
  unsigned n;
  unsigned m;

  if (!PyArg_ParseTuple(args,"nII:utilsc_uratio", &int_ptr, &n, &m))
    return NULL;

  obj = reinterpret_cast< Stirling * >(int_ptr);
  double result = obj->uratio(n, m);

  return Py_BuildValue("d", result);
}

static PyObject *utilsc_vratio(PyObject *self, PyObject *args)
{
  Stirling *obj;
  size_t int_ptr;
  unsigned n;
  unsigned m;

  if (!PyArg_ParseTuple(args,"nII:utilsc_vratio", &int_ptr, &n, &m))
    return NULL;

  obj = reinterpret_cast< Stirling * >(int_ptr);
  double result = obj->vratio(n, m);

  return Py_BuildValue("d", result);
}

static PyObject *utilsc_wratio(PyObject *self, PyObject *args)
{
  Stirling *obj;
  size_t int_ptr;
  unsigned n;
  unsigned m;

  if (!PyArg_ParseTuple(args,"nII:utilsc_wratio", &int_ptr, &n, &m))
    return NULL;

  obj = reinterpret_cast< Stirling * >(int_ptr);
  double result = obj->wratio(n, m);

  return Py_BuildValue("d", result);
}

PyMODINIT_FUNC PyInit_utilsc(void)
{
  PyObject *m;
  static PyMethodDef UtilscMethods[] = {
    {"read_corpus", utilsc_read_corpus, METH_VARARGS, "Read a corpus from text file."},
    {"ref_count", utilsc_ref_count, METH_VARARGS, "Reference count of python objects."},
    {"kmeanspp", utilsc_kmeanspp, METH_VARARGS, "Initialization for k-means and GMM."},
    {"new_stirling", utilsc_new_stirling, METH_VARARGS, "Initialization for stirling numbers."},
    {"log_stirling_num", utilsc_log_stirling_num, METH_VARARGS, "Get log of stirling number of first kind."},
    {"uratio", utilsc_uratio, METH_VARARGS, "Get ratio of stirling number of first kind S(n+1,m)/S(n,m)."},
    {"vratio", utilsc_vratio, METH_VARARGS, "Get ratio of stirling number of first kind S(n+1,m+1)/S(n+1,m)."},
    {"wratio", utilsc_wratio, METH_VARARGS, "Get ratio of stirling number of first kind S(n+1,m+1)/S(n,m)."},
    {NULL, NULL, 0, NULL}
  };
  static struct PyModuleDef mdef = {PyModuleDef_HEAD_INIT,
                                 "utilsc",
                                 "Example module that creates an extension type.",
                                 -1,
                                 UtilscMethods};
  m = PyModule_Create(&mdef);
  if ( m == NULL )
    return NULL;

  /* IMPORTANT: this must be called */
  import_array();

  UtilscError = PyErr_NewException("utilsc.error", NULL, NULL);
  Py_INCREF(UtilscError);
  PyModule_AddObject(m, "error", UtilscError);
  
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
  //PyImport_AppendInittab("utilsc", PyInit_covertreec);
    
  /* Pass argv[0] to the Python interpreter */
  Py_SetProgramName(program);

  /* Initialize the Python interpreter.  Required. */
  Py_Initialize();

  /* Add a static module */
  //PyInit_utilsc();
}

