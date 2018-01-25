//#define EIGEN_USE_MKL_ALL        //uncomment if available
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN 

#include <Python.h>
#include "numpy/arrayobject.h"
#include "hdp.h"

int DataIO::corpus::from_python(PyObject* collection)
{  
  // free existing memory
  if(docs)
    delete[] docs;
    
  // allocate memory for corpus
  _size = PyList_GET_SIZE(collection);
  docs = new DataIO::document[_size];
  
  for(size_t i = 0; i < _size; ++i)
  {
    PyArrayObject* npy_doc = (PyArrayObject*) PyList_GET_ITEM(collection, i);
    if(PyArray_NDIM(npy_doc) != 1)
        throw std::runtime_error("Each document must a 1D numpy array");
    if(PyArray_TYPE(npy_doc) != NPY_UINT32)
        throw std::runtime_error("Each document must a uint32 numpy array");
    unsigned* fnp = (unsigned*) PyArray_GETPTR1(npy_doc, 0);
    docs[i].reassign(PyArray_SIZE(npy_doc), fnp);
  }
  
  return 0;
}

static PyObject *HdpcError;

static PyObject *new_hdpc(PyObject *self, PyObject *args)
{
  unsigned K = 100;
  unsigned iters = 1000;
  PyObject *in_vocab;

  if (!PyArg_ParseTuple(args,"IIO!:new_hdpc", &K, &iters, &PyList_Type, &in_vocab))
    return NULL;

  /* Convert existing vocab into a map for fast search and insertion */
  std::vector<std::string> word_map;
  unsigned V = (unsigned)PyList_GET_SIZE(in_vocab);
  for(unsigned w = 0; w < V; ++w)
      word_map.emplace_back((char*)PyUnicode_DATA(PyList_GET_ITEM(in_vocab, w)));

  utils::ParsedArgs params(K, iters, "aliasHDP");
  model* hdp = model::init(params, word_map, 0);
  size_t int_ptr = reinterpret_cast< size_t >(hdp);

  return Py_BuildValue("n", int_ptr);
}

static PyObject *delete_hdpc(PyObject *self, PyObject *args)
{
  model *obj;
  size_t int_ptr;
  PyObject* ext_pwk;

  if (!PyArg_ParseTuple(args,"nO:delete_hdpc", &int_ptr, &ext_pwk))
    return NULL;

  obj = reinterpret_cast< model* >(int_ptr);

  // check if external references
  if(ext_pwk != Py_None)
      obj->release();

  delete obj;

  return Py_BuildValue("n", int_ptr);
}


static PyObject *hdpc_fit(PyObject *self, PyObject *args)
{
  model *obj;
  size_t int_ptr;
  PyObject *in_trng;
  PyObject *in_test;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "nO!O!:hdpc_fit", &int_ptr, &PyList_Type, &in_trng, &PyList_Type, &in_test))
    return NULL;

  DataIO::corpus trngdata, testdata;
  trngdata.from_python(in_trng);
  testdata.from_python(in_test);
  std::cout << "Converted!" << std::endl;

  obj = reinterpret_cast< model * >(int_ptr);
  double score;
  try{
    score = obj->fit(trngdata, testdata);
  }
  catch(std::exception & e)
  {
    std::cout << e.what() << std::endl;
  }
  
  trngdata.release();
  testdata.release();

  return Py_BuildValue("d", score);
}

static PyObject *hdpc_evaluate(PyObject *self, PyObject *args) {

  model *obj;
  size_t int_ptr;
  PyObject *in_data;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "nO!:hdpc_evaluate", &int_ptr, &PyList_Type, &in_data))
    return NULL;

  DataIO::corpus data;
  data.from_python(in_data);

  obj = reinterpret_cast< model * >(int_ptr);
  double score = obj->evaluate(data);
  
  data.release();

  return Py_BuildValue("d", score);
}

static PyObject *hdpc_predict(PyObject *self, PyObject *args) {

  model *obj;
  size_t int_ptr;
  PyObject *in_data;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "nO!:hdpc_predict", &int_ptr, &PyList_Type, &in_data))
    return NULL;

  //obj = reinterpret_cast< model * >(int_ptr);
  //std::vector<unsigned> results = obj->predict(queryPts);

  //Py_INCREF(out_array);
  Py_RETURN_NONE;
}

static PyObject *hdpc_topic_matrix(PyObject *self, PyObject *args) {

  model *obj;
  size_t int_ptr;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "n:hdpc_topic_matrix", &int_ptr))
    return NULL;

  obj = reinterpret_cast< model * >(int_ptr);
  std::tuple<unsigned short, unsigned, double*> results = obj->get_topic_matrix();
  
  npy_intp numTopics = std::get<0>(results);
  npy_intp vocabSize = std::get<1>(results);
  //std::cout<<numPoints<<", "<<numDims<<std::endl;
  npy_intp dims[2] = {vocabSize, numTopics};
  PyObject *out_array = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT64, std::get<2>(results));
  PyArray_ENABLEFLAGS((PyArrayObject *)out_array, NPY_ARRAY_OWNDATA);

  //Py_INCREF(out_array);
  return out_array;
}

static PyObject *hdpc_top_words(PyObject *self, PyObject *args) {

  model *obj;
  size_t int_ptr;
  unsigned num_top;
  PyObject* out_data;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "nI:hdpc_top_words", &int_ptr, &num_top))
    return NULL;

  obj = reinterpret_cast< model * >(int_ptr);
  std::vector<std::vector<std::string>> results = obj->get_top_words(num_top);
  
  size_t K = results.size();
  out_data = PyList_New(K);
  for(size_t k = 0; k < K; ++k)
  {
    auto& t = results[k];
    size_t ntw = t.size();
    if(ntw-num_top)
        throw std::runtime_error("Returned object does not match requested number!");
    PyObject *topic_words = PyList_New(ntw);
    for(size_t w = 0; w < ntw; ++w)
        PyList_SET_ITEM(topic_words, w, PyUnicode_FromString(t[w].c_str()));
    PyList_SET_ITEM(out_data, k, topic_words);
  }

  //Py_INCREF(out_array);
  return out_data;
}

static PyObject *hdpc_serialize(PyObject *self, PyObject *args)
{
  model *obj;
  size_t int_ptr;

  if (!PyArg_ParseTuple(args,"n:hdpc_serialize", &int_ptr))
    return NULL;
  
  obj = reinterpret_cast< model * >(int_ptr);
  char* buff = obj->serialize();
  size_t len = obj->msg_size();
  
  return Py_BuildValue("y#", buff, len);
}

static PyObject *hdpc_deserialize(PyObject *self, PyObject *args)
{
  char* buff;
  size_t len;
  if (!PyArg_ParseTuple(args,"y#:hdpc_deserialize", &buff, &len))
    return NULL;
      
  model* hdp = new model();
  hdp->deserialize(buff);
  size_t int_ptr = reinterpret_cast< size_t >(hdp);

  return Py_BuildValue("n", int_ptr);
}

PyMODINIT_FUNC PyInit_hdpc(void)
{
  PyObject *m;
  static PyMethodDef HdpcMethods[] = {
    {"new", new_hdpc, METH_VARARGS, "Initialize a new HDP."},
    {"delete", delete_hdpc, METH_VARARGS, "Delete the HDP."},
    {"fit", hdpc_fit, METH_VARARGS, "Train HDP using Gibbs."},
    {"evaluate", hdpc_evaluate, METH_VARARGS, "Get perplexity."},
    {"predict", hdpc_predict, METH_VARARGS, "Get topic assignment."},
    {"topic_matrix", hdpc_topic_matrix, METH_VARARGS, "Get the learnt word|topic distributions."},
    {"top_words", hdpc_top_words, METH_VARARGS, "Get the top words for all topics."},
    {"serialize", hdpc_serialize, METH_VARARGS, "Serialize the current HDP."},
    {"deserialize", hdpc_deserialize, METH_VARARGS, "Construct a HDP from deserializing."},
    {NULL, NULL, 0, NULL}
  };
  static struct PyModuleDef mdef = {PyModuleDef_HEAD_INIT,
                                 "hdpc",
                                 "Example module that creates an extension type.",
                                 -1,
                                 HdpcMethods};
  m = PyModule_Create(&mdef);
  if ( m == NULL )
    return NULL;

  /* IMPORTANT: this must be called */
  import_array();

  HdpcError = PyErr_NewException("hdpc.error", NULL, NULL);
  Py_INCREF(HdpcError);
  PyModule_AddObject(m, "error", HdpcError);
  
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
  //PyImport_AppendInittab("hdpc", PyInit_covertreec);
    
  /* Pass argv[0] to the Python interpreter */
  Py_SetProgramName(program);

  /* Initialize the Python interpreter.  Required. */
  Py_Initialize();

  /* Add a static module */
  //PyInit_hdpc();
}

