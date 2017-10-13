//#define EIGEN_USE_MKL_ALL        //uncomment if available
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN 

#include <Python.h>
#include "numpy/arrayobject.h"
#include "glda.h"

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

static PyObject *GldacError;

static PyObject *new_gldac(PyObject *self, PyObject *args)
{
  unsigned K = 100;
  unsigned iters = 1000;
  PyObject *in_vocab;
  PyArrayObject *in_array;

  if (!PyArg_ParseTuple(args,"IIO!O!:new_gldac", &K, &iters, &PyList_Type, &in_vocab, &PyArray_Type, &in_array))
    return NULL;

  /* Convert existing vocab into a map for fast search and insertion */
  std::vector<std::string> word_map;
  unsigned V = (unsigned)PyList_GET_SIZE(in_vocab);
  for(unsigned w = 0; w < V; ++w)
      word_map.emplace_back((char*)PyUnicode_DATA(PyList_GET_ITEM(in_vocab, w)));

  /* Map word vectors to eigen matrix */
  npy_intp numWords = PyArray_DIM(in_array, 0);
  npy_intp numDims = PyArray_DIM(in_array, 1);
  //std::cout<<numPoints<<", "<<numDims<<std::endl;
  if ((unsigned)numWords != V)
  {
      throw std::runtime_error("Size of word map and word vectors do not match!");
  }
  npy_intp idx[2] = {0, 0};
  double * fnp = reinterpret_cast< double * >( PyArray_GetPtr(in_array, idx) );
  Eigen::Map<Eigen::MatrixXd> word_vec(fnp, numDims, numWords);

  utils::ParsedArgs params(K, iters, "scaGLDA");
  model* glda = model::init(params, word_map, word_vec, 0);
  size_t int_ptr = reinterpret_cast< size_t >(glda);

  return Py_BuildValue("n", int_ptr);
}

static PyObject *delete_gldac(PyObject *self, PyObject *args)
{
  model *obj;
  size_t int_ptr;
  PyObject* ext_pwk;

  if (!PyArg_ParseTuple(args,"nO:delete_gldac", &int_ptr, &ext_pwk))
    return NULL;

  obj = reinterpret_cast< model* >(int_ptr);

  delete obj;

  return Py_BuildValue("n", int_ptr);
}


static PyObject *gldac_fit(PyObject *self, PyObject *args)
{
  model *obj;
  size_t int_ptr;
  PyObject *in_trng;
  PyObject *in_test;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "nO!O!:gldac_fit", &int_ptr, &PyList_Type, &in_trng, &PyList_Type, &in_test))
    return NULL;

  DataIO::corpus trngdata, testdata;
  trngdata.from_python(in_trng);
  testdata.from_python(in_test);
  std::cout << "Converted!" << std::endl;

  obj = reinterpret_cast< model * >(int_ptr);
  double score = obj->fit(trngdata, testdata);
  
  trngdata.release();
  testdata.release();

  return Py_BuildValue("d", score);
}

static PyObject *gldac_evaluate(PyObject *self, PyObject *args) {

  model *obj;
  size_t int_ptr;
  PyObject *in_data;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "nO!:gldac_evaluate", &int_ptr, &PyList_Type, &in_data))
    return NULL;

  DataIO::corpus data;
  data.from_python(in_data);

  obj = reinterpret_cast< model * >(int_ptr);
  double score = obj->evaluate(data);
  
  data.release();

  return Py_BuildValue("d", score);
}

static PyObject *gldac_predict(PyObject *self, PyObject *args) {

  model *obj;
  size_t int_ptr;
  PyObject *in_data;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "nO!:gldac_predict", &int_ptr, &PyList_Type, &in_data))
    return NULL;

  //obj = reinterpret_cast< model * >(int_ptr);
  //std::vector<unsigned> results = obj->predict(queryPts);

  //Py_INCREF(out_array);
  Py_RETURN_NONE;
}

static PyObject *gldac_topic_matrix(PyObject *self, PyObject *args) {

  model *obj;
  size_t int_ptr;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "n:gldac_topic_matrix", &int_ptr))
    return NULL;

  obj = reinterpret_cast< model * >(int_ptr);
  std::pair<Eigen::Map<Eigen::MatrixXd>, Eigen::Map<Eigen::MatrixXd>> results = obj->get_topic_matrix();
  
  npy_intp numTopics = results.first.cols();
  npy_intp vocabSize = results.first.rows();
  //std::cout<<numPoints<<", "<<numDims<<std::endl;
  npy_intp dims[2] = {vocabSize, numTopics};

  // means
  PyObject *out_mean = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT64, results.first.data());
  PyArray_ENABLEFLAGS((PyArrayObject *)out_mean, NPY_ARRAY_OWNDATA);

  // variance
  PyObject *out_var = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT64, results.second.data());
  PyArray_ENABLEFLAGS((PyArrayObject *)out_var, NPY_ARRAY_OWNDATA);

  return Py_BuildValue("NN", out_mean, out_var);
}

static PyObject *gldac_top_words(PyObject *self, PyObject *args) {

  model *obj;
  size_t int_ptr;
  unsigned num_top;
  PyObject* out_data;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "nI:gldac_top_words", &int_ptr, &num_top))
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

static PyObject *gldac_serialize(PyObject *self, PyObject *args)
{
  model *obj;
  size_t int_ptr;

  if (!PyArg_ParseTuple(args,"n:gldac_serialize", &int_ptr))
    return NULL;
  
  obj = reinterpret_cast< model * >(int_ptr);
  char* buff = obj->serialize();
  size_t len = obj->msg_size();
  
  return Py_BuildValue("y#", buff, len);
}

static PyObject *gldac_deserialize(PyObject *self, PyObject *args)
{
  char* buff;
  size_t len;
  if (!PyArg_ParseTuple(args,"y#:gldac_deserialize", &buff, &len))
    return NULL;
      
  model* glda = new model();
  glda->deserialize(buff);
  size_t int_ptr = reinterpret_cast< size_t >(glda);

  return Py_BuildValue("n", int_ptr);
}

static PyObject *gldac_word_vec(PyObject *self, PyObject *args)
{
  model *obj;
  size_t int_ptr;

  if (!PyArg_ParseTuple(args,"n:gldac_word_vec", &int_ptr))
    return NULL;
  
  obj = reinterpret_cast< model * >(int_ptr);
  Eigen::Map<Eigen::MatrixXd>* results = obj->get_word_vec();
  
  npy_intp vocabSize = results->cols();
  npy_intp embbedDim = results->rows();
  //std::cout<<numPoints<<", "<<numDims<<std::endl;
  npy_intp dims[2] = {embbedDim, vocabSize};

  // word vecs
  PyObject *out_array = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT64, results->data());
  PyArray_ENABLEFLAGS((PyArrayObject *)out_array, NPY_ARRAY_OWNDATA);
  
  return out_array;
}

PyMODINIT_FUNC PyInit_gldac(void)
{
  PyObject *m;
  static PyMethodDef GldacMethods[] = {
    {"new", new_gldac, METH_VARARGS, "Initialize a new GLDA."},
    {"delete", delete_gldac, METH_VARARGS, "Delete the GLDA."},
    {"fit", gldac_fit, METH_VARARGS, "Train GLDA using ESCA."},
    {"evaluate", gldac_evaluate, METH_VARARGS, "Get perplexity."},
    {"predict", gldac_predict, METH_VARARGS, "Get topic assignment."},
    {"topic_matrix", gldac_topic_matrix, METH_VARARGS, "Get the learnt word|topic distributions."},
    {"top_words", gldac_top_words, METH_VARARGS, "Get the top words for all topics."},
    {"serialize", gldac_serialize, METH_VARARGS, "Serialize the current GLDA."},
    {"deserialize", gldac_deserialize, METH_VARARGS, "Construct a GLDA from deserializing."},
    {"word_vec", gldac_word_vec, METH_VARARGS, "Get word vectors."},
    {NULL, NULL, 0, NULL}
  };
  static struct PyModuleDef mdef = {PyModuleDef_HEAD_INIT,
                                 "gldac",
                                 "Example module that creates an extension type.",
                                 -1,
                                 GldacMethods};
  m = PyModule_Create(&mdef);
  if ( m == NULL )
    return NULL;

  /* IMPORTANT: this must be called */
  import_array();

  GldacError = PyErr_NewException("ldac.error", NULL, NULL);
  Py_INCREF(GldacError);
  PyModule_AddObject(m, "error", GldacError);
  
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
  //PyImport_AppendInittab("ldac", PyInit_covertreec);
    
  /* Pass argv[0] to the Python interpreter */
  Py_SetProgramName(program);

  /* Initialize the Python interpreter.  Required. */
  Py_Initialize();

  /* Add a static module */
  //PyInit_gldac();
}

