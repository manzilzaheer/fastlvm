from distutils.core import setup, Extension
import numpy as np

covertreec_module = Extension('covertreec',
        sources = ['src/cover_tree/covertreecmodule.cxx', 'src/commons/utils.cpp',  'src/cover_tree/cover_tree.cpp'],
        include_dirs=[np.get_include(), 'lib/'],
        extra_compile_args=['-march=corei7-avx', '-pthread', '-std=c++14'],
        extra_link_args=['-march=corei7-avx', '-pthread', '-std=c++14']
)

kmeans_module = Extension('kmeansc',
        sources = ['src/k_means/kmeanscmodule.cxx', 'src/commons/utils.cpp', 'src/commons/suff_stats.cpp', 'src/cover_tree/cover_tree.cpp', 'src/k_means/model.cpp', 'src/k_means/fmm.cpp'],
        include_dirs=[np.get_include(), 'lib/'],
        extra_compile_args=['-march=corei7-avx', '-pthread', '-std=c++14'],
        extra_link_args=['-march=corei7-avx', '-pthread', '-std=c++14']
)

gmm_module = Extension('gmmc',
        sources = ['src/gmm/gmmcmodule.cxx', 'src/commons/utils.cpp', 'src/commons/vose.cpp', 'src/commons/suff_stats.cpp', 'src/gmm/model.cpp', 'src/gmm/fmm.cpp'],
        include_dirs=[np.get_include(), 'lib/'],
        extra_compile_args=['-march=corei7-avx', '-pthread', '-std=c++14'],
        extra_link_args=['-march=corei7-avx', '-pthread', '-std=c++14']
)

lda_module = Extension('ldac',
        sources = ['src/lda/ldacmodule.cxx', 'src/commons/utils.cpp', 'src/commons/stringtokenizer.cpp', 'src/commons/dataio.cpp', 'src/commons/vose.cpp', 'src/lda/model.cpp', 'src/lda/lda.cpp'],
        include_dirs=[np.get_include(), 'lib/'],
        extra_compile_args=['-march=corei7-avx', '-pthread', '-std=c++14'],
        extra_link_args=['-march=corei7-avx', '-pthread', '-std=c++14']
)

hdp_module = Extension('hdpc',
        sources = ['src/hdp/hdpcmodule.cxx', 'src/commons/utils.cpp', 'src/commons/stringtokenizer.cpp', 'src/commons/dataio.cpp', 'src/commons/vose.cpp', 'src/commons/stirling.cpp', 'src/hdp/model.cpp', 'src/hdp/hdp.cpp'],
        include_dirs=[np.get_include(), 'lib/'],
        extra_compile_args=['-march=corei7-avx', '-pthread', '-std=c++14'],
        extra_link_args=['-march=corei7-avx', '-pthread', '-std=c++14']
)

glda_module = Extension('gldac',
        sources = ['src/glda/gldacmodule.cxx', 'src/commons/utils.cpp', 'src/commons/stringtokenizer.cpp', 'src/commons/dataio.cpp', 'src/commons/vose.cpp', 'src/commons/suff_stats.cpp', 'src/glda/model.cpp', 'src/glda/glda.cpp'],
        include_dirs=[np.get_include(), 'lib/'],
        extra_compile_args=['-march=corei7-avx', '-pthread', '-std=c++14'],
        extra_link_args=['-march=corei7-avx', '-pthread', '-std=c++14']
)

utils_module = Extension('utilsc',
        sources = ['src/commons/utilscmodule.cxx', 'src/commons/utils.cpp', 'src/commons/dataio.cpp', 'src/commons/stirling.cpp', 'src/commons/stringtokenizer.cpp'],
        include_dirs=[np.get_include(), 'lib/'],
        extra_compile_args=['-march=corei7-avx', '-pthread', '-std=c++14'],
        extra_link_args=['-march=corei7-avx', '-pthread', '-std=c++14']
)

setup ( name = 'fastlvm',
    version = '1.0',
    description = 'fastlvm -- fast search, clustering, and mixture modelling',
    ext_modules = [ covertreec_module, kmeans_module, gmm_module, lda_module, hdp_module, glda_module, utils_module ],
    packages = ['fastlvm']
)
