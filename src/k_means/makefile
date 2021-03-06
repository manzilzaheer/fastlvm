CDIR = ../commons
IDIR = ../../lib
MKLROOT=/opt/intel/mkl

ARCH = core-avx2
DEBUG = -DNDEBUG

INTEL_CC = mpiicpc
INTEL_CFLAGS = $(DEBUG) -I"$(CDIR)" -I"$(IDIR)" -I"${MKLROOT}"/include -DMULTIMACHINE -O3 -march=$(ARCH) -std=c++14 -inline-factor=500 -no-inline-max-size -no-inline-max-total-size -use-intel-optimized-headers -parallel -qopt-prefetch=4 -qopt-mem-layout-trans=2 -pthread -c
INTEL_LFLAGS = $(DEBUG) -I"$(CDIR)" -I"$(IDIR)" -I"${MKLROOT}"/include -DMULTIMACHINE -O3 -march=$(ARCH) -std=c++14 -inline-factor=500 -no-inline-max-size -no-inline-max-total-size -use-intel-optimized-headers -parallel -qopt-prefetch=4 -qopt-mem-layout-trans=2 -pthread -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a -Wl,--end-group -lpthread -lm -ldl
INTEL_TFLAGS = -I"$(CDIR)" -fast -DNDEBUG -std=c++14 -inline-factor=500 -no-inline-max-size -no-inline-max-total-size -use-intel-optimized-headers -parallel -qopt-prefetch=4 -qopt-mem-layout-trans=3 -pthread 

GNU_CC = mpic++ -m64
GNU_CFLAGS = $(DEBUG) -I"$(CDIR)" -I"$(IDIR)" -DMULTIMACHINE -O3 -march=$(ARCH) -pthread -std=c++14 -c
GNU_LFLAGS = $(DEBUG) -I"$(CDIR)" -I"$(IDIR)" -DMULTIMACHINE -O3 -march=$(ARCH) -pthread -std=c++14

CC = $(GNU_CC)
CFLAGS = $(GNU_CFLAGS)
LFLAGS = $(GNU_LFLAGS)

SOURCEDIR = .
BUILDDIR = ../../build
EXECUTABLE = ../../dist/k_means

SOURCES = $(wildcard $(SOURCEDIR)/*.cpp)
OBJECTS = $(patsubst $(SOURCEDIR)/%.cpp,$(BUILDDIR)/%.o,$(SOURCES))

COMMONS = $(wildcard $(SOURCEDIR)/../commons/*.cpp)
COBJECT = $(patsubst $(SOURCEDIR)/../commons/%.cpp,$(BUILDDIR)/%.o,$(COMMONS))

CTSOURCE = $(wildcard $(SOURCEDIR)/../cover_tree/*.cpp)
CTOBJECT = $(patsubst $(SOURCEDIR)/../cover_tree/%.cpp,$(BUILDDIR)/%.o,$(CTSOURCE))

all: $(EXECUTABLE)

gcc: $(EXECUTABLE)

intel: CC=$(INTEL_CC)
intel: CFLAGS=$(INTEL_CFLAGS)
intel: LFLAGS=$(INTEL_LFLAGS)
intel: $(EXECUTABLE)
 
$(EXECUTABLE): $(COBJECT) $(CTOBJECT) $(OBJECTS)
	$(CC) $(LFLAGS) $^ -o $@

$(COBJECT): $(BUILDDIR)/%.o : $(SOURCEDIR)/../commons/%.cpp
	$(CC) $(CFLAGS) $< -o $@

$(CTOBJECT): $(BUILDDIR)/%.o : $(SOURCEDIR)/../cover_tree/%.cpp
	$(CC) $(CFLAGS) $< -o $@

$(OBJECTS): $(BUILDDIR)/%.o : $(SOURCEDIR)/%.cpp
	$(CC) $(CFLAGS) $< -o $@

inteltogether:
	$(INTEL_CC) $(INTEL_TFLAGS) $(SOURCES) $(COMMONS) $(CTSOURCE) -o $(EXECUTABLE)

clean:
	rm -rf $(BUILDDIR)/*
	rm -rf $(EXECUTABLE)
