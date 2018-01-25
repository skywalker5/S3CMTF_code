
CXX=g++


ifeq (unix,macos)
  LIB_FLAGS = -larmadillo -framework Accelerate
else
  LIB_FLAGS = -larmadillo -llapack -lblas -DARMA_DONT_USE_WRAPPER 
endif


OPT = -O2 -mcmodel=medium  -fopenmp

CXXFLAGS = $(DEBUG) $(FINAL) $(OPT) $(EXTRA_OPT)

all: S3CMTF-base S3CMTF-opt


S3CMTF-base: S3CMTF-base.cpp 
	$(CXX) $(CXXFLAGS)  -o $@  $< $(LIB_FLAGS)

S3CMTF-opt: S3CMTF-opt.cpp 
	$(CXX) $(CXXFLAGS)  -o $@  $< $(LIB_FLAGS) 


.PHONY: clean

clean:
	rm -f S3CMTF-base S3CMTF-opt