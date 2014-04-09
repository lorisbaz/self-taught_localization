
PROJECT := grabcutobf
#$(info $(PROJECT))

# name of the final binaries
BINS := grabcutobf_test.bin

# name of the final Python modules
PYMODULES := grabcutobf.so

# source and header files
#CXX_SRCS := $(shell find . -maxdepth 1 -name "*.cpp" | cut -c 3-)
#HXX_SRCS := $(shell find . -maxdepth 1 -name "*.hpp" | cut -c 3-)
CXX_SRCS := grabcutobf.cpp grabcutobf_test.cpp

# object names
CXX_OBJS := ${CXX_SRCS:.cpp=.o}
OBJS := $(CXX_OBJS)

# compilation directives
CXX := g++
INCLUDE_DIRS +=
LIBRARY_DIRS += 
LIBRARIES := opencv_core opencv_highgui opencv_imgproc
CXXFLAGS += -std=c++0x -Wall -fPIC \
            $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))
CXXFLAGS += -g
LDFLAGS += $(foreach librarydir,$(LIBRARY_DIRS),-L$(librarydir)) \
	   $(foreach library,$(LIBRARIES),-l$(library))

.PHONY: all clean

.SUFFIXES: .o .cpp

all: $(BINS)

$(BINS): %.bin : $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS) $(CXXFLAGS)

$(OBJS): $(CXX_SRCS)

pymodule:
	python setup.py install --install-lib ./

clean:
	@- $(RM) $(BINS) $(OBJS) $(PYMODULES)
	@- $(RM) -r build

