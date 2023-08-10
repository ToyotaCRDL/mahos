# This Makefile is for development on Linux or compatible environment.

PYTHON_INCLUDES := $(shell python3-config --includes)
# PYBIND11_INCLUDES := -I/usr/include/pybind11
# INCLUDES := $(PYTHON_INCLUDES) $(PYBIND11_INCLUDES)
INCLUDES := $(PYTHON_INCLUDES)
FLAGS := -O3 -Wall -shared -std=c++11 -fPIC
NAME := cqdyne_analyzer
EXT := $(shell python3-config --extension-suffix)

gcc:
	g++ $(FLAGS) $(INCLUDES) $(NAME).cc -o $(NAME)$(EXT)

clang:
	clang++ -stdlib=libc++ $(FLAGS) $(INCLUDES) $(NAME).cc -o $(NAME)$(EXT)

clean:
	$(RM) $(NAME)$(EXT)
