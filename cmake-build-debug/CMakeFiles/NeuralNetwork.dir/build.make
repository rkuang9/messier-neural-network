# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.20

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "C:\Program Files\JetBrains\CLion 2021.2.2\bin\cmake\win\bin\cmake.exe"

# The command to remove a file.
RM = "C:\Program Files\JetBrains\CLion 2021.2.2\bin\cmake\win\bin\cmake.exe" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\Users\R\Desktop\NeuralNetwork

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\R\Desktop\NeuralNetwork\cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/NeuralNetwork.dir/depend.make
# Include the progress variables for this target.
include CMakeFiles/NeuralNetwork.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/NeuralNetwork.dir/flags.make

CMakeFiles/NeuralNetwork.dir/main.cpp.obj: CMakeFiles/NeuralNetwork.dir/flags.make
CMakeFiles/NeuralNetwork.dir/main.cpp.obj: CMakeFiles/NeuralNetwork.dir/includes_CXX.rsp
CMakeFiles/NeuralNetwork.dir/main.cpp.obj: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\R\Desktop\NeuralNetwork\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/NeuralNetwork.dir/main.cpp.obj"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\NeuralNetwork.dir\main.cpp.obj -c C:\Users\R\Desktop\NeuralNetwork\main.cpp

CMakeFiles/NeuralNetwork.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/NeuralNetwork.dir/main.cpp.i"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\R\Desktop\NeuralNetwork\main.cpp > CMakeFiles\NeuralNetwork.dir\main.cpp.i

CMakeFiles/NeuralNetwork.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/NeuralNetwork.dir/main.cpp.s"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\R\Desktop\NeuralNetwork\main.cpp -o CMakeFiles\NeuralNetwork.dir\main.cpp.s

CMakeFiles/NeuralNetwork.dir/src/activation/sigmoid.cpp.obj: CMakeFiles/NeuralNetwork.dir/flags.make
CMakeFiles/NeuralNetwork.dir/src/activation/sigmoid.cpp.obj: CMakeFiles/NeuralNetwork.dir/includes_CXX.rsp
CMakeFiles/NeuralNetwork.dir/src/activation/sigmoid.cpp.obj: ../src/activation/sigmoid.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\R\Desktop\NeuralNetwork\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/NeuralNetwork.dir/src/activation/sigmoid.cpp.obj"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\NeuralNetwork.dir\src\activation\sigmoid.cpp.obj -c C:\Users\R\Desktop\NeuralNetwork\src\activation\sigmoid.cpp

CMakeFiles/NeuralNetwork.dir/src/activation/sigmoid.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/NeuralNetwork.dir/src/activation/sigmoid.cpp.i"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\R\Desktop\NeuralNetwork\src\activation\sigmoid.cpp > CMakeFiles\NeuralNetwork.dir\src\activation\sigmoid.cpp.i

CMakeFiles/NeuralNetwork.dir/src/activation/sigmoid.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/NeuralNetwork.dir/src/activation/sigmoid.cpp.s"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\R\Desktop\NeuralNetwork\src\activation\sigmoid.cpp -o CMakeFiles\NeuralNetwork.dir\src\activation\sigmoid.cpp.s

CMakeFiles/NeuralNetwork.dir/src/activation/tanh.cpp.obj: CMakeFiles/NeuralNetwork.dir/flags.make
CMakeFiles/NeuralNetwork.dir/src/activation/tanh.cpp.obj: CMakeFiles/NeuralNetwork.dir/includes_CXX.rsp
CMakeFiles/NeuralNetwork.dir/src/activation/tanh.cpp.obj: ../src/activation/tanh.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\R\Desktop\NeuralNetwork\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/NeuralNetwork.dir/src/activation/tanh.cpp.obj"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\NeuralNetwork.dir\src\activation\tanh.cpp.obj -c C:\Users\R\Desktop\NeuralNetwork\src\activation\tanh.cpp

CMakeFiles/NeuralNetwork.dir/src/activation/tanh.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/NeuralNetwork.dir/src/activation/tanh.cpp.i"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\R\Desktop\NeuralNetwork\src\activation\tanh.cpp > CMakeFiles\NeuralNetwork.dir\src\activation\tanh.cpp.i

CMakeFiles/NeuralNetwork.dir/src/activation/tanh.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/NeuralNetwork.dir/src/activation/tanh.cpp.s"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\R\Desktop\NeuralNetwork\src\activation\tanh.cpp -o CMakeFiles\NeuralNetwork.dir\src\activation\tanh.cpp.s

CMakeFiles/NeuralNetwork.dir/src/activation/relu.cpp.obj: CMakeFiles/NeuralNetwork.dir/flags.make
CMakeFiles/NeuralNetwork.dir/src/activation/relu.cpp.obj: CMakeFiles/NeuralNetwork.dir/includes_CXX.rsp
CMakeFiles/NeuralNetwork.dir/src/activation/relu.cpp.obj: ../src/activation/relu.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\R\Desktop\NeuralNetwork\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/NeuralNetwork.dir/src/activation/relu.cpp.obj"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\NeuralNetwork.dir\src\activation\relu.cpp.obj -c C:\Users\R\Desktop\NeuralNetwork\src\activation\relu.cpp

CMakeFiles/NeuralNetwork.dir/src/activation/relu.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/NeuralNetwork.dir/src/activation/relu.cpp.i"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\R\Desktop\NeuralNetwork\src\activation\relu.cpp > CMakeFiles\NeuralNetwork.dir\src\activation\relu.cpp.i

CMakeFiles/NeuralNetwork.dir/src/activation/relu.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/NeuralNetwork.dir/src/activation/relu.cpp.s"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\R\Desktop\NeuralNetwork\src\activation\relu.cpp -o CMakeFiles\NeuralNetwork.dir\src\activation\relu.cpp.s

CMakeFiles/NeuralNetwork.dir/src/activation/leakyrelu.cpp.obj: CMakeFiles/NeuralNetwork.dir/flags.make
CMakeFiles/NeuralNetwork.dir/src/activation/leakyrelu.cpp.obj: CMakeFiles/NeuralNetwork.dir/includes_CXX.rsp
CMakeFiles/NeuralNetwork.dir/src/activation/leakyrelu.cpp.obj: ../src/activation/leakyrelu.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\R\Desktop\NeuralNetwork\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/NeuralNetwork.dir/src/activation/leakyrelu.cpp.obj"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\NeuralNetwork.dir\src\activation\leakyrelu.cpp.obj -c C:\Users\R\Desktop\NeuralNetwork\src\activation\leakyrelu.cpp

CMakeFiles/NeuralNetwork.dir/src/activation/leakyrelu.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/NeuralNetwork.dir/src/activation/leakyrelu.cpp.i"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\R\Desktop\NeuralNetwork\src\activation\leakyrelu.cpp > CMakeFiles\NeuralNetwork.dir\src\activation\leakyrelu.cpp.i

CMakeFiles/NeuralNetwork.dir/src/activation/leakyrelu.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/NeuralNetwork.dir/src/activation/leakyrelu.cpp.s"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\R\Desktop\NeuralNetwork\src\activation\leakyrelu.cpp -o CMakeFiles\NeuralNetwork.dir\src\activation\leakyrelu.cpp.s

CMakeFiles/NeuralNetwork.dir/src/activation/softmax.cpp.obj: CMakeFiles/NeuralNetwork.dir/flags.make
CMakeFiles/NeuralNetwork.dir/src/activation/softmax.cpp.obj: CMakeFiles/NeuralNetwork.dir/includes_CXX.rsp
CMakeFiles/NeuralNetwork.dir/src/activation/softmax.cpp.obj: ../src/activation/softmax.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\R\Desktop\NeuralNetwork\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/NeuralNetwork.dir/src/activation/softmax.cpp.obj"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\NeuralNetwork.dir\src\activation\softmax.cpp.obj -c C:\Users\R\Desktop\NeuralNetwork\src\activation\softmax.cpp

CMakeFiles/NeuralNetwork.dir/src/activation/softmax.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/NeuralNetwork.dir/src/activation/softmax.cpp.i"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\R\Desktop\NeuralNetwork\src\activation\softmax.cpp > CMakeFiles\NeuralNetwork.dir\src\activation\softmax.cpp.i

CMakeFiles/NeuralNetwork.dir/src/activation/softmax.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/NeuralNetwork.dir/src/activation/softmax.cpp.s"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\R\Desktop\NeuralNetwork\src\activation\softmax.cpp -o CMakeFiles\NeuralNetwork.dir\src\activation\softmax.cpp.s

CMakeFiles/NeuralNetwork.dir/src/activation/swish.cpp.obj: CMakeFiles/NeuralNetwork.dir/flags.make
CMakeFiles/NeuralNetwork.dir/src/activation/swish.cpp.obj: CMakeFiles/NeuralNetwork.dir/includes_CXX.rsp
CMakeFiles/NeuralNetwork.dir/src/activation/swish.cpp.obj: ../src/activation/swish.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\R\Desktop\NeuralNetwork\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/NeuralNetwork.dir/src/activation/swish.cpp.obj"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\NeuralNetwork.dir\src\activation\swish.cpp.obj -c C:\Users\R\Desktop\NeuralNetwork\src\activation\swish.cpp

CMakeFiles/NeuralNetwork.dir/src/activation/swish.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/NeuralNetwork.dir/src/activation/swish.cpp.i"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\R\Desktop\NeuralNetwork\src\activation\swish.cpp > CMakeFiles\NeuralNetwork.dir\src\activation\swish.cpp.i

CMakeFiles/NeuralNetwork.dir/src/activation/swish.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/NeuralNetwork.dir/src/activation/swish.cpp.s"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\R\Desktop\NeuralNetwork\src\activation\swish.cpp -o CMakeFiles\NeuralNetwork.dir\src\activation\swish.cpp.s

CMakeFiles/NeuralNetwork.dir/src/activation/softplus.cpp.obj: CMakeFiles/NeuralNetwork.dir/flags.make
CMakeFiles/NeuralNetwork.dir/src/activation/softplus.cpp.obj: CMakeFiles/NeuralNetwork.dir/includes_CXX.rsp
CMakeFiles/NeuralNetwork.dir/src/activation/softplus.cpp.obj: ../src/activation/softplus.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\R\Desktop\NeuralNetwork\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/NeuralNetwork.dir/src/activation/softplus.cpp.obj"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\NeuralNetwork.dir\src\activation\softplus.cpp.obj -c C:\Users\R\Desktop\NeuralNetwork\src\activation\softplus.cpp

CMakeFiles/NeuralNetwork.dir/src/activation/softplus.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/NeuralNetwork.dir/src/activation/softplus.cpp.i"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\R\Desktop\NeuralNetwork\src\activation\softplus.cpp > CMakeFiles\NeuralNetwork.dir\src\activation\softplus.cpp.i

CMakeFiles/NeuralNetwork.dir/src/activation/softplus.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/NeuralNetwork.dir/src/activation/softplus.cpp.s"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\R\Desktop\NeuralNetwork\src\activation\softplus.cpp -o CMakeFiles\NeuralNetwork.dir\src\activation\softplus.cpp.s

CMakeFiles/NeuralNetwork.dir/src/activation/softsign.cpp.obj: CMakeFiles/NeuralNetwork.dir/flags.make
CMakeFiles/NeuralNetwork.dir/src/activation/softsign.cpp.obj: CMakeFiles/NeuralNetwork.dir/includes_CXX.rsp
CMakeFiles/NeuralNetwork.dir/src/activation/softsign.cpp.obj: ../src/activation/softsign.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\R\Desktop\NeuralNetwork\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object CMakeFiles/NeuralNetwork.dir/src/activation/softsign.cpp.obj"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\NeuralNetwork.dir\src\activation\softsign.cpp.obj -c C:\Users\R\Desktop\NeuralNetwork\src\activation\softsign.cpp

CMakeFiles/NeuralNetwork.dir/src/activation/softsign.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/NeuralNetwork.dir/src/activation/softsign.cpp.i"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\R\Desktop\NeuralNetwork\src\activation\softsign.cpp > CMakeFiles\NeuralNetwork.dir\src\activation\softsign.cpp.i

CMakeFiles/NeuralNetwork.dir/src/activation/softsign.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/NeuralNetwork.dir/src/activation/softsign.cpp.s"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\R\Desktop\NeuralNetwork\src\activation\softsign.cpp -o CMakeFiles\NeuralNetwork.dir\src\activation\softsign.cpp.s

CMakeFiles/NeuralNetwork.dir/src/weights/xavier.cpp.obj: CMakeFiles/NeuralNetwork.dir/flags.make
CMakeFiles/NeuralNetwork.dir/src/weights/xavier.cpp.obj: CMakeFiles/NeuralNetwork.dir/includes_CXX.rsp
CMakeFiles/NeuralNetwork.dir/src/weights/xavier.cpp.obj: ../src/weights/xavier.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\R\Desktop\NeuralNetwork\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object CMakeFiles/NeuralNetwork.dir/src/weights/xavier.cpp.obj"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\NeuralNetwork.dir\src\weights\xavier.cpp.obj -c C:\Users\R\Desktop\NeuralNetwork\src\weights\xavier.cpp

CMakeFiles/NeuralNetwork.dir/src/weights/xavier.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/NeuralNetwork.dir/src/weights/xavier.cpp.i"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\R\Desktop\NeuralNetwork\src\weights\xavier.cpp > CMakeFiles\NeuralNetwork.dir\src\weights\xavier.cpp.i

CMakeFiles/NeuralNetwork.dir/src/weights/xavier.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/NeuralNetwork.dir/src/weights/xavier.cpp.s"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\R\Desktop\NeuralNetwork\src\weights\xavier.cpp -o CMakeFiles\NeuralNetwork.dir\src\weights\xavier.cpp.s

CMakeFiles/NeuralNetwork.dir/src/weights/he.cpp.obj: CMakeFiles/NeuralNetwork.dir/flags.make
CMakeFiles/NeuralNetwork.dir/src/weights/he.cpp.obj: CMakeFiles/NeuralNetwork.dir/includes_CXX.rsp
CMakeFiles/NeuralNetwork.dir/src/weights/he.cpp.obj: ../src/weights/he.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\R\Desktop\NeuralNetwork\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object CMakeFiles/NeuralNetwork.dir/src/weights/he.cpp.obj"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\NeuralNetwork.dir\src\weights\he.cpp.obj -c C:\Users\R\Desktop\NeuralNetwork\src\weights\he.cpp

CMakeFiles/NeuralNetwork.dir/src/weights/he.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/NeuralNetwork.dir/src/weights/he.cpp.i"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\R\Desktop\NeuralNetwork\src\weights\he.cpp > CMakeFiles\NeuralNetwork.dir\src\weights\he.cpp.i

CMakeFiles/NeuralNetwork.dir/src/weights/he.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/NeuralNetwork.dir/src/weights/he.cpp.s"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\R\Desktop\NeuralNetwork\src\weights\he.cpp -o CMakeFiles\NeuralNetwork.dir\src\weights\he.cpp.s

CMakeFiles/NeuralNetwork.dir/src/weights/normal.cpp.obj: CMakeFiles/NeuralNetwork.dir/flags.make
CMakeFiles/NeuralNetwork.dir/src/weights/normal.cpp.obj: CMakeFiles/NeuralNetwork.dir/includes_CXX.rsp
CMakeFiles/NeuralNetwork.dir/src/weights/normal.cpp.obj: ../src/weights/normal.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\R\Desktop\NeuralNetwork\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building CXX object CMakeFiles/NeuralNetwork.dir/src/weights/normal.cpp.obj"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\NeuralNetwork.dir\src\weights\normal.cpp.obj -c C:\Users\R\Desktop\NeuralNetwork\src\weights\normal.cpp

CMakeFiles/NeuralNetwork.dir/src/weights/normal.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/NeuralNetwork.dir/src/weights/normal.cpp.i"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\R\Desktop\NeuralNetwork\src\weights\normal.cpp > CMakeFiles\NeuralNetwork.dir\src\weights\normal.cpp.i

CMakeFiles/NeuralNetwork.dir/src/weights/normal.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/NeuralNetwork.dir/src/weights/normal.cpp.s"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\R\Desktop\NeuralNetwork\src\weights\normal.cpp -o CMakeFiles\NeuralNetwork.dir\src\weights\normal.cpp.s

CMakeFiles/NeuralNetwork.dir/src/loss/quadratic.cpp.obj: CMakeFiles/NeuralNetwork.dir/flags.make
CMakeFiles/NeuralNetwork.dir/src/loss/quadratic.cpp.obj: CMakeFiles/NeuralNetwork.dir/includes_CXX.rsp
CMakeFiles/NeuralNetwork.dir/src/loss/quadratic.cpp.obj: ../src/loss/quadratic.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\R\Desktop\NeuralNetwork\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Building CXX object CMakeFiles/NeuralNetwork.dir/src/loss/quadratic.cpp.obj"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\NeuralNetwork.dir\src\loss\quadratic.cpp.obj -c C:\Users\R\Desktop\NeuralNetwork\src\loss\quadratic.cpp

CMakeFiles/NeuralNetwork.dir/src/loss/quadratic.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/NeuralNetwork.dir/src/loss/quadratic.cpp.i"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\R\Desktop\NeuralNetwork\src\loss\quadratic.cpp > CMakeFiles\NeuralNetwork.dir\src\loss\quadratic.cpp.i

CMakeFiles/NeuralNetwork.dir/src/loss/quadratic.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/NeuralNetwork.dir/src/loss/quadratic.cpp.s"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\R\Desktop\NeuralNetwork\src\loss\quadratic.cpp -o CMakeFiles\NeuralNetwork.dir\src\loss\quadratic.cpp.s

CMakeFiles/NeuralNetwork.dir/src/loss/cross_entropy.cpp.obj: CMakeFiles/NeuralNetwork.dir/flags.make
CMakeFiles/NeuralNetwork.dir/src/loss/cross_entropy.cpp.obj: CMakeFiles/NeuralNetwork.dir/includes_CXX.rsp
CMakeFiles/NeuralNetwork.dir/src/loss/cross_entropy.cpp.obj: ../src/loss/cross_entropy.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\R\Desktop\NeuralNetwork\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_14) "Building CXX object CMakeFiles/NeuralNetwork.dir/src/loss/cross_entropy.cpp.obj"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\NeuralNetwork.dir\src\loss\cross_entropy.cpp.obj -c C:\Users\R\Desktop\NeuralNetwork\src\loss\cross_entropy.cpp

CMakeFiles/NeuralNetwork.dir/src/loss/cross_entropy.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/NeuralNetwork.dir/src/loss/cross_entropy.cpp.i"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\R\Desktop\NeuralNetwork\src\loss\cross_entropy.cpp > CMakeFiles\NeuralNetwork.dir\src\loss\cross_entropy.cpp.i

CMakeFiles/NeuralNetwork.dir/src/loss/cross_entropy.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/NeuralNetwork.dir/src/loss/cross_entropy.cpp.s"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\R\Desktop\NeuralNetwork\src\loss\cross_entropy.cpp -o CMakeFiles\NeuralNetwork.dir\src\loss\cross_entropy.cpp.s

CMakeFiles/NeuralNetwork.dir/src/loss/kldivergence.cpp.obj: CMakeFiles/NeuralNetwork.dir/flags.make
CMakeFiles/NeuralNetwork.dir/src/loss/kldivergence.cpp.obj: CMakeFiles/NeuralNetwork.dir/includes_CXX.rsp
CMakeFiles/NeuralNetwork.dir/src/loss/kldivergence.cpp.obj: ../src/loss/kldivergence.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\R\Desktop\NeuralNetwork\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_15) "Building CXX object CMakeFiles/NeuralNetwork.dir/src/loss/kldivergence.cpp.obj"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\NeuralNetwork.dir\src\loss\kldivergence.cpp.obj -c C:\Users\R\Desktop\NeuralNetwork\src\loss\kldivergence.cpp

CMakeFiles/NeuralNetwork.dir/src/loss/kldivergence.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/NeuralNetwork.dir/src/loss/kldivergence.cpp.i"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\R\Desktop\NeuralNetwork\src\loss\kldivergence.cpp > CMakeFiles\NeuralNetwork.dir\src\loss\kldivergence.cpp.i

CMakeFiles/NeuralNetwork.dir/src/loss/kldivergence.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/NeuralNetwork.dir/src/loss/kldivergence.cpp.s"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\R\Desktop\NeuralNetwork\src\loss\kldivergence.cpp -o CMakeFiles\NeuralNetwork.dir\src\loss\kldivergence.cpp.s

CMakeFiles/NeuralNetwork.dir/src/loss/mean_absolute_error.cpp.obj: CMakeFiles/NeuralNetwork.dir/flags.make
CMakeFiles/NeuralNetwork.dir/src/loss/mean_absolute_error.cpp.obj: CMakeFiles/NeuralNetwork.dir/includes_CXX.rsp
CMakeFiles/NeuralNetwork.dir/src/loss/mean_absolute_error.cpp.obj: ../src/loss/mean_absolute_error.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\R\Desktop\NeuralNetwork\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_16) "Building CXX object CMakeFiles/NeuralNetwork.dir/src/loss/mean_absolute_error.cpp.obj"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\NeuralNetwork.dir\src\loss\mean_absolute_error.cpp.obj -c C:\Users\R\Desktop\NeuralNetwork\src\loss\mean_absolute_error.cpp

CMakeFiles/NeuralNetwork.dir/src/loss/mean_absolute_error.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/NeuralNetwork.dir/src/loss/mean_absolute_error.cpp.i"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\R\Desktop\NeuralNetwork\src\loss\mean_absolute_error.cpp > CMakeFiles\NeuralNetwork.dir\src\loss\mean_absolute_error.cpp.i

CMakeFiles/NeuralNetwork.dir/src/loss/mean_absolute_error.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/NeuralNetwork.dir/src/loss/mean_absolute_error.cpp.s"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\R\Desktop\NeuralNetwork\src\loss\mean_absolute_error.cpp -o CMakeFiles\NeuralNetwork.dir\src\loss\mean_absolute_error.cpp.s

CMakeFiles/NeuralNetwork.dir/src/optimizer/optimizer.cpp.obj: CMakeFiles/NeuralNetwork.dir/flags.make
CMakeFiles/NeuralNetwork.dir/src/optimizer/optimizer.cpp.obj: CMakeFiles/NeuralNetwork.dir/includes_CXX.rsp
CMakeFiles/NeuralNetwork.dir/src/optimizer/optimizer.cpp.obj: ../src/optimizer/optimizer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\R\Desktop\NeuralNetwork\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_17) "Building CXX object CMakeFiles/NeuralNetwork.dir/src/optimizer/optimizer.cpp.obj"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\NeuralNetwork.dir\src\optimizer\optimizer.cpp.obj -c C:\Users\R\Desktop\NeuralNetwork\src\optimizer\optimizer.cpp

CMakeFiles/NeuralNetwork.dir/src/optimizer/optimizer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/NeuralNetwork.dir/src/optimizer/optimizer.cpp.i"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\R\Desktop\NeuralNetwork\src\optimizer\optimizer.cpp > CMakeFiles\NeuralNetwork.dir\src\optimizer\optimizer.cpp.i

CMakeFiles/NeuralNetwork.dir/src/optimizer/optimizer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/NeuralNetwork.dir/src/optimizer/optimizer.cpp.s"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\R\Desktop\NeuralNetwork\src\optimizer\optimizer.cpp -o CMakeFiles\NeuralNetwork.dir\src\optimizer\optimizer.cpp.s

CMakeFiles/NeuralNetwork.dir/src/optimizer/SGD.cpp.obj: CMakeFiles/NeuralNetwork.dir/flags.make
CMakeFiles/NeuralNetwork.dir/src/optimizer/SGD.cpp.obj: CMakeFiles/NeuralNetwork.dir/includes_CXX.rsp
CMakeFiles/NeuralNetwork.dir/src/optimizer/SGD.cpp.obj: ../src/optimizer/SGD.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\R\Desktop\NeuralNetwork\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_18) "Building CXX object CMakeFiles/NeuralNetwork.dir/src/optimizer/SGD.cpp.obj"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\NeuralNetwork.dir\src\optimizer\SGD.cpp.obj -c C:\Users\R\Desktop\NeuralNetwork\src\optimizer\SGD.cpp

CMakeFiles/NeuralNetwork.dir/src/optimizer/SGD.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/NeuralNetwork.dir/src/optimizer/SGD.cpp.i"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\R\Desktop\NeuralNetwork\src\optimizer\SGD.cpp > CMakeFiles\NeuralNetwork.dir\src\optimizer\SGD.cpp.i

CMakeFiles/NeuralNetwork.dir/src/optimizer/SGD.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/NeuralNetwork.dir/src/optimizer/SGD.cpp.s"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\R\Desktop\NeuralNetwork\src\optimizer\SGD.cpp -o CMakeFiles\NeuralNetwork.dir\src\optimizer\SGD.cpp.s

CMakeFiles/NeuralNetwork.dir/src/optimizer/adam.cpp.obj: CMakeFiles/NeuralNetwork.dir/flags.make
CMakeFiles/NeuralNetwork.dir/src/optimizer/adam.cpp.obj: CMakeFiles/NeuralNetwork.dir/includes_CXX.rsp
CMakeFiles/NeuralNetwork.dir/src/optimizer/adam.cpp.obj: ../src/optimizer/adam.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\R\Desktop\NeuralNetwork\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_19) "Building CXX object CMakeFiles/NeuralNetwork.dir/src/optimizer/adam.cpp.obj"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\NeuralNetwork.dir\src\optimizer\adam.cpp.obj -c C:\Users\R\Desktop\NeuralNetwork\src\optimizer\adam.cpp

CMakeFiles/NeuralNetwork.dir/src/optimizer/adam.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/NeuralNetwork.dir/src/optimizer/adam.cpp.i"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\R\Desktop\NeuralNetwork\src\optimizer\adam.cpp > CMakeFiles\NeuralNetwork.dir\src\optimizer\adam.cpp.i

CMakeFiles/NeuralNetwork.dir/src/optimizer/adam.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/NeuralNetwork.dir/src/optimizer/adam.cpp.s"
	C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\R\Desktop\NeuralNetwork\src\optimizer\adam.cpp -o CMakeFiles\NeuralNetwork.dir\src\optimizer\adam.cpp.s

# Object files for target NeuralNetwork
NeuralNetwork_OBJECTS = \
"CMakeFiles/NeuralNetwork.dir/main.cpp.obj" \
"CMakeFiles/NeuralNetwork.dir/src/activation/sigmoid.cpp.obj" \
"CMakeFiles/NeuralNetwork.dir/src/activation/tanh.cpp.obj" \
"CMakeFiles/NeuralNetwork.dir/src/activation/relu.cpp.obj" \
"CMakeFiles/NeuralNetwork.dir/src/activation/leakyrelu.cpp.obj" \
"CMakeFiles/NeuralNetwork.dir/src/activation/softmax.cpp.obj" \
"CMakeFiles/NeuralNetwork.dir/src/activation/swish.cpp.obj" \
"CMakeFiles/NeuralNetwork.dir/src/activation/softplus.cpp.obj" \
"CMakeFiles/NeuralNetwork.dir/src/activation/softsign.cpp.obj" \
"CMakeFiles/NeuralNetwork.dir/src/weights/xavier.cpp.obj" \
"CMakeFiles/NeuralNetwork.dir/src/weights/he.cpp.obj" \
"CMakeFiles/NeuralNetwork.dir/src/weights/normal.cpp.obj" \
"CMakeFiles/NeuralNetwork.dir/src/loss/quadratic.cpp.obj" \
"CMakeFiles/NeuralNetwork.dir/src/loss/cross_entropy.cpp.obj" \
"CMakeFiles/NeuralNetwork.dir/src/loss/kldivergence.cpp.obj" \
"CMakeFiles/NeuralNetwork.dir/src/loss/mean_absolute_error.cpp.obj" \
"CMakeFiles/NeuralNetwork.dir/src/optimizer/optimizer.cpp.obj" \
"CMakeFiles/NeuralNetwork.dir/src/optimizer/SGD.cpp.obj" \
"CMakeFiles/NeuralNetwork.dir/src/optimizer/adam.cpp.obj"

# External object files for target NeuralNetwork
NeuralNetwork_EXTERNAL_OBJECTS =

../bin/NeuralNetwork.exe: CMakeFiles/NeuralNetwork.dir/main.cpp.obj
../bin/NeuralNetwork.exe: CMakeFiles/NeuralNetwork.dir/src/activation/sigmoid.cpp.obj
../bin/NeuralNetwork.exe: CMakeFiles/NeuralNetwork.dir/src/activation/tanh.cpp.obj
../bin/NeuralNetwork.exe: CMakeFiles/NeuralNetwork.dir/src/activation/relu.cpp.obj
../bin/NeuralNetwork.exe: CMakeFiles/NeuralNetwork.dir/src/activation/leakyrelu.cpp.obj
../bin/NeuralNetwork.exe: CMakeFiles/NeuralNetwork.dir/src/activation/softmax.cpp.obj
../bin/NeuralNetwork.exe: CMakeFiles/NeuralNetwork.dir/src/activation/swish.cpp.obj
../bin/NeuralNetwork.exe: CMakeFiles/NeuralNetwork.dir/src/activation/softplus.cpp.obj
../bin/NeuralNetwork.exe: CMakeFiles/NeuralNetwork.dir/src/activation/softsign.cpp.obj
../bin/NeuralNetwork.exe: CMakeFiles/NeuralNetwork.dir/src/weights/xavier.cpp.obj
../bin/NeuralNetwork.exe: CMakeFiles/NeuralNetwork.dir/src/weights/he.cpp.obj
../bin/NeuralNetwork.exe: CMakeFiles/NeuralNetwork.dir/src/weights/normal.cpp.obj
../bin/NeuralNetwork.exe: CMakeFiles/NeuralNetwork.dir/src/loss/quadratic.cpp.obj
../bin/NeuralNetwork.exe: CMakeFiles/NeuralNetwork.dir/src/loss/cross_entropy.cpp.obj
../bin/NeuralNetwork.exe: CMakeFiles/NeuralNetwork.dir/src/loss/kldivergence.cpp.obj
../bin/NeuralNetwork.exe: CMakeFiles/NeuralNetwork.dir/src/loss/mean_absolute_error.cpp.obj
../bin/NeuralNetwork.exe: CMakeFiles/NeuralNetwork.dir/src/optimizer/optimizer.cpp.obj
../bin/NeuralNetwork.exe: CMakeFiles/NeuralNetwork.dir/src/optimizer/SGD.cpp.obj
../bin/NeuralNetwork.exe: CMakeFiles/NeuralNetwork.dir/src/optimizer/adam.cpp.obj
../bin/NeuralNetwork.exe: CMakeFiles/NeuralNetwork.dir/build.make
../bin/NeuralNetwork.exe: CMakeFiles/NeuralNetwork.dir/linklibs.rsp
../bin/NeuralNetwork.exe: CMakeFiles/NeuralNetwork.dir/objects1.rsp
../bin/NeuralNetwork.exe: CMakeFiles/NeuralNetwork.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\Users\R\Desktop\NeuralNetwork\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_20) "Linking CXX executable ..\bin\NeuralNetwork.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\NeuralNetwork.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/NeuralNetwork.dir/build: ../bin/NeuralNetwork.exe
.PHONY : CMakeFiles/NeuralNetwork.dir/build

CMakeFiles/NeuralNetwork.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\NeuralNetwork.dir\cmake_clean.cmake
.PHONY : CMakeFiles/NeuralNetwork.dir/clean

CMakeFiles/NeuralNetwork.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" C:\Users\R\Desktop\NeuralNetwork C:\Users\R\Desktop\NeuralNetwork C:\Users\R\Desktop\NeuralNetwork\cmake-build-debug C:\Users\R\Desktop\NeuralNetwork\cmake-build-debug C:\Users\R\Desktop\NeuralNetwork\cmake-build-debug\CMakeFiles\NeuralNetwork.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/NeuralNetwork.dir/depend

