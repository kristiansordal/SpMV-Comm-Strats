# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.30

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

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/Cellar/cmake/3.30.1/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Cellar/cmake/3.30.1/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/kristiansordal/dev/uib/master/project/SpMV-Comm-Strats/P3

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/kristiansordal/dev/uib/master/project/SpMV-Comm-Strats/P3/build/Debug

# Include any dependencies generated for this target.
include CMakeFiles/1e.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/1e.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/1e.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/1e.dir/flags.make

CMakeFiles/1e.dir/src/main_dist.c.o: CMakeFiles/1e.dir/flags.make
CMakeFiles/1e.dir/src/main_dist.c.o: /Users/kristiansordal/dev/uib/master/project/SpMV-Comm-Strats/P3/src/main_dist.c
CMakeFiles/1e.dir/src/main_dist.c.o: CMakeFiles/1e.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/kristiansordal/dev/uib/master/project/SpMV-Comm-Strats/P3/build/Debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/1e.dir/src/main_dist.c.o"
	/opt/homebrew/Cellar/gcc/14.1.0_2/bin/gcc-14 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/1e.dir/src/main_dist.c.o -MF CMakeFiles/1e.dir/src/main_dist.c.o.d -o CMakeFiles/1e.dir/src/main_dist.c.o -c /Users/kristiansordal/dev/uib/master/project/SpMV-Comm-Strats/P3/src/main_dist.c

CMakeFiles/1e.dir/src/main_dist.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/1e.dir/src/main_dist.c.i"
	/opt/homebrew/Cellar/gcc/14.1.0_2/bin/gcc-14 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/kristiansordal/dev/uib/master/project/SpMV-Comm-Strats/P3/src/main_dist.c > CMakeFiles/1e.dir/src/main_dist.c.i

CMakeFiles/1e.dir/src/main_dist.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/1e.dir/src/main_dist.c.s"
	/opt/homebrew/Cellar/gcc/14.1.0_2/bin/gcc-14 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/kristiansordal/dev/uib/master/project/SpMV-Comm-Strats/P3/src/main_dist.c -o CMakeFiles/1e.dir/src/main_dist.c.s

CMakeFiles/1e.dir/src/spmv.c.o: CMakeFiles/1e.dir/flags.make
CMakeFiles/1e.dir/src/spmv.c.o: /Users/kristiansordal/dev/uib/master/project/SpMV-Comm-Strats/P3/src/spmv.c
CMakeFiles/1e.dir/src/spmv.c.o: CMakeFiles/1e.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/kristiansordal/dev/uib/master/project/SpMV-Comm-Strats/P3/build/Debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/1e.dir/src/spmv.c.o"
	/opt/homebrew/Cellar/gcc/14.1.0_2/bin/gcc-14 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/1e.dir/src/spmv.c.o -MF CMakeFiles/1e.dir/src/spmv.c.o.d -o CMakeFiles/1e.dir/src/spmv.c.o -c /Users/kristiansordal/dev/uib/master/project/SpMV-Comm-Strats/P3/src/spmv.c

CMakeFiles/1e.dir/src/spmv.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/1e.dir/src/spmv.c.i"
	/opt/homebrew/Cellar/gcc/14.1.0_2/bin/gcc-14 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/kristiansordal/dev/uib/master/project/SpMV-Comm-Strats/P3/src/spmv.c > CMakeFiles/1e.dir/src/spmv.c.i

CMakeFiles/1e.dir/src/spmv.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/1e.dir/src/spmv.c.s"
	/opt/homebrew/Cellar/gcc/14.1.0_2/bin/gcc-14 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/kristiansordal/dev/uib/master/project/SpMV-Comm-Strats/P3/src/spmv.c -o CMakeFiles/1e.dir/src/spmv.c.s

CMakeFiles/1e.dir/src/mtx.c.o: CMakeFiles/1e.dir/flags.make
CMakeFiles/1e.dir/src/mtx.c.o: /Users/kristiansordal/dev/uib/master/project/SpMV-Comm-Strats/P3/src/mtx.c
CMakeFiles/1e.dir/src/mtx.c.o: CMakeFiles/1e.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/kristiansordal/dev/uib/master/project/SpMV-Comm-Strats/P3/build/Debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object CMakeFiles/1e.dir/src/mtx.c.o"
	/opt/homebrew/Cellar/gcc/14.1.0_2/bin/gcc-14 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/1e.dir/src/mtx.c.o -MF CMakeFiles/1e.dir/src/mtx.c.o.d -o CMakeFiles/1e.dir/src/mtx.c.o -c /Users/kristiansordal/dev/uib/master/project/SpMV-Comm-Strats/P3/src/mtx.c

CMakeFiles/1e.dir/src/mtx.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/1e.dir/src/mtx.c.i"
	/opt/homebrew/Cellar/gcc/14.1.0_2/bin/gcc-14 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/kristiansordal/dev/uib/master/project/SpMV-Comm-Strats/P3/src/mtx.c > CMakeFiles/1e.dir/src/mtx.c.i

CMakeFiles/1e.dir/src/mtx.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/1e.dir/src/mtx.c.s"
	/opt/homebrew/Cellar/gcc/14.1.0_2/bin/gcc-14 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/kristiansordal/dev/uib/master/project/SpMV-Comm-Strats/P3/src/mtx.c -o CMakeFiles/1e.dir/src/mtx.c.s

# Object files for target 1e
1e_OBJECTS = \
"CMakeFiles/1e.dir/src/main_dist.c.o" \
"CMakeFiles/1e.dir/src/spmv.c.o" \
"CMakeFiles/1e.dir/src/mtx.c.o"

# External object files for target 1e
1e_EXTERNAL_OBJECTS =

1e: CMakeFiles/1e.dir/src/main_dist.c.o
1e: CMakeFiles/1e.dir/src/spmv.c.o
1e: CMakeFiles/1e.dir/src/mtx.c.o
1e: CMakeFiles/1e.dir/build.make
1e: /opt/homebrew/Cellar/open-mpi/5.0.6/lib/libmpi.dylib
1e: /opt/homebrew/lib/libmetis.dylib
1e: /opt/homebrew/Cellar/gcc/14.1.0_2/lib/gcc/current/libgomp.dylib
1e: CMakeFiles/1e.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/kristiansordal/dev/uib/master/project/SpMV-Comm-Strats/P3/build/Debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking C executable 1e"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/1e.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/1e.dir/build: 1e
.PHONY : CMakeFiles/1e.dir/build

CMakeFiles/1e.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/1e.dir/cmake_clean.cmake
.PHONY : CMakeFiles/1e.dir/clean

CMakeFiles/1e.dir/depend:
	cd /Users/kristiansordal/dev/uib/master/project/SpMV-Comm-Strats/P3/build/Debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/kristiansordal/dev/uib/master/project/SpMV-Comm-Strats/P3 /Users/kristiansordal/dev/uib/master/project/SpMV-Comm-Strats/P3 /Users/kristiansordal/dev/uib/master/project/SpMV-Comm-Strats/P3/build/Debug /Users/kristiansordal/dev/uib/master/project/SpMV-Comm-Strats/P3/build/Debug /Users/kristiansordal/dev/uib/master/project/SpMV-Comm-Strats/P3/build/Debug/CMakeFiles/1e.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/1e.dir/depend

