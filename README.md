nsight-gtc
==============

Companion code for GTC 2016 Nsight VSE/EE tutorials

Build on Windows
----------------
To build on Windows you need Visual Studio 2013 and CUDA 6.0 or later. There is a Configuration for each step. So:

    1- Select the configuration 
    2- Build
    3- Run

You can build all the Configurations at once using BUILD > Batch Build...

Also, even if we don't provide Visual Studio 2008, 2010 or 2012 projects, you can easily build the step with NVCC from the command line:

    nvcc -O3 -arch=sm_35 -DOPTIMIZATION_STEP=0x50 -o nsight-gtc2015.exe nsight-gtc2015.cu
    
That command will build the optimization step 5. Replace 0x50 by another value 0xXX to build another step.

Build on Linux
--------------
To build on Linux you need CUDA 6.0 or later. There is a Makefile and a rule "step-XX" for each step. You can build all steps at once:

    make -j4

To run the code, simply launch ./nsight-gtc2015-XX where XX is the step.

Run the Code
------------
The executables all expect an input image. We have copied the claw.ppm image used during the tutorial in the folder data. 

To run the code on Windows from the command-line, type:

    x64\Step-00\nsight-gtc2015.exe data\claw.ppm

To run the code on Linux, type:

    ./nsight-gtc2015-00 data/claw.ppm

Build the OpenGL version
------------------------
As you may have seen the code contains everything to visualize the images. To build with the OpenGL support, we assume you have the CUDA SDK Samples installed on your system. On my Windows, it is installed in %PROGRAMDATA%\NVIDIA Corporation\CUDA Samples\v6.0

To build a step with OpenGL support simply copy the following line to a Visual Studio Command Line:

    nvcc -O3 -arch=sm_35 -I"%PROGRAMDATA%\NVIDIA Corporation\CUDA Samples\v6.0\common\inc" -DOPTIMIZATION_STEP=0x50 -DWITH_OPENGL -o nsight-gtc2015.exe nsight-gtc2015.cu "%PROGRAMDATA%\NVIDIA Corporation\CUDA Samples\v6.0\common\lib\x64\freeglut.lib" "%PROGRAMDATA%\NVIDIA Corporation\CUDA Samples\v6.0\common\lib\x64\glew64.lib"

You may want to change the target architecture (-arch) or the optimization step (-DOPTIMIZATION_STEP). Of course, on Windows, you'll have to copy the FreeGlut and GLEW DLLs from "%PROGRAMDATA%\NVIDIA Corporation\CUDA Samples\v6.0\common\bin\x64" to the folder where you are executing the code.
