###################################################################################################
#
# Paths to modify
#
###################################################################################################

# The CUDA path

CUDA_PATH ?= /usr/local/cuda-7.0

###################################################################################################
#
# Compiler options
#
###################################################################################################

# The compiler

NVCC = $(CUDA_PATH)/bin/nvcc

# The flags

NVCC_FLAGS = -O3 -lineinfo -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 

###################################################################################################
#
# Experiments options
#
###################################################################################################
#to run the experiments on device #x: make DEVICE=x run
DEVICE ?= 0

###################################################################################################
###################################################################################################

BINARIES  = step-00
BINARIES += step-1a
BINARIES += step-1b
BINARIES += step-20
BINARIES += step-30
BINARIES += step-40
BINARIES += step-50
BINARIES += step-60
BINARIES += step-70
BINARIES += step-80
BINARIES += step-90
BINARIES += step-91

###################################################################################################
#
# The rules to build the code
#
###################################################################################################
all: $(BINARIES)
  
step-00: nsight-gtc2015.cu
	$(NVCC) $(NVCC_FLAGS) -DOPTIMIZATION_STEP=0x00 -o nsight-gtc2015-step-00 $<
                                                                   
step-1a: nsight-gtc2015.cu                                         
	$(NVCC) $(NVCC_FLAGS) -DOPTIMIZATION_STEP=0x1a -o nsight-gtc2015-step-1a $<
                                                                   
step-1b: nsight-gtc2015.cu                                         
	$(NVCC) $(NVCC_FLAGS) -DOPTIMIZATION_STEP=0x1b -o nsight-gtc2015-step-1b $<
                                                                   
step-20: nsight-gtc2015.cu                                         
	$(NVCC) $(NVCC_FLAGS) -DOPTIMIZATION_STEP=0x20 -o nsight-gtc2015-step-20 $<
                                                                   
step-30: nsight-gtc2015.cu                                         
	$(NVCC) $(NVCC_FLAGS) -DOPTIMIZATION_STEP=0x30 -o nsight-gtc2015-step-30 $<
                                                                   
step-40: nsight-gtc2015.cu                                         
	$(NVCC) $(NVCC_FLAGS) -DOPTIMIZATION_STEP=0x40 -o nsight-gtc2015-step-40 $<
                                                                   
step-50: nsight-gtc2015.cu                                         
	$(NVCC) $(NVCC_FLAGS) -DOPTIMIZATION_STEP=0x50 -o nsight-gtc2015-step-50 $<
                                                                   
step-60: nsight-gtc2015.cu                                         
	$(NVCC) $(NVCC_FLAGS) -DOPTIMIZATION_STEP=0x60 -o nsight-gtc2015-step-60 $<
                                                                   
step-70: nsight-gtc2015.cu                                         
	$(NVCC) $(NVCC_FLAGS) -DOPTIMIZATION_STEP=0x70 -o nsight-gtc2015-step-70 $<
                                                                   
step-80: nsight-gtc2015.cu                                         
	$(NVCC) $(NVCC_FLAGS) -DOPTIMIZATION_STEP=0x80 -o nsight-gtc2015-step-80 $<
                                                                   
step-90: nsight-gtc2015.cu                                         
	$(NVCC) $(NVCC_FLAGS) -DOPTIMIZATION_STEP=0x90 -o nsight-gtc2015-step-90 $<

step-91: nsight-gtc2015.cu                                         
	$(NVCC) $(NVCC_FLAGS) --use_fast_math -DOPTIMIZATION_STEP=0x91 -o nsight-gtc2015-step-91 $<

clean:
	rm -f nsight-gtc2015-step-00
	rm -f nsight-gtc2015-step-1a
	rm -f nsight-gtc2015-step-1b
	rm -f nsight-gtc2015-step-20
	rm -f nsight-gtc2015-step-30
	rm -f nsight-gtc2015-step-40
	rm -f nsight-gtc2015-step-50
	rm -f nsight-gtc2015-step-60
	rm -f nsight-gtc2015-step-70
	rm -f nsight-gtc2015-step-80
	rm -f nsight-gtc2015-step-90
	rm -f nsight-gtc2015-step-91

run:
	./nsight-gtc2015-step-00 $(DEVICE) data/claw.ppm
	./nsight-gtc2015-step-1a $(DEVICE) data/claw.ppm
	./nsight-gtc2015-step-1b $(DEVICE) data/claw.ppm
	./nsight-gtc2015-step-20 $(DEVICE) data/claw.ppm
	./nsight-gtc2015-step-30 $(DEVICE) data/claw.ppm
	./nsight-gtc2015-step-40 $(DEVICE) data/claw.ppm
	./nsight-gtc2015-step-50 $(DEVICE) data/claw.ppm
	./nsight-gtc2015-step-60 $(DEVICE) data/claw.ppm
	./nsight-gtc2015-step-70 $(DEVICE) data/claw.ppm
	./nsight-gtc2015-step-80 $(DEVICE) data/claw.ppm
	./nsight-gtc2015-step-90 $(DEVICE) data/claw.ppm
	./nsight-gtc2015-step-91 $(DEVICE) data/claw.ppm
