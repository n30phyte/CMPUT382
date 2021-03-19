################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../main.cu 

OBJS += \
./main.o 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda/bin/nvcc -O3 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 -ccbin g++ -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


