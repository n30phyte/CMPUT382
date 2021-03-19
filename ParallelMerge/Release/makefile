################################################################################
# Automatically-generated file. Do not edit!
################################################################################

-include ../makefile.init

RM := rm

# All of the sources participating in the build are defined here
-include sources.mk
-include libwb/vendor/subdir.mk
-include libwb/Release/vendor/subdir.mk
-include libwb/Release/subdir.mk
-include libwb/subdir.mk
-include subdir.mk
-include objects.mk

-include ../makefile.defs

# Add inputs and outputs from these tool invocations to the build variables 

# All Target
all: ParallelMerge

# Tool invocations
ParallelMerge: $(OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: NVCC linker'
	/usr/local/cuda/bin/nvcc --cudart=static -ccbin g++ -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 -o "ParallelMerge" $(OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) $(EXECUTABLES)$(OBJS) ParallelMerge
	-@echo ' '

.PHONY: all clean dependents

-include ../makefile.targets
