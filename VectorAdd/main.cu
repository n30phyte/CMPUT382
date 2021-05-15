#include "kernel.h"

#include "wb.h"

int main(int argc, char **argv) {
    wbArg_t args;
    int inputLength;
    float *hostInput1;
    float *hostInput2;
    float *hostOutput;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput1 =
            (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 =
            (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);
    hostOutput = (float *) malloc(inputLength * sizeof(float));

    addVectors(hostInput1, hostInput2, hostOutput, inputLength);

    wbSolution(args, hostOutput, inputLength);

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

#if LAB_DEBUG
    system("pause");
#endif

    return 0;
}

