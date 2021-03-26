#include <algorithm>
#include <wb.h>
#include <chrono>

int main(int argc, char **argv) {
    wbArg_t args;
    int *hostInput;  // The input 1D list
    int numElements; // number of elements in the list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (int *) wbImport(wbArg_getInputFile(args, 0), &numElements, "integral_vector");
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ",
          numElements);

    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    wbTime_start(Compute, "Performing Sort computation");
    auto t1 = high_resolution_clock::now();
    std::sort(hostInput, hostInput + numElements);
    auto t2 = high_resolution_clock::now();
    auto ms_int = duration_cast<milliseconds>(t2 - t1);
    std::cout << ms_int.count() << "ms\n";
    wbTime_stop(Compute, "Performing Sort computation");

    wbSolution(args, hostInput, numElements);

    free(hostInput);

#if LAB_DEBUG
    system("pause");
#endif

    return 0;
}
