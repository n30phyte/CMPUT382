#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "kernel.h"

TEST(testVectorAdd, test0) {
    std::vector<float> input1 = {-1, 0.4, 2.4, 0.8, -0.2, 2.6, 1.4, -1, -0.4, 1.6, 0.6, 0.6, -1, 1.8, 0, 0.4};
    std::vector<float> input2 = {-1, 0.4, 2.4, 0.8, -0.2, 2.6, 1.4, -1, -0.4, 1.6, 0.6, 0.6, -1, 1.8, 0, 0.4};
    std::vector<float> output;
    std::vector<float> expected = {-2, 0.8, 4.8, 1.6, -0.4, 5.2, 2.8, -2, -0.8, 3.2, 1.2, 1.2, -2, 3.6, 0, 0.8};

    output.resize(16);

    addVectors(input1, input2, output, 16);

    EXPECT_THAT(output, testing::Pointwise(testing::FloatNear(1e-5), expected));
}