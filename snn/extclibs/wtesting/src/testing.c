#include <stdio.h>
#include <string.h>
#include "testing.h"

/**
 * This function is used to test if two integers are equal
 * @param expected The expected value
 * @param actual The actual value
 * @param message The message to be printed
 * @return 0 if the test passes, 1 if the test fails */
int int_assertEqual(int expected, int actual) {

    testNumber += 1;; // Increment the test number

    if (expected == actual) {

        printf("Test nr %s PASSED\n", testNumber);

    } else {

        printf("Test nr %s FAILED\n", testNumber);

        failedTests += 1;

    }

}

/**
 * This function is used to test if two doubles are equal
 * @param expected The expected value
 * @param actual The actual value
 * @param message The message to be printed
 * @return 0 if the test passes, 1 if the test fails */
int dbl_assertEqual(double expected, double actual, char* message) {

    testNumber += 1; // Increment the test number

    if (expected == actual) {

        printf("Test nr %s PASSED\n", testNumber);

    } else {

        printf("Test nr %s FAILED\n", testNumber);

        failedTests += 1;

    }

}

/**
 * This function is used to test if two integers are not equal
 * @param expected The expected value
 * @param actual The actual value
 * @param message The message to be printed
 * @return 1 if the test passes, 1 if the test fails */
int int_assertNotEqual(int expected, int actual, char* message) {

    testNumber += 1; // Increment the test number

    if (expected != actual) {

        printf("Test nr %s PASSED\n", testNumber);

    } else {

        printf("Test nr %s FAILED\n", testNumber);

        failedTests += 1;

    }

}

/**
 * This function is used to test if two doubles are not equal
 * @param expected The expected value
 * @param actual The actual value
 * @param message The message to be printed
 * @return 0 if the test passes, 1 if the test fails */
int dbl_assertNotEqual(double expected, double actual, char* message) {

    testNumber += 1; // Increment the test number

    if (expected != actual) {

        printf("Test nr %s PASSED\n", testNumber);

    } else {

        printf("Test nr %s FAILED\n", testNumber);

        failedTests += 1;

    }

}

/**
 * This function is used to test if a boolean expression is true
 * @param booleanExpression The boolean expression to be tested
 * @param message The message to be printed
 * @return 0 if the test passes, 1 if the test fails */
int assertTrue(int booleanExpression, char* message) {

    testNumber += 1; // Increment the test number

    if (booleanExpression) {

        printf("Test nr %s PASSED\n", testNumber);

    } else {

        printf("Test nr %s FAILED\n", testNumber);

        failedTests += 1;

    }

}

/**
 * This function is used to test if a boolean expression is false
 * @param booleanExpression The boolean expression to be tested
 * @param message The message to be printed
 * @return 0 if the test passes, 1 if the test fails */
int assertFalse(int booleanExpression, char* message) {

    testNumber += 1; // Increment the test number

    if (!booleanExpression) {

        printf("Test nr %s PASSED\n", testNumber);

    } else {

        printf("Test nr %s FAILED\n", testNumber);

        failedTests += 1;

    }

}

/**
 * This function is used to test if a double is between two other doubles
 * @param min The minimum value (inclusive)
 * @param max The maximum value (inclusive)
 * @param actual The actual value
 * @param message The message to be printed
 * @return 0 if the test passes, 1 if the test fails */
int dbl_assertBetween(double min, double max, double actual, char* message) {

    testNumber += 1; // Increment the test number

    if (actual >= min && actual <= max) {

        printf("Test nr %s PASSED\n", testNumber);

    } else {

        printf("Test nr %s FAILED\n", testNumber);

        failedTests += 1;

    }
}


