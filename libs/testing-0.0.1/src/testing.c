#include <stdio.h>
#include "testing.h"

/**
 * This function is used to test if two integers are equal
 * @param expected The expected value
 * @param actual The actual value
 * @param message The message to be printed
 * @return 1 if the test passes, 0 if the test fails */
int int_assertEqual(int expected, int actual, char* message) {
    if (expected == actual) {
        printf("Test PASSED: %s\n", message);
        return 1;
    } else {
        printf("Test FAILED: %s\n", message);
        return 0;
    }
}

/**
 * This function is used to test if two doubles are equal
 * @param expected The expected value
 * @param actual The actual value
 * @param message The message to be printed
 * @return 1 if the test passes, 0 if the test fails */
int dbl_assertEqual(double expected, double actual, char* message) {
    if (expected == actual) {
        printf("Test PASSED: %s\n", message);
        return 1;
    } else {
        printf("Test FAILED: %s\n", message);
        return 0;
    }
}

/**
 * This function is used to test if two integers are not equal
 * @param expected The expected value
 * @param actual The actual value
 * @param message The message to be printed
 * @return 1 if the test passes, 0 if the test fails */
int int_assertNotEqual(int expected, int actual, char* message) {
    if (expected != actual) {
        printf("Test PASSED: %s\n", message);
        return 1;
    } else {
        printf("Test FAILED: %s\n", message);
        return 0;
    }
}

/**
 * This function is used to test if two doubles are not equal
 * @param expected The expected value
 * @param actual The actual value
 * @param message The message to be printed
 * @return 1 if the test passes, 0 if the test fails */
int dbl_assertNotEqual(double expected, double actual, char* message) {
    if (expected != actual) {
        printf("Test PASSED: %s\n", message);
        return 1;
    } else {
        printf("Test FAILED: %s\n", message);
        return 0;
    }
}

/**
 * This function is used to test if a boolean expression is true
 * @param booleanExpression The boolean expression to be tested
 * @param message The message to be printed
 * @return 1 if the test passes, 0 if the test fails */
int assertTrue(int booleanExpression, char* message) {
    if (booleanExpression) {
        printf("Test PASSED: %s\n", message);
        return 1;
    } else {
        printf("Test FAILED: %s\n", message);
        return 0;
    }
}

/**
 * This function is used to test if a boolean expression is false
 * @param booleanExpression The boolean expression to be tested
 * @param message The message to be printed
 * @return 1 if the test passes, 0 if the test fails */
int assertFalse(int booleanExpression, char* message) {
    if (!booleanExpression) {
        printf("Test PASSED: %s\n", message);
        return 1;
    } else {
        printf("Test FAILED: %s\n", message);
        return 0;
    }
}