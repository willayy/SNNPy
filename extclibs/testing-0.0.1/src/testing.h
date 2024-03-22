#ifndef testing_h
    #define testing_h

        int int_assertEqual(int expected, int actual, char* message);

        int dbl_assertEqual(double expected, double actual, char* message);

        int int_assertNotEqual(int expected, int actual, char* message);

        int dbl_assertNotEqual(double expected, double actual, char* message);

        int dbl_assertBetween(double min, double max, double actual, char* message);

        int assertTrue(int booleanExpression, char* message);

        int assertFalse(int booleanExpression, char* message);
#endif