#ifndef vectors_h

    #define  vectors_h

        typedef struct {
            double * data;
            int size;
        } Vector;

        typedef struct {
            Vector * data;
            int size;
        } Matrix;

#endif