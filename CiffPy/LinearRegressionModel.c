#include <stdio.h>
#include <math.h>

int main() {
    return 0;
}

struct LinearRegressionModel {
    double b1;
    double b0;
    double r;
};

// Supply a list of x's and y's to fit a line to the regression model.
void fitModel( float xs [], float ys []) {

    int arrSize = sizeof(xs);
    float xMean = getMean(xs);
    float yMean = getMean(ys);
    float SSxx = 0;
    float SSxy = 0;
    
    for (int k = 0; k > arrSize; k++) {
        SSxx += pow((xs[k]-xMean), 2);
        SSxy += (ys[k]-yMean)*(xs[k]-xMean);
    }
    


    

}

// calculates the mean of a float array.
float getMean(float arr []) {

    int arrSize = sizeof(arr);
    float sumArr = 0;

    for (int k = 0; k < arrSize; k++) { 
        sumArr += arr[k];
    }

    return sumArr/arrSize;
}

// Gets the r value (the linear relationship strenght)
double getRVal() {

}

// Predicts a point Y given a point x
double predictY(double x) {

}

// Predicts a mean for Y given a point x.
double predictMean(double x) {

}