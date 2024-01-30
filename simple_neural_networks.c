#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "./Headers/simple_neural_networks.h"

double weighted_sum(double * input, double * weight, int LEN) {
    double output;
    for(int i = 0; i < LEN; i++) {
        output += input[i]*weight[i];
    }
    return output;
}

double multiple_input_single_output(double * input, double * weight, int LEN) {
    double predicted_value;
    predicted_value = weighted_sum(input, weight, LEN);
    return predicted_value;
}

void matrix_vector_multiply(double * input_vector, 
                            int INPUT_LEN, 
                            double * output_vector, 
                            int OUTPUT_LEN, 
                            double weight_matrix[OUTPUT_LEN][INPUT_LEN]) {

    for(int k = 0; k<OUTPUT_LEN; k++) {
        for(int i = 0; i<INPUT_LEN; i++) {
            output_vector[k] += input_vector[i] * weight_matrix[k][i];
        }
    }                            
}

void multiple_input_multiple_output_nn(double * input_vector, 
                                       int INPUT_LEN, 
                                       double * output_vector, 
                                       int OUTPUT_LEN, 
                                       double weight_matrix[OUTPUT_LEN][INPUT_LEN]) {

    matrix_vector_multiply(input_vector, INPUT_LEN, output_vector, OUTPUT_LEN, weight_matrix);
}


void weight_random_initialization(int HIDDEN_LEN, 
                                  int INPUT_LEN,
                                  double weights_matrix[HIDDEN_LEN][INPUT_LEN]) {
    double d_rand;
    // seed random number generator
    srand(2);
    for(int i = 0; i < HIDDEN_LEN; i++) {
        for(int j = 0; j < INPUT_LEN; j++) {
            // Gernerate rand numbers between 0 and 1
            d_rand = (rand() % 10);
            d_rand /= 10;
            weights_matrix[i][j] = d_rand;
        }
    }
}

void normalize_data_2d(int ROW, int COL, double input_matrix[ROW][COL], double output_matrix[ROW][COL]) {
    // find max
     double max = -999999999;
     for(int i = 0; i < ROW; i++) {
        for(int j = 0; j < COL; j++) {
            if(input_matrix[i][j]) {
                max = input_matrix[i][j];
            }
        }
     }
     // normalize
     for(int i = 0; i < ROW; i++) {
        for(int j = 0; j < COL; j++) {
            output_matrix[i][j] = input_matrix[i][j] / max;
        }
     }
}

void weight_random_initialization_1d(double * output_vector, int LEN) {
    double d_rand;
    srand(2);
    for(int j = 0; j < LEN; j++) {
        d_rand = (rand() % 10);
        d_rand /= 10;
        output_vector[j] = d_rand;
    }
}

double sigmoid(double x) {
    double result = 1 / (1 + exp(-x));
    return result;
}

void vector_sigmoid(double * input_vector, double * output_vector, int LEN) {
    for(int i = 0; i < LEN; i++) {
        output_vector[i] = sigmoid(input_vector[i]);
    }
}