#ifndef _SIMPLE_NEURAL_NETWORKS_H
#define _SIMPLE_NEURAL_NETWORKS_H

#define  HID_LEN 3

double multiple_input_single_output(double * input, double * weight, int LEN);

void multiple_input_multiple_output_nn(double * input_vector, 
                                       int INPUT_LEN, 
                                       double * output_vector, 
                                       int OUTPUT_LEN, 
                                       double weight_matrix[OUTPUT_LEN][INPUT_LEN]);

void weight_random_initialization(int HIDDEN_LEN, 
                                  int INPUT_LEN,
                                  double weights_matrix[HIDDEN_LEN][INPUT_LEN]);
void normalize_data_2d(int ROW, int COL, double input_matrix[ROW][COL], double output_matrix[ROW][COL]);
void weight_random_initialization_1d(double * output_vector, int LEN);
void vector_sigmoid(double * input_vector, double * output_vector, int LEN);
double sigmoid(double x);

#endif // _SIMPLE_NEURAL_NETWORKS_H