#include <stdio.h>
#include "./Headers/simple_neural_networks.h"

#define NUM_OF_FEATURES 2 // n values
#define NUM_OF_EXAMPLE  3 // m values

#define NUM_OF_HID_NODES 3
#define NUM_OF_OUT_NODES 1

double raw_x[NUM_OF_FEATURES][NUM_OF_EXAMPLE] = {{2,5,1},   // часы тренировок
                                                 {8,5,8}}; // часы отдыха
double raw_y[1][NUM_OF_EXAMPLE] = {{200, 90, 190}}; // масса 

// train x
// 2/8  5/8  1/8
// 8/8  5/8  8/8
// dim = nx X m
double train_x[NUM_OF_FEATURES][NUM_OF_EXAMPLE];
// train y
// 200/200  90/200  190/200
// dim = 1 X m
double train_y[1][NUM_OF_EXAMPLE];

// Input layer to hidden layer weights buffer 
double syn0[NUM_OF_HID_NODES][NUM_OF_FEATURES];
// Hidden layer to output layer weights buffer
double syn1[NUM_OF_HID_NODES];

double train_x_eg1[NUM_OF_FEATURES];
double train_y_eg1;
double z1_eg1[NUM_OF_HID_NODES];
double a1_eg1[NUM_OF_HID_NODES];
double z2_eg1;
double yhat_eg1;

int main() {

  // Normalize x and y
  normalize_data_2d(NUM_OF_FEATURES, NUM_OF_EXAMPLE, raw_x, train_x);
  normalize_data_2d(1, NUM_OF_EXAMPLE, raw_y, train_y);

  train_x_eg1[0] = train_x[0][0];
  train_x_eg1[1] = train_x[1][0];

  train_y_eg1 = train_y[0][0];

  printf("\n\n train_x_eg1: [%f %f]\n", train_x_eg1[0], train_x_eg1[1]);
  printf("\n\n train_y_eg1: %f\n", train_y_eg1);

  // Initialize syn0 and syn1 weights
  weight_random_initialization(NUM_OF_HID_NODES, NUM_OF_FEATURES, syn0);

  // synapse 0 weights
  printf("\n\nSynapse 0 weights: \r\n");
  for(int i = 0; i < NUM_OF_HID_NODES; i++) {
    for(int j = 0; j < NUM_OF_FEATURES; j++) {
      printf(" %f ", syn0[i][j]);
    }
    printf("\n\r");
    printf("\n\r");
  } 

  weight_random_initialization_1d(syn1, NUM_OF_OUT_NODES);
  for(int i = 0; i < NUM_OF_OUT_NODES; i++) {
    printf("\nSynapse 1 [%f %f %f]\n", syn1[0], syn1[1], syn1[2]);
  }

  // Compute z1 
  multiple_input_multiple_output_nn(train_x_eg1, 
                                    NUM_OF_FEATURES, 
                                    z1_eg1, 
                                    NUM_OF_HID_NODES, 
                                    syn0);
  printf("\n\n z1_eg1 = [%f %f %f]\n", z1_eg1[0], z1_eg1[1], z1_eg1[2]);

  // Compute a1 
  vector_sigmoid(z1_eg1, a1_eg1, NUM_OF_HID_NODES);
  printf("\n\n a1_eg1 = [%f %f %f]\n", a1_eg1[0], a1_eg1[1], a1_eg1[2]);

  // Compute z2 
  z2_eg1 = multiple_input_single_output(a1_eg1, syn1, NUM_OF_HID_NODES);
  printf("\n\n z1_eg1 = %f\n", z2_eg1);

  //Compute yhat
  yhat_eg1 = sigmoid(z2_eg1);
  printf("\n\n yhat_eg1 = %f\n", yhat_eg1);
  
  return 0;
}