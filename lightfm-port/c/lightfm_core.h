#ifndef LIGHTFM_CORE_H
#define LIGHTFM_CORE_H

#include <stdlib.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

// Sparse matrix in Compressed Sparse Row format
typedef struct {
    int *indices;     // Column indices for non-zero elements
    int *indptr;      // Row pointer array  
    float *data;      // Non-zero values
    int rows;         // Number of rows
    int cols;         // Number of columns
    int nnz;          // Number of non-zero elements
} CSRMatrix;

// Main LightFM model state
typedef struct {
    // Item embeddings and gradients
    float **item_features;           // [n_item_features][no_components]
    float **item_feature_gradients;  // [n_item_features][no_components]
    float **item_feature_momentum;   // [n_item_features][no_components]
    float *item_biases;              // [n_item_features]
    float *item_bias_gradients;      // [n_item_features]
    float *item_bias_momentum;       // [n_item_features]
    
    // User embeddings and gradients
    float **user_features;           // [n_user_features][no_components]
    float **user_feature_gradients;  // [n_user_features][no_components]
    float **user_feature_momentum;   // [n_user_features][no_components]
    float *user_biases;              // [n_user_features]
    float *user_bias_gradients;      // [n_user_features]
    float *user_bias_momentum;       // [n_user_features]
    
    // Model dimensions
    int n_user_features;
    int n_item_features;
    int no_components;
    
    // Optimizer parameters
    int adadelta;         // Whether to use AdaDelta optimizer
    float learning_rate;  // Learning rate for AdaGrad
    float rho;           // AdaDelta decay parameter
    float eps;           // AdaDelta epsilon
    int max_sampled;     // Max negative samples for WARP
    
    // Scaling factors for lazy regularization
    double item_scale;
    double user_scale;
} FastLightFM;

// Memory management functions
CSRMatrix* csr_matrix_new(int rows, int cols, int nnz);
void csr_matrix_free(CSRMatrix *matrix);
FastLightFM* lightfm_new(int n_user_features, int n_item_features, int no_components);
void lightfm_free(FastLightFM *model);

// Core training functions
int fit_logistic(
    CSRMatrix *item_features,
    CSRMatrix *user_features,
    int *user_ids,
    int *item_ids,
    float *Y,
    float *sample_weight,
    int *shuffle_indices,
    FastLightFM *lightfm,
    double learning_rate,
    double item_alpha,
    double user_alpha,
    int num_threads,
    int num_examples
);

int fit_warp(
    CSRMatrix *item_features,
    CSRMatrix *user_features,
    CSRMatrix *interactions,
    int *user_ids,
    int *item_ids,
    float *Y,
    float *sample_weight,
    int *shuffle_indices,
    FastLightFM *lightfm,
    double learning_rate,
    double item_alpha,
    double user_alpha,
    int num_threads,
    unsigned int *random_states,
    int num_examples
);

int fit_bpr(
    CSRMatrix *item_features,
    CSRMatrix *user_features,
    CSRMatrix *interactions,
    int *user_ids,
    int *item_ids,
    float *Y,
    float *sample_weight,
    int *shuffle_indices,
    FastLightFM *lightfm,
    double learning_rate,
    double item_alpha,
    double user_alpha,
    int num_threads,
    unsigned int *random_states,
    int num_examples
);

int fit_warp_kos(
    CSRMatrix *item_features,
    CSRMatrix *user_features,
    CSRMatrix *data,
    int *user_ids,
    int *shuffle_indices,
    FastLightFM *lightfm,
    double learning_rate,
    double item_alpha,
    double user_alpha,
    int k,
    int n,
    int num_threads,
    unsigned int *random_states,
    int num_examples
);

// Prediction functions
int predict_lightfm(
    CSRMatrix *item_features,
    CSRMatrix *user_features,
    int *user_ids,
    int *item_ids,
    float *predictions,
    FastLightFM *lightfm,
    int num_threads,
    int num_examples
);

int predict_ranks(
    CSRMatrix *item_features,
    CSRMatrix *user_features,
    CSRMatrix *test_interactions,
    CSRMatrix *train_interactions,
    float *ranks,
    FastLightFM *lightfm,
    int num_threads
);

// Core algorithm components
void compute_representation(
    CSRMatrix *features,
    float **feature_embeddings,
    float *feature_biases,
    FastLightFM *lightfm,
    int row_id,
    double scale,
    float *representation
);

float compute_prediction_from_repr(
    float *user_repr,
    float *item_repr,
    int no_components
);

double update_biases(
    CSRMatrix *feature_indices,
    int start,
    int stop,
    float *biases,
    float *gradients,
    float *momentum,
    double gradient,
    int adadelta,
    double learning_rate,
    double alpha,
    float rho,
    float eps
);

double update_features(
    CSRMatrix *feature_indices,
    float **features,
    float **gradients,
    float **momentum,
    int component,
    int start,
    int stop,
    double gradient,
    int adadelta,
    double learning_rate,
    double alpha,
    float rho,
    float eps
);

// Helper functions
float sigmoid(float v);
int in_positives(int item_id, int user_id, CSRMatrix *interactions);
int sample_range(int min_val, int max_val, unsigned int *seed);

// Evaluation functions
float precision_at_k(
    CSRMatrix *test_interactions,
    CSRMatrix *train_interactions,
    float *predictions,
    int k,
    int n_users,
    int n_items
);

#ifdef __cplusplus
}
#endif

#endif // LIGHTFM_CORE_H