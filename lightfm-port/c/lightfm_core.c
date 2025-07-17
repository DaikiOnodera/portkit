#include "lightfm_core.h"
#include <string.h>
#include <stdio.h>

// Memory management functions
CSRMatrix* csr_matrix_new(int rows, int cols, int nnz) {
    CSRMatrix *matrix = (CSRMatrix*)malloc(sizeof(CSRMatrix));
    if (!matrix) return NULL;
    
    matrix->rows = rows;
    matrix->cols = cols;
    matrix->nnz = nnz;
    
    matrix->indices = (int*)calloc(nnz, sizeof(int));
    matrix->indptr = (int*)calloc(rows + 1, sizeof(int));
    matrix->data = (float*)calloc(nnz, sizeof(float));
    
    if (!matrix->indices || !matrix->indptr || !matrix->data) {
        csr_matrix_free(matrix);
        return NULL;
    }
    
    return matrix;
}

void csr_matrix_free(CSRMatrix *matrix) {
    if (!matrix) return;
    
    free(matrix->indices);
    free(matrix->indptr);
    free(matrix->data);
    free(matrix);
}

FastLightFM* lightfm_new(int n_user_features, int n_item_features, int no_components) {
    FastLightFM *model = (FastLightFM*)malloc(sizeof(FastLightFM));
    if (!model) return NULL;
    
    model->n_user_features = n_user_features;
    model->n_item_features = n_item_features;
    model->no_components = no_components;
    
    // Allocate item features
    model->item_features = (float**)malloc(n_item_features * sizeof(float*));
    model->item_feature_gradients = (float**)malloc(n_item_features * sizeof(float*));
    model->item_feature_momentum = (float**)malloc(n_item_features * sizeof(float*));
    
    for (int i = 0; i < n_item_features; i++) {
        model->item_features[i] = (float*)calloc(no_components, sizeof(float));
        model->item_feature_gradients[i] = (float*)calloc(no_components, sizeof(float));
        model->item_feature_momentum[i] = (float*)calloc(no_components, sizeof(float));
    }
    
    model->item_biases = (float*)calloc(n_item_features, sizeof(float));
    model->item_bias_gradients = (float*)calloc(n_item_features, sizeof(float));
    model->item_bias_momentum = (float*)calloc(n_item_features, sizeof(float));
    
    // Allocate user features
    model->user_features = (float**)malloc(n_user_features * sizeof(float*));
    model->user_feature_gradients = (float**)malloc(n_user_features * sizeof(float*));
    model->user_feature_momentum = (float**)malloc(n_user_features * sizeof(float*));
    
    for (int i = 0; i < n_user_features; i++) {
        model->user_features[i] = (float*)calloc(no_components, sizeof(float));
        model->user_feature_gradients[i] = (float*)calloc(no_components, sizeof(float));
        model->user_feature_momentum[i] = (float*)calloc(no_components, sizeof(float));
    }
    
    model->user_biases = (float*)calloc(n_user_features, sizeof(float));
    model->user_bias_gradients = (float*)calloc(n_user_features, sizeof(float));
    model->user_bias_momentum = (float*)calloc(n_user_features, sizeof(float));
    
    // Initialize optimizer parameters
    model->adadelta = 0;
    model->learning_rate = 0.05f;
    model->rho = 0.95f;
    model->eps = 1e-6f;
    model->max_sampled = 10;
    model->item_scale = 1.0;
    model->user_scale = 1.0;
    
    // Initialize embeddings with small random values
    initialize_embeddings(model);
    
    return model;
}

void lightfm_free(FastLightFM *model) {
    if (!model) return;
    
    // Free item features
    if (model->item_features) {
        for (int i = 0; i < model->n_item_features; i++) {
            free(model->item_features[i]);
            free(model->item_feature_gradients[i]);
            free(model->item_feature_momentum[i]);
        }
        free(model->item_features);
        free(model->item_feature_gradients);
        free(model->item_feature_momentum);
    }
    
    free(model->item_biases);
    free(model->item_bias_gradients);
    free(model->item_bias_momentum);
    
    // Free user features
    if (model->user_features) {
        for (int i = 0; i < model->n_user_features; i++) {
            free(model->user_features[i]);
            free(model->user_feature_gradients[i]);
            free(model->user_feature_momentum[i]);
        }
        free(model->user_features);
        free(model->user_feature_gradients);
        free(model->user_feature_momentum);
    }
    
    free(model->user_biases);
    free(model->user_bias_gradients);
    free(model->user_bias_momentum);
    
    free(model);
}

// Initialize embeddings with small random values
void initialize_embeddings(FastLightFM *model) {
    unsigned int seed = 42;  // Fixed seed for reproducibility
    const float scale = 0.05f;  // Small initialization scale
    
    // Initialize item embeddings
    for (int i = 0; i < model->n_item_features; i++) {
        for (int j = 0; j < model->no_components; j++) {
            // Generate random number between -scale and scale
            float random_val = (float)rand_r(&seed) / RAND_MAX;
            model->item_features[i][j] = (random_val * 2.0f - 1.0f) * scale;
        }
        // Initialize gradients to small values for AdaGrad
        for (int j = 0; j < model->no_components; j++) {
            model->item_feature_gradients[i][j] = 1e-4f;
        }
        model->item_bias_gradients[i] = 1e-4f;
    }
    
    // Initialize user embeddings
    for (int i = 0; i < model->n_user_features; i++) {
        for (int j = 0; j < model->no_components; j++) {
            // Generate random number between -scale and scale
            float random_val = (float)rand_r(&seed) / RAND_MAX;
            model->user_features[i][j] = (random_val * 2.0f - 1.0f) * scale;
        }
        // Initialize gradients to small values for AdaGrad
        for (int j = 0; j < model->no_components; j++) {
            model->user_feature_gradients[i][j] = 1e-4f;
        }
        model->user_bias_gradients[i] = 1e-4f;
    }
}

// Helper functions
float sigmoid(float v) {
    return 1.0f / (1.0f + expf(-v));
}

int in_positives(int item_id, int user_id, CSRMatrix *interactions) {
    if (!interactions || user_id >= interactions->rows) return 0;
    
    int start = interactions->indptr[user_id];
    int end = interactions->indptr[user_id + 1];
    
    for (int i = start; i < end; i++) {
        if (interactions->indices[i] == item_id) {
            return 1;
        }
    }
    return 0;
}

int sample_range(int min_val, int max_val, unsigned int *seed) {
    if (min_val >= max_val) return min_val;
    return min_val + (rand_r(seed) % (max_val - min_val));
}

void compute_representation(
    CSRMatrix *features,
    float **feature_embeddings,
    float *feature_biases,
    FastLightFM *lightfm,
    int row_id,
    double scale,
    float *representation
) {
    // Initialize representation to zero
    for (int i = 0; i <= lightfm->no_components; i++) {
        representation[i] = 0.0f;
    }
    
    if (!features || row_id >= features->rows) return;
    
    int start = features->indptr[row_id];
    int end = features->indptr[row_id + 1];
    
    // Sum feature embeddings
    for (int i = start; i < end; i++) {
        int feature_id = features->indices[i];
        float feature_weight = features->data[i];
        
        // Add to representation vector
        for (int j = 0; j < lightfm->no_components; j++) {
            representation[j] += feature_weight * feature_embeddings[feature_id][j] * scale;
        }
        
        // Add bias term (stored in last position)
        representation[lightfm->no_components] += feature_weight * feature_biases[feature_id] * scale;
    }
}

float compute_prediction_from_repr(
    float *user_repr,
    float *item_repr,
    int no_components
) {
    float prediction = 0.0f;
    
    // Dot product of user and item embeddings
    for (int i = 0; i < no_components; i++) {
        prediction += user_repr[i] * item_repr[i];
    }
    
    // Add bias terms (stored in last position)
    prediction += user_repr[no_components] + item_repr[no_components];
    
    return prediction;
}

// Full WARP implementation
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
) {
    float *user_repr = (float*)malloc((lightfm->no_components + 1) * sizeof(float));
    float *pos_it_repr = (float*)malloc((lightfm->no_components + 1) * sizeof(float));
    float *neg_it_repr = (float*)malloc((lightfm->no_components + 1) * sizeof(float));
    
    const double MAX_LOSS = 10.0;
    const double MAX_REG_SCALE = 1000000.0;
    
    lightfm->learning_rate = learning_rate;
    
    for (int i = 0; i < num_examples; i++) {
        int row = shuffle_indices[i];
        int user_id = user_ids[row];
        int positive_item_id = item_ids[row];
        
        // Skip if not a positive example
        if (Y[row] <= 0) continue;
        
        float weight = sample_weight[row];
        
        // Compute user and positive item representations
        compute_representation(user_features, lightfm->user_features, lightfm->user_biases,
                             lightfm, user_id, lightfm->user_scale, user_repr);
        compute_representation(item_features, lightfm->item_features, lightfm->item_biases,
                             lightfm, positive_item_id, lightfm->item_scale, pos_it_repr);
        
        double positive_prediction = compute_prediction_from_repr(user_repr, pos_it_repr, lightfm->no_components);
        
        int sampled = 0;
        while (sampled < lightfm->max_sampled) {
            sampled++;
            
            // Sample negative item
            int negative_item_id = rand_r(&random_states[0]) % item_features->rows;
            
            compute_representation(item_features, lightfm->item_features, lightfm->item_biases,
                                 lightfm, negative_item_id, lightfm->item_scale, neg_it_repr);
            
            double negative_prediction = compute_prediction_from_repr(user_repr, neg_it_repr, lightfm->no_components);
            
            // Check if we found a violating negative example
            if (negative_prediction > positive_prediction - 1.0) {
                // Skip if negative is actually positive
                if (in_positives(negative_item_id, user_id, interactions)) {
                    continue;
                }
                
                // Compute WARP loss with importance weighting
                double loss = weight * log(fmax(1.0, floor((double)(item_features->rows - 1) / sampled)));
                
                // Clip gradients for numerical stability
                if (loss > MAX_LOSS) {
                    loss = MAX_LOSS;
                }
                
                // Apply WARP gradient updates
                warp_update(loss, item_features, user_features, user_id, positive_item_id,
                           negative_item_id, user_repr, pos_it_repr, neg_it_repr, lightfm,
                           item_alpha, user_alpha);
                break;
            }
        }
        
        // Apply regularization if scales get too large
        if (lightfm->item_scale > MAX_REG_SCALE || lightfm->user_scale > MAX_REG_SCALE) {
            regularize(lightfm, item_alpha, user_alpha);
        }
    }
    
    free(user_repr);
    free(pos_it_repr);
    free(neg_it_repr);
    
    // Final regularization
    regularize(lightfm, item_alpha, user_alpha);
    
    return 0;
}

// Basic prediction implementation
int predict_lightfm(
    CSRMatrix *item_features,
    CSRMatrix *user_features,
    int *user_ids,
    int *item_ids,
    float *predictions,
    FastLightFM *lightfm,
    int num_threads,
    int num_examples
) {
    float *user_repr = (float*)malloc((lightfm->no_components + 1) * sizeof(float));
    float *item_repr = (float*)malloc((lightfm->no_components + 1) * sizeof(float));
    
    for (int i = 0; i < num_examples; i++) {
        int user_id = user_ids[i];
        int item_id = item_ids[i];
        
        // Compute representations
        compute_representation(user_features, lightfm->user_features, lightfm->user_biases,
                             lightfm, user_id, lightfm->user_scale, user_repr);
        compute_representation(item_features, lightfm->item_features, lightfm->item_biases,
                             lightfm, item_id, lightfm->item_scale, item_repr);
        
        // Compute prediction
        predictions[i] = compute_prediction_from_repr(user_repr, item_repr, lightfm->no_components);
    }
    
    free(user_repr);
    free(item_repr);
    
    return 0;
}

// Placeholder implementations for other functions
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
) {
    (void)item_features; (void)user_features; (void)user_ids; (void)item_ids;
    (void)Y; (void)sample_weight; (void)shuffle_indices; (void)lightfm;
    (void)learning_rate; (void)item_alpha; (void)user_alpha; 
    (void)num_threads; (void)num_examples;
    return -1; /* Not implemented */
}

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
) {
    (void)item_features; (void)user_features; (void)interactions; (void)user_ids; (void)item_ids;
    (void)Y; (void)sample_weight; (void)shuffle_indices; (void)lightfm;
    (void)learning_rate; (void)item_alpha; (void)user_alpha; 
    (void)num_threads; (void)random_states; (void)num_examples;
    return -1; /* Not implemented */
}

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
) {
    (void)item_features; (void)user_features; (void)data; (void)user_ids;
    (void)shuffle_indices; (void)lightfm; (void)learning_rate; 
    (void)item_alpha; (void)user_alpha; (void)k; (void)n;
    (void)num_threads; (void)random_states; (void)num_examples;
    return -1; /* Not implemented */
}

int predict_ranks(
    CSRMatrix *item_features,
    CSRMatrix *user_features,
    CSRMatrix *test_interactions,
    CSRMatrix *train_interactions,
    float *ranks,
    FastLightFM *lightfm,
    int num_threads
) {
    (void)item_features; (void)user_features; (void)test_interactions;
    (void)train_interactions; (void)ranks; (void)lightfm; (void)num_threads;
    return -1; /* Not implemented */
}

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
) {
    double sum_learning_rate = 0.0;
    
    if (adadelta) {
        for (int i = start; i < stop; i++) {
            int feature = feature_indices->indices[i];
            float feature_weight = feature_indices->data[i];
            
            gradients[feature] = rho * gradients[feature] + (1 - rho) * pow(feature_weight * gradient, 2);
            double local_learning_rate = sqrt(momentum[feature] + eps) / sqrt(gradients[feature] + eps);
            double update = local_learning_rate * gradient * feature_weight;
            momentum[feature] = rho * momentum[feature] + (1 - rho) * update * update;
            biases[feature] -= update;
            
            // Lazy regularization
            biases[feature] *= (1.0 + alpha * local_learning_rate);
            sum_learning_rate += local_learning_rate;
        }
    } else {
        for (int i = start; i < stop; i++) {
            int feature = feature_indices->indices[i];
            float feature_weight = feature_indices->data[i];
            
            double local_learning_rate = learning_rate / sqrt(gradients[feature]);
            biases[feature] -= local_learning_rate * feature_weight * gradient;
            gradients[feature] += pow(gradient * feature_weight, 2);
            
            // Lazy regularization
            biases[feature] *= (1.0 + alpha * local_learning_rate);
            sum_learning_rate += local_learning_rate;
        }
    }
    
    return sum_learning_rate;
}

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
) {
    double sum_learning_rate = 0.0;
    
    if (adadelta) {
        for (int i = start; i < stop; i++) {
            int feature = feature_indices->indices[i];
            float feature_weight = feature_indices->data[i];
            
            gradients[feature][component] = rho * gradients[feature][component] + 
                                          (1 - rho) * pow(feature_weight * gradient, 2);
            double local_learning_rate = sqrt(momentum[feature][component] + eps) / 
                                       sqrt(gradients[feature][component] + eps);
            double update = local_learning_rate * gradient * feature_weight;
            momentum[feature][component] = rho * momentum[feature][component] + 
                                         (1 - rho) * update * update;
            features[feature][component] -= update;
            
            // Lazy regularization
            features[feature][component] *= (1.0 + alpha * local_learning_rate);
            sum_learning_rate += local_learning_rate;
        }
    } else {
        for (int i = start; i < stop; i++) {
            int feature = feature_indices->indices[i];
            float feature_weight = feature_indices->data[i];
            
            double local_learning_rate = learning_rate / sqrt(gradients[feature][component]);
            features[feature][component] -= local_learning_rate * feature_weight * gradient;
            gradients[feature][component] += pow(gradient * feature_weight, 2);
            
            // Lazy regularization
            features[feature][component] *= (1.0 + alpha * local_learning_rate);
            sum_learning_rate += local_learning_rate;
        }
    }
    
    return sum_learning_rate;
}

// WARP gradient update function
void warp_update(
    double loss,
    CSRMatrix *item_features,
    CSRMatrix *user_features,
    int user_id,
    int positive_item_id,
    int negative_item_id,
    float *user_repr,
    float *pos_it_repr,
    float *neg_it_repr,
    FastLightFM *lightfm,
    double item_alpha,
    double user_alpha
) {
    double avg_learning_rate = 0.0;
    
    // Get the iteration ranges for features
    int positive_item_start = item_features->indptr[positive_item_id];
    int positive_item_stop = item_features->indptr[positive_item_id + 1];
    int negative_item_start = item_features->indptr[negative_item_id];
    int negative_item_stop = item_features->indptr[negative_item_id + 1];
    int user_start = user_features->indptr[user_id];
    int user_stop = user_features->indptr[user_id + 1];
    
    // Update biases
    avg_learning_rate += update_biases(item_features, positive_item_start, positive_item_stop,
                                     lightfm->item_biases, lightfm->item_bias_gradients,
                                     lightfm->item_bias_momentum, -loss, lightfm->adadelta,
                                     lightfm->learning_rate, item_alpha, lightfm->rho, lightfm->eps);
    
    avg_learning_rate += update_biases(item_features, negative_item_start, negative_item_stop,
                                     lightfm->item_biases, lightfm->item_bias_gradients,
                                     lightfm->item_bias_momentum, loss, lightfm->adadelta,
                                     lightfm->learning_rate, item_alpha, lightfm->rho, lightfm->eps);
    
    avg_learning_rate += update_biases(user_features, user_start, user_stop,
                                     lightfm->user_biases, lightfm->user_bias_gradients,
                                     lightfm->user_bias_momentum, loss, lightfm->adadelta,
                                     lightfm->learning_rate, user_alpha, lightfm->rho, lightfm->eps);
    
    // Update latent representations
    for (int i = 0; i < lightfm->no_components; i++) {
        float user_component = user_repr[i];
        float positive_item_component = pos_it_repr[i];
        float negative_item_component = neg_it_repr[i];
        
        avg_learning_rate += update_features(item_features, lightfm->item_features,
                                           lightfm->item_feature_gradients, lightfm->item_feature_momentum,
                                           i, positive_item_start, positive_item_stop,
                                           -loss * user_component, lightfm->adadelta,
                                           lightfm->learning_rate, item_alpha, lightfm->rho, lightfm->eps);
        
        avg_learning_rate += update_features(item_features, lightfm->item_features,
                                           lightfm->item_feature_gradients, lightfm->item_feature_momentum,
                                           i, negative_item_start, negative_item_stop,
                                           loss * user_component, lightfm->adadelta,
                                           lightfm->learning_rate, item_alpha, lightfm->rho, lightfm->eps);
        
        avg_learning_rate += update_features(user_features, lightfm->user_features,
                                           lightfm->user_feature_gradients, lightfm->user_feature_momentum,
                                           i, user_start, user_stop,
                                           loss * (negative_item_component - positive_item_component),
                                           lightfm->adadelta, lightfm->learning_rate, user_alpha,
                                           lightfm->rho, lightfm->eps);
    }
    
    // Update regularization scales
    lightfm->item_scale *= (1.0 + item_alpha * avg_learning_rate);
    lightfm->user_scale *= (1.0 + user_alpha * avg_learning_rate);
}

// Regularization function
void regularize(FastLightFM *lightfm, double item_alpha, double user_alpha) {
    // Apply regularization to all features
    for (int i = 0; i < lightfm->n_item_features; i++) {
        for (int j = 0; j < lightfm->no_components; j++) {
            lightfm->item_features[i][j] *= lightfm->item_scale;
        }
        lightfm->item_biases[i] *= lightfm->item_scale;
    }
    
    for (int i = 0; i < lightfm->n_user_features; i++) {
        for (int j = 0; j < lightfm->no_components; j++) {
            lightfm->user_features[i][j] *= lightfm->user_scale;
        }
        lightfm->user_biases[i] *= lightfm->user_scale;
    }
    
    // Reset scales
    lightfm->item_scale = 1.0;
    lightfm->user_scale = 1.0;
}

float precision_at_k(
    CSRMatrix *test_interactions,
    CSRMatrix *train_interactions,
    float *predictions,
    int k,
    int n_users,
    int n_items
) {
    (void)test_interactions; (void)train_interactions; (void)predictions;
    (void)k; (void)n_users; (void)n_items;
    return 0.0f; /* Not implemented */
}