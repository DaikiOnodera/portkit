# LightFM Call Graph Analysis

## Overview
This document provides a comprehensive call graph analysis of the LightFM library, detailing the function dependencies, data flow, and key components needed for porting to PHP.

## Core Architecture

### 1. Main Classes and Entry Points

```
LightFM (lightfm/lightfm.py)
├── __init__(no_components, loss, learning_schedule, user_alpha, item_alpha, ...)
├── fit(interactions, user_features, item_features, sample_weight, epochs, ...)
├── fit_partial(interactions, user_features, item_features, ...)
├── predict(user_ids, item_ids, user_features, item_features, ...)
├── predict_rank(test_interactions, user_features, item_features, ...)
├── get_user_representations(features)
├── get_item_representations(features)
└── get_params() / set_params()
```

### 2. Training Flow Call Graph

```
LightFM.fit()
└── LightFM.fit_partial()
    ├── _check_finite() [input validation]
    ├── _construct_feature_matrices(user_features, item_features)
    ├── _process_sample_weight(interactions, sample_weight)
    ├── _initialize() [first call only]
    │   ├── _initialize_biases()
    │   ├── _initialize_embeddings()
    │   └── _initialize_gradients()
    └── _run_epoch() [for each epoch]
        ├── _get_positives_lookup_matrix()
        ├── _get_lightfm_data()
        └── [Cython Functions]
            ├── fit_logistic()
            ├── fit_bpr()
            ├── fit_warp()
            └── fit_warp_kos()
```

### 3. Prediction Flow Call Graph

```
LightFM.predict()
├── _check_initialized()
├── _construct_feature_matrices()
├── _get_lightfm_data()
└── predict_lightfm() [Cython]

LightFM.predict_rank()
├── _check_initialized()
├── _construct_feature_matrices()
├── _get_lightfm_data()
└── predict_ranks() [Cython]
```

### 4. Data Processing Call Graph

```
Dataset (lightfm/data.py)
├── fit()
│   └── fit_partial()
│       ├── _check_new_user_item_ids()
│       └── _update_id_mappings()
├── build_interactions()
│   ├── _unpack_datum()
│   └── _build_coo_matrix()
├── build_user_features()
│   └── _FeatureBuilder.build()
│       ├── _process_features()
│       └── _iter_features()
└── build_item_features()
    └── _FeatureBuilder.build()
```

## 5. Cython Fast Functions (_lightfm_fast.py)

### Core Training Functions
```
fit_logistic(CSRMatrix interactions, CSRMatrix user_features, CSRMatrix item_features, ...)
├── sample_range()
├── update_biases()
├── update_embeddings()
└── compute_loss()

fit_bpr(CSRMatrix interactions, ...)
├── sample_negative_item()
├── compute_bpr_loss()
├── update_biases()
└── update_embeddings()

fit_warp(CSRMatrix interactions, ...)
├── sample_negative_item()
├── compute_warp_loss()
├── rank_violation_count()
├── update_biases()
└── update_embeddings()

fit_warp_kos(CSRMatrix interactions, ...)
├── sample_negative_items()
├── compute_kos_loss()
├── update_biases()
└── update_embeddings()
```

### Prediction Functions
```
predict_lightfm(CSRMatrix user_features, CSRMatrix item_features, ...)
├── dot_product()
├── compute_user_repr()
├── compute_item_repr()
└── add_biases()

predict_ranks(CSRMatrix test_interactions, ...)
├── predict_lightfm()
├── argsort()
└── rank_computation()
```

## 6. Matrix Operations and Data Structures

### CSRMatrix Operations (Cython)
```
CSRMatrix
├── get_row()
├── get_col()
├── get_row_start()
├── get_row_end()
├── multiply()
└── add()

FastLightFM
├── get_user_embedding()
├── get_item_embedding()
├── get_user_bias()
├── get_item_bias()
├── update_user_embedding()
├── update_item_embedding()
├── update_user_bias()
└── update_item_bias()
```

### Random Number Generation
```
rand_r() [Custom implementation]
├── uniform_sampling()
├── negative_item_sampling()
└── bootstrapping()
```

## 7. Evaluation Call Graph

```
Evaluation Functions (lightfm/evaluation.py)
├── precision_at_k()
│   ├── predict_ranks()
│   ├── _get_test_interactions_matrix()
│   └── _precision_at_k_score()
├── recall_at_k()
│   ├── predict_ranks()
│   └── _recall_at_k_score()
├── auc_score()
│   ├── predict()
│   └── _auc_score()
└── reciprocal_rank()
    ├── predict_ranks()
    └── _reciprocal_rank_score()
```

## 8. Dependencies and External Calls

### NumPy Operations
```
numpy
├── array creation and manipulation
├── random number generation
├── mathematical operations
├── sparse matrix conversion
└── array indexing and slicing
```

### SciPy Operations
```
scipy.sparse
├── csr_matrix
├── coo_matrix
├── hstack()
├── vstack()
└── matrix operations
```

### Scikit-learn
```
sklearn.preprocessing
└── normalize()
```

## 9. Key Functions for PHP Porting

### Critical Path Functions (Must Port)
1. **fit_logistic()** - Core logistic loss training
2. **fit_bpr()** - Bayesian Personalized Ranking
3. **fit_warp()** - WARP loss training
4. **predict_lightfm()** - Score prediction
5. **predict_ranks()** - Ranking prediction

### Supporting Functions (Important)
1. **CSRMatrix operations** - Sparse matrix handling
2. **Random sampling** - Negative item sampling
3. **Gradient updates** - SGD with momentum
4. **Feature matrix construction** - Data preprocessing
5. **Evaluation metrics** - Model validation

### Data Structures to Implement
1. **CSRMatrix** - Compressed Sparse Row matrix
2. **FastLightFM** - Model parameter storage
3. **Dataset** - Data preprocessing and mapping
4. **Feature mappings** - ID to index conversions

## 10. Porting Strategy

### Phase 1: Core Algorithm
- Port Cython functions to C
- Create PHP FFI bindings
- Implement basic matrix operations

### Phase 2: Data Processing
- Port Dataset class functionality
- Implement sparse matrix operations
- Create feature builders

### Phase 3: Evaluation
- Port evaluation metrics
- Implement ranking functions
- Create validation framework

### Phase 4: Integration
- Integrate all components
- Add fuzz testing
- Optimize performance

## 11. Function Complexity Analysis

### High Complexity (Cython/C required)
- Matrix factorization algorithms
- Sparse matrix operations
- Gradient computations
- Negative sampling

### Medium Complexity (PHP feasible)
- Data preprocessing
- Feature mapping
- Evaluation metrics
- I/O operations

### Low Complexity (Pure PHP)
- Parameter validation
- Configuration management
- Result formatting
- Dataset loading

This call graph provides the foundation for understanding LightFM's architecture and planning the PHP port with FFI bindings.