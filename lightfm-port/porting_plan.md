# LightFM to PHP Porting Plan

## Overview
This document outlines the detailed plan for porting LightFM from Python to PHP using FFI bindings, following the zopfli-port methodology.

## Implementation Phases

### Phase 1: Core Algorithm Extraction and C Implementation

#### 1.1 Extract Cython Code
- Extract algorithms from `lightfm/_lightfm_fast.py`
- Identify core computational functions
- Map Cython types to C types
- Preserve algorithm logic and numerical precision

#### 1.2 C Implementation Structure
```c
// lightfm_core.h
typedef struct {
    int* indices;
    int* indptr;
    float* data;
    int rows, cols, nnz;
} CSRMatrix;

typedef struct {
    float* user_embeddings;
    float* item_embeddings; 
    float* user_biases;
    float* item_biases;
    // ... gradient and momentum arrays
    int n_users, n_items, n_components;
} LightFMModel;

// Core training functions
int fit_logistic(CSRMatrix* interactions, CSRMatrix* user_features, 
                CSRMatrix* item_features, LightFMModel* model, ...);
int fit_bpr(CSRMatrix* interactions, CSRMatrix* user_features,
           CSRMatrix* item_features, LightFMModel* model, ...);
int fit_warp(CSRMatrix* interactions, CSRMatrix* user_features,
            CSRMatrix* item_features, LightFMModel* model, ...);

// Prediction functions  
int predict_lightfm(CSRMatrix* user_features, CSRMatrix* item_features,
                   LightFMModel* model, float* predictions);
int predict_ranks(CSRMatrix* test_interactions, CSRMatrix* user_features,
                 CSRMatrix* item_features, LightFMModel* model, int* ranks);
```

#### 1.3 Priority Functions to Implement
1. **fit_warp()** - Used by main.py, most complex algorithm
2. **predict_lightfm()** - Core prediction function  
3. **precision_at_k()** - Evaluation metric
4. **CSR matrix operations** - Sparse matrix support
5. **Random number generation** - Consistent with Python

### Phase 2: PHP FFI Integration

#### 2.1 FFI Binding Architecture
```php
<?php
// php/FFIBindings.php
class LightFMFFI {
    private FFI $ffi;
    
    public function __construct() {
        $this->ffi = FFI::cdef("
            typedef struct { ... } CSRMatrix;
            typedef struct { ... } LightFMModel;
            
            int fit_warp(CSRMatrix* interactions, ...);
            int predict_lightfm(CSRMatrix* user_features, ...);
            // ... other function declarations
        ", "./c/liblightfm.so");
    }
    
    public function fitWarp(array $interactions, ...) {
        // Convert PHP arrays to C structs
        // Call C function via FFI
        // Handle return values and errors
    }
}
```

#### 2.2 Data Structure Conversion
- PHP associative arrays ↔ C structs
- PHP arrays ↔ C CSR matrices
- Handle memory management carefully
- Implement efficient serialization/deserialization

#### 2.3 Sparse Matrix Implementation
```php
<?php
// php/SparseMatrix.php
class CSRMatrix {
    public array $indices;
    public array $indptr;
    public array $data;
    public int $rows, $cols, $nnz;
    
    public function __construct(array $coo_data) {
        // Convert from COO to CSR format
        $this->convertCOOToCSR($coo_data);
    }
    
    public function toFFIStruct(): FFI\CData {
        // Convert to C struct for FFI calls
    }
}
```

### Phase 3: Data Processing Layer

#### 3.1 Dataset Class Implementation
```php
<?php  
// php/Dataset.php
class Dataset {
    private array $user_id_mapping = [];
    private array $item_id_mapping = [];
    private array $user_feature_mapping = [];
    private array $item_feature_mapping = [];
    
    public function fit(array $interactions): void {
        // Build ID mappings like Python version
    }
    
    public function buildInteractions(array $interactions): CSRMatrix {
        // Convert to sparse matrix format
    }
    
    public function buildUserFeatures(array $user_features): CSRMatrix {
        // Build user feature matrix
    }
    
    public function buildItemFeatures(array $item_features): CSRMatrix {
        // Build item feature matrix  
    }
}
```

#### 3.2 MovieLens Dataset Loader
```php
<?php
// php/MovieLensDataset.php
class MovieLensDataset {
    public function fetchMovielens(float $min_rating = 4.0): array {
        // Download and process MovieLens data
        // Return train/test split like Python version
        // Ensure identical preprocessing steps
    }
}
```

### Phase 4: Main LightFM Class

#### 4.1 LightFM PHP Implementation
```php
<?php
// php/LightFM.php
class LightFM {
    private LightFMFFI $ffi;
    private array $params;
    private ?LightFMModel $model = null;
    
    public function __construct(array $params = []) {
        $this->ffi = new LightFMFFI();
        $this->params = array_merge([
            'no_components' => 10,
            'loss' => 'warp', 
            'learning_rate' => 0.05,
            'item_alpha' => 0.0,
            'user_alpha' => 0.0,
            'random_state' => null
        ], $params);
        
        // Set random seed for reproducibility
        if ($this->params['random_state'] !== null) {
            mt_srand($this->params['random_state']);
        }
    }
    
    public function fit(CSRMatrix $interactions, array $options = []): void {
        // Initialize model if needed
        // Call appropriate C training function
        // Handle different loss functions
    }
    
    public function predict(array $user_ids, array $item_ids): array {
        // Call C prediction function
        // Return predictions array
    }
}
```

### Phase 5: Evaluation Metrics

#### 5.1 Evaluation Class
```php
<?php
// php/Evaluation.php  
class Evaluation {
    public static function precisionAtK(LightFM $model, CSRMatrix $test_interactions, 
                                       int $k = 10): float {
        // Port precision@k calculation from Python
        // Use same algorithm and numerical precision
    }
    
    public static function recallAtK(LightFM $model, CSRMatrix $test_interactions,
                                    int $k = 10): float {
        // Port recall@k calculation
    }
    
    public static function aucScore(LightFM $model, CSRMatrix $test_interactions): float {
        // Port AUC calculation
    }
}
```

### Phase 6: Fuzz Testing Framework

#### 6.1 Differential Testing
```php
<?php
// fuzz/fuzz_test.php
class LightFMFuzzTest {
    public function runDifferentialTest(): void {
        // Generate random test data
        // Run both Python and PHP versions
        // Compare outputs with tolerance
        // Report differences
    }
    
    public function testTraining(): void {
        // Test training algorithms
        // Compare final model parameters
        // Validate convergence behavior
    }
    
    public function testPrediction(): void {
        // Test prediction functions
        // Compare prediction accuracy
        // Test ranking consistency
    }
}
```

#### 6.2 Test Case Generation
- Random sparse matrices of various sizes
- Different hyperparameter combinations
- Edge cases (empty features, single user/item)
- Convergence tests with different epochs

### Phase 7: Integration and Optimization

#### 7.1 main.php Implementation
```php
<?php
// main.php - Equivalent to main.py
require_once 'php/LightFM.php';
require_once 'php/MovieLensDataset.php';
require_once 'php/Evaluation.php';

// Fixed seed for reproducible results
mt_srand(42);

// Load MovieLens dataset
$dataset = new MovieLensDataset();
$data = $dataset->fetchMovielens(5.0);

// Train model
$model = new LightFM([
    'loss' => 'warp',
    'random_state' => 42
]);

$model->fit($data['train'], [
    'epochs' => 30,
    'num_threads' => 2
]);

// Evaluate
$test_precision = Evaluation::precisionAtK($model, $data['test'], 5);
echo "Test precision@5: " . $test_precision . "\n";
?>
```

## Critical Success Factors

### 1. Numerical Precision
- Use identical floating-point precision (float32)
- Handle edge cases consistently
- Maintain gradient computation accuracy
- Validate against Python reference

### 2. Random Seed Handling
- Fix seeds in both versions
- Use consistent RNG algorithms
- Handle negative sampling identically
- Test reproducibility extensively

### 3. Sparse Matrix Operations
- Implement efficient CSR matrix operations
- Handle memory allocation carefully
- Optimize for large matrices
- Validate matrix operations

### 4. Memory Management
- Proper cleanup of C memory
- Handle PHP memory limits
- Optimize for large embedding matrices
- Profile memory usage

## Validation Strategy

### Unit Tests
- Individual function testing
- Matrix operation validation
- Edge case handling
- Error condition testing

### Integration Tests
- End-to-end workflow testing
- Multiple dataset testing
- Hyperparameter sensitivity
- Performance benchmarking

### Differential Tests
- Python vs PHP comparison
- Numerical precision validation
- Convergence behavior testing
- Random seed reproducibility

## Timeline and Milestones

### Week 1-2: Core C Implementation
- Extract and port Cython algorithms
- Implement sparse matrix operations
- Create build system
- Basic function testing

### Week 3-4: PHP FFI Integration
- Implement FFI bindings
- Create PHP wrapper classes
- Data structure conversion
- Integration testing

### Week 5-6: Full Implementation
- Complete LightFM class
- Dataset and evaluation classes
- main.php implementation
- Differential testing

### Week 7-8: Testing and Optimization
- Comprehensive fuzz testing
- Performance optimization
- Bug fixes and validation
- Documentation completion

This plan provides a comprehensive roadmap for successfully porting LightFM to PHP while maintaining numerical accuracy and providing robust validation through differential testing.