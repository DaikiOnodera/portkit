# CLAUDE.md - LightFM to PHP Porting Project

This file provides guidance to Claude Code when working on the LightFM to PHP porting project.

## Project Overview

This project ports the LightFM Python recommendation system library to PHP using FFI bindings, following the same methodology as the zopfli-port project. The goal is to create a stable PHP implementation with fuzz testing to validate equivalence between Python and PHP versions.

## Commands

### Python Environment Setup
- `source ~/py311/bin/activate` - Activate Python 3.11 environment
- `cd lightfm-port/lightfm && pip install -e .` - Install lightfm in development mode

### Development Commands  
- `php main.php` - Run PHP version (equivalent to main.py)
- `python main.py` - Run Python reference version
- `php fuzz_test.php` - Run differential fuzz tests between Python and PHP

### Testing and Validation
- `php -l *.php` - Check PHP syntax
- `php composer.phar install` - Install PHP dependencies
- `php vendor/bin/phpunit tests/` - Run PHPUnit tests

## Architecture

### Core Components

1. **lightfm-port/main.py** - Python reference implementation
2. **lightfm-port/main.php** - PHP equivalent implementation
3. **lightfm-port/php/** - PHP FFI bindings and wrapper classes
4. **lightfm-port/c/** - C implementations of core algorithms
5. **lightfm-port/fuzz/** - Differential fuzz testing framework

### Expected Project Structure
```
lightfm-port/
├── lightfm/              # Original LightFM Python library
├── main.py               # Python reference implementation
├── main.php              # PHP equivalent
├── call_graph.md         # Function dependency analysis
├── CLAUDE.md             # This file
├── php/                  # PHP implementation
│   ├── LightFM.php       # Main LightFM class
│   ├── Dataset.php       # Data processing
│   ├── Evaluation.php    # Evaluation metrics
│   └── FFIBindings.php   # C FFI bindings
├── c/                    # C implementations
│   ├── lightfm_core.c    # Core algorithms
│   ├── lightfm_core.h    # Header file
│   └── Makefile          # Build system
└── fuzz/                 # Fuzz testing
    ├── fuzz_test.php     # Main fuzz test runner
    └── test_cases/       # Individual test cases
```

## Implementation Strategy

### Phase 1: Core Algorithm Porting
1. Extract Cython code from LightFM Python library
2. Port critical algorithms to C:
   - `fit_logistic()` - Logistic loss training
   - `fit_bpr()` - Bayesian Personalized Ranking
   - `fit_warp()` - WARP loss training
   - `predict_lightfm()` - Score prediction
   - `predict_ranks()` - Ranking prediction

### Phase 2: PHP FFI Integration
1. Create C shared library with exported functions
2. Implement PHP FFI bindings
3. Create PHP wrapper classes mimicking Python API
4. Handle sparse matrix operations (CSR format)

### Phase 3: Data Processing
1. Port Dataset class functionality
2. Implement feature matrix construction
3. Handle user/item ID mapping
4. Support MovieLens dataset loading

### Phase 4: Evaluation and Testing
1. Port evaluation metrics (precision@k, recall@k, AUC)
2. Create differential fuzz tests
3. Validate numerical equivalence
4. Performance benchmarking

## Key Requirements

### Reproducible Results
- **CRITICAL**: Fix random seed in both Python and PHP versions
- Use fixed seeds for:
  - Model initialization
  - Negative sampling
  - Data shuffling
  - Dataset splitting

### Numerical Precision
- Use consistent floating-point precision (float32)
- Handle sparse matrix operations identically
- Maintain gradient computation accuracy
- Validate convergence behavior

### Memory Management
- Efficient sparse matrix handling
- Proper memory cleanup in C code
- Handle large embedding matrices
- Optimize for PHP memory constraints

## Development Principles

### Code Quality
- Follow PSR-12 PHP coding standards
- Use type hints where possible
- Implement proper error handling
- Document all public APIs

### Testing Strategy
- Differential testing between Python and PHP
- Unit tests for individual components
- Integration tests for full workflows
- Fuzz testing with random inputs

### Performance Considerations
- Minimize PHP-C FFI calls
- Batch operations where possible
- Use efficient data structures
- Profile critical paths

## Critical Functions to Port

### Training Functions (from _lightfm_fast.py)
```c
// Core training algorithms
int fit_logistic(CSRMatrix* interactions, CSRMatrix* user_features, ...);
int fit_bpr(CSRMatrix* interactions, CSRMatrix* user_features, ...);
int fit_warp(CSRMatrix* interactions, CSRMatrix* user_features, ...);
int fit_warp_kos(CSRMatrix* interactions, CSRMatrix* user_features, ...);
```

### Prediction Functions
```c
// Prediction and ranking
int predict_lightfm(CSRMatrix* user_features, CSRMatrix* item_features, ...);
int predict_ranks(CSRMatrix* test_interactions, ...);
```

### Matrix Operations
```c
// Sparse matrix support
typedef struct {
    int* indices;
    int* indptr; 
    float* data;
    int rows, cols, nnz;
} CSRMatrix;

int csr_dot_product(CSRMatrix* a, CSRMatrix* b, float* result);
int csr_add(CSRMatrix* a, CSRMatrix* b, CSRMatrix* result);
```

## Dependencies

### PHP Requirements
- PHP 8.1+ with FFI extension enabled
- Composer for dependency management
- OpenMP support (optional, for parallelization)

### C Dependencies
- GCC or Clang compiler
- BLAS/LAPACK libraries (optional, for optimization)
- OpenMP (optional, for parallel computation)

### External Libraries
- SuiteSparse (for advanced sparse matrix operations)
- Or custom sparse matrix implementation

## Validation Approach

### Differential Testing
1. Run identical test cases on both Python and PHP versions
2. Compare outputs with tolerance for floating-point differences
3. Test edge cases and boundary conditions
4. Validate convergence properties

### Test Cases
- Simple matrix factorization problems
- MovieLens dataset end-to-end
- Various loss functions (logistic, BPR, WARP)
- Different hyperparameters
- Edge cases (empty features, single user/item)

## File Organization

### Python Reference (main.py)
```python
from lightfm import LightFM
from lightfm.datasets import fetch_movielens
from lightfm.evaluation import precision_at_k

# Fixed seed for reproducibility
import numpy as np
np.random.seed(42)

# Load data, train model, evaluate
data = fetch_movielens(min_rating=5.0)
model = LightFM(loss='warp', random_state=42)
model.fit(data['train'], epochs=30, num_threads=2)
test_precision = precision_at_k(model, data['test'], k=5).mean()
```

### PHP Implementation (main.php)
```php
<?php
require_once 'php/LightFM.php';
require_once 'php/Dataset.php';
require_once 'php/Evaluation.php';

// Fixed seed for reproducibility
mt_srand(42);

// Load data, train model, evaluate
$dataset = new MovieLensDataset();
$data = $dataset->fetchMovielens(5.0);
$model = new LightFM(['loss' => 'warp', 'random_state' => 42]);
$model->fit($data['train'], ['epochs' => 30, 'num_threads' => 2]);
$test_precision = Evaluation::precisionAtK($model, $data['test'], 5);
?>
```

## Success Criteria

1. **Functional Equivalence**: PHP version produces same results as Python version
2. **Performance**: PHP version runs within 2x of Python performance
3. **Stability**: No memory leaks or crashes under normal usage
4. **Test Coverage**: >90% code coverage with differential tests
5. **Documentation**: Complete API documentation and usage examples

## Notes for Claude Code

- Always use `source ~/py311/bin/activate` before Python operations
- Prioritize numerical accuracy over performance optimizations
- Test frequently with small datasets before scaling up
- Pay special attention to random seed handling for reproducibility
- Use the call_graph.md file to understand function dependencies
- Follow the zopfli-port project structure as a reference model