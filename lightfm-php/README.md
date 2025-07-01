# LightFM PHP

PHP port of the LightFM Python library using FFI bindings to C implementation for hybrid recommender systems.

## Installation

Install via Composer:

```bash
composer require lightfm/lightfm-php
```

## Requirements

- PHP 8.0 or higher
- FFI extension enabled
- The bundled C library (`liblightfm.so`)

## Usage

```php
<?php

require_once 'vendor/autoload.php';

// Create a LightFM model
$model = new LightFM([
    'no_components' => 30,
    'loss' => 'warp',
    'learning_rate' => 0.05
]);

// Fit the model with interaction data
$interactions = new CSRMatrix($data, $rows, $cols);
$model->fit($interactions);

// Get recommendations
$predictions = $model->predict($userIds, $itemIds);
```

## Features

- **Hybrid Recommender Systems**: Supports both collaborative filtering and content-based filtering
- **Multiple Loss Functions**: WARP, BPR, and logistic loss functions
- **Fast C Implementation**: Uses FFI bindings to a compiled C library for performance
- **CSR Matrix Support**: Efficient sparse matrix operations
- **Evaluation Metrics**: Built-in precision, recall, and AUC metrics

## Classes

- `LightFM`: Main model class for training and prediction
- `CSRMatrix`: Sparse matrix implementation for efficient data handling
- `Evaluation`: Evaluation metrics for model performance
- `MovieLensDataset`: Dataset loader for MovieLens data
- `LightFMFFI`: FFI bindings to the C library

## License

Apache-2.0