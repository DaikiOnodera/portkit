<?php

require_once 'vendor/autoload.php';

echo "Testing LightFM PHP Installation...\n";

try {
    // Test basic class instantiation
    echo "1. Testing LightFM class instantiation...\n";
    $model = new LightFM([
        'no_components' => 10,
        'loss' => 'warp',
        'learning_rate' => 0.05
    ]);
    echo "   ✓ LightFM model created successfully\n";
    
    // Test CSR Matrix
    echo "2. Testing CSRMatrix class...\n";
    $data = [1.0, 2.0];
    $indices = [0, 1];
    $indptr = [0, 1, 2];
    $matrix = new CSRMatrix($indices, $indptr, $data, 2, 2);
    echo "   ✓ CSRMatrix created successfully\n";
    
    // Test Evaluation class
    echo "3. Testing Evaluation class...\n";
    $evaluation = new Evaluation();
    echo "   ✓ Evaluation class instantiated successfully\n";
    
    // Test MovieLens dataset
    echo "4. Testing MovieLensDataset class...\n";
    $dataset = new MovieLensDataset();
    echo "   ✓ MovieLensDataset class instantiated successfully\n";
    
    echo "\n✅ All tests passed! LightFM PHP package is working correctly.\n";
    
} catch (Exception $e) {
    echo "\n❌ Test failed with error: " . $e->getMessage() . "\n";
    echo "Stack trace:\n" . $e->getTraceAsString() . "\n";
    exit(1);
}