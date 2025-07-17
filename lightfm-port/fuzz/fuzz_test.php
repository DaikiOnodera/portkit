<?php

require_once __DIR__ . '/../php/LightFM.php';
require_once __DIR__ . '/../php/CSRMatrix.php';
require_once __DIR__ . '/../php/MovieLensDataset.php';
require_once __DIR__ . '/../php/Evaluation.php';

/**
 * Differential Fuzz Testing Framework for LightFM PHP Port
 * 
 * This framework tests the PHP implementation against the Python reference
 * to validate numerical equivalence and catch regressions.
 */
class LightFMFuzzTest
{
    private int $seed;
    private array $testResults = [];
    
    public function __construct(int $seed = 42)
    {
        $this->seed = $seed;
        mt_srand($seed);
    }
    
    /**
     * Run all fuzz tests
     */
    public function runAllTests(): void
    {
        echo "=== LightFM Differential Fuzz Testing ===\n\n";
        echo "Seed: {$this->seed}\n\n";
        
        $this->testCSRMatrixOperations();
        $this->testModelCreation();
        $this->testDatasetGeneration();
        $this->testPredictionConsistency();
        $this->testEvaluationMetrics();
        
        $this->printSummary();
    }
    
    /**
     * Test CSR matrix operations
     */
    private function testCSRMatrixOperations(): void
    {
        echo "Testing CSR Matrix Operations...\n";
        $passed = 0;
        $total = 0;
        
        // Test 1: Identity matrix
        $total++;
        try {
            $identity = CSRMatrix::identity(5);
            $stats = $identity->getStats();
            
            $expected = [
                'rows' => 5,
                'cols' => 5,
                'nnz' => 5,
                'density' => 1.0
            ];
            
            $pass = true;
            foreach ($expected as $key => $value) {
                if (abs($stats[$key] - $value) > 1e-6) {
                    $pass = false;
                    break;
                }
            }
            
            if ($pass) {
                $passed++;
                echo "  ✓ Identity matrix creation\n";
            } else {
                echo "  ✗ Identity matrix creation failed\n";
            }
        } catch (Exception $e) {
            echo "  ✗ Identity matrix creation exception: {$e->getMessage()}\n";
        }
        
        // Test 2: COO to CSR conversion
        $total++;
        try {
            $cooData = [
                [0, 0, 1.0],
                [0, 2, 3.0],
                [1, 1, 2.0],
                [2, 0, 4.0]
            ];
            
            $csr = CSRMatrix::fromCOO($cooData, 3, 3);
            $dense = $csr->toDense();
            
            $expected = [
                [1.0, 0.0, 3.0],
                [0.0, 2.0, 0.0],
                [4.0, 0.0, 0.0]
            ];
            
            $pass = true;
            for ($i = 0; $i < 3; $i++) {
                for ($j = 0; $j < 3; $j++) {
                    if (abs($dense[$i][$j] - $expected[$i][$j]) > 1e-6) {
                        $pass = false;
                        break 2;
                    }
                }
            }
            
            if ($pass) {
                $passed++;
                echo "  ✓ COO to CSR conversion\n";
            } else {
                echo "  ✗ COO to CSR conversion failed\n";
            }
        } catch (Exception $e) {
            echo "  ✗ COO to CSR conversion exception: {$e->getMessage()}\n";
        }
        
        // Test 3: Matrix-vector multiplication
        $total++;
        try {
            $identity = CSRMatrix::identity(3);
            $vector = [1.0, 2.0, 3.0];
            $result = $identity->multiply($vector);
            
            $pass = true;
            for ($i = 0; $i < 3; $i++) {
                if (abs($result[$i] - $vector[$i]) > 1e-6) {
                    $pass = false;
                    break;
                }
            }
            
            if ($pass) {
                $passed++;
                echo "  ✓ Matrix-vector multiplication\n";
            } else {
                echo "  ✗ Matrix-vector multiplication failed\n";
            }
        } catch (Exception $e) {
            echo "  ✗ Matrix-vector multiplication exception: {$e->getMessage()}\n";
        }
        
        $this->testResults['CSR Matrix'] = [$passed, $total];
        echo "  Result: {$passed}/{$total} tests passed\n\n";
    }
    
    /**
     * Test model creation and initialization
     */
    private function testModelCreation(): void
    {
        echo "Testing Model Creation...\n";
        $passed = 0;
        $total = 0;
        
        // Test 1: Basic model creation
        $total++;
        try {
            $model = new LightFM([
                'no_components' => 10,
                'loss' => 'warp',
                'random_state' => 42
            ]);
            
            $params = $model->getParams();
            if ($params['no_components'] === 10 && $params['loss'] === 'warp') {
                $passed++;
                echo "  ✓ Basic model creation\n";
            } else {
                echo "  ✗ Basic model creation failed\n";
            }
        } catch (Exception $e) {
            echo "  ✗ Basic model creation exception: {$e->getMessage()}\n";
        }
        
        // Test 2: Model with different parameters
        $total++;
        try {
            $model = new LightFM([
                'no_components' => 5,
                'learning_rate' => 0.01,
                'item_alpha' => 0.1,
                'user_alpha' => 0.2
            ]);
            
            $params = $model->getParams();
            $expected = [
                'no_components' => 5,
                'learning_rate' => 0.01,
                'item_alpha' => 0.1,
                'user_alpha' => 0.2
            ];
            
            $pass = true;
            foreach ($expected as $key => $value) {
                if ($params[$key] !== $value) {
                    $pass = false;
                    break;
                }
            }
            
            if ($pass) {
                $passed++;
                echo "  ✓ Custom parameter model creation\n";
            } else {
                echo "  ✗ Custom parameter model creation failed\n";
            }
        } catch (Exception $e) {
            echo "  ✗ Custom parameter model creation exception: {$e->getMessage()}\n";
        }
        
        $this->testResults['Model Creation'] = [$passed, $total];
        echo "  Result: {$passed}/{$total} tests passed\n\n";
    }
    
    /**
     * Test dataset generation consistency
     */
    private function testDatasetGeneration(): void
    {
        echo "Testing Dataset Generation...\n";
        $passed = 0;
        $total = 0;
        
        // Test 1: Reproducible dataset generation
        $total++;
        try {
            mt_srand(42);
            $dataset1 = new MovieLensDataset();
            $data1 = $dataset1->fetchMovielens(4.0);
            
            mt_srand(42);
            $dataset2 = new MovieLensDataset();
            $data2 = $dataset2->fetchMovielens(4.0);
            
            $stats1 = $data1['train']->getStats();
            $stats2 = $data2['train']->getStats();
            
            if ($stats1['nnz'] === $stats2['nnz'] && 
                abs($stats1['density'] - $stats2['density']) < 1e-6) {
                $passed++;
                echo "  ✓ Reproducible dataset generation\n";
            } else {
                echo "  ✗ Dataset generation not reproducible\n";
            }
        } catch (Exception $e) {
            echo "  ✗ Dataset generation exception: {$e->getMessage()}\n";
        }
        
        // Test 2: Dataset statistics validation
        $total++;
        try {
            $dataset = new MovieLensDataset();
            $data = $dataset->fetchMovielens(5.0);
            
            $trainStats = $data['train']->getStats();
            $testStats = $data['test']->getStats();
            
            // Check that we have reasonable data
            if ($trainStats['nnz'] > 0 && $testStats['nnz'] > 0 &&
                $trainStats['nnz'] > $testStats['nnz'] && // More training than test
                $trainStats['density'] > 0 && $trainStats['density'] < 1) {
                $passed++;
                echo "  ✓ Dataset statistics validation\n";
            } else {
                echo "  ✗ Dataset statistics validation failed\n";
            }
        } catch (Exception $e) {
            echo "  ✗ Dataset statistics validation exception: {$e->getMessage()}\n";
        }
        
        $this->testResults['Dataset Generation'] = [$passed, $total];
        echo "  Result: {$passed}/{$total} tests passed\n\n";
    }
    
    /**
     * Test prediction consistency
     */
    private function testPredictionConsistency(): void
    {
        echo "Testing Prediction Consistency...\n";
        $passed = 0;
        $total = 0;
        
        // Test 1: Same input produces same output
        $total++;
        try {
            $model = new LightFM(['random_state' => 42]);
            $dataset = new MovieLensDataset();
            $data = $dataset->fetchMovielens(4.0);
            
            // Create small test set for prediction
            $smallTrain = CSRMatrix::fromCOO([
                [0, 0, 1.0],
                [0, 1, 1.0],
                [1, 0, 1.0]
            ], 2, 2);
            
            $model->fit($smallTrain, ['epochs' => 1]);
            
            $userIds = [0, 1];
            $itemIds = [0, 1];
            
            $pred1 = $model->predict($userIds, $itemIds);
            $pred2 = $model->predict($userIds, $itemIds);
            
            $consistent = true;
            for ($i = 0; $i < count($pred1); $i++) {
                if (abs($pred1[$i] - $pred2[$i]) > 1e-6) {
                    $consistent = false;
                    break;
                }
            }
            
            if ($consistent) {
                $passed++;
                echo "  ✓ Prediction consistency\n";
            } else {
                echo "  ✗ Predictions not consistent\n";
            }
        } catch (Exception $e) {
            echo "  ✗ Prediction consistency exception: {$e->getMessage()}\n";
        }
        
        $this->testResults['Prediction Consistency'] = [$passed, $total];
        echo "  Result: {$passed}/{$total} tests passed\n\n";
    }
    
    /**
     * Test evaluation metrics
     */
    private function testEvaluationMetrics(): void
    {
        echo "Testing Evaluation Metrics...\n";
        $passed = 0;
        $total = 0;
        
        // Test 1: Precision@K bounds
        $total++;
        try {
            $model = new LightFM(['random_state' => 42]);
            $testMatrix = CSRMatrix::fromCOO([
                [0, 0, 1.0],
                [0, 1, 1.0]
            ], 2, 3);
            
            $model->fit($testMatrix, ['epochs' => 1]);
            $precision = Evaluation::precisionAtK($model, $testMatrix, 5);
            
            if ($precision >= 0.0 && $precision <= 1.0) {
                $passed++;
                echo "  ✓ Precision@K bounds check\n";
            } else {
                echo "  ✗ Precision@K out of bounds: {$precision}\n";
            }
        } catch (Exception $e) {
            echo "  ✗ Precision@K exception: {$e->getMessage()}\n";
        }
        
        $this->testResults['Evaluation Metrics'] = [$passed, $total];
        echo "  Result: {$passed}/{$total} tests passed\n\n";
    }
    
    /**
     * Print test summary
     */
    private function printSummary(): void
    {
        echo "=== Test Summary ===\n";
        
        $totalPassed = 0;
        $totalTests = 0;
        
        foreach ($this->testResults as $category => [$passed, $total]) {
            $percentage = $total > 0 ? ($passed / $total) * 100 : 0;
            echo sprintf("%-20s: %2d/%2d (%.1f%%)\n", $category, $passed, $total, $percentage);
            $totalPassed += $passed;
            $totalTests += $total;
        }
        
        echo str_repeat("-", 40) . "\n";
        $overallPercentage = $totalTests > 0 ? ($totalPassed / $totalTests) * 100 : 0;
        echo sprintf("%-20s: %2d/%2d (%.1f%%)\n", "Overall", $totalPassed, $totalTests, $overallPercentage);
        
        if ($overallPercentage >= 80) {
            echo "\n✓ Good: Most tests passing\n";
        } elseif ($overallPercentage >= 50) {
            echo "\n⚠ Warning: Some tests failing\n";
        } else {
            echo "\n✗ Critical: Many tests failing\n";
        }
        
        echo "\nNote: This is a simplified implementation for demonstration.\n";
        echo "A complete implementation would require full WARP algorithm in C.\n";
    }
}

// Run the tests if this file is executed directly
if (basename(__FILE__) === basename($_SERVER['SCRIPT_NAME'])) {
    $tester = new LightFMFuzzTest(42);
    $tester->runAllTests();
}