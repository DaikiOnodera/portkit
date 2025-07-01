<?php

require_once __DIR__ . '/LightFMFFI.php';
require_once __DIR__ . '/CSRMatrix.php';

/**
 * LightFM - Matrix Factorization for Hybrid Recommender Systems
 * 
 * PHP port of the LightFM Python library using FFI bindings to C implementation.
 * Supports collaborative filtering with item and user features.
 */
class LightFM
{
    private LightFMFFI $ffi;
    private ?FFI\CData $model = null;
    private array $params;
    private bool $fitted = false;
    
    // Model dimensions
    private int $nUserFeatures = 0;
    private int $nItemFeatures = 0;
    
    public function __construct(array $params = [])
    {
        $this->ffi = new LightFMFFI();
        $this->params = array_merge([
            'no_components' => 10,
            'loss' => 'warp',
            'learning_rate' => 0.05,
            'item_alpha' => 0.0,
            'user_alpha' => 0.0,
            'max_sampled' => 10,
            'random_state' => null
        ], $params);
        
        // Set random seed for reproducibility
        if ($this->params['random_state'] !== null) {
            mt_srand($this->params['random_state']);
        }
    }
    
    public function __destruct()
    {
        if ($this->model !== null) {
            $this->ffi->freeLightFM($this->model);
        }
    }
    
    /**
     * Fit the LightFM model
     * 
     * @param CSRMatrix $interactions User-item interaction matrix
     * @param array $options Training options
     */
    public function fit(CSRMatrix $interactions, array $options = []): void
    {
        $options = array_merge([
            'epochs' => 1,
            'num_threads' => 1,
            'user_features' => null,
            'item_features' => null
        ], $options);
        
        // Create identity feature matrices if none provided
        $userFeatures = $options['user_features'] ?? $this->createIdentityMatrix($interactions->getRows());
        $itemFeatures = $options['item_features'] ?? $this->createIdentityMatrix($interactions->getCols());
        
        $this->nUserFeatures = $userFeatures->getRows();
        $this->nItemFeatures = $itemFeatures->getRows();
        
        // Initialize model if not already done
        if ($this->model === null) {
            $this->model = $this->ffi->createLightFM(
                $this->nUserFeatures,
                $this->nItemFeatures,
                $this->params['no_components']
            );
            $this->initializeModelWeights();
        }
        
        // Convert matrices to FFI format
        $interactionsFFI = $this->matrixToFFI($interactions);
        $userFeaturesFFI = $this->matrixToFFI($userFeatures);
        $itemFeaturesFFI = $this->matrixToFFI($itemFeatures);
        
        // Training loop
        for ($epoch = 0; $epoch < $options['epochs']; $epoch++) {
            $this->fitEpoch($interactionsFFI, $userFeaturesFFI, $itemFeaturesFFI, $options);
        }
        
        // Cleanup FFI matrices
        $this->ffi->freeCSRMatrix($interactionsFFI);
        $this->ffi->freeCSRMatrix($userFeaturesFFI);
        $this->ffi->freeCSRMatrix($itemFeaturesFFI);
        
        $this->fitted = true;
    }
    
    /**
     * Make predictions for user-item pairs
     * 
     * @param array $userIds Array of user IDs
     * @param array $itemIds Array of item IDs
     * @param CSRMatrix|null $userFeatures User feature matrix
     * @param CSRMatrix|null $itemFeatures Item feature matrix
     * @return array Predictions
     */
    public function predict(
        array $userIds, 
        array $itemIds, 
        ?CSRMatrix $userFeatures = null, 
        ?CSRMatrix $itemFeatures = null,
        int $numThreads = 1
    ): array {
        if (!$this->fitted) {
            throw new RuntimeException('Model must be fitted before making predictions');
        }
        
        if (count($userIds) !== count($itemIds)) {
            throw new InvalidArgumentException('User IDs and item IDs must have the same length');
        }
        
        // Use identity matrices if features not provided
        $userFeatures = $userFeatures ?? $this->createIdentityMatrix($this->nUserFeatures);
        $itemFeatures = $itemFeatures ?? $this->createIdentityMatrix($this->nItemFeatures);
        
        // Convert to FFI format
        $userFeaturesFFI = $this->matrixToFFI($userFeatures);
        $itemFeaturesFFI = $this->matrixToFFI($itemFeatures);
        
        // Make predictions
        $predictions = $this->ffi->predict(
            $itemFeaturesFFI,
            $userFeaturesFFI,
            $userIds,
            $itemIds,
            $this->model,
            $numThreads
        );
        
        // Cleanup
        $this->ffi->freeCSRMatrix($userFeaturesFFI);
        $this->ffi->freeCSRMatrix($itemFeaturesFFI);
        
        return $predictions;
    }
    
    /**
     * Get model parameters
     */
    public function getParams(): array
    {
        return $this->params;
    }
    
    /**
     * Check if model is fitted
     */
    public function isFitted(): bool
    {
        return $this->fitted;
    }
    
    /**
     * Get model dimensions
     */
    public function getModelDimensions(): array
    {
        return [
            'n_user_features' => $this->nUserFeatures,
            'n_item_features' => $this->nItemFeatures,
            'no_components' => $this->params['no_components']
        ];
    }
    
    /**
     * Fit one epoch
     */
    private function fitEpoch(FFI\CData $interactions, FFI\CData $userFeatures, FFI\CData $itemFeatures, array $options): void
    {
        // Generate training examples from interactions
        $examples = $this->generateTrainingExamples($interactions);
        
        if (empty($examples)) {
            return;
        }
        
        // Shuffle examples
        shuffle($examples);
        
        $userIds = array_column($examples, 'user');
        $itemIds = array_column($examples, 'item');
        $Y = array_column($examples, 'rating');
        $sampleWeight = array_fill(0, count($Y), 1.0);
        $shuffleIndices = range(0, count($Y) - 1);
        $randomStates = [mt_rand()];
        
        // Call appropriate training function based on loss
        switch ($this->params['loss']) {
            case 'warp':
                $this->ffi->fitWarp(
                    $itemFeatures,
                    $userFeatures,
                    $interactions,
                    $userIds,
                    $itemIds,
                    $Y,
                    $sampleWeight,
                    $shuffleIndices,
                    $this->model,
                    $this->params['learning_rate'],
                    $this->params['item_alpha'],
                    $this->params['user_alpha'],
                    $options['num_threads'],
                    $randomStates
                );
                break;
                
            default:
                throw new InvalidArgumentException("Unsupported loss function: {$this->params['loss']}");
        }
    }
    
    /**
     * Generate training examples from interaction matrix
     */
    private function generateTrainingExamples(FFI\CData $interactions): array
    {
        $examples = [];
        
        for ($user = 0; $user < $interactions->rows; $user++) {
            $start = $interactions->indptr[$user];
            $end = $interactions->indptr[$user + 1];
            
            for ($idx = $start; $idx < $end; $idx++) {
                $item = $interactions->indices[$idx];
                $rating = $interactions->data[$idx];
                
                $examples[] = [
                    'user' => $user,
                    'item' => $item,
                    'rating' => $rating
                ];
            }
        }
        
        return $examples;
    }
    
    /**
     * Convert CSRMatrix to FFI format
     */
    private function matrixToFFI(CSRMatrix $matrix): FFI\CData
    {
        $ffiMatrix = $this->ffi->createCSRMatrix($matrix->getRows(), $matrix->getCols(), $matrix->getNnz());
        
        $this->ffi->fillCSRMatrix(
            $ffiMatrix,
            $matrix->getIndices(),
            $matrix->getIndptr(),
            $matrix->getData()
        );
        
        return $ffiMatrix;
    }
    
    /**
     * Create identity matrix (each row has single 1.0 at diagonal)
     */
    private function createIdentityMatrix(int $size): CSRMatrix
    {
        $indices = range(0, $size - 1);
        $indptr = range(0, $size);
        $data = array_fill(0, $size, 1.0);
        
        return new CSRMatrix($indices, $indptr, $data, $size, $size);
    }
    
    /**
     * Initialize model weights with small random values
     */
    private function initializeModelWeights(): void
    {
        // This would normally initialize the embeddings and biases
        // For now, the C code handles initialization with zeros
        // In a complete implementation, we'd set random initial values
    }
}