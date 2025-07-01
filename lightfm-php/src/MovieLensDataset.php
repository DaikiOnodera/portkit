<?php

require_once __DIR__ . '/CSRMatrix.php';

/**
 * MovieLens Dataset Loader for PHP
 * 
 * Simplified version of the LightFM MovieLens dataset loader
 * that creates synthetic data for testing purposes.
 */
class MovieLensDataset
{
    private array $data = [];
    private int $nUsers = 943;
    private int $nItems = 1682;
    
    /**
     * Fetch MovieLens dataset (simplified synthetic version)
     * 
     * @param float $minRating Minimum rating threshold (ratings >= this are positive)
     * @return array Array with 'train' and 'test' CSR matrices
     */
    public function fetchMovielens(float $minRating = 4.0): array
    {
        // Generate synthetic MovieLens-like data for testing
        $this->generateSyntheticData($minRating);
        
        // Split into train/test (80/20 split)
        $trainData = [];
        $testData = [];
        
        foreach ($this->data as $interaction) {
            if (mt_rand() / mt_getrandmax() < 0.8) {
                $trainData[] = $interaction;
            } else {
                $testData[] = $interaction;
            }
        }
        
        // Convert to CSR matrices
        $trainMatrix = $this->buildInteractionMatrix($trainData);
        $testMatrix = $this->buildInteractionMatrix($testData);
        
        return [
            'train' => $trainMatrix,
            'test' => $testMatrix,
            'item_features' => CSRMatrix::identity($this->nItems),
            'user_features' => CSRMatrix::identity($this->nUsers)
        ];
    }
    
    /**
     * Generate synthetic data similar to MovieLens 100k
     */
    private function generateSyntheticData(float $minRating): void
    {
        mt_srand(42); // Fixed seed for reproducibility
        
        $this->data = [];
        
        // Generate approximately 15,000 positive interactions (above minRating)
        // This simulates the density of real MovieLens data after filtering
        $numInteractions = 15000;
        
        for ($i = 0; $i < $numInteractions; $i++) {
            $userId = mt_rand(0, $this->nUsers - 1);
            $itemId = mt_rand(0, $this->nItems - 1);
            
            // Generate rating above threshold (since we only keep positive interactions)
            $rating = $minRating + (mt_rand() / mt_getrandmax()) * (5.0 - $minRating);
            
            // Avoid duplicate interactions
            $key = "{$userId}_{$itemId}";
            if (!isset($this->data[$key])) {
                $this->data[$key] = [
                    'user_id' => $userId,
                    'item_id' => $itemId,
                    'rating' => $rating
                ];
            }
        }
        
        // Convert to indexed array
        $this->data = array_values($this->data);
        
        echo "Generated " . count($this->data) . " synthetic interactions\n";
        echo "Users: {$this->nUsers}, Items: {$this->nItems}\n";
        echo "Density: " . sprintf("%.4f%%", (count($this->data) / ($this->nUsers * $this->nItems)) * 100) . "\n";
    }
    
    /**
     * Build CSR interaction matrix from interaction data
     * 
     * @param array $interactions Array of interaction records
     * @return CSRMatrix
     */
    private function buildInteractionMatrix(array $interactions): CSRMatrix
    {
        if (empty($interactions)) {
            return CSRMatrix::zeros($this->nUsers, $this->nItems);
        }
        
        // Convert to COO format: [row, col, value] triplets
        $cooData = [];
        foreach ($interactions as $interaction) {
            $cooData[] = [
                $interaction['user_id'],
                $interaction['item_id'],
                1.0  // Binary ratings (1 for positive interaction)
            ];
        }
        
        return CSRMatrix::fromCOO($cooData, $this->nUsers, $this->nItems);
    }
    
    /**
     * Get dataset statistics
     */
    public function getStats(): array
    {
        return [
            'n_users' => $this->nUsers,
            'n_items' => $this->nItems,
            'n_interactions' => count($this->data),
            'density' => count($this->data) / ($this->nUsers * $this->nItems)
        ];
    }
}