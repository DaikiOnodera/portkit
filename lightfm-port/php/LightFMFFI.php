<?php

/**
 * LightFM FFI Bindings for PHP
 * 
 * This class provides FFI bindings to the LightFM C library
 * for matrix factorization and recommendation systems.
 */
class LightFMFFI
{
    private FFI $ffi;
    private string $libPath;
    
    public function __construct(?string $libPath = null)
    {
        $this->libPath = $libPath ?? dirname(__DIR__) . '/c/liblightfm.so';
        
        if (!file_exists($this->libPath)) {
            throw new RuntimeException("LightFM library not found at: {$this->libPath}");
        }
        
        $this->ffi = FFI::cdef($this->getCDefinitions(), $this->libPath);
    }
    
    /**
     * Get C function definitions for FFI
     */
    private function getCDefinitions(): string
    {
        return '
            typedef struct {
                int *indices;
                int *indptr;
                float *data;
                int rows;
                int cols;
                int nnz;
            } CSRMatrix;
            
            typedef struct {
                float **item_features;
                float **item_feature_gradients;
                float **item_feature_momentum;
                float *item_biases;
                float *item_bias_gradients;
                float *item_bias_momentum;
                
                float **user_features;
                float **user_feature_gradients;
                float **user_feature_momentum;
                float *user_biases;
                float *user_bias_gradients;
                float *user_bias_momentum;
                
                int n_user_features;
                int n_item_features;
                int no_components;
                
                int adadelta;
                float learning_rate;
                float rho;
                float eps;
                int max_sampled;
                
                double item_scale;
                double user_scale;
            } FastLightFM;
            
            // Memory management
            CSRMatrix* csr_matrix_new(int rows, int cols, int nnz);
            void csr_matrix_free(CSRMatrix *matrix);
            FastLightFM* lightfm_new(int n_user_features, int n_item_features, int no_components);
            void lightfm_free(FastLightFM *model);
            
            // Training functions
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
            
            // Helper functions
            float sigmoid(float v);
            int sample_range(int min_val, int max_val, unsigned int *seed);
        ';
    }
    
    /**
     * Create new CSR matrix
     */
    public function createCSRMatrix(int $rows, int $cols, int $nnz): FFI\CData
    {
        return $this->ffi->csr_matrix_new($rows, $cols, $nnz);
    }
    
    /**
     * Free CSR matrix
     */
    public function freeCSRMatrix(FFI\CData $matrix): void
    {
        $this->ffi->csr_matrix_free($matrix);
    }
    
    /**
     * Create new LightFM model
     */
    public function createLightFM(int $nUserFeatures, int $nItemFeatures, int $noComponents): FFI\CData
    {
        return $this->ffi->lightfm_new($nUserFeatures, $nItemFeatures, $noComponents);
    }
    
    /**
     * Free LightFM model
     */
    public function freeLightFM(FFI\CData $model): void
    {
        $this->ffi->lightfm_free($model);
    }
    
    /**
     * Convert PHP array to C array
     */
    public function arrayToC(array $data, string $type): FFI\CData
    {
        $size = count($data);
        $cArray = $this->ffi->new($type . "[{$size}]");
        
        for ($i = 0; $i < $size; $i++) {
            $cArray[$i] = $data[$i];
        }
        
        return $cArray;
    }
    
    /**
     * Convert C array to PHP array
     */
    public function cArrayToPHP(FFI\CData $cArray, int $size): array
    {
        $phpArray = [];
        for ($i = 0; $i < $size; $i++) {
            $phpArray[] = $cArray[$i];
        }
        return $phpArray;
    }
    
    /**
     * Fill CSR matrix with data
     */
    public function fillCSRMatrix(FFI\CData $matrix, array $indices, array $indptr, array $data): void
    {
        // Fill indices
        for ($i = 0; $i < count($indices); $i++) {
            $matrix->indices[$i] = $indices[$i];
        }
        
        // Fill indptr
        for ($i = 0; $i < count($indptr); $i++) {
            $matrix->indptr[$i] = $indptr[$i];
        }
        
        // Fill data
        for ($i = 0; $i < count($data); $i++) {
            $matrix->data[$i] = $data[$i];
        }
    }
    
    /**
     * Fit WARP model
     */
    public function fitWarp(
        FFI\CData $itemFeatures,
        FFI\CData $userFeatures,
        FFI\CData $interactions,
        array $userIds,
        array $itemIds,
        array $Y,
        array $sampleWeight,
        array $shuffleIndices,
        FFI\CData $lightfm,
        float $learningRate,
        float $itemAlpha,
        float $userAlpha,
        int $numThreads,
        array $randomStates
    ): int {
        $userIdsC = $this->arrayToC($userIds, 'int');
        $itemIdsC = $this->arrayToC($itemIds, 'int');
        $YC = $this->arrayToC($Y, 'float');
        $sampleWeightC = $this->arrayToC($sampleWeight, 'float');
        $shuffleIndicesC = $this->arrayToC($shuffleIndices, 'int');
        $randomStatesC = $this->arrayToC($randomStates, 'unsigned int');
        
        return $this->ffi->fit_warp(
            $itemFeatures,
            $userFeatures,
            $interactions,
            FFI::addr($userIdsC[0]),
            FFI::addr($itemIdsC[0]),
            FFI::addr($YC[0]),
            FFI::addr($sampleWeightC[0]),
            FFI::addr($shuffleIndicesC[0]),
            $lightfm,
            $learningRate,
            $itemAlpha,
            $userAlpha,
            $numThreads,
            FFI::addr($randomStatesC[0]),
            count($userIds)
        );
    }
    
    /**
     * Predict with LightFM model
     */
    public function predict(
        FFI\CData $itemFeatures,
        FFI\CData $userFeatures,
        array $userIds,
        array $itemIds,
        FFI\CData $lightfm,
        int $numThreads = 1
    ): array {
        $numExamples = count($userIds);
        $userIdsC = $this->arrayToC($userIds, 'int');
        $itemIdsC = $this->arrayToC($itemIds, 'int');
        $predictionsC = $this->ffi->new('float[' . $numExamples . ']');
        
        $result = $this->ffi->predict_lightfm(
            $itemFeatures,
            $userFeatures,
            FFI::addr($userIdsC[0]),
            FFI::addr($itemIdsC[0]),
            FFI::addr($predictionsC[0]),
            $lightfm,
            $numThreads,
            $numExamples
        );
        
        if ($result != 0) {
            throw new RuntimeException("Prediction failed with error code: {$result}");
        }
        
        return $this->cArrayToPHP($predictionsC, $numExamples);
    }
    
    /**
     * Sigmoid function
     */
    public function sigmoid(float $v): float
    {
        return $this->ffi->sigmoid($v);
    }
    
    /**
     * Sample random number in range
     */
    public function sampleRange(int $min, int $max, int $seed): int
    {
        $seedC = $this->ffi->new('unsigned int');
        $seedC->cdata = $seed;
        return $this->ffi->sample_range($min, $max, FFI::addr($seedC));
    }
}