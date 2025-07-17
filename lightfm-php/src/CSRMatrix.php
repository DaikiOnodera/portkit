<?php

/**
 * Compressed Sparse Row (CSR) Matrix implementation for PHP
 * 
 * This class provides a sparse matrix representation compatible with
 * the LightFM C library's CSRMatrix structure.
 */
class CSRMatrix
{
    private array $indices;
    private array $indptr;
    private array $data;
    private int $rows;
    private int $cols;
    private int $nnz;
    
    /**
     * Create CSR matrix from arrays
     * 
     * @param array $indices Column indices for non-zero elements
     * @param array $indptr Row pointer array
     * @param array $data Non-zero values
     * @param int $rows Number of rows
     * @param int $cols Number of columns
     */
    public function __construct(array $indices, array $indptr, array $data, int $rows, int $cols)
    {
        $this->indices = $indices;
        $this->indptr = $indptr;
        $this->data = $data;
        $this->rows = $rows;
        $this->cols = $cols;
        $this->nnz = count($data);
        
        $this->validate();
    }
    
    /**
     * Create CSR matrix from COO (Coordinate) format
     * 
     * @param array $cooData Array of [row, col, value] triplets
     * @param int $rows Number of rows
     * @param int $cols Number of columns
     * @return CSRMatrix
     */
    public static function fromCOO(array $cooData, int $rows, int $cols): self
    {
        if (empty($cooData)) {
            return new self([], array_fill(0, $rows + 1, 0), [], $rows, $cols);
        }
        
        // Sort by row, then by column
        usort($cooData, function($a, $b) {
            if ($a[0] === $b[0]) {
                return $a[1] <=> $b[1];
            }
            return $a[0] <=> $b[0];
        });
        
        $indices = [];
        $data = [];
        $indptr = array_fill(0, $rows + 1, 0);
        
        $currentRow = 0;
        foreach ($cooData as $i => [$row, $col, $value]) {
            // Fill indptr for empty rows
            while ($currentRow <= $row) {
                $indptr[$currentRow] = $i;
                $currentRow++;
            }
            
            $indices[] = $col;
            $data[] = $value;
        }
        
        // Fill remaining indptr entries
        while ($currentRow <= $rows) {
            $indptr[$currentRow] = count($data);
            $currentRow++;
        }
        
        return new self($indices, $indptr, $data, $rows, $cols);
    }
    
    /**
     * Create identity matrix
     * 
     * @param int $size Matrix size
     * @return CSRMatrix
     */
    public static function identity(int $size): self
    {
        $indices = range(0, $size - 1);
        $indptr = range(0, $size);
        $data = array_fill(0, $size, 1.0);
        
        return new self($indices, $indptr, $data, $size, $size);
    }
    
    /**
     * Create zero matrix
     * 
     * @param int $rows Number of rows
     * @param int $cols Number of columns
     * @return CSRMatrix
     */
    public static function zeros(int $rows, int $cols): self
    {
        return new self([], array_fill(0, $rows + 1, 0), [], $rows, $cols);
    }
    
    /**
     * Get matrix element
     * 
     * @param int $row Row index
     * @param int $col Column index
     * @return float Element value
     */
    public function get(int $row, int $col): float
    {
        if ($row >= $this->rows || $col >= $this->cols || $row < 0 || $col < 0) {
            throw new OutOfBoundsException("Index ({$row}, {$col}) out of bounds");
        }
        
        $start = $this->indptr[$row];
        $end = $this->indptr[$row + 1];
        
        for ($i = $start; $i < $end; $i++) {
            if ($this->indices[$i] === $col) {
                return $this->data[$i];
            }
        }
        
        return 0.0;
    }
    
    /**
     * Get row as array
     * 
     * @param int $row Row index
     * @return array Row values
     */
    public function getRow(int $row): array
    {
        if ($row >= $this->rows || $row < 0) {
            throw new OutOfBoundsException("Row index {$row} out of bounds");
        }
        
        $rowData = array_fill(0, $this->cols, 0.0);
        $start = $this->indptr[$row];
        $end = $this->indptr[$row + 1];
        
        for ($i = $start; $i < $end; $i++) {
            $rowData[$this->indices[$i]] = $this->data[$i];
        }
        
        return $rowData;
    }
    
    /**
     * Get non-zero elements for row
     * 
     * @param int $row Row index
     * @return array Array of [column, value] pairs
     */
    public function getRowNonZero(int $row): array
    {
        if ($row >= $this->rows || $row < 0) {
            throw new OutOfBoundsException("Row index {$row} out of bounds");
        }
        
        $nonZero = [];
        $start = $this->indptr[$row];
        $end = $this->indptr[$row + 1];
        
        for ($i = $start; $i < $end; $i++) {
            $nonZero[] = [$this->indices[$i], $this->data[$i]];
        }
        
        return $nonZero;
    }
    
    /**
     * Convert to dense matrix
     * 
     * @return array 2D array representation
     */
    public function toDense(): array
    {
        $dense = [];
        for ($i = 0; $i < $this->rows; $i++) {
            $dense[] = $this->getRow($i);
        }
        return $dense;
    }
    
    /**
     * Matrix-vector multiplication
     * 
     * @param array $vector Vector to multiply
     * @return array Result vector
     */
    public function multiply(array $vector): array
    {
        if (count($vector) !== $this->cols) {
            throw new InvalidArgumentException("Vector size must match number of columns");
        }
        
        $result = array_fill(0, $this->rows, 0.0);
        
        for ($row = 0; $row < $this->rows; $row++) {
            $start = $this->indptr[$row];
            $end = $this->indptr[$row + 1];
            
            for ($i = $start; $i < $end; $i++) {
                $result[$row] += $this->data[$i] * $vector[$this->indices[$i]];
            }
        }
        
        return $result;
    }
    
    /**
     * Get matrix statistics
     * 
     * @return array Statistics array
     */
    public function getStats(): array
    {
        $density = $this->rows * $this->cols > 0 ? $this->nnz / ($this->rows * $this->cols) : 0;
        
        return [
            'rows' => $this->rows,
            'cols' => $this->cols,
            'nnz' => $this->nnz,
            'density' => $density,
            'sparsity' => 1.0 - $density
        ];
    }
    
    // Getters
    public function getIndices(): array { return $this->indices; }
    public function getIndptr(): array { return $this->indptr; }
    public function getData(): array { return $this->data; }
    public function getRows(): int { return $this->rows; }
    public function getCols(): int { return $this->cols; }
    public function getNnz(): int { return $this->nnz; }
    
    /**
     * Validate matrix structure
     */
    private function validate(): void
    {
        if (count($this->indptr) !== $this->rows + 1) {
            throw new InvalidArgumentException("indptr size must be rows + 1");
        }
        
        if (count($this->indices) !== $this->nnz) {
            throw new InvalidArgumentException("indices size must equal nnz");
        }
        
        if (count($this->data) !== $this->nnz) {
            throw new InvalidArgumentException("data size must equal nnz");
        }
        
        // Check indptr is non-decreasing
        for ($i = 1; $i < count($this->indptr); $i++) {
            if ($this->indptr[$i] < $this->indptr[$i - 1]) {
                throw new InvalidArgumentException("indptr must be non-decreasing");
            }
        }
        
        // Check indices are in valid range
        foreach ($this->indices as $index) {
            if ($index < 0 || $index >= $this->cols) {
                throw new InvalidArgumentException("Index {$index} out of column range");
            }
        }
    }
    
    /**
     * String representation
     */
    public function __toString(): string
    {
        $stats = $this->getStats();
        return sprintf(
            "CSRMatrix(%dx%d, %d non-zeros, %.2f%% density)",
            $stats['rows'],
            $stats['cols'],
            $stats['nnz'],
            $stats['density'] * 100
        );
    }
}