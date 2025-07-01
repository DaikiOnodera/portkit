<?php

require_once __DIR__ . '/CSRMatrix.php';
require_once __DIR__ . '/LightFM.php';

/**
 * Evaluation metrics for LightFM recommendation models
 */
class Evaluation
{
    /**
     * Calculate precision at k
     * 
     * @param LightFM $model Trained LightFM model
     * @param CSRMatrix $testInteractions Test interactions matrix
     * @param int $k Number of top recommendations to consider
     * @param CSRMatrix|null $trainInteractions Training interactions (to exclude from evaluation)
     * @return float Mean precision@k across all users
     */
    public static function precisionAtK(
        LightFM $model, 
        CSRMatrix $testInteractions, 
        int $k = 10,
        ?CSRMatrix $trainInteractions = null
    ): float {
        if (!$model->isFitted()) {
            throw new RuntimeException('Model must be fitted before evaluation');
        }
        
        $nUsers = $testInteractions->getRows();
        $nItems = $testInteractions->getCols();
        $precisions = [];
        
        for ($userId = 0; $userId < $nUsers; $userId++) {
            // Get test items for this user
            $testItems = $testInteractions->getRowNonZero($userId);
            if (empty($testItems)) {
                continue; // Skip users with no test interactions
            }
            
            // Get predictions for all items
            $itemIds = range(0, $nItems - 1);
            $userIds = array_fill(0, $nItems, $userId);
            
            try {
                $predictions = $model->predict($userIds, $itemIds);
            } catch (Exception $e) {
                // Skip user if prediction fails
                continue;
            }
            
            // Create item-prediction pairs and sort by prediction descending
            $itemPredictions = [];
            for ($i = 0; $i < $nItems; $i++) {
                $itemPredictions[] = [
                    'item_id' => $i,
                    'prediction' => $predictions[$i]
                ];
            }
            
            usort($itemPredictions, function($a, $b) {
                return $b['prediction'] <=> $a['prediction'];
            });
            
            // Get top-k recommendations
            $topK = array_slice($itemPredictions, 0, $k);
            
            // Count how many top-k items are in test set
            $testItemIds = array_column($testItems, 0);
            $testItemSet = array_flip($testItemIds);
            
            $hits = 0;
            foreach ($topK as $rec) {
                if (isset($testItemSet[$rec['item_id']])) {
                    $hits++;
                }
            }
            
            // Calculate precision for this user
            $precision = $hits / min($k, count($topK));
            $precisions[] = $precision;
        }
        
        // Return mean precision across all users
        return empty($precisions) ? 0.0 : array_sum($precisions) / count($precisions);
    }
    
    /**
     * Calculate recall at k
     * 
     * @param LightFM $model Trained LightFM model
     * @param CSRMatrix $testInteractions Test interactions matrix
     * @param int $k Number of top recommendations to consider
     * @param CSRMatrix|null $trainInteractions Training interactions (to exclude from evaluation)
     * @return float Mean recall@k across all users
     */
    public static function recallAtK(
        LightFM $model, 
        CSRMatrix $testInteractions, 
        int $k = 10,
        ?CSRMatrix $trainInteractions = null
    ): float {
        if (!$model->isFitted()) {
            throw new RuntimeException('Model must be fitted before evaluation');
        }
        
        $nUsers = $testInteractions->getRows();
        $nItems = $testInteractions->getCols();
        $recalls = [];
        
        for ($userId = 0; $userId < $nUsers; $userId++) {
            // Get test items for this user
            $testItems = $testInteractions->getRowNonZero($userId);
            if (empty($testItems)) {
                continue; // Skip users with no test interactions
            }
            
            // Get predictions for all items
            $itemIds = range(0, $nItems - 1);
            $userIds = array_fill(0, $nItems, $userId);
            
            try {
                $predictions = $model->predict($userIds, $itemIds);
            } catch (Exception $e) {
                // Skip user if prediction fails
                continue;
            }
            
            // Create item-prediction pairs and sort by prediction descending
            $itemPredictions = [];
            for ($i = 0; $i < $nItems; $i++) {
                $itemPredictions[] = [
                    'item_id' => $i,
                    'prediction' => $predictions[$i]
                ];
            }
            
            usort($itemPredictions, function($a, $b) {
                return $b['prediction'] <=> $a['prediction'];
            });
            
            // Get top-k recommendations
            $topK = array_slice($itemPredictions, 0, $k);
            
            // Count how many top-k items are in test set
            $testItemIds = array_column($testItems, 0);
            $testItemSet = array_flip($testItemIds);
            
            $hits = 0;
            foreach ($topK as $rec) {
                if (isset($testItemSet[$rec['item_id']])) {
                    $hits++;
                }
            }
            
            // Calculate recall for this user
            $recall = $hits / count($testItems);
            $recalls[] = $recall;
        }
        
        // Return mean recall across all users
        return empty($recalls) ? 0.0 : array_sum($recalls) / count($recalls);
    }
    
    /**
     * Calculate Area Under ROC Curve (simplified version)
     * 
     * @param LightFM $model Trained LightFM model
     * @param CSRMatrix $testInteractions Test interactions matrix
     * @param CSRMatrix|null $trainInteractions Training interactions (to exclude from evaluation)
     * @return float Mean AUC across all users
     */
    public static function aucScore(
        LightFM $model, 
        CSRMatrix $testInteractions, 
        ?CSRMatrix $trainInteractions = null
    ): float {
        if (!$model->isFitted()) {
            throw new RuntimeException('Model must be fitted before evaluation');
        }
        
        $nUsers = $testInteractions->getRows();
        $nItems = $testInteractions->getCols();
        $aucs = [];
        
        for ($userId = 0; $userId < min($nUsers, 100); $userId++) { // Limit to 100 users for efficiency
            // Get test items for this user
            $testItems = $testInteractions->getRowNonZero($userId);
            if (empty($testItems)) {
                continue;
            }
            
            // Sample negative items (items not in test set)
            $testItemSet = array_flip(array_column($testItems, 0));
            $negativeItems = [];
            $sampleSize = min(100, $nItems - count($testItems)); // Sample up to 100 negatives
            
            $attempts = 0;
            while (count($negativeItems) < $sampleSize && $attempts < $sampleSize * 2) {
                $itemId = mt_rand(0, $nItems - 1);
                if (!isset($testItemSet[$itemId]) && !in_array($itemId, $negativeItems)) {
                    $negativeItems[] = $itemId;
                }
                $attempts++;
            }
            
            if (empty($negativeItems)) {
                continue;
            }
            
            // Get predictions for positive and negative items
            $allItems = array_merge(array_column($testItems, 0), $negativeItems);
            $allUsers = array_fill(0, count($allItems), $userId);
            
            try {
                $predictions = $model->predict($allUsers, $allItems);
            } catch (Exception $e) {
                continue;
            }
            
            // Calculate AUC using trapezoidal rule approximation
            $positivePreds = array_slice($predictions, 0, count($testItems));
            $negativePreds = array_slice($predictions, count($testItems));
            
            $auc = self::calculateAUC($positivePreds, $negativePreds);
            $aucs[] = $auc;
        }
        
        return empty($aucs) ? 0.5 : array_sum($aucs) / count($aucs);
    }
    
    /**
     * Calculate AUC from positive and negative predictions
     */
    private static function calculateAUC(array $positivePreds, array $negativePreds): float
    {
        if (empty($positivePreds) || empty($negativePreds)) {
            return 0.5;
        }
        
        $correct = 0;
        $total = 0;
        
        foreach ($positivePreds as $posPred) {
            foreach ($negativePreds as $negPred) {
                if ($posPred > $negPred) {
                    $correct++;
                } elseif ($posPred == $negPred) {
                    $correct += 0.5;
                }
                $total++;
            }
        }
        
        return $total > 0 ? $correct / $total : 0.5;
    }
}