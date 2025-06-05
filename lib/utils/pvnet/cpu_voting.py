import numpy as np
import torch

def cpu_voting(mask, vertex, n_hyp_points=128, inlier_thresh=0.99, max_num=100):
    """
    CPU implementation of RANSAC voting
    Args:
        mask: (H, W) tensor of binary mask
        vertex: (H, W, N, 2) tensor of vertex predictions
        n_hyp_points: number of hypothesis points to sample
        inlier_thresh: threshold for inlier ratio
        max_num: maximum number of RANSAC iterations
    Returns:
        keypoints: (N, 2) tensor of estimated keypoint positions
    """
    device = mask.device
    mask = mask.cpu().numpy()
    vertex = vertex.cpu().numpy()
    
    h, w = mask.shape
    n_kpts = vertex.shape[2]
    keypoints = np.zeros((n_kpts, 2))
    
    # For each keypoint
    for i in range(n_kpts):
        # Get valid votes for this keypoint
        valid_mask = mask > 0
        if valid_mask.sum() < 2:  # Need at least 2 points
            continue
            
        votes = vertex[valid_mask, i]
        points = np.stack(np.where(valid_mask), axis=1)
        
        best_inliers = 0
        best_center = None
        
        # RANSAC iterations
        for _ in range(min(max_num, n_hyp_points)):
            # Randomly sample 2 points
            idx = np.random.choice(len(points), 2, replace=False)
            p1, p2 = points[idx]
            v1, v2 = votes[idx]
            
            # Compute intersection of voting lines
            p1 = p1 + v1
            p2 = p2 + v2
            
            # Use this as hypothesis
            center = (p1 + p2) / 2
            
            # Count inliers
            pred_vectors = center[None] - points
            pred_vectors = pred_vectors / (np.linalg.norm(pred_vectors, axis=1, keepdims=True) + 1e-6)
            
            actual_vectors = votes / (np.linalg.norm(votes, axis=1, keepdims=True) + 1e-6)
            
            # Compute angle between predicted and actual vectors
            dot_product = (pred_vectors * actual_vectors).sum(axis=1)
            dot_product = np.clip(dot_product, -1, 1)
            angles = np.arccos(dot_product)
            
            # Count inliers (angles within threshold)
            inliers = (angles < np.pi/6).sum()  # 30 degrees threshold
            
            if inliers > best_inliers:
                best_inliers = inliers
                best_center = center
                
                # Early stopping if we found a good solution
                if best_inliers / len(points) > inlier_thresh:
                    break
        
        if best_center is not None:
            keypoints[i] = best_center
    
    return torch.from_numpy(keypoints).float().to(device) 