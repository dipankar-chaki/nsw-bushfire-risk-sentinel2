"""
Bushfire Risk Assessment Module
NSW Bushfire Risk Assessment Project

This module combines vegetation indices to calculate bushfire risk scores
and classify risk levels for different areas.
"""

import numpy as np
from typing import Dict, Tuple, Union, Optional
from pathlib import Path
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class RiskCategory(Enum):
    """Risk category classifications."""
    VERY_LOW = "Very Low"
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    VERY_HIGH = "Very High"

class BushfireRiskAssessor:
    """
    Calculate bushfire risk scores based on vegetation indices.
    
    The risk assessment combines multiple vegetation indices to provide
    a comprehensive view of bushfire risk across the study area.
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize risk assessor with index weights.
        
        Args:
            weights: Dictionary of weights for each index
                    Default weights emphasize moisture content and vegetation health
        """
        self.weights = weights or {
            'NDVI': 0.3,    # Vegetation health (lower = higher risk)
            'NDMI': 0.5,    # Moisture content (lower = higher risk)
            'NBR': 0.2      # Burn ratio (lower = higher risk)
        }
        
        # Normalize weights to sum to 1
        weight_sum = sum(self.weights.values())
        self.weights = {k: v/weight_sum for k, v in self.weights.items()}
        
        logger.info(f"Initialized risk assessor with weights: {self.weights}")
    
    def calculate_vegetation_health_score(self, ndvi: np.ndarray) -> np.ndarray:
        """
        Calculate vegetation health score from NDVI.
        
        Args:
            ndvi: NDVI array (range -1 to 1)
            
        Returns:
            Health score array (0-100, higher = healthier vegetation = lower risk)
        """
        # Normalize NDVI to 0-100 scale
        # NDVI values typically range from -0.1 to 0.8 for vegetation
        health_score = np.clip((ndvi + 0.1) / 0.9 * 100, 0, 100)
        
        logger.info(f"Vegetation health score - Min: {health_score.min():.1f}, "
                   f"Max: {health_score.max():.1f}, Mean: {health_score.mean():.1f}")
        
        return health_score
    
    def calculate_moisture_score(self, ndmi: np.ndarray) -> np.ndarray:
        """
        Calculate moisture score from NDMI.
        
        Args:
            ndmi: NDMI array (range -1 to 1)
            
        Returns:
            Moisture score array (0-100, higher = more moisture = lower risk)
        """
        # Normalize NDMI to 0-100 scale
        # NDMI values typically range from -0.2 to 0.6 for vegetation
        moisture_score = np.clip((ndmi + 0.2) / 0.8 * 100, 0, 100)
        
        logger.info(f"Moisture score - Min: {moisture_score.min():.1f}, "
                   f"Max: {moisture_score.max():.1f}, Mean: {moisture_score.mean():.1f}")
        
        return moisture_score
    
    def calculate_burn_susceptibility_score(self, nbr: np.ndarray) -> np.ndarray:
        """
        Calculate burn susceptibility score from NBR.
        
        Args:
            nbr: NBR array (range -1 to 1)
            
        Returns:
            Susceptibility score array (0-100, higher = less susceptible = lower risk)
        """
        # Normalize NBR to 0-100 scale
        # Higher NBR values indicate less burn susceptibility
        susceptibility_score = np.clip((nbr + 1) / 2 * 100, 0, 100)
        
        logger.info(f"Burn susceptibility score - Min: {susceptibility_score.min():.1f}, "
                   f"Max: {susceptibility_score.max():.1f}, Mean: {susceptibility_score.mean():.1f}")
        
        return susceptibility_score
    
    def calculate_risk_score(self, indices: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate comprehensive risk score from vegetation indices.
        
        Args:
            indices: Dictionary containing vegetation indices
                    Keys should match the weights dictionary
        
        Returns:
            Risk score array (0-100, higher = higher risk)
        """
        if not indices:
            raise ValueError("No indices provided for risk calculation")
        
        # Initialize risk components
        risk_components = {}
        
        # Calculate individual risk components
        if 'NDVI' in indices:
            health_score = self.calculate_vegetation_health_score(indices['NDVI'])
            # Convert to risk (inverse of health)
            risk_components['NDVI'] = 100 - health_score
        
        if 'NDMI' in indices:
            moisture_score = self.calculate_moisture_score(indices['NDMI'])
            # Convert to risk (inverse of moisture)
            risk_components['NDMI'] = 100 - moisture_score
        
        if 'NBR' in indices:
            susceptibility_score = self.calculate_burn_susceptibility_score(indices['NBR'])
            # Convert to risk (inverse of resistance to burning)
            risk_components['NBR'] = 100 - susceptibility_score
        
        # Calculate weighted risk score
        risk_score = np.zeros_like(list(risk_components.values())[0])
        total_weight = 0
        
        for index_name, risk_component in risk_components.items():
            if index_name in self.weights:
                weight = self.weights[index_name]
                risk_score += weight * risk_component
                total_weight += weight
                logger.debug(f"Added {index_name} component with weight {weight}")
        
        # Normalize by total weight used (in case some indices are missing)
        if total_weight > 0:
            risk_score = risk_score / total_weight
        
        # Ensure risk score is in valid range
        risk_score = np.clip(risk_score, 0, 100)
        
        logger.info(f"Calculated risk score - Min: {risk_score.min():.1f}, "
                   f"Max: {risk_score.max():.1f}, Mean: {risk_score.mean():.1f}")
        
        return risk_score
    
    def classify_risk(self, risk_score: np.ndarray) -> np.ndarray:
        """
        Classify risk scores into categorical risk levels.
        
        Args:
            risk_score: Risk score array (0-100)
            
        Returns:
            Risk category array with integer codes:
            1 = Very Low, 2 = Low, 3 = Moderate, 4 = High, 5 = Very High
        """
        risk_categories = np.zeros_like(risk_score, dtype=int)
        
        # Define thresholds
        risk_categories[risk_score < 20] = 1  # Very Low
        risk_categories[(risk_score >= 20) & (risk_score < 40)] = 2  # Low
        risk_categories[(risk_score >= 40) & (risk_score < 60)] = 3  # Moderate
        risk_categories[(risk_score >= 60) & (risk_score < 80)] = 4  # High
        risk_categories[risk_score >= 80] = 5  # Very High
        
        # Log distribution
        unique, counts = np.unique(risk_categories, return_counts=True)
        total_pixels = np.sum(counts)
        
        logger.info("Risk category distribution:")
        for cat_code, count in zip(unique, counts):
            if cat_code > 0:  # Skip no-data values
                cat_name = list(RiskCategory)[cat_code-1].value
                percentage = (count / total_pixels) * 100
                logger.info(f"  {cat_name}: {count} pixels ({percentage:.1f}%)")
        
        return risk_categories
    
    def get_risk_category_name(self, category_code: int) -> str:
        """
        Get risk category name from code.
        
        Args:
            category_code: Integer category code (1-5)
            
        Returns:
            Risk category name
        """
        if 1 <= category_code <= 5:
            return list(RiskCategory)[category_code-1].value
        return "Unknown"
    
    def calculate_area_statistics(self, risk_categories: np.ndarray, 
                                pixel_area_m2: float = 100) -> Dict[str, Dict[str, float]]:
        """
        Calculate area statistics for each risk category.
        
        Args:
            risk_categories: Risk category array
            pixel_area_m2: Area per pixel in square meters (default 100 for 10m pixels)
            
        Returns:
            Dictionary with area statistics per category
        """
        stats = {}
        unique, counts = np.unique(risk_categories, return_counts=True)
        total_pixels = np.sum(counts[unique > 0])  # Exclude no-data
        
        for cat_code, count in zip(unique, counts):
            if cat_code > 0:  # Skip no-data values
                cat_name = self.get_risk_category_name(cat_code)
                area_m2 = count * pixel_area_m2
                area_ha = area_m2 / 10000  # Convert to hectares
                percentage = (count / total_pixels) * 100
                
                stats[cat_name] = {
                    'pixels': int(count),
                    'area_m2': area_m2,
                    'area_hectares': area_ha,
                    'percentage': percentage
                }
        
        return stats


def simple_risk_assessment(ndvi: np.ndarray, ndmi: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Simple risk assessment function for quick calculations.
    
    Args:
        ndvi: NDVI array
        ndmi: Optional NDMI array
        
    Returns:
        Risk score array (0-100)
    """
    # Basic risk calculation based primarily on vegetation health
    # Lower NDVI = dry/stressed vegetation = higher risk
    vegetation_risk = (1 - np.clip(ndvi, -1, 1)) * 50  # Scale to 0-100
    
    if ndmi is not None:
        # Add moisture component
        moisture_risk = (1 - np.clip(ndmi, -1, 1)) * 50
        risk_score = (vegetation_risk + moisture_risk) / 2
    else:
        risk_score = vegetation_risk
    
    return np.clip(risk_score, 0, 100)


def assess_risk_from_indices(indices: Dict[str, np.ndarray], 
                           weights: Optional[Dict[str, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to assess risk from a dictionary of indices.
    
    Args:
        indices: Dictionary of vegetation indices
        weights: Optional custom weights for indices
        
    Returns:
        Tuple of (risk_scores, risk_categories)
    """
    assessor = BushfireRiskAssessor(weights)
    risk_scores = assessor.calculate_risk_score(indices)
    risk_categories = assessor.classify_risk(risk_scores)
    
    return risk_scores, risk_categories 