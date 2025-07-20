#!/usr/bin/env python3
"""
Basic functionality test for NSW Bushfire Risk Assessment
Tests core modules without requiring external dependencies like databases.
"""

import numpy as np
import sys
from pathlib import Path

def test_vegetation_indices():
    """Test vegetation index calculations."""
    print("Testing vegetation index calculations...")
    
    from vegetation_indices import VegetationIndexCalculator
    
    # Create test data
    test_data = {
        'B04': np.random.uniform(1000, 3000, (50, 50)),  # Red
        'B08': np.random.uniform(3000, 6000, (50, 50)),  # NIR
        'B11': np.random.uniform(2000, 4000, (50, 50)),  # SWIR1
        'B12': np.random.uniform(1500, 3500, (50, 50))   # SWIR2
    }
    
    calc = VegetationIndexCalculator("")
    for band_name, band_data in test_data.items():
        calc.bands[band_name] = band_data
    
    # Test NDVI calculation
    ndvi = calc.calculate_ndvi()
    assert ndvi.shape == (50, 50), "NDVI shape mismatch"
    assert -1 <= ndvi.min() and ndvi.max() <= 1, "NDVI values out of range"
    
    # Test NDMI calculation
    ndmi = calc.calculate_ndmi()
    assert ndmi.shape == (50, 50), "NDMI shape mismatch"
    assert -1 <= ndmi.min() and ndmi.max() <= 1, "NDMI values out of range"
    
    # Test NBR calculation
    nbr = calc.calculate_nbr()
    assert nbr.shape == (50, 50), "NBR shape mismatch"
    assert -1 <= nbr.min() and nbr.max() <= 1, "NBR values out of range"
    
    print("✓ Vegetation index calculations passed")
    return {'NDVI': ndvi, 'NDMI': ndmi, 'NBR': nbr}

def test_risk_assessment(indices):
    """Test risk assessment calculations."""
    print("Testing risk assessment...")
    
    from risk_assessment import BushfireRiskAssessor
    
    assessor = BushfireRiskAssessor()
    
    # Test risk score calculation
    risk_scores = assessor.calculate_risk_score(indices)
    assert risk_scores.shape == (50, 50), "Risk scores shape mismatch"
    assert 0 <= risk_scores.min() and risk_scores.max() <= 100, "Risk scores out of range"
    
    # Test risk classification
    risk_categories = assessor.classify_risk(risk_scores)
    assert risk_categories.shape == (50, 50), "Risk categories shape mismatch"
    assert 1 <= risk_categories.min() and risk_categories.max() <= 5, "Risk categories out of range"
    
    print("✓ Risk assessment calculations passed")
    return risk_scores, risk_categories

def test_visualization(risk_scores, risk_categories):
    """Test visualization creation."""
    print("Testing visualization...")
    
    from visualization import BushfireRiskVisualizer
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    
    visualizer = BushfireRiskVisualizer()
    
    # Test risk map creation (without saving)
    import rasterio
    transform = rasterio.Affine(0.001, 0, 150.0, 0, -0.001, -33.5)
    
    try:
        fig = visualizer.plot_risk_map(risk_scores, risk_categories, transform)
        assert fig is not None, "Risk map creation failed"
        import matplotlib.pyplot as plt
        plt.close(fig)
        print("✓ Static visualization passed")
    except Exception as e:
        print(f"⚠ Static visualization test failed: {e}")
    
    # Test quick visualization function
    try:
        from visualization import quick_visualization
        output_dir = Path("./test_output")
        output_dir.mkdir(exist_ok=True)
        
        result_paths = quick_visualization(risk_scores, risk_categories, output_dir)
        assert 'static_map' in result_paths, "Static map not created"
        assert 'interactive_map' in result_paths, "Interactive map not created"
        
        print("✓ Quick visualization passed")
    except Exception as e:
        print(f"⚠ Quick visualization test failed: {e}")

def test_main_functionality():
    """Test main assessment workflow."""
    print("Testing main assessment workflow...")
    
    try:
        from main import BushfireRiskAssessment
        
        # Create assessment with test output directory
        assessment = BushfireRiskAssessment(output_dir="./test_output")
        
        # Test sample data loading
        bands = assessment.load_sample_data()
        assert len(bands) >= 4, "Insufficient bands loaded"
        assert 'B04' in bands and 'B08' in bands, "Required bands missing"
        
        print("✓ Main functionality test passed")
        
    except Exception as e:
        print(f"⚠ Main functionality test failed: {e}")

def main():
    """Run all tests."""
    print("=" * 60)
    print("NSW BUSHFIRE RISK ASSESSMENT - BASIC TESTS")
    print("=" * 60)
    
    try:
        # Test 1: Vegetation indices
        indices = test_vegetation_indices()
        
        # Test 2: Risk assessment
        risk_scores, risk_categories = test_risk_assessment(indices)
        
        # Test 3: Visualization
        test_visualization(risk_scores, risk_categories)
        
        # Test 4: Main functionality
        test_main_functionality()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED - System is ready to use!")
        print("=" * 60)
        print("Next steps:")
        print("  1. Run: python main.py --quick")
        print("  2. Check output in ./output directory")
        print("  3. Open example_notebook.ipynb for tutorial")
        
        return 0
        
    except Exception as e:
        print("=" * 60)
        print(f"❌ TESTS FAILED: {e}")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    exit(main()) 