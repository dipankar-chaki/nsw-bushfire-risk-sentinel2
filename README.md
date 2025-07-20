# NSW Bushfire Risk Assessment using Sentinel-2 Satellite Imagery

A Python-based system for assessing bushfire risk in New South Wales using Sentinel-2 satellite imagery, demonstrating remote sensing and GIS capabilities for environmental monitoring.

## 🎯 Project Overview

This project implements an end-to-end pipeline for:

- Processing Sentinel-2 satellite imagery
- Calculating vegetation indices (NDVI, NDMI, NBR)
- Assessing bushfire risk using combined indices
- Creating interactive visualizations
- Storing results in PostGIS spatial database

**Study Area**: Blue Mountains, NSW (150.0°E to 150.8°E, -34.0°N to -33.5°N)

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- PostgreSQL with PostGIS extension (optional, for database features)
- Git

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/nsw-bushfire-risk-sentinel2.git
   cd nsw-bushfire-risk-sentinel2
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run quick assessment**
   ```bash
   python main.py --quick
   ```

This will generate sample visualizations in the `./output` directory.

## 📊 Features

### Core Functionality

- ✅ **Vegetation Index Calculation**

  - NDVI (Normalized Difference Vegetation Index)
  - NDMI (Normalized Difference Moisture Index)
  - NBR (Normalized Burn Ratio)

- ✅ **Risk Assessment Model**

  - Weighted combination of vegetation indices
  - 5-category risk classification (Very Low to Very High)
  - Configurable risk weights and thresholds

- ✅ **Visualization**

  - Static matplotlib maps with customizable colormaps
  - Interactive Folium maps with multiple layers
  - Risk distribution charts and statistics

- ✅ **Database Integration**
  - PostGIS spatial database storage
  - Spatial queries and analysis
  - Area statistics and reporting

### MVP Features (Implemented)

- [x] Sample data processing (synthetic Sentinel-2 imagery)
- [x] Vegetation index calculations
- [x] Risk score generation and classification
- [x] Static and interactive map creation
- [x] PostGIS database integration
- [x] Command-line interface
- [x] Jupyter notebook demonstration

## 🗂️ Project Structure

```
nsw-bushfire-risk-sentinel2/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── main.py                     # Main entry point and CLI
├── vegetation_indices.py       # Vegetation index calculations
├── risk_assessment.py          # Risk modeling and classification
├── visualization.py            # Static and interactive mapping
├── database.py                 # PostGIS database operations
├── example_notebook.ipynb      # Tutorial and demonstration
└── output/                     # Generated results (created at runtime)
    ├── risk_assessment_map.png
    ├── vegetation_indices.png
    └── interactive_risk_map.html
```

## 💻 Usage

### Command Line Interface

```bash
# Run full assessment with all features
python main.py

# Quick assessment with minimal output
python main.py --quick

# Specify custom output directory
python main.py --output-dir ./my_results

# Verbose logging
python main.py --verbose

# Skip database operations
python main.py --no-database
```

### Python API

```python
from main import BushfireRiskAssessment

# Initialize assessment
assessment = BushfireRiskAssessment(output_dir="./results")

# Run complete workflow
results = assessment.run_full_assessment()

# Access results
print(f"Mean risk score: {results['summary']['mean_risk_score']}")
print(f"High risk areas: {results['summary']['high_risk_percentage']:.1f}%")
```

### Individual Module Usage

```python
# Calculate vegetation indices
from vegetation_indices import VegetationIndexCalculator

calc = VegetationIndexCalculator("data/sentinel2_scene/")
calc.load_bands_from_paths({
    'B04': 'data/B04.tif',
    'B08': 'data/B08.tif'
})
ndvi = calc.calculate_ndvi()

# Assess risk
from risk_assessment import BushfireRiskAssessor

assessor = BushfireRiskAssessor()
risk_scores = assessor.calculate_risk_score({'NDVI': ndvi})
risk_categories = assessor.classify_risk(risk_scores)

# Create visualizations
from visualization import BushfireRiskVisualizer

viz = BushfireRiskVisualizer()
fig = viz.plot_risk_map(risk_scores, risk_categories, transform)
```

## 🗃️ Database Setup (Optional)

The system supports PostGIS for spatial data storage and analysis.

### PostgreSQL/PostGIS Installation

**Ubuntu/Debian:**

```bash
sudo apt-get install postgresql postgresql-contrib postgis
```

**macOS (using Homebrew):**

```bash
brew install postgresql postgis
```

**Windows:**
Download from [PostgreSQL official website](https://www.postgresql.org/download/windows/)

### Database Configuration

1. **Create database and enable PostGIS**

   ```sql
   CREATE DATABASE bushfire_risk;
   \c bushfire_risk
   CREATE EXTENSION postgis;
   ```

2. **Set environment variables**

   ```bash
   export DB_HOST=localhost
   export DB_NAME=bushfire_risk
   export DB_USER=postgres
   export DB_PASSWORD=your_password
   export DB_PORT=5432
   ```

3. **Or create .env file**
   ```
   DB_HOST=localhost
   DB_NAME=bushfire_risk
   DB_USER=postgres
   DB_PASSWORD=your_password
   DB_PORT=5432
   ```

### Database Schema

The system creates the following tables:

- `bushfire_risk.risk_assessments` - Main assessment results
- `bushfire_risk.pixel_results` - Detailed pixel-level data

## 📈 Methodology

### Risk Assessment Model

The bushfire risk is calculated using a weighted combination of vegetation indices:

```
Risk Score = (NDVI_risk × 0.3) + (NDMI_risk × 0.5) + (NBR_risk × 0.2)
```

Where each index is converted to risk using inverse normalization:

- Higher vegetation health (NDVI) = Lower risk
- Higher moisture content (NDMI) = Lower risk
- Higher burn resistance (NBR) = Lower risk

### Risk Classification

| Risk Score | Category  | Color       | Description        |
| ---------- | --------- | ----------- | ------------------ |
| 0-20       | Very Low  | 🟢 Green    | Minimal fire risk  |
| 20-40      | Low       | 🟡 Yellow   | Low fire risk      |
| 40-60      | Moderate  | 🟠 Orange   | Moderate fire risk |
| 60-80      | High      | 🔴 Red      | High fire risk     |
| 80-100     | Very High | 🔴 Dark Red | Extreme fire risk  |

## 🔬 Sample Data

For demonstration purposes, the system generates realistic synthetic Sentinel-2 data that mimics:

- Spatial patterns of vegetation distribution
- Typical reflectance values for different land cover types
- Moisture content variations across the landscape

In production, replace with actual Sentinel-2 Level-2A products from:

- [Copernicus Open Access Hub](https://scihub.copernicus.eu/)
- [Google Earth Engine](https://earthengine.google.com/)
- [AWS Open Data](https://aws.amazon.com/opendata/public-datasets/)

## 🎓 Tutorial

See `example_notebook.ipynb` for a complete step-by-step tutorial covering:

1. Data loading and preparation
2. Vegetation index calculation
3. Risk assessment methodology
4. Visualization creation
5. Database integration
6. Results interpretation

Run the notebook:

```bash
jupyter notebook example_notebook.ipynb
```

## 🔧 Configuration

### Risk Model Parameters

Customize risk assessment in `risk_assessment.py`:

```python
# Custom weights for indices
weights = {
    'NDVI': 0.4,    # Vegetation health
    'NDMI': 0.4,    # Moisture content
    'NBR': 0.2      # Burn ratio
}

assessor = BushfireRiskAssessor(weights=weights)
```

### Visualization Settings

Modify colors and styling in `visualization.py`:

```python
# Custom risk category colors
risk_colors = {
    1: '#2E8B57',  # Very Low - Sea Green
    2: '#FFD700',  # Low - Gold
    3: '#FF8C00',  # Moderate - Dark Orange
    4: '#FF4500',  # High - Orange Red
    5: '#DC143C'   # Very High - Crimson
}
```

## 📊 Output Files

### Generated Outputs

- **Static Maps** (`*.png`): High-resolution risk maps for reports
- **Interactive Maps** (`*.html`): Web-based maps for exploration
- **Data Files** (`*.tif`): GeoTIFF rasters for GIS software
- **Database Records**: Spatial data for queries and analysis

### File Locations

```
output/
├── risk_assessment_map.png      # Combined risk visualization
├── vegetation_indices.png       # NDVI, NDMI, NBR maps
├── interactive_risk_map.html    # Folium interactive map
└── bushfire_assessment.log      # Processing log
```

## 🧪 Testing

Run basic functionality tests:

```python
# Test database connection
from database import test_database_connection
print("Database test:", test_database_connection())

# Test quick assessment
from main import main
main()  # Runs with default parameters
```

## 🚧 Development Roadmap

### Phase 1: MVP (✅ Complete)

- [x] Core vegetation index calculations
- [x] Basic risk assessment model
- [x] Visualization capabilities
- [x] Database integration
- [x] Command-line interface

### Phase 2: Production Features

- [ ] Real Sentinel-2 data integration
- [ ] Automated data download from Copernicus Hub
- [ ] Cloud masking and quality filtering
- [ ] Time-series analysis and change detection
- [ ] Web dashboard interface

### Phase 3: Advanced Analytics

- [ ] Machine learning risk models
- [ ] Integration with weather data
- [ ] Historical fire validation
- [ ] Real-time monitoring alerts
- [ ] Mobile application

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

For questions, issues, or contributions:

- 📧 Email: your.email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/your-username/nsw-bushfire-risk-sentinel2/issues)
- 📖 Documentation: This README and `example_notebook.ipynb`

## 🙏 Acknowledgments

- **Copernicus Programme**: For Sentinel-2 satellite data
- **PostGIS Community**: For spatial database capabilities
- **Python Geospatial Ecosystem**: rasterio, geopandas, folium
- **NSW Rural Fire Service**: For domain expertise and validation

## 📚 References

1. Tucker, C.J. (1979). Red and photographic infrared linear combinations for monitoring vegetation. _Remote Sensing of Environment_, 8(2), 127-150.

2. Gao, B.C. (1996). NDWI—A normalized difference water index for remote sensing of vegetation liquid water from space. _Remote Sensing of Environment_, 58(3), 257-266.

3. Key, C.H. & Benson, N.C. (2006). Landscape Assessment: Ground measure of severity, the Composite Burn Index; and Remote sensing of severity, the Normalized Burn Ratio. _FIREMON: Fire Effects Monitoring and Inventory System_.

---

**🔥 Ready to assess bushfire risk with satellite imagery!** 🛰️
