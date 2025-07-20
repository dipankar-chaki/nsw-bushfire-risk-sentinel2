"""
Database Module for PostGIS Operations
NSW Bushfire Risk Assessment Project

This module handles database connections, schema creation, and spatial queries
for storing and retrieving bushfire risk assessment results.
"""

import psycopg2
import psycopg2.extras
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, Point
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
import os
from datetime import datetime, date
import json

logger = logging.getLogger(__name__)

class BushfireRiskDatabase:
    """
    Database manager for bushfire risk assessment data.
    
    Handles connections to PostGIS database and provides methods for
    storing and querying spatial risk assessment results.
    """
    
    def __init__(self, connection_params: Optional[Dict[str, str]] = None):
        """
        Initialize database connection.
        
        Args:
            connection_params: Database connection parameters
                             If None, will attempt to read from environment variables
        """
        self.connection_params = connection_params or self._get_default_connection_params()
        self.connection = None
        self.schema_name = "bushfire_risk"
        
    def _get_default_connection_params(self) -> Dict[str, str]:
        """Get default connection parameters from environment variables."""
        return {
            'host': os.getenv('DB_HOST', 'localhost'),
            'database': os.getenv('DB_NAME', 'bushfire_risk'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'password'),
            'port': os.getenv('DB_PORT', '5432')
        }
    
    def connect(self) -> None:
        """Establish database connection."""
        try:
            self.connection = psycopg2.connect(**self.connection_params)
            self.connection.autocommit = False
            logger.info("Successfully connected to PostgreSQL database")
            
            # Test PostGIS extension
            self._test_postgis()
            
        except psycopg2.Error as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def _test_postgis(self) -> None:
        """Test if PostGIS extension is available."""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT PostGIS_Version();")
                version = cursor.fetchone()[0]
                logger.info(f"PostGIS version: {version}")
        except psycopg2.Error as e:
            logger.warning(f"PostGIS test failed: {e}")
            logger.warning("Make sure PostGIS extension is installed and enabled")
    
    def disconnect(self) -> None:
        """Close database connection."""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type:
            self.connection.rollback()
        else:
            self.connection.commit()
        self.disconnect()
    
    def create_schema(self) -> None:
        """Create the bushfire risk assessment schema and tables."""
        if not self.connection:
            raise ValueError("Database connection not established")
        
        try:
            with self.connection.cursor() as cursor:
                # Create schema
                cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema_name};")
                
                # Create main risk assessment table
                create_table_sql = f"""
                CREATE TABLE IF NOT EXISTS {self.schema_name}.risk_assessments (
                    id SERIAL PRIMARY KEY,
                    assessment_date DATE NOT NULL,
                    geom GEOMETRY(Polygon, 4326) NOT NULL,
                    risk_score FLOAT CHECK (risk_score >= 0 AND risk_score <= 100),
                    risk_category VARCHAR(20),
                    ndvi_mean FLOAT,
                    ndmi_mean FLOAT,
                    nbr_mean FLOAT,
                    area_hectares FLOAT,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
                cursor.execute(create_table_sql)
                
                # Create spatial index
                cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_risk_assessments_geom 
                ON {self.schema_name}.risk_assessments USING GIST (geom);
                """)
                
                # Create date index
                cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_risk_assessments_date 
                ON {self.schema_name}.risk_assessments (assessment_date);
                """)
                
                # Create risk score index
                cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_risk_assessments_risk_score 
                ON {self.schema_name}.risk_assessments (risk_score);
                """)
                
                # Create pixel-level results table for detailed analysis
                cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema_name}.pixel_results (
                    id SERIAL PRIMARY KEY,
                    assessment_id INTEGER REFERENCES {self.schema_name}.risk_assessments(id),
                    geom GEOMETRY(Point, 4326) NOT NULL,
                    risk_score FLOAT,
                    risk_category INTEGER,
                    ndvi FLOAT,
                    ndmi FLOAT,
                    nbr FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """)
                
                # Create spatial index for pixel results
                cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_pixel_results_geom 
                ON {self.schema_name}.pixel_results USING GIST (geom);
                """)
                
                self.connection.commit()
                logger.info(f"Created schema {self.schema_name} and tables successfully")
                
        except psycopg2.Error as e:
            self.connection.rollback()
            logger.error(f"Failed to create schema: {e}")
            raise
    
    def insert_risk_assessment(self, 
                              assessment_date: date,
                              geometry: Polygon,
                              risk_score: float,
                              risk_category: str,
                              indices: Dict[str, float],
                              area_hectares: float,
                              metadata: Optional[Dict] = None) -> int:
        """
        Insert a risk assessment record.
        
        Args:
            assessment_date: Date of assessment
            geometry: Polygon geometry of assessed area
            risk_score: Overall risk score (0-100)
            risk_category: Risk category name
            indices: Dictionary of mean index values
            area_hectares: Area in hectares
            metadata: Optional metadata dictionary
            
        Returns:
            ID of inserted record
        """
        if not self.connection:
            raise ValueError("Database connection not established")
        
        try:
            with self.connection.cursor() as cursor:
                insert_sql = f"""
                INSERT INTO {self.schema_name}.risk_assessments 
                (assessment_date, geom, risk_score, risk_category, 
                 ndvi_mean, ndmi_mean, nbr_mean, area_hectares, metadata)
                VALUES (%s, ST_GeomFromText(%s, 4326), %s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
                """
                
                cursor.execute(insert_sql, (
                    assessment_date,
                    geometry.wkt,
                    risk_score,
                    risk_category,
                    indices.get('NDVI'),
                    indices.get('NDMI'),
                    indices.get('NBR'),
                    area_hectares,
                    json.dumps(metadata) if metadata else None
                ))
                
                record_id = cursor.fetchone()[0]
                self.connection.commit()
                
                logger.info(f"Inserted risk assessment record with ID: {record_id}")
                return record_id
                
        except psycopg2.Error as e:
            self.connection.rollback()
            logger.error(f"Failed to insert risk assessment: {e}")
            raise
    
    def query_risk_by_area(self, bbox: Tuple[float, float, float, float]) -> gpd.GeoDataFrame:
        """
        Query risk assessments within a bounding box.
        
        Args:
            bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat)
            
        Returns:
            GeoDataFrame containing matching records
        """
        if not self.connection:
            raise ValueError("Database connection not established")
        
        min_lon, min_lat, max_lon, max_lat = bbox
        
        query_sql = f"""
        SELECT id, assessment_date, ST_AsText(geom) as geometry,
               risk_score, risk_category, ndvi_mean, ndmi_mean, nbr_mean,
               area_hectares, metadata, created_at
        FROM {self.schema_name}.risk_assessments
        WHERE ST_Intersects(geom, ST_MakeEnvelope(%s, %s, %s, %s, 4326))
        ORDER BY assessment_date DESC;
        """
        
        try:
            gdf = gpd.read_postgis(
                query_sql, 
                self.connection, 
                params=(min_lon, min_lat, max_lon, max_lat),
                geom_col='geometry'
            )
            
            logger.info(f"Retrieved {len(gdf)} risk assessment records for bbox")
            return gdf
            
        except Exception as e:
            logger.error(f"Failed to query risk assessments: {e}")
            raise
    
    def query_high_risk_areas(self, min_risk_score: float = 60.0) -> gpd.GeoDataFrame:
        """
        Query areas with high risk scores.
        
        Args:
            min_risk_score: Minimum risk score threshold
            
        Returns:
            GeoDataFrame containing high-risk areas
        """
        if not self.connection:
            raise ValueError("Database connection not established")
        
        query_sql = f"""
        SELECT id, assessment_date, ST_AsText(geom) as geometry,
               risk_score, risk_category, ndvi_mean, ndmi_mean, nbr_mean,
               area_hectares, created_at
        FROM {self.schema_name}.risk_assessments
        WHERE risk_score >= %s
        ORDER BY risk_score DESC, assessment_date DESC;
        """
        
        try:
            gdf = gpd.read_postgis(
                query_sql, 
                self.connection, 
                params=(min_risk_score,),
                geom_col='geometry'
            )
            
            logger.info(f"Retrieved {len(gdf)} high-risk areas (score >= {min_risk_score})")
            return gdf
            
        except Exception as e:
            logger.error(f"Failed to query high-risk areas: {e}")
            raise
    
    def get_risk_statistics(self) -> Dict[str, Union[int, float]]:
        """
        Get summary statistics for all risk assessments.
        
        Returns:
            Dictionary containing summary statistics
        """
        if not self.connection:
            raise ValueError("Database connection not established")
        
        try:
            with self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(f"""
                SELECT 
                    COUNT(*) as total_assessments,
                    AVG(risk_score) as avg_risk_score,
                    MIN(risk_score) as min_risk_score,
                    MAX(risk_score) as max_risk_score,
                    SUM(area_hectares) as total_area_hectares,
                    COUNT(CASE WHEN risk_category = 'Very High' THEN 1 END) as very_high_count,
                    COUNT(CASE WHEN risk_category = 'High' THEN 1 END) as high_count,
                    COUNT(CASE WHEN risk_category = 'Moderate' THEN 1 END) as moderate_count,
                    COUNT(CASE WHEN risk_category = 'Low' THEN 1 END) as low_count,
                    COUNT(CASE WHEN risk_category = 'Very Low' THEN 1 END) as very_low_count
                FROM {self.schema_name}.risk_assessments;
                """)
                
                stats = dict(cursor.fetchone())
                logger.info("Retrieved risk assessment statistics")
                return stats
                
        except psycopg2.Error as e:
            logger.error(f"Failed to get risk statistics: {e}")
            raise
    
    def spatial_join_with_boundaries(self, boundary_table: str, 
                                   boundary_geom_col: str = 'geom',
                                   boundary_name_col: str = 'name') -> gpd.GeoDataFrame:
        """
        Perform spatial join with administrative boundaries.
        
        Args:
            boundary_table: Name of boundary table
            boundary_geom_col: Name of geometry column in boundary table
            boundary_name_col: Name of name column in boundary table
            
        Returns:
            GeoDataFrame with joined boundary information
        """
        if not self.connection:
            raise ValueError("Database connection not established")
        
        query_sql = f"""
        SELECT 
            r.id, r.assessment_date, ST_AsText(r.geom) as geometry,
            r.risk_score, r.risk_category, r.area_hectares,
            b.{boundary_name_col} as boundary_name
        FROM {self.schema_name}.risk_assessments r
        LEFT JOIN {boundary_table} b 
            ON ST_Intersects(r.geom, b.{boundary_geom_col})
        ORDER BY r.assessment_date DESC;
        """
        
        try:
            gdf = gpd.read_postgis(query_sql, self.connection, geom_col='geometry')
            logger.info(f"Performed spatial join with {boundary_table}")
            return gdf
            
        except Exception as e:
            logger.error(f"Failed to perform spatial join: {e}")
            raise


def create_sample_data_insert() -> str:
    """
    Generate sample SQL for inserting test data.
    
    Returns:
        SQL string for inserting sample data
    """
    return """
    -- Sample data for testing
    INSERT INTO bushfire_risk.risk_assessments 
    (assessment_date, geom, risk_score, risk_category, ndvi_mean, ndmi_mean, area_hectares)
    VALUES 
    ('2024-01-15', ST_GeomFromText('POLYGON((150.1 -33.9, 150.2 -33.9, 150.2 -33.8, 150.1 -33.8, 150.1 -33.9))', 4326), 
     75.5, 'High', 0.3, 0.1, 1000.0),
    ('2024-01-15', ST_GeomFromText('POLYGON((150.3 -33.9, 150.4 -33.9, 150.4 -33.8, 150.3 -33.8, 150.3 -33.9))', 4326), 
     25.2, 'Low', 0.7, 0.5, 800.0);
    """


def setup_database_from_env() -> BushfireRiskDatabase:
    """
    Set up database connection using environment variables.
    
    Returns:
        Configured database instance
    """
    # Load environment variables from .env file if it exists
    env_file = Path('.env')
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file)
    
    db = BushfireRiskDatabase()
    return db


# Example usage and testing functions
def test_database_connection(db_params: Optional[Dict[str, str]] = None) -> bool:
    """
    Test database connection and basic operations.
    
    Args:
        db_params: Optional database parameters
        
    Returns:
        True if test successful, False otherwise
    """
    try:
        with BushfireRiskDatabase(db_params) as db:
            db.create_schema()
            stats = db.get_risk_statistics()
            logger.info(f"Database test successful. Current assessments: {stats.get('total_assessments', 0)}")
            return True
            
    except Exception as e:
        logger.error(f"Database test failed: {e}")
        return False 