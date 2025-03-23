import json
import time
import shapely
from shapely.geometry import shape, mapping, LineString, MultiLineString, Polygon, MultiPolygon, LinearRing
import shapely.ops
import traceback

def is_valid_line(line_coords, feature_name):
    """Check if a line has enough points to be valid"""
    if len(line_coords) < 2:
        print(f"Warning: Feature '{feature_name}' has a line with {len(line_coords)} points (needs at least 2)")
        return False
    return True

def convert_boundary_to_polygon(geometry, feature_name="unknown"):
    """Convert LineString or MultiLineString geometry to Polygon with a robust fallback approach"""
    try:
        # Validate the input geometry format first
        if geometry['type'] == 'MultiLineString':
            # Check for empty or invalid coordinate arrays
            geometry['coordinates'] = [
                line_coords for line_coords in geometry['coordinates'] if is_valid_line(line_coords, feature_name)
            ]
        
        # Create the shapely geometry
        geom = shape(geometry)
        
        # If already a polygon type, return as is
        if isinstance(geom, (Polygon, MultiPolygon)):
            return geometry, True
        
        # Handle LineString or MultiLineString
        if isinstance(geom, (LineString, MultiLineString)):
            try:
                # Try to polygonize directly
                merged = shapely.ops.linemerge(geom)
                polygons = list(shapely.ops.polygonize(merged))
                
                if polygons:
                    # If we got polygons, return the first one
                    # (or you could union them if there are multiple)
                    result = polygons[0]
                    if len(polygons) > 1:
                        result = shapely.ops.unary_union(polygons)
                    
                    # Convert back to GeoJSON format
                    return mapping(result), True
                else:
                    # Fallback 1: Try to close the linestrings ourselves
                    # This works when the linestring almost forms a ring but doesn't quite close
                    if isinstance(merged, LineString):
                        coords = list(merged.coords)
                        if len(coords) >= 3:  # Need at least 3 points for a valid polygon
                            # If the start and end points are very close, connect them
                            if merged.coords[0] != merged.coords[-1]:
                                coords.append(coords[0])  # Close the ring
                            
                            try:
                                ring = LinearRing(coords)
                                polygon = Polygon(ring)
                                if polygon.is_valid:
                                    return mapping(polygon), True
                            except Exception as e:
                                print(f"Failed to create polygon from closed linestring: {e}")
                    
                    # Fallback 2: Try creating a convex hull
                    # This is a more aggressive approach but can work for complex boundaries
                    print(f"Using convex hull as fallback for feature '{feature_name}'")
                    hull = geom.convex_hull
                    if isinstance(hull, Polygon) and hull.is_valid:
                        return mapping(hull), True
                    
                    # Fallback 3: Try buffering with a small value, which can sometimes close small gaps
                    buffered = geom.buffer(0.0000001)  # Very small buffer
                    if isinstance(buffered, Polygon) and buffered.is_valid:
                        return mapping(buffered), True
                    
                print(f"Failed to convert boundary: {feature_name} - Could not create valid polygon from lines")
            except Exception as e:
                print(f"Error polygonizing geometry for '{feature_name}': {e}")
    except Exception as e:
        print(f"Error processing geometry for '{feature_name}':")
        print(f"Geometry type: {geometry['type']}")
        print(f"Error: {e}")
        traceback.print_exc()
    
    # Return original geometry and False if conversion failed
    return geometry, False

def process_boundaries():
    """Process boundary features and convert to polygons"""
    # Load the GeoJSON file
    print("Loading GeoJSON file...")
    start_time = time.time()
    
    with open('../data/aus_boundaries.json', 'r') as f:
        geojson_data = json.load(f)
    
    print(f"File loaded in {time.time() - start_time:.2f} seconds")
    
    # Process features - convert boundaries to polygons
    print("Converting boundary features to polygons...")
    processed_features = []
    boundary_conversion_count = 0
    failed_conversion_count = 0
    
    for feature in geojson_data['features']:
        # Only process features that are boundaries
        is_boundary = feature['properties'].get('boundary') is not None
        
        if is_boundary and feature['geometry']['type'] in ['LineString', 'MultiLineString']:
            # Try to convert to polygon
            original_type = feature['geometry']['type']
            original_geometry = feature['geometry']
            feature['geometry'], success = convert_boundary_to_polygon(
                feature['geometry'],
                feature['properties'].get('name', 'unnamed')
            )
            
            # Check if conversion happened
            if success and feature['geometry']['type'] in ['Polygon', 'MultiPolygon']:
                boundary_conversion_count += 1
                print(f"Converted {original_type} to {feature['geometry']['type']} for boundary {feature['properties'].get('name', 'unnamed')}")
                processed_features.append(feature)
            else:
                failed_conversion_count += 1
                print(f"Failed to convert {original_type} for boundary {feature['properties'].get('name', 'unnamed')}")
                # Skip this feature as it couldn't be converted to a polygon
        else:
            # Keep non-boundary features and features that are already polygons
            processed_features.append(feature)
    
    # Create new GeoJSON FeatureCollection with processed features
    output_geojson = {
        "type": "FeatureCollection",
        "features": processed_features
    }
    
    # Save the processed GeoJSON
    print("Saving processed GeoJSON...")
    with open('../data/aus_boundaries_polygons.json', 'w') as f:
        json.dump(output_geojson, f)
    
    print(f"Converted {boundary_conversion_count} boundary features to polygons")
    print(f"Dropped {failed_conversion_count} boundary features that could not be converted")
    print(f"Saved processed features to '../data/aus_boundaries_polygons.json'")

if __name__ == "__main__":
    process_boundaries()