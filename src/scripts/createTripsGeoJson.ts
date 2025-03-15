import fs from 'fs';
import path from 'path';
import type { TripSummary } from '../rwgps/types';
import type { Feature, FeatureCollection, Point } from 'geojson';

/**
 * Converts RWGPS trips to a GeoJSON FeatureCollection
 * This version extracts the first point of each ride
 */
async function createTripsGeoJson() {
  // Path to the saved trips data file
  const tripsFilePath = path.join(process.cwd(), 'data', 'rwgps_trips.json');
  
  console.log(`Reading trips from ${tripsFilePath}...`);
  
  // Read and parse the trips file
  if (!fs.existsSync(tripsFilePath)) {
    console.error(`Error: File ${tripsFilePath} not found`);
    return;
  }
  
  const tripsData = fs.readFileSync(tripsFilePath, 'utf8');
  const trips: TripSummary[] = JSON.parse(tripsData);
  
  console.log(`Processing ${trips.length} trips...`);
  
  // Create GeoJSON features for each trip
  const features: Feature<Point>[] = trips.map(trip => {
    // Create a point feature using the first point of the trip
    return {
      type: 'Feature',
      geometry: {
        type: 'Point',
        coordinates: [trip.first_lng, trip.first_lat]
      },
      properties: {
        id: trip.id,
        name: trip.name,
        distance: trip.distance,
        elevation_gain: trip.elevation_gain,
        activity_type: trip.activity_type,
        user: trip.byline_name,
        url: `https://ridewithgps.com${trip.url}`,
        locality: trip.locality,
        administrative_area: trip.administrative_area,
        track_type: trip.track_type,
        terrain: trip.terrain,
        difficulty: trip.difficulty
      }
    };
  });
  
  // Create the full FeatureCollection
  const featureCollection: FeatureCollection = {
    type: 'FeatureCollection',
    features: features
  };
  
  // Save the GeoJSON to a file
  const outputPath = path.join(process.cwd(), 'data', 'rwgps_trips_points.geojson');
  fs.writeFileSync(outputPath, JSON.stringify(featureCollection, null, 2), 'utf8');
  
  console.log(`GeoJSON file created at ${outputPath}`);
  console.log(`Created features for ${features.length} trips`);
}

// Execute the conversion script
createTripsGeoJson().catch(error => {
  console.error('Error creating GeoJSON:', error);
});
