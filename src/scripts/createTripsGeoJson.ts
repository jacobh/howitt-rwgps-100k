import fs from 'fs';
import path from 'path';
import type { TripSummary } from '../rwgps/types';
import type { Feature, FeatureCollection, Point } from 'geojson';

/**
 * Converts RWGPS trips to a GeoJSON FeatureCollection
 * This version extracts the first point of each ride
 * and samples a random subset of 1000 rides
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
  const allTrips: TripSummary[] = JSON.parse(tripsData);
  
  console.log(`Found ${allTrips.length} trips total`);
  
  // // Sample size
  // const sampleSize = Math.min(10_000, allTrips.length);
  
  // // Take a random sample of trips
  // const sampledTrips = getRandomSample(allTrips, sampleSize);
  
  // console.log(`Processing random sample of ${sampledTrips.length} trips...`);
  
  // Create GeoJSON features for each trip
  const features: Feature<Point>[] = allTrips.map(trip => {
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
  console.log(`Created features for ${features.length} trips (from original ${allTrips.length})`);
}

/**
 * Get a random sample of items from an array
 * @param array The source array
 * @param sampleSize Number of items to sample
 * @returns Random sample of the specified size
 */
function getRandomSample<T>(array: T[], sampleSize: number): T[] {
  // Create a copy of the array to avoid modifying the original
  const arrayCopy = [...array];
  
  // If sample size is larger than array, return the whole array
  if (sampleSize >= arrayCopy.length) {
    return arrayCopy;
  }
  
  // Fisher-Yates shuffle algorithm
  for (let i = arrayCopy.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arrayCopy[i], arrayCopy[j]] = [arrayCopy[j], arrayCopy[i]];
  }
  
  // Return the first N elements of the shuffled array
  return arrayCopy.slice(0, sampleSize);
}

// Execute the conversion script
createTripsGeoJson().catch(error => {
  console.error('Error creating GeoJSON:', error);
});