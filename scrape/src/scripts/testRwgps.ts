import { RwgpsApiClient } from '../rwgps/api';
import type { BoundingBox } from '../rwgps/types';

async function main() {
  const client = new RwgpsApiClient();
  
  // Search for trips in Mount Buller area
  const boundingBox: BoundingBox = [-37.34, 146.396672, -37.097749, 146.647226];
  const exploreResults = await client.exploreTrips(boundingBox);
  
  console.log(`Found ${exploreResults.total_count} trips`);
  
  if (exploreResults.results.length > 0) {
    const tripSummary = exploreResults.results[0];
    console.log(`First trip: ${tripSummary.name} (${tripSummary.distance / 1000} km)`);
    
    // Get details for the first trip
    const tripId = tripSummary.id;
    const tripDetails = await client.getTripDetails(tripId);
    
    // Using typed response
    const { trip } = tripDetails;

    console.log(`Trip "${trip.name}" has ${trip.track_points.length} track points`);
    console.log(`Elevation: ${trip.metrics.ele.min}m to ${trip.metrics.ele.max}m`);
    if (tripDetails.user) {
      console.log(`Created by: ${tripDetails.user.name} from ${tripDetails.user.locality}`);
    }
    
    // Hills analysis
    if (trip.metrics.hills.length > 0) {
      trip.metrics.hills.forEach((hill, index) => {
        console.log(`Hill #${index + 1}: ${hill.distance}m at ${hill.avg_grade.toFixed(1)}%`);
      });
    }
  
  }
}

main().catch(console.error);