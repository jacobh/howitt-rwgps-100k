import fs from 'fs';
import path from 'path';
import pLimit from 'p-limit';
import ProgressBar from 'progress';
import { RwgpsApiClient } from '../rwgps/api';
import type { TripSummary, TripDetailsResponse } from '../rwgps/types';

/**
 * Fetches and stores trip details for a single trip
 * @param client - RWGPS API client
 * @param tripId - ID of the trip to fetch
 * @param outputDir - Directory to save the trip data
 * @param progressBar - Progress bar to update
 */
async function fetchAndStoreTrip(
  client: RwgpsApiClient,
  tripId: number,
  outputDir: string,
  progressBar?: ProgressBar
): Promise<void> {
  try {
    const outputPath = path.join(outputDir, `trip_${tripId}.json`);
    
    // Skip if already downloaded
    if (fs.existsSync(outputPath)) {
      if (progressBar) progressBar.tick({ tripId, status: 'skipped' });
      return;
    }
    
    // Fetch trip details
    const tripDetails = await client.getTripDetails(tripId);
    
    // Create the output directory if it doesn't exist
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }
    
    // Save the trip data
    fs.writeFileSync(outputPath, JSON.stringify(tripDetails, null, 2), 'utf8');
    
    // Update progress bar
    if (progressBar) progressBar.tick({ tripId, status: 'success' });
  } catch (error) {
    // Update progress bar with error
    if (progressBar) progressBar.tick({ tripId, status: 'failed' });
    throw error;
  }
}

/**
 * Downloads trip details for all trips in the trips.json file
 * Limits concurrent requests to 100
 */
async function downloadTripDetails() {
  // Path to the saved trips data file
  const tripsFilePath = path.join(process.cwd(), 'data', 'rwgps_trips.json');
  const outputDir = path.join(process.cwd(), 'data', 'trips');
  
  console.log('Starting RWGPS Trip Details Downloader');
  console.log(`Reading trips from ${tripsFilePath}...`);
  
  // Check if trips file exists
  if (!fs.existsSync(tripsFilePath)) {
    console.error(`Error: File ${tripsFilePath} not found`);
    return;
  }
  
  // Read and parse the trips file
  const tripsData = fs.readFileSync(tripsFilePath, 'utf8');
  const allTrips: TripSummary[] = JSON.parse(tripsData);
  
  console.log(`Found ${allTrips.length} trips to download`);
  
  // Count already downloaded trips
  let alreadyDownloaded = 0;
  if (fs.existsSync(outputDir)) {
    const files = fs.readdirSync(outputDir);
    alreadyDownloaded = files.filter(file => file.startsWith('trip_') && file.endsWith('.json')).length;
    console.log(`${alreadyDownloaded} trips already downloaded`);
  }
  
  // Initialize the RWGPS API client
  const client = new RwgpsApiClient();
  
  // Create a limit function that allows only 100 concurrent requests
  const limit = pLimit(100);
  
  // Create a progress bar
  const progressBar = new ProgressBar(
    '[:bar] :current/:total (:percent) | ETA: :eta s | Trip :tripId - :status', 
    {
      complete: '=',
      incomplete: ' ',
      width: 30,
      total: allTrips.length
    }
  );
  
  // Track statistics for summary
  let succeeded = 0;
  let failed = 0;
  let skipped = 0;
  
  // Create an array of limited promises for each trip download
  const downloadPromises = allTrips.map(trip => 
    limit(() => fetchAndStoreTrip(client, trip.id, outputDir, progressBar)
      .then(() => {
        // Count as success or skipped based on file existence before the call
        if (fs.existsSync(path.join(outputDir, `trip_${trip.id}.json`))) {
          succeeded++;
        } else {
          skipped++;
        }
        return { status: 'fulfilled', tripId: trip.id };
      })
      .catch(error => {
        failed++;
        return { status: 'rejected', reason: error, tripId: trip.id };
      })
    )
  );
  
  console.log(`Starting download of ${downloadPromises.length} trips with max 100 concurrent requests...`);
  
  // Execute all downloads with Promise.allSettled to handle errors gracefully
  await Promise.allSettled(downloadPromises);
  
  // Print summary
  console.log('\nDownload summary:');
  console.log(`Total trips: ${allTrips.length}`);
  console.log(`Successfully downloaded: ${succeeded}`);
  console.log(`Already existed (skipped): ${skipped}`);
  console.log(`Failed to download: ${failed}`);
  
  if (failed > 0) {
    console.log('\nSome trips failed to download. Check the logs for details.');
  }
}

// Execute the download
if (require.main === module) {
  downloadTripDetails().catch(error => {
    console.error('Error in download process:', error);
    process.exit(1);
  });
}

export { fetchAndStoreTrip, downloadTripDetails };