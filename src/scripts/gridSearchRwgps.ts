import { sliceIntoBboxes } from "../geo/sliceIntoBboxes";
import { RwgpsApiClient } from "../rwgps/api";
import type { BoundingBox, TripSummary } from "../rwgps/types";
import { searchPolygon } from "./sliceBbox";
import pLimit from "p-limit";
import path from "path";
import fs from "fs";

/**
 * Converts a GeoJSON bbox polygon to a RWGPS BoundingBox format [south, west, north, east]
 */
function convertToRwgpsBbox(
  bboxPolygon: GeoJSON.Feature<GeoJSON.Polygon>
): BoundingBox {
  // Extract the coordinates from the bbox polygon
  const coordinates = bboxPolygon.geometry.coordinates[0];

  // A bbox polygon has 5 points (last one is the same as first)
  // We need to extract the min/max lat/lng
  let minLng = Infinity,
    maxLng = -Infinity,
    minLat = Infinity,
    maxLat = -Infinity;

  for (const [lng, lat] of coordinates) {
    minLng = Math.min(minLng, lng);
    maxLng = Math.max(maxLng, lng);
    minLat = Math.min(minLat, lat);
    maxLat = Math.max(maxLat, lat);
  }

  // Return as [south, west, north, east]
  return [minLat, minLng, maxLat, maxLng];
}

/**
 * Search RWGPS for trips in a specific bounding box
 * Automatically paginates through results (up to 10 pages)
 */
async function searchBbox(
  client: RwgpsApiClient,
  bbox: GeoJSON.Feature<GeoJSON.Polygon>,
  index: number,
  total: number
) {
  const rwgpsBbox = convertToRwgpsBbox(bbox);
  console.log(`Searching cell ${index + 1}/${total}: ${rwgpsBbox}`);

  let allTrips: TripSummary[] = [];
  let offset = 0;
  let pageCount = 0;
  let hasMorePages = true;

  // Fetch up to 10 pages of results
  while (hasMorePages && pageCount < 20) {
    pageCount++;
    console.log(
      `Cell ${index + 1}: Fetching page ${pageCount} (offset: ${offset})`
    );

    const results = await client.exploreTrips(rwgpsBbox, {
      limit: 10,
      offset: offset,
    });

    // Add results to collection
    allTrips = allTrips.concat(results.results);

    // Update offset for next page
    offset += results.results.length;

    // Check if there are more pages
    hasMorePages = results.meta.explore.next_page_url !== null;

    if (hasMorePages) {
      console.log(
        `Cell ${index + 1}: More results available, next page available`
      );
    }
  }

  const totalCount = allTrips.length;

  // Calculate average distance for trips in this cell
  let cellTotalDistance = 0;
  for (const trip of allTrips) {
    cellTotalDistance += trip.distance;
  }

  const cellAvgDistance =
    allTrips.length > 0 ? cellTotalDistance / allTrips.length / 1000 : 0;

  console.log(
    `Cell ${index + 1}: Found ${totalCount} trips, fetched ${
      allTrips.length
    }, avg distance: ${cellAvgDistance.toFixed(2)} km`
  );

  return {
    cellIndex: index,
    tripsCount: totalCount,
    totalDistance: cellTotalDistance,
    trips: allTrips,
  };
}

/**
 * Searches RWGPS for trips in each bbox of the sliced polygon
 * Automatically paginates through results (up to 10 pages per bbox)
 */
async function gridSearchRwgps(
  maxAreaKm2: number = 200,
  concurrency: number = 20
) {
  console.log("Starting grid search of RWGPS...");

  // Slice the search polygon into smaller bboxes
  const bboxes = sliceIntoBboxes(
    searchPolygon.features[0] as GeoJSON.Feature<GeoJSON.Polygon>,
    maxAreaKm2
  );

  console.log(
    `Searching ${bboxes.length} grid cells with max ${concurrency} concurrent requests...`
  );

  // Initialize the RWGPS API client
  const client = new RwgpsApiClient();

  // Create a limit function that allows only a specific number of concurrent executions
  const limit = pLimit(concurrency);

  // Create an array of limited promises for each bbox search
  const promises = bboxes.map((bbox, index) =>
    limit(() => searchBbox(client, bbox, index, bboxes.length))
  );

  // Wait for all searches to complete
  const results = await Promise.all(promises);

  // Compile statistics
  let totalTripsFound = 0;
  let totalDistance = 0;

  results.forEach((result) => {
    totalTripsFound += result.tripsCount;
    totalDistance += result.totalDistance;
  });

  // Calculate overall statistics
  const avgTripDistance =
    totalTripsFound > 0 ? totalDistance / totalTripsFound / 1000 : 0;

  console.log("\nSearch completed!");
  console.log(`Total trips found: ${totalTripsFound}`);
  console.log(`Average trip distance: ${avgTripDistance.toFixed(2)} km`);

  // Collect all trips from all results into a single array
  const allTrips = results.flatMap((result) => result.trips);

  // Save trips to JSON file in current working directory
  const fs = require("fs");
  const path = require("path");
  const outputPath = path.join(process.cwd(), "data", "rwgps_trips.json");

  console.log(`\nSaving ${allTrips.length} trips to ${outputPath}...`);
  fs.writeFileSync(outputPath, JSON.stringify(allTrips, null, 2), "utf8");
  console.log("Trip data saved successfully!");
}

// Execute the grid search
gridSearchRwgps().catch((error) => {
  console.error("Error in grid search:", error);
});
