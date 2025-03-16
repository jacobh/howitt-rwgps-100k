import fs from "fs";
import path from "path";
import { unpack } from "msgpackr";
import type { TripDetailsResponse } from "../rwgps/types";

/**
 * Formats distance in meters to a readable format (in kilometers)
 */
function formatDistance(meters: number): string {
  // Convert to kilometers
  const km = meters / 1000;
  return `${km.toFixed(1)} km`;
}

/**
 * Lists all trips in a batch file
 */
async function listTrips(batchPath: string): Promise<void> {
  try {
    console.log(`Inspecting trip batch: ${path.basename(batchPath)}`);

    // Read and unpack the MessagePack file
    const batchData = await fs.promises.readFile(batchPath);
    const trips = unpack(batchData) as TripDetailsResponse[];

    console.log(`Found ${trips.length} trips in batch\n`);

    // Print header
    console.log("Trip ID    | Username             | Distance      | Title");
    console.log(
      "----------|-----------------------|---------------|------------------"
    );

    // Print information for each trip
    trips.forEach((trip) => {
      const tripId = trip.trip.id.toString().padEnd(10);
      const username = (trip.user?.name || "Unknown").padEnd(21);
      const distance = formatDistance(trip.trip.distance || 0).padEnd(13);
      const title = trip.trip.name || "Untitled";

      console.log(`${tripId} | ${username} | ${distance} | ${title}`);
    });

    console.log("\nBatch inspection complete.");
  } catch (error: any) {
    console.error(`Error inspecting trip batch: ${error.message}`);
    process.exit(1);
  }
}

/**
 * Extracts and pretty prints a specific trip by ID
 */
async function extractTrip(batchPath: string, tripId: number): Promise<void> {
  try {
    console.log(`Extracting trip ${tripId} from ${path.basename(batchPath)}`);

    // Read and unpack the MessagePack file
    const batchData = await fs.promises.readFile(batchPath);
    const trips = unpack(batchData) as TripDetailsResponse[];

    // Find the trip with the matching ID
    const trip = trips.find((trip) => trip.trip.id === tripId);

    if (!trip) {
      console.error(`Trip with ID ${tripId} not found in the batch file.`);
      process.exit(1);
    }

    // Pretty print the trip data as JSON
    console.log(JSON.stringify(trip, null, 2));

    console.log(`\nSuccessfully extracted trip ${tripId}.`);
  } catch (error: any) {
    console.error(`Error extracting trip: ${error.message}`);
    process.exit(1);
  }
}

/**
 * Displays usage information
 */
function printUsage(): void {
  console.log("Usage: npm run inspect-batch -- <command> [options]");
  console.log("");
  console.log("Commands:");
  console.log(
    "  list <batch-file-path>              List all trips in the batch"
  );
  console.log(
    "  extract-trip <batch-file-path> <trip-id>  Extract and print a specific trip"
  );
  console.log("");
  console.log("Examples:");
  console.log(
    "  npm run inspect-batch -- list ./data/trip_batches/trip_batch_0.msgpack"
  );
  console.log(
    "  npm run inspect-batch -- extract-trip ./data/trip_batches/trip_batch_0.msgpack 12345"
  );
}

/**
 * Main function to parse command line arguments and run the appropriate command
 */
async function main(): Promise<void> {
  const args = process.argv.slice(2);

  if (args.length === 0 || args.includes("--help") || args.includes("-h")) {
    printUsage();
    return;
  }

  const command = args[0];

  switch (command) {
    case "list":
      if (args.length < 2) {
        console.error("Error: Missing batch file path");
        printUsage();
        process.exit(1);
      }

      const listBatchPath = args[1];

      // Check if the file exists
      if (!fs.existsSync(listBatchPath)) {
        console.error(`Error: File not found: ${listBatchPath}`);
        process.exit(1);
      }

      await listTrips(listBatchPath);
      break;

    case "extract-trip":
      if (args.length < 3) {
        console.error("Error: Missing batch file path or trip ID");
        printUsage();
        process.exit(1);
      }

      const extractBatchPath = args[1];
      const tripId = parseInt(args[2], 10);

      if (isNaN(tripId)) {
        console.error("Error: Trip ID must be a number");
        process.exit(1);
      }

      // Check if the file exists
      if (!fs.existsSync(extractBatchPath)) {
        console.error(`Error: File not found: ${extractBatchPath}`);
        process.exit(1);
      }

      await extractTrip(extractBatchPath, tripId);
      break;

    default:
      console.error(`Error: Unknown command '${command}'`);
      printUsage();
      process.exit(1);
  }
}

// Execute the script if run directly
if (require.main === module) {
  main().catch((error) => {
    console.error("Unexpected error:", error);
    process.exit(1);
  });
}
