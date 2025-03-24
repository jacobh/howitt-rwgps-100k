import fs from "fs";
import path from "path";
import { unpack } from "msgpackr";
import yargs from "yargs";
import { hideBin } from "yargs/helpers";
import type { TripDetailsResponse } from "../rwgps/types";
import { table } from "table";

/**
 * Formats distance in meters to a readable format (in kilometers)
 */
function formatDistance(meters: number): string {
  // Convert to kilometers
  const km = meters / 1000;
  return `${km.toFixed(1)} km`;
}

function sanitizeString(str: string): string {
    if (!str) return '';
    
    // Replace control characters with empty string
    // This regex matches all ASCII control characters (0-31 and 127)
    return str.replace(/[\x00-\x1F\x7F]/g, '');
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

    const tableData = [
      ["Trip ID", "User ID", "Distance", "Title"], // Header row
      ...trips.map((trip) => [
        trip.trip.id.toString(),
        trip.trip.user_id || "Unknown",
        formatDistance(trip.trip.distance || 0),
        sanitizeString(trip.trip.name || "Untitled"),
      ]),
    ];

    console.log(table(tableData, { singleLine: true }));

    console.log("\nBatch inspection complete.");
  } catch (error: any) {
    console.error(`Error inspecting trip batch: ${error.message}`);
    process.exit(1);
  }
}

/**
 * Lists all trips in all batch files in a directory
 */
/**
 * Lists all trips in all batch files in a directory
 */
async function listAllTripsInDirectory(directoryPath: string): Promise<void> {
  try {
    console.log(`Inspecting trip batches in directory: ${directoryPath}`);

    // Check if directory exists
    if (!fs.existsSync(directoryPath)) {
      console.error(`Error: Directory not found: ${directoryPath}`);
      process.exit(1);
    }

    // Get all files in the directory
    const files = await fs.promises.readdir(directoryPath);

    // Filter for .msgpack files
    const batchFiles = files
      .filter((file) => file.endsWith(".msgpack"))
      .map((file) => path.join(directoryPath, file));

    if (batchFiles.length === 0) {
      console.log("No batch files (.msgpack) found in the directory.");
      return;
    }

    console.log(`Found ${batchFiles.length} batch files\n`);

    // Process each batch file
    for (const batchFile of batchFiles) {
      await listTrips(batchFile);
      console.log("\n----------------------------------------\n");
    }

    console.log("Directory inspection complete.");
  } catch (error: any) {
    console.error(`Error inspecting trip batch directory: ${error.message}`);
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
 * Main function to parse command line arguments and run the appropriate command
 */
async function main(): Promise<void> {
  // Configure yargs
  const argv = yargs(hideBin(process.argv))
    .scriptName("inspect-batch")
    .usage("Usage: $0 <command> [options]")
    .command(
      "list <batchPath>",
      "List all trips in the batch",
      (yargs) => {
        return yargs.positional("batchPath", {
          describe: "Path to the batch file",
          type: "string",
          demandOption: true,
        });
      },
      async (argv) => {
        // Check if the file exists
        if (!fs.existsSync(argv.batchPath)) {
          console.error(`Error: File not found: ${argv.batchPath}`);
          process.exit(1);
        }
        await listTrips(argv.batchPath);
      }
    )
    .command(
      "list-dir <dirPath>",
      "List all trips from all batch files in a directory",
      (yargs) => {
        return yargs.positional("dirPath", {
          describe: "Path to the directory containing batch files",
          type: "string",
          demandOption: true,
        });
      },
      async (argv) => {
        await listAllTripsInDirectory(argv.dirPath);
      }
    )
    .command(
      "extract-trip <batchPath> <tripId>",
      "Extract and print a specific trip",
      (yargs) => {
        return yargs
          .positional("batchPath", {
            describe: "Path to the batch file",
            type: "string",
            demandOption: true,
          })
          .positional("tripId", {
            describe: "ID of the trip to extract",
            type: "number",
            demandOption: true,
          });
      },
      async (argv) => {
        // Check if the file exists
        if (!fs.existsSync(argv.batchPath)) {
          console.error(`Error: File not found: ${argv.batchPath}`);
          process.exit(1);
        }
        await extractTrip(argv.batchPath, argv.tripId);
      }
    )
    .example(
      "$0 list ./data/trip_batches/trip_batch_0.msgpack",
      "List all trips in the batch"
    )
    .example(
      "$0 list-dir ./data/trip_batches",
      "List all trips from all batch files in a directory"
    )
    .example(
      "$0 extract-trip ./data/trip_batches/trip_batch_0.msgpack 12345",
      "Extract and print trip with ID 12345"
    )
    .demandCommand(1, "You must specify a command")
    .strict()
    .help()
    .alias("h", "help")
    .alias("v", "version").argv;
}

// Execute the script if run directly
if (require.main === module) {
  main().catch((error) => {
    console.error("Unexpected error:", error);
    process.exit(1);
  });
}
