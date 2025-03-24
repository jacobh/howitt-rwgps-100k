import fs from 'fs';
import path from 'path';
import { pack } from 'msgpackr';
import { chunk } from 'lodash-es';
import pLimit from 'p-limit';

/**
 * Gets all trip JSON filenames from the specified directory
 */
async function getTripFilenames(inputDir: string): Promise<string[]> {
  console.log(`Reading trip files from ${inputDir}...`);
  
  const files = await fs.promises.readdir(inputDir);
  const tripFiles = files.filter(file => file.startsWith('trip_') && file.endsWith('.json'));
  
  console.log(`Found ${tripFiles.length} trip files.`);
  return tripFiles;
}

/**
 * Reads and parses a JSON file, returning null if there's an error
 */
async function readAndParseFile<T>(filePath: string): Promise<T | null> {
  try {
    const content = await fs.promises.readFile(filePath, 'utf8');
    return JSON.parse(content);
  } catch (error: any) {
    console.error(`Error reading/parsing file ${path.basename(filePath)}: ${error.message}`);
    return null;
  }
}

/**
 * Reads all files in a batch with concurrency limit
 */
async function readFileBatch(filenames: string[], inputDir: string, concurrencyLimit: number = 100): Promise<any[]> {
  const limit = pLimit(concurrencyLimit);
  
  const readPromises = filenames.map(filename => 
    limit(() => readAndParseFile(path.join(inputDir, filename)))
  );
  
  const results = await Promise.all(readPromises);
  // Filter out null values (failed reads/parses)
  return results.filter(data => data !== null);
}

/**
 * Converts data to MessagePack and saves to file
 */
async function saveAsMsgPack(data: any, outputPath: string): Promise<void> {
  const packed = pack(data);
  await fs.promises.writeFile(outputPath, packed);
  console.log(`Saved to ${outputPath}`);
}

/**
 * Ensures a directory exists, creating it if necessary
 */
async function ensureDirectoryExists(dirPath: string): Promise<void> {
  if (!fs.existsSync(dirPath)) {
    await fs.promises.mkdir(dirPath, { recursive: true });
  }
}

/**
 * Processes a single batch of files
 */
async function processBatch(
  batchIndex: number,
  filenames: string[], 
  inputDir: string, 
  outputDir: string,
  concurrencyLimit: number = 100
): Promise<void> {
  console.log(`Processing batch ${batchIndex + 1} with ${filenames.length} files...`);
  
  // Read and parse all files in the batch
  const tripData = await readFileBatch(filenames, inputDir, concurrencyLimit);
  
  console.log(`Successfully processed ${tripData.length}/${filenames.length} files in batch ${batchIndex + 1}.`);
  
  // Convert to MessagePack and save
  console.log(`Converting batch ${batchIndex + 1} to MessagePack...`);
  const outputPath = path.join(outputDir, `trip_batch_${batchIndex}.msgpack`);
  await saveAsMsgPack(tripData, outputPath);
}

/**
 * Main function to batch trip files into MessagePack files
 */
async function batchTrips(batchSize: number = 1000): Promise<void> {
  const inputDir = path.join(process.cwd(), 'data', 'trips');
  const outputDir = path.join(process.cwd(), 'data', 'trip_batches');
  
  console.log('Starting trip batching process...');
  
  // Ensure output directory exists
  await ensureDirectoryExists(outputDir);
  
  // Get all trip filenames
  const tripFiles = await getTripFilenames(inputDir);
  
  // Split into batches
  const batches = chunk(tripFiles, batchSize);
  console.log(`Created ${batches.length} batches of approximately ${batchSize} files each.`);
  
  // Process each batch sequentially
  for (let batchIndex = 0; batchIndex < batches.length; batchIndex++) {
    await processBatch(batchIndex, batches[batchIndex], inputDir, outputDir);
  }
  
  console.log('\nBatching process completed!');
  console.log(`Total batches created: ${batches.length}`);
}

// Execute the batching
if (require.main === module) {
  batchTrips().catch(error => {
    console.error('Error in batching process:', error);
    process.exit(1);
  });
}