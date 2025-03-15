import * as turf from "@turf/turf";
import type { Feature, Polygon } from "geojson";

/**
 * Slices a polygon into north-oriented bounding boxes of approximately the specified area
 * @param polygon - GeoJSON polygon to slice
 * @param maxAreaKm2 - Maximum area of each bbox in km²
 * @returns Array of bounding box polygons
 */
function sliceIntoBboxes(
  polygon: GeoJSON.Feature<GeoJSON.Polygon>,
  maxAreaKm2: number = 200
): GeoJSON.Feature<GeoJSON.Polygon>[] {
  // Calculate the polygon area in square kilometers
  const area = turf.area(polygon) / 1000000; // Convert from m² to km²
  console.log(`Polygon area: ${area.toFixed(2)} km²`);

  // Calculate the bbox of the whole polygon
  const bbox = turf.bbox(polygon);
  const bboxWidth = turf.distance([bbox[0], bbox[1]], [bbox[2], bbox[1]], {
    units: "kilometers",
  });
  const bboxHeight = turf.distance([bbox[0], bbox[1]], [bbox[0], bbox[3]], {
    units: "kilometers",
  });
  console.log(
    `Bbox dimensions: ${bboxWidth.toFixed(2)} km × ${bboxHeight.toFixed(2)} km`
  );

  // Calculate cell dimensions in kilometers
  const cellSideKm = Math.sqrt(maxAreaKm2);
  console.log(
    `Target cell size: ${cellSideKm.toFixed(2)} km × ${cellSideKm.toFixed(
      2
    )} km`
  );

  // Estimate number of cells needed along each axis
  const cellsX = Math.ceil(bboxWidth / cellSideKm);
  const cellsY = Math.ceil(bboxHeight / cellSideKm);
  console.log(
    `Grid dimensions: ${cellsX} × ${cellsY} = ${cellsX * cellsY} cells max`
  );

  console.log(
    `Using cell size: ${cellSideKm.toFixed(2)} km × ${cellSideKm.toFixed(2)} km`
  );

  // Turf will take care of converting kilometers to degrees
  const grid = turf.rectangleGrid(bbox, cellSideKm, cellSideKm, {
    units: "kilometers",
    mask: polygon,
  });

  console.log(
    `Generated ${grid.features.length} bounding boxes that intersect the polygon`
  );
  return grid.features as GeoJSON.Feature<GeoJSON.Polygon>[];
}

export { sliceIntoBboxes };
