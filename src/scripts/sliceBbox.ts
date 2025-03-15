import { sliceIntoBboxes } from "../geo/sliceIntoBboxes";

export const searchPolygon = {
  type: "FeatureCollection",
  features: [
    {
      type: "Feature",
      properties: {},
      geometry: {
        coordinates: [
          [
            [145.929448872725, -37.38927224975594],
            [145.88913653317132, -37.869761093001316],
            [146.48777477554478, -37.99217732005912],
            [147.7435041526423, -37.552443605857],
            [147.5056613492742, -37.119743808090554],
            [147.1660790197394, -37.037602668529054],
            [146.69079498037803, -36.99104045731059],
            [146.52737710857946, -37.163219960923286],
            [146.07953729197004, -37.24568744266266],
            [145.929448872725, -37.38927224975594],
          ],
        ],
        type: "Polygon",
      },
    },
  ],
};

// Example usage with your searchPolygon
const bboxes = sliceIntoBboxes(searchPolygon.features[0] as GeoJSON.Feature<GeoJSON.Polygon>);
console.log(`Generated ${bboxes.length} bounding boxes`);

// Optional: Convert to a feature collection if needed
const bboxCollection: GeoJSON.FeatureCollection = {
  type: 'FeatureCollection',
  features: bboxes
};

console.log(JSON.stringify(bboxCollection, null, 2));