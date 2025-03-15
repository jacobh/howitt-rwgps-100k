import { sliceIntoBboxes } from "../geo/sliceIntoBboxes";

export const searchPolygon = {
    "type": "FeatureCollection",
    "features": [
      {
        "type": "Feature",
        "properties": {},
        "geometry": {
          "coordinates": [
            [
              [
                145.2087597703823,
                -37.88468362833891
              ],
              [
                146.37505934449143,
                -38.22077618886137
              ],
              [
                147.3626917498001,
                -37.8768491697833
              ],
              [
                148.68284360815204,
                -37.8102226314988
              ],
              [
                149.19899320690598,
                -37.144609932209725
              ],
              [
                148.9260294767953,
                -36.76785182015516
              ],
              [
                149.23869702219514,
                -36.137127286347614
              ],
              [
                149.2436599991072,
                -35.05968610418373
              ],
              [
                148.78706612328614,
                -34.868527165739984
              ],
              [
                148.34536117819675,
                -34.94992590240486
              ],
              [
                148.2758795014421,
                -35.23013122488193
              ],
              [
                147.80439669488743,
                -35.5093721279283
              ],
              [
                146.7075787975349,
                -36.00474725706536
              ],
              [
                145.75468723060357,
                -37.03375921079269
              ],
              [
                145.28320442404885,
                -37.39737356547042
              ],
              [
                145.2087597703823,
                -37.88468362833891
              ]
            ]
          ],
          "type": "Polygon"
        }
      }
    ]
  }

// Example usage with your searchPolygon
const bboxes = sliceIntoBboxes(searchPolygon.features[0] as GeoJSON.Feature<GeoJSON.Polygon>, 100);
console.log(`Generated ${bboxes.length} bounding boxes`);

// Optional: Convert to a feature collection if needed
const bboxCollection: GeoJSON.FeatureCollection = {
  type: 'FeatureCollection',
  features: bboxes
};

console.log(JSON.stringify(bboxCollection, null, 2));