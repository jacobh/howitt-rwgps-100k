import type {
  BoundingBox,
  ExploreTripsParams,
  ExploreTripsResponse,
  TripDetailsResponse,
} from "./types";

/**
 * Client for interacting with the Ride with GPS API
 */
export class RwgpsApiClient {
  private baseUrl: string = "https://ridewithgps.com"
  private apiKey: string;
  
  constructor(
    apiKey?: string,
  ) {
    // Use provided API key, or fall back to environment variable, or use default
    this.apiKey = apiKey ??
                  (typeof process !== 'undefined' ? process.env.RWGPS_API_KEY : undefined) ?? ''
  }

  /**
   * Get common headers used for all API requests
   */
  private getHeaders(): HeadersInit {
    return {
      Accept: "application/json",
      "Accept-Language": "en-US,en;q=0.5",
      "x-rwgps-api-key": this.apiKey,
      "content-type": "application/json",
      "Sec-Fetch-Dest": "empty",
      "Sec-Fetch-Mode": "cors",
      "Sec-Fetch-Site": "same-origin",
      Priority: "u=4",
      Pragma: "no-cache",
      "Cache-Control": "no-cache",
    };
  }

  /**
   * Get trips in a bounding box
   */
  async exploreTrips(
    boundingBox: BoundingBox,
    options: Omit<ExploreTripsParams, "boundingBox"> = {}
  ): Promise<ExploreTripsResponse> {
    const [south, west, north, east] = boundingBox;
    const boundingBoxParam = `${south},${west},${north},${east}`;

    const params = new URLSearchParams({
      bounding_box: boundingBoxParam,
      models: options.models?.join(",") || "trips",
      sort_by: options.sortBy || "relevance desc",
    });

    if (options.offset) {
      params.append("offset", options.offset.toString());
    }

    if (options.nextAssets) {
      params.append("next_assets", options.nextAssets);
    }

    if (options.limit) {
      params.append("limit", options.limit.toString());
    }

    const url = `${this.baseUrl}/explore.json?${params.toString()}`;

    console.log(url);

    const response = await fetch(url, {
      credentials: "include",
      headers: this.getHeaders(),
      referrer: `${this.baseUrl}/explore?b=b!${west}!${south}!${east}!${north}&m=rides`,
      method: "GET",
      mode: "cors",
    });

    if (!response.ok) {
      throw new Error(
        `Failed to fetch trips: ${response.status} ${response.statusText}`
      );
    }

    return (await response.json()) as ExploreTripsResponse;
  }

  /**
   * Get detailed trip data by ID
   */
  async getTripDetails(tripId: number): Promise<TripDetailsResponse> {
    const url = `${this.baseUrl}/trips/${tripId}.json`;

    const response = await fetch(url, {
      credentials: "include",
      headers: this.getHeaders(),
      referrer: `${this.baseUrl}/trips/${tripId}`,
      method: "GET",
      mode: "cors",
    });

    if (!response.ok) {
      throw new Error(
        `Failed to fetch trip details: ${response.status} ${response.statusText}`
      );
    }

    return (await response.json()) as TripDetailsResponse;
  }
}
