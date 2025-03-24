await fetch("https://ridewithgps.com/explore.json?bounding_box=-37.34%2C146.396672%2C-37.097749%2C146.647226&models=trips&sort_by=relevance%20desc", {
    "credentials": "include",
    "headers": {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:137.0) Gecko/20100101 Firefox/137.0",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.5",
        "x-rwgps-api-key": "ak17s7k3",
        "x-rwgps-api-version": "3",
        "content-type": "application/json",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "Priority": "u=4",
        "Pragma": "no-cache",
        "Cache-Control": "no-cache"
    },
    "referrer": "https://ridewithgps.com/explore?b=b!146.396672!-37.340000!146.647226!-37.097749&m=rides",
    "method": "GET",
    "mode": "cors"
});

const resp = {
    "results": [
      {
        "id": 155259958,
        "type": "trip",
        "url": "/trips/155259958",
        "administrative_area": "Victoria",
        "avg_speed": 12.0276,
        "distance": 10457.2,
        "duration": 3693,
        "elevation_gain": 333.047,
        "elevation_loss": 336.895,
        "first_lat": -37.14603,
        "first_lng": 146.453445,
        "last_lat": -37.146034,
        "last_lng": 146.45343,
        "locality": "Mount Buller",
        "name": "Mount Buller Mountain Biking",
        "activity_type": "cycling:mountain",
        "byline_name": "John Griffiths",
        "simplified_polyline": "tavaF_e{|Z?gtuH@Mua@wo@...[shortened for brevity]",
        "track_type": "out_and_back",
        "terrain": "climbing",
        "difficulty": "easy"
      },
      // Additional trip objects with similar structure
      // Just showing one example for brevity
    ],
    "total_count": 10,
    "extras": [
      {
        "type": "user",
        "id": 872340,
        "user": {
          "id": 872340,
          "name": "John Griffiths",
          "profile_photo_path": "/users/872340/profile-default-40@2x.png",
          "locality": "Melbourne",
          "country_code": "AU"
          // Other user properties
        }
      }
      // Additional user objects
    ],
    "permissions": {
      "155259958": {
        "navigate": true,
        "customize_export_options": true,
        "export_as_history": true,
        "can_be_copied": true,
        "can_be_modified": false
      }
      // Permissions for other trip IDs
    },
    "meta": {
      "api_version": 3,
      "explore": {
        "bounding_box": [
          { "lat": -37.172544, "lng": 146.434321 },
          { "lat": -37.129559, "lng": 146.478739 }
        ],
        "models": ["trips"],
        "total_count": 1429,
        "next_page_url": "/explore.json?bounding_box=-37.172544%2C146.434321%2C-37.129559%2C146.478739&models=trips&next_assets=Trip-210013705%2CTrip-206849733%2CTrip-229877044%2CTrip-235943807%2CTrip-244209760&offset=15&sort_by=relevance+desc"
      },
      "type": "trip"
    }
  }

await fetch("https://ridewithgps.com/user_activities/record?object_id=155259958&object_type=Trip&ua_action=viewed", {
    "credentials": "include",
    "headers": {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:137.0) Gecko/20100101 Firefox/137.0",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.5",
        "x-rwgps-api-key": "ak17s7k3",
        "x-rwgps-api-version": "3",
        "content-type": "application/json",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "Idempotency-Key": "\"3044825633234570417\"",
        "Priority": "u=4",
        "Pragma": "no-cache",
        "Cache-Control": "no-cache"
    },
    "referrer": "https://ridewithgps.com/trips/155259958",
    "method": "POST",
    "mode": "cors"
});

const resp2 = {
  "trip": {
    "id": 155259958,
    "name": "Mount Buller Mountain Biking",
    "administrative_area": "Victoria",
    "country_code": "AU",
    "locality": "Mount Buller",
    "created_at": "2024-03-16T17:33:31+11:00",
    "departed_at": "2021-11-20T15:23:31+11:00",
    "description": null,
    "distance": 10457.2,
    "duration": 3693,
    "moving_time": 3130,
    "elevation_gain": 333.047,
    "elevation_loss": 336.895,
    "avg_speed": 12.0276,
    "max_speed": 35.118,
    "min_hr": 74,
    "max_hr": 163,
    "activity_type": "cycling:mountain",
    "visibility": 0,
    "likes_count": 0,
    "views": 1,
    
    // Bounding box coordinates
    "first_lat": -37.14603,
    "first_lng": 146.453445,
    "last_lat": -37.146034,
    "last_lng": 146.45343,
    "ne_lat": -37.144371,
    "ne_lng": 146.475342,
    "sw_lat": -37.157738,
    "sw_lng": 146.45343,
    
    // Sample track point (of 3089 total)
    "track_points": [
      {
        "x": 146.453445, // longitude
        "y": -37.14603,  // latitude
        "e": 1590.6,     // elevation (m)
        "d": 0.0,        // distance (m)
        "s": 0.0,        // speed (m/s)
        "T": 12.0,       // temperature (C)
        "t": 1637382211, // timestamp
        "h": 78          // heart rate
      }
    ],
    
    // Calculated metrics
    "metrics": {
      "ele": {
        "max": 1641.2,
        "min": 1517,
        "avg": 1575.88
      },
      "hr": {
        "max": 163,
        "min": 74,
        "avg": 130.93
      },
      "speed": {
        "max": 35.103,
        "min": 0,
        "avg": 12.00
      },
      "grade": {
        "max": 15.53,
        "min": -16.36,
        "avg": 3.07
      },
      "temp": {
        "max": 16,
        "min": 10,
        "avg": 12.35
      },
      "stationary": false,
      "pace": 353.15,
      "movingPace": 299.98,
      "vam": 542.48,
      
      // Hill segments analysis
      "hills": [
        {
          "first_i": 247,
          "last_i": 1077,
          "ele_gain": 125,
          "ele_loss": 3,
          "distance": 2071,
          "avg_grade": 7.798,
          "is_climb": true
        }
      ]
    }
  },
  
  "permissions": {
    "navigate": true,
    "customize_export_options": true,
    "export_as_history": true,
    "can_be_copied": true,
    "can_be_modified": false
  },
  
  "user": {
    "id": 872340,
    "name": "John Griffiths",
    "locality": "Melbourne",
    "administrative_area": "07",
    "country_code": "AU",
    "created_at": "2016-11-04T15:48:22+11:00",
    "account_level": 0
  }
}