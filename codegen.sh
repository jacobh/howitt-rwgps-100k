#!/bin/bash

# codegen.sh - Script to generate TypeScript API from Strava's Swagger spec

# Set variables
SCHEMAS_DIR="./schemas"
SRC_DIR="./src"
SWAGGER_URL="https://developers.strava.com/swagger/swagger.json"
SWAGGER_FILE="${SCHEMAS_DIR}/strava_swagger.json"
BUNDLED_SWAGGER="${SCHEMAS_DIR}/bundled_strava_swagger.json"
OPENAPI3_FILE="${SCHEMAS_DIR}/bundled_strava_openapi3.json"
OUTPUT_TS="${SRC_DIR}/stravaApi.ts"

# Create directories if they don't exist
mkdir -p "${SCHEMAS_DIR}"
mkdir -p "${SRC_DIR}"

echo "🚴 Starting Strava API TypeScript generation process..."

# Step 1: Download Swagger spec if it doesn't exist
if [ ! -f "${SWAGGER_FILE}" ]; then
  echo "📥 Downloading Strava Swagger spec..."
  wget "${SWAGGER_URL}" -O "${SWAGGER_FILE}"
else
  echo "✅ Using existing Swagger file: ${SWAGGER_FILE}"
fi

# Step 2: Bundle the Swagger spec
echo "📦 Bundling Swagger spec..."
npx swagger-cli bundle "${SWAGGER_FILE}" -o "${BUNDLED_SWAGGER}" -t json

# Step 3: Convert Swagger 2.0 to OpenAPI 3.0
echo "🔄 Converting from Swagger 2.0 to OpenAPI 3.0..."
npx api-spec-converter --from=swagger_2 --to=openapi_3 "${BUNDLED_SWAGGER}" > "${OPENAPI3_FILE}"

# Step 4: Generate TypeScript code
echo "⚙️ Generating TypeScript API client..."
npx swagger-typescript-api -p "${OPENAPI3_FILE}" \
  -o "${SRC_DIR}" \
  -n "stravaApi.ts" \
  --api-class-name StravaApi \
  -r \
  --extract-response-body \
  --extract-request-params \
  --extract-request-body

echo "✨ Done! TypeScript Strava API client has been generated at ${OUTPUT_TS}"
echo "🚀 You can now import the StravaApi class in your project."