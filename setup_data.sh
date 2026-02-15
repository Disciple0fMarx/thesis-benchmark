#!/bin/bash

# Define the root data directory
BASE_DIR="data/raw"
TEMP_DIR="opentraj_temp"

echo "🚀 Starting dataset download and organization..."

# Create the target directory structure
mkdir -p "$BASE_DIR/eth/univ"
mkdir -p "$BASE_DIR/eth/hotel"
mkdir -p "$BASE_DIR/ucy/univ"
mkdir -p "$BASE_DIR/ucy/zara1"
mkdir -p "$BASE_DIR/ucy/zara2"

# Clone OpenTraj (shallow clone to save time/space)
if [ ! -d "$TEMP_DIR" ]; then
    echo "📥 Cloning OpenTraj repository..."
    git clone --depth 1 https://github.com/crowdbotp/OpenTraj.git "$TEMP_DIR"
else
    echo "📦 OpenTraj folder already exists, skipping clone."
fi

echo "📂 Moving files to target folders..."

# 1. ETH Datasets
cp -r "$TEMP_DIR/datasets/ETH/seq_eth/." "$BASE_DIR/eth/univ/"
cp -r "$TEMP_DIR/datasets/ETH/seq_hotel/." "$BASE_DIR/eth/hotel/"

# 2. UCY Datasets
cp -r "$TEMP_DIR/datasets/UCY/students03/." "$BASE_DIR/ucy/univ/"
cp -r "$TEMP_DIR/datasets/UCY/zara01/." "$BASE_DIR/ucy/zara1/"
cp -r "$TEMP_DIR/datasets/UCY/zara02/." "$BASE_DIR/ucy/zara2/"

# Optional: Clean up the cloned repository
echo "🧹 Cleaning up temporary files..."
rm -rf "$TEMP_DIR"

echo "✅ Done! Your datasets are located in: $(pwd)/$BASE_DIR"
ls -R "$BASE_DIR"
