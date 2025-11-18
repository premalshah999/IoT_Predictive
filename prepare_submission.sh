#!/bin/bash

################################################################################
# Prepare Submission - IoT Anomaly Detection Project
################################################################################
# This script creates a clean submission copy without Extra directory
################################################################################

echo "================================================================================"
echo "PREPARING SUBMISSION COPY"
echo "================================================================================"
echo ""

# Define directories
SOURCE_DIR="$(pwd)"
PARENT_DIR="$(dirname "$SOURCE_DIR")"
SUBMISSION_DIR="$PARENT_DIR/iot_anomaly_detection_submission"

# Remove old submission directory if exists
if [ -d "$SUBMISSION_DIR" ]; then
    echo "🗑️  Removing old submission directory..."
    rm -rf "$SUBMISSION_DIR"
fi

# Create new submission directory
echo "📁 Creating submission copy..."
mkdir -p "$SUBMISSION_DIR"

# Copy all files except Extra, venv, and hidden files
echo "📋 Copying files..."
rsync -av --progress \
    --exclude='Extra/' \
    --exclude='venv/' \
    --exclude='.claude/' \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    --exclude='prepare_submission.sh' \
    --exclude='SUBMISSION_READY.md' \
    "$SOURCE_DIR/" "$SUBMISSION_DIR/"

echo ""
echo "================================================================================"
echo "VERIFICATION"
echo "================================================================================"
echo ""

# Count files
TOTAL_FILES=$(find "$SUBMISSION_DIR" -type f | wc -l | xargs)
echo "✅ Total files copied: $TOTAL_FILES"

# Verify key files
echo ""
echo "Checking core deliverables:"
[ -f "$SUBMISSION_DIR/README.md" ] && echo "  ✅ README.md" || echo "  ❌ README.md MISSING"
[ -f "$SUBMISSION_DIR/iot_anomaly_utils.py" ] && echo "  ✅ iot_anomaly_utils.py" || echo "  ❌ iot_anomaly_utils.py MISSING"
[ -f "$SUBMISSION_DIR/iot_anomaly.API.md" ] && echo "  ✅ iot_anomaly.API.md" || echo "  ❌ iot_anomaly.API.md MISSING"
[ -f "$SUBMISSION_DIR/iot_anomaly.API.ipynb" ] && echo "  ✅ iot_anomaly.API.ipynb" || echo "  ❌ iot_anomaly.API.ipynb MISSING"
[ -f "$SUBMISSION_DIR/iot_anomaly.example.ipynb" ] && echo "  ✅ iot_anomaly.example.ipynb" || echo "  ❌ iot_anomaly.example.ipynb MISSING"
[ -f "$SUBMISSION_DIR/Dockerfile" ] && echo "  ✅ Dockerfile" || echo "  ❌ Dockerfile MISSING"
[ -d "$SUBMISSION_DIR/models" ] && echo "  ✅ models/ directory" || echo "  ❌ models/ MISSING"
[ -d "$SUBMISSION_DIR/results" ] && echo "  ✅ results/ directory" || echo "  ❌ results/ MISSING"
[ -d "$SUBMISSION_DIR/charts" ] && echo "  ✅ charts/ directory" || echo "  ❌ charts/ MISSING"

echo ""
echo "Checking excluded directories:"
[ ! -d "$SUBMISSION_DIR/Extra" ] && echo "  ✅ Extra/ excluded" || echo "  ⚠️  Extra/ still present"
[ ! -d "$SUBMISSION_DIR/venv" ] && echo "  ✅ venv/ excluded" || echo "  ⚠️  venv/ still present"
[ ! -d "$SUBMISSION_DIR/.claude" ] && echo "  ✅ .claude/ excluded" || echo "  ⚠️  .claude/ still present"

echo ""
echo "================================================================================"
echo "DIRECTORY SIZE"
echo "================================================================================"
du -sh "$SUBMISSION_DIR"

echo ""
echo "================================================================================"
echo "✅ SUBMISSION READY"
echo "================================================================================"
echo ""
echo "Submission directory created at:"
echo "  📁 $SUBMISSION_DIR"
echo ""
echo "Next steps:"
echo "  1. Review the submission directory"
echo "  2. Test that scripts run correctly"
echo "  3. Create zip file or submit directory"
echo ""
echo "To create zip:"
echo "  cd $SUBMISSION_DIR"
echo "  zip -r ../iot_anomaly_detection.zip ."
echo ""
echo "✅ Done!"
