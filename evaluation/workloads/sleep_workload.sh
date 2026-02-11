#!/bin/bash

# Default to 60 if no argument given
DURATION=${1:-60}

echo "Sleeping for ${DURATION} seconds..."
sleep ${DURATION}
echo "Done sleeping."