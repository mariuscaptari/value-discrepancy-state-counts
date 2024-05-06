#!/bin/bash

# Define the base directory where results are stored
BASE_DIR="tmp/atari"

# Method being used
METHOD="vpd_simhash"

# File with the list of Atari games
GAME_LIST_FILE="atari-game-list.txt"

# Output file to store the run count
OUTPUT_FILE="atari-game-run-count.txt"

# Clear the output file or create it if it doesn't exist
> "$OUTPUT_FILE"

# Iterate over each game
while IFS= read -r GAME
do
    # Initialize a string to store run details for the game
    RUN_DETAILS=""

    # Check for all possible iteration folders
    ITERATION=0
    while [ -d "$BASE_DIR/$GAME/$METHOD"_"$ITERATION" ]
    do
        # Find the maximum training run number in this iteration
        MAX_RUN=$(ls "$BASE_DIR/$GAME/$METHOD"_"$ITERATION/logs/" | grep 'log_' | sed 's/log_//' | sort -n | tail -1)

        # Add 1 to MAX_RUN to adjust for 0-indexing
        MAX_RUN=$((MAX_RUN + 1))

        # Append the maximum run number for this iteration to the run details string
        [ -z "$RUN_DETAILS" ] && RUN_DETAILS="$MAX_RUN" || RUN_DETAILS="$RUN_DETAILS, $MAX_RUN"

        # Move to the next iteration
        ((ITERATION++))
    done

    # Append the game and its run details to the output file
    echo "$GAME - Iterations: $RUN_DETAILS" >> "$OUTPUT_FILE"
done < "$GAME_LIST_FILE"

echo "Detailed run information for each game has been saved to $OUTPUT_FILE:"

# Display the contents of the output file
cat "$OUTPUT_FILE"
