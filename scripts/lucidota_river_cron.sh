#!/bin/bash

# Source the environment script
source lucidota_safe_ops_env.sh

# Run the River training loop
python3 scripts/lucidota_river_reflex.py --limit 1000 --json > 05_OUTPUTS/receipts/river_cron_$(date +%Y%m%dT%H%M%SZ).json

# Print summary
echo "River training loop completed. Output logged to 05_OUTPUTS/receipts/river_cron_$(date +%Y%m%dT%H%M%SZ).json"

To run this script every 15 minutes, add the following line to your crontab:
*/15 * * * * bash /home/mfspx/LUCIDOTA/scripts/lucidota_river_cron.sh
