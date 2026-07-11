#!/bin/bash
# Submit the three final jobs with: ./scripts/run_orfeo.sh

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

RUNNER="scripts/slurm/stage.sbatch"

CORE_JOB=$(sbatch --parsable --job-name=gamo_core "$RUNNER" core)
UNSTRUCTURED_JOB=$(sbatch --parsable --job-name=gamo_unstructured \
  --dependency="afterok:${CORE_JOB}" "$RUNNER" unstructured)
ABLATION_JOB=$(sbatch --parsable --job-name=gamo_ablation \
  --dependency="afterok:${CORE_JOB}" "$RUNNER" ablation)

echo "Submitted core training + structured job: $CORE_JOB"
echo "Submitted unstructured job after core: $UNSTRUCTURED_JOB"
echo "Submitted ablation job after core: $ABLATION_JOB"
echo "Pipeline: $CORE_JOB -> [$UNSTRUCTURED_JOB + $ABLATION_JOB]"
