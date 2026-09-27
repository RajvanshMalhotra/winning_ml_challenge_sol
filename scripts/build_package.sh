#!/usr/bin/env bash
# Assemble <team>_submission.zip.  usage: build_package.sh <team_name> <submission_dir with the two TSVs>
set -euo pipefail
TEAM=$1; SUBDIR=$2; MEMBERS=${3:-TEAM_MEMBERS}
P=artifacts/package/${TEAM}_submission
rm -rf "$P"; mkdir -p "$P/output" "$P/code/business_entity_resolution/src"
C=$P/code/business_entity_resolution
cp -r src/ber "$C/src/"; cp -r scripts "$C/src/scripts"; cp -r configs "$C/src/configs"; cp pyproject.toml "$C/src/"
find "$C/src" -name "__pycache__" -prune -exec rm -rf {} +
rm -f "$C/src/scripts/hpc.sh"
cp docs/package_README.md "$C/README.md"
cp docs/requirements_pinned.txt "$C/requirements.txt"
cp docs/requirements_vllm_pinned.txt "$C/requirements-vllm.txt"
sed -e "s/TEAM_NAME/${TEAM_DISPLAY:-$TEAM}/" -e "s/TEAM_MEMBERS/${MEMBERS}/" docs/Documentation_final.md > "$P/Documentation_template.md"
cp "$SUBDIR/matching_results.tsv" "$SUBDIR/candidate_pairs.tsv" "$P/output/"
python3 data/student_resource/utils/validate_submission.py --matching "$P/output/matching_results.tsv" \
    --candidate "$P/output/candidate_pairs.tsv" --test-dir data/student_resource/dataset/test | tail -2
(cd artifacts/package && rm -f "${TEAM}_submission.zip" && zip -qr -9 "${TEAM}_submission.zip" "${TEAM}_submission")
ls -la "artifacts/package/${TEAM}_submission.zip"; unzip -l "artifacts/package/${TEAM}_submission.zip" | tail -1
