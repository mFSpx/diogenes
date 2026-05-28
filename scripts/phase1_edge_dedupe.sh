#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-09_STORAGE/krampuschewing_unpacked}"
OUT_DIR="${ROOT}/05_OUTPUTS/runtime"
STAMP="$(date -u +%Y%m%dT%H%M%S%6NZ)"
OUT_TSV="${OUT_DIR}/phase1_edge_dedupe_${STAMP}.tsv"
OUT_JSON="${OUT_DIR}/phase1_edge_dedupe_${STAMP}.json"

mkdir -p "${OUT_DIR}"

if [[ "${TARGET}" = /* ]]; then
  TARGET_PATH="${TARGET}"
else
  TARGET_PATH="${ROOT}/${TARGET}"
fi
if [[ ! -d "${TARGET_PATH}" ]]; then
  printf '{"ok":false,"blocker":"missing_target","target":"%s","receipt_path":"%s"}\n' "${TARGET}" "${OUT_JSON}" > "${OUT_JSON}"
  echo "${OUT_JSON}"
  exit 2
fi

tmp="$(mktemp)"
trap 'rm -f "${tmp}"' EXIT
declare -A seen

while IFS= read -r -d '' line; do
  hash="${line%% *}"
  path="${line#*  }"
  if [[ -n "${hash}" && -n "${path}" && -z "${seen[$hash]:-}" ]]; then
    printf '%s\t%s\n' "${path}" "${hash}" >> "${tmp}"
    seen["$hash"]=1
  fi
done < <(find "${TARGET_PATH}" -type f -print0 | sort -z | xargs -r -0 sha256sum -z)

mv "${tmp}" "${OUT_TSV}"

total_files="$(find "${TARGET_PATH}" -type f | wc -l | awk '{print $1}')"
unique_hashes="$(wc -l < "${OUT_TSV}" | awk '{print $1}')"
deduped="$(( total_files - unique_hashes ))"

cat > "${OUT_JSON}" <<EOF
{
  "schema": "lucidota.absurd_flows.phase1_edge_dedupe.v1",
  "ok": true,
  "target": "$(printf '%s' "${TARGET}" | sed 's/"/\\"/g')",
  "target_path": "$(printf '%s' "${TARGET_PATH}" | sed 's/"/\\"/g')",
  "mapping_path": "$(printf '%s' "${OUT_TSV#${ROOT}/}" | sed 's/"/\\"/g')",
  "unique_hashes": ${unique_hashes},
  "total_files": ${total_files},
  "deduped_files": ${deduped},
  "receipt_path": "$(printf '%s' "${OUT_JSON#${ROOT}/}" | sed 's/"/\\"/g')"
}
EOF

echo "${OUT_JSON}"
