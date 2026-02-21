#!/bin/bash
LOG=batch_repair_results/worst_cases_run.log
REPORT=batch_repair_results/batch_report.csv
TOTAL=39

echo "=== Progress ==="
done=$(grep -E "^[0-9]+_original" "$REPORT" 2>/dev/null | cut -d, -f1 | sort -u | grep -Ff <(python3 -c "
import os, re
for f in os.listdir('results/Input_MnM2/worst_cases/'):
    m = re.match(r'^[\d.]+_\w+_\w+_(\d+_original_lax_\w+_\d+)\.png$', f)
    if m: print(m.group(1))
") | wc -l)
echo "Completed: $done / $TOTAL"

echo ""
echo "=== Latest Verdicts ==="
grep -E "^[0-9]+_original" "$REPORT" 2>/dev/null | tail -10 | awk -F, '{printf "  %-45s %s  %.4f→%.4f (Δ%+.4f)\n", $1, $2, $5, $6, $7}'

echo ""
echo "=== Summary ==="
grep -E "^[0-9]+_original" "$REPORT" 2>/dev/null | cut -d, -f2 | sort | uniq -c | sort -rn

echo ""
echo "=== Currently processing ==="
tail -5 "$LOG"
