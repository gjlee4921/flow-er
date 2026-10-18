#!/usr/bin/env bash
# Downloads the Figshare spleen FCS files (article 26063551) with resume, 4 at a time.
set -u
DEST="$HOME/code/reinvent-the-wheel/data/figshare_spleen"
LOCK="$DEST/.download.lock"
mkdir -p "$DEST/fcs"
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "Another download is already running (lock: $LOCK)."
  echo "If you're sure none is running, remove that folder and start again."; exit 1
fi
cleanup() { kill "$PROG" 2>/dev/null; pkill -P $$ 2>/dev/null; rmdir "$LOCK" 2>/dev/null; }
trap cleanup EXIT INT TERM
cp -n "/c/Users/leegj/Downloads/Spleen_Y_A_wsp_file.wsp" "$DEST/" 2>/dev/null || true
curl -s https://api.figshare.com/v2/articles/26063551 | python -I -c "
import sys,json
for f in json.load(sys.stdin)['files']: print(f\"{f['download_url']}\t{f['name']}\t{f['size']}\")" | tr -d '\r' > "$DEST/.manifest.tsv"
TOTAL=$(awk -F'\t' '{s+=$3} END{print s}' "$DEST/.manifest.tsv")
echo "Downloading $(wc -l < "$DEST/.manifest.tsv") files, $((TOTAL/1000000)) MB, into $DEST/fcs"
( prev=0; while sleep 20; do now=$(du -sb "$DEST/fcs" | cut -f1)
    echo "  progress: $((now/1000000)) / $((TOTAL/1000000)) MB ($((now*100/TOTAL))%)  ~$(( (now-prev)/20000 )) KB/s"; prev=$now; done ) &
PROG=$!
dl() { url="$1"; name="$2"; size="$3"; out="$DEST/fcs/$name"
  for try in 1 2 3 4 5; do
    have=$(stat -c %s "$out" 2>/dev/null || echo 0)
    [ "$have" = "$size" ] && { echo "  done: $name"; return 0; }
    [ "$have" -gt "$size" ] && { echo "  $name larger than expected; restarting it"; rm -f -- "$out"; }
    curl -sL -C - --retry 3 --connect-timeout 30 -o "$out" "$url"
  done
  echo "  FAILED: $name ($(stat -c %s "$out" 2>/dev/null || echo 0)/$size bytes)"; }
export -f dl; export DEST
start=$(date +%s)
tr '\t' '\n' < "$DEST/.manifest.tsv" | xargs -d '\n' -n 3 -P 4 bash -c 'dl "$0" "$1" "$2"'
ok=0; while IFS=$'\t' read -r u n s; do [ "$(stat -c %s "$DEST/fcs/$n" 2>/dev/null || echo 0)" = "$s" ] && ok=$((ok+1)); done < "$DEST/.manifest.tsv"
echo "Finished in $(( ($(date +%s)-start)/60 )) min: $ok / $(wc -l < "$DEST/.manifest.tsv") files complete and the right size."
