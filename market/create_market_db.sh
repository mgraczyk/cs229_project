#!//bin/bash
grams_path="/local/ssd/dev/learning/stanford/cs229/darknets/data/grams"

rm -f market.db
find "${grams_path}" -name "Agora.csv" | sort | \
  xargs -I{} sh import_market_data.sh {}

# Remove the column labels
sqlite3 market.db "delete from agora where hash == 'hash';"

# Make the database read-only for safety.
chmod 400 market.db
