folder=$(dirname "$1");
crawl_date=$(basename "$folder")
echo "$crawl_date"

sqlite3 market.db <<EOF
drop table if exists this_csv_dump;
create table this_csv_dump(
    hash TEXT,
    market_name TEXT,
    item_link TEXT,
    vendor_name TEXT,
    price REAL,
    name TEXT,
    description TEXT,
    image_link TEXT,
    add_time INTEGER,
    ship_from TEXT
);

.separator ","
.import "$1" this_csv_dump

create table if not exists agora(
    oid INTEGER PRIMARY KEY,
    crawl_date TEXT,
    hash TEXT,
    market_name TEXT,
    item_link TEXT,
    vendor_name TEXT,
    price REAL,
    name TEXT,
    description TEXT,
    image_link TEXT,
    add_time INTEGER,
    ship_from TEXT
);

INSERT INTO agora (
    crawl_date, hash, market_name, item_link, vendor_name,
    price, name, description, image_link, add_time, ship_from)
  SELECT "$crawl_date",
    hash, market_name, item_link, vendor_name, price, name,
    description, image_link, add_time, ship_from
  FROM this_csv_dump;
drop table this_csv_dump;
EOF
