sqlite3 $1 <<EOF
CREATE TABLE labels (
    oid INTEGER,
    hash TEXT,
    count INTEGER,
    dose REAL,
    unit TEXT,
    drug TEXT,
    category TEXT
);

.import "labels.csv" labels

EOF
