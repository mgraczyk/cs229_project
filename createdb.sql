PRAGMA foreign_keys=ON;

BEGIN TRANSACTION;

CREATE TABLE posts
        (oid INTEGER PRIMARY KEY,
         datetime DATE NOT NULL check(datetime(datetime)),
         message TEXT CHECK(length(message)>0),
         message_id INTEGER NOT NULL,
         poster_id INTEGER NOT NULL,
         scrape_file TEXT CHECK(length(scrape_file) > 0),
         topic TEXT CHECK(length(topic) > 0),
         UNIQUE(datetime, message_id, topic) ON CONFLICT IGNORE);

COMMIT;
