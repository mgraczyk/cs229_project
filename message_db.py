import os
import sqlite3 as sql
from util import memoized

class MessageDB(object):
    UnknownResultType = "unspecified"

    def __init__(self, dbfileName):
        self._created_db = False

        noDataYet = not os.path.isfile(dbfileName)
        self._conn = sql.connect(dbfileName)

        self._conn.row_factory = sql.Row
        self._conn.isolation_level = "DEFERRED"
        self._conn.execute("pragma foreign_keys=on")
        self._conn.execute("pragma synchronous=0")
        self._conn.execute("pragma journal_mode=MEMORY")
        self._conn.execute("pragma temp_store=MEMORY")

        if noDataYet:
            try:
                self._create_db()
            except e:
                os.remove(dbfileName)
                raise IOError("Couldn't create the database.") from e

    def get_cursor(self):
        return self._conn.cursor()

    def close(self):
        self._conn.commit()
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exec_type, exec_value, traceback):
        self.close()

    def commit(self):
        self._conn.commit()

    def clear_data(self):
        cur = self.get_cursor()
        cur.execute("DELETE FROM posts")
        self.commit()
        cur.execute("PRAGMA VACUUM")

    def created_db(self):
        return self._created_db

    def select_random_posts(self, count=1):
        cur = self.get_cursor()
        cur.execute("SELECT * FROM posts ORDER BY RANDOM() LIMIT {};".format(count));
        return cur.fetchall()

    def _create_db(self):
        cur = self.get_cursor()

        with open("createdb.sql", "r") as createScript:
            cur.executescript(createScript.read())

        self._created_db = True
        self.commit()

    def insert_from_dict(self, queryFmt, tblColNames, dataObj):
        cur = self.get_cursor()

        data = self._get_non_none_column_data(tblColNames, dataObj)

        query = queryFmt.format(', '.join(data.keys()),
                                        ', '.join(":"+n for n in data.keys()))

        cur.execute(query, data)
        return cur.lastrowid

    def insert_post_tuple(self, post_tuple):
      try:
        cur = self.get_cursor()
        cur.execute("""
            INSERT INTO posts
              (datetime, message, message_id, poster_id, scrape_file, topic)
            SELECT ?, ?, ?, ?, ?, ?
            """, post_tuple)
        post_id = cur.lastrowid
        return post_id
      except Exception as e:
        raise ValueError("Could not insert {}".format(post_tuple)) from e

    @memoized
    def _get_non_pk_column_names(self, tblName):
        """ Get an iterable of the column names in the table with name
        tblName which are not part of the primary key.
        """
        cur = self.get_cursor()
        return set(d["name"] for d in
                      cur.execute("PRAGMA table_info({})".format(tblName)) if not d["pk"])

    @staticmethod
    def _get_non_none_column_data(tblColNames, dataObj):
        return {k:v for k,v in dataObj.items() if k in tblColNames and v is not None}
