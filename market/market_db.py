import os
import sqlite3 as sql
from util import memoized
import subprocess

import constants

class MarketDB(object):
  def __init__(self, dbfileName, readonly=True):
    self._created_db = False

    noDataYet = not os.path.isfile(dbfileName)

    uri_fmt = "file:{}{}"
    mode_str = "?mode=ro" if readonly else ""

    self._conn = sql.connect(uri_fmt.format(dbfileName, mode_str), uri=True)

    # self._conn.row_factory = sql.Row
    self._conn.isolation_level = "DEFERRED"
    self._conn.execute("pragma foreign_keys=on")
    self._conn.execute("pragma synchronous=0")
    self._conn.execute("pragma journal_mode=MEMORY")
    self._conn.execute("pragma temp_store=MEMORY")

    self._create_labels_db()

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
    cur.execute("DELETE FROM agora")
    self.commit()
    cur.execute("PRAGMA VACUUM")

  def created_db(self):
    return self._created_db

  def select_listings(self, max_count=2**40):
    cur = self.get_cursor()
    cur.execute("SELECT * FROM agora LIMIT {};".format(max_count));
    return cur.fetchall()

  def select_columns(self, column_names="*", max_count=2**40):
    cur = self.get_cursor()
    cur.execute("SELECT {} FROM agora LIMIT {};".format(
                column_names, max_count));
    return cur.fetchall()

  def select_normalized_data_columns(self, column_names="*", max_count=2**40):
    self._create_normalized_data_table()
    cur.execute(
        "SELECT {} FROM temp.normalized LIMIT {};".format(column_names, max_count));
    return cur.fetchall()

  def select_random_listings(self, column_names="*", max_count=1):
    cur = self.get_cursor()
    cur.execute("SELECT {} FROM agora ORDER BY RANDOM() LIMIT {};".format(
                column_names, max_count));
    return cur.fetchall()

  def select_labeled_listings(self, column_names="*", max_count=2**40):
    self._create_normalized_data_table()
    cur = self.get_cursor()
    cur.execute(
        """SELECT {}
           FROM temp.normalized INNER JOIN labels
           ON temp.normalized.oid == labels.oid;
        """.format(column_names))
    return cur.fetchall()

  def _create_db(self):
    raise NotImplementedError()

  def _create_normalized_data_table(self):
    cur = self.get_cursor()
    # TODO(mgraczyk): Add normalization back once we can run more quickly.
    cur.execute(
        """CREATE TEMP TABLE IF NOT EXISTS normalized as
           SELECT oid, hash, market_name, vendor_name, price,
                  replace(name, " 39 ", "'") as name,
                  replace(description, " 39 ", "'") as description
           FROM agora;
        """)
    # cur.execute(
        # """CREATE TEMP TABLE IF NOT EXISTS normalized as
           # SELECT oid, hash, vendor_name, price, name, description
           # FROM agora;
        # """)
    self.commit()

  def _create_labels_db(self):
    if not os.path.isfile(constants.DEFAULT_LABELS_DATABASE_PATH):
        status = subprocess.check_output([
            "./create_labels_db.sh",
            constants.DEFAULT_LABELS_DATABASE_PATH
        ], shell=True)

    # Link the labels table into the market database.
    cur = self.get_cursor()
    cur.execute('ATTACH database "{}" as labels;'.format(
        constants.DEFAULT_LABELS_DATABASE_PATH))
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
