#!/usr/bin/env ipython3
# coding=UTF-8

from __future__ import print_function

import glob
import logging
import tarfile
from lxml import etree
import os
import sys
from os.path import isdir
from os.path import isfile
import re
import IPython
from datetime import datetime
from datetime import date
from datetime import time

from message_db import MessageDB

logger = logging.getLogger("darknet_parser")

TEST_FILE_PATH = r"./testdata/2014-01-02.tar.xz"
DATA_PATH = r"/Volumes/My Passport/dnmarchives/agora-forums"
TESTDATA_PATH = r"/local/ssd/dev/learning/stanford/cs229/darknets/testdata"
TESTDATA_PATH = r"/local/ssd/dev/learning/stanford/cs229/darknets/data/agora-forums"
REGEXP_NS = "http://exslt.org/regular-expressions"

tar_file_date_regex = re.compile(r"(?P<year>\d+)[-_](?P<month>\d+)[-_](?P<day>\d+)")
reply_time_regex = re.compile(r"(?:Reply #\d+\b|) on:")

# Example: December 05, 2013, 09:54:24 pm
time_string_fmt = r" %B %d, %Y, %I:%M:%S %p »"
today_time_string_fmt = r" at %I:%M:%S %p »"

EPOCH_TIME = datetime.utcfromtimestamp(0)
def unix_time_millis(dt):
  return int((dt - EPOCH_TIME).total_seconds() * 1000.0)

def as_iter(tar):
  while True:
    result = tar.next()
    if result is None:
      raise StopIteration
    else:
      yield result

def today_from_crawl_name(crawl_name):
  today_match = tar_file_date_regex.match(crawl_name)
  if not today_match:
    raise ValueError("crawl had unexpected format".format(crawl_name))
  return { k: int(v) for k, v in today_match.groupdict().items() }

parser = etree.HTMLParser(remove_blank_text=True, remove_comments=True)
POST_WRAPPER_XPATH = etree.XPath("""//div[@class="post_wrapper"]""")
POST_POSTER_XPATH = etree.XPath("""div[@class="poster"]/h4/a""")
POST_AREA_XPATH = etree.XPath("""div[@class="postarea"]""")
POST_TIME_XPATH = etree.XPath("""div[@class="flow_hidden"]/div[@class="keyinfo"]/div[@class="smalltext"]""")
POST_MESSAGE_XPATH = etree.XPath("""div[@class="post"]/div[@class="inner"]""")

def extract_post_from_wrapper(post_wrapper, today):
  # TODO(mgraczyk): Handle cases where inner <a/> is missing.
  posters = POST_POSTER_XPATH(post_wrapper)
  if len(posters) != 1:
    raise ValueError("found {} posters".format(len(posters)))
  poster = posters[0]
  poster_name = poster.text
  poster_id_index = poster.get("href").rfind("u=") + 2
  poster_id = int(poster.get("href")[poster_id_index:])

  post_areas = POST_AREA_XPATH(post_wrapper)
  if len(post_areas) != 1:
    raise ValueError("found {} post_area sub divs".format(len(post_areas)))
  post_area = post_areas[0]

  post_time_wrappers = POST_TIME_XPATH(post_area)
  if len(post_time_wrappers) != 1:
    raise ValueError("found {} post_time_wrappers".format(len(post_time_wrappers)))
  post_time_wrapper = post_time_wrappers[0]
  post_time_string = post_time_wrapper.getchildren()[0].tail
  try:
    post_datetime = datetime.strptime(post_time_string, time_string_fmt)
  except ValueError as e:
    # Try parsing the today format.
    today_node = post_time_wrapper.getchildren()[1]
    today_string = etree.tostring(today_node)
    if today_string.startswith(b"<strong>Today<"):
      post_time = datetime.strptime(today_node.tail, today_time_string_fmt).time()
      post_date = date(today["year"], today["month"], today["day"])
      post_datetime = datetime.combine(post_date, post_time)
    else:
      raise e

  message_results = POST_MESSAGE_XPATH(post_area)
  if len(message_results) != 1:
    raise ValueError("found {} message_results".format(len(message_results)))
  message_result = message_results[0]

  # TODO(mgraczyk): Find a way to capture image tokens
  message_text = message_result.text
  message_id = message_result.get("id")[4:]
  return (unix_time_millis(post_datetime), message_text, message_id, poster_id)

def insert_messages_in_db(db, message_html_file, scrape_info, today):
  tree = etree.parse(message_html_file, parser)
  post_wrappers = POST_WRAPPER_XPATH(tree)
  for post_wrapper in post_wrappers:
    try:
      post = extract_post_from_wrapper(post_wrapper, today)
      db.insert_post_tuple(post + scrape_info)
    except Exception as e:
      print(etree.tostring(post_wrapper))
      raise ValueError("Could not parse post {}".format(scrape_info)) from e

def insert_tared_messages_in_db(db, tar_path, topic_message_identifier):
  path_basename = os.path.basename(tar_path)
  today = today_from_crawl_name(path_basename)

  try:
    with tarfile.open(tar_path, mode="r|xz") as tar:
      for tarinfo in as_iter(tar):
        if (tarinfo.isfile() and
            topic_message_identifier in tarinfo.name and
            tarinfo.name.endswith(".html")):
          with tar.extractfile(tarinfo) as f:
            scrape_info = (path_basename, os.path.basename(tarinfo.name))
            insert_messages_in_db(db, f, scrape_info, today)
  except Exception as e:
    raise ValueError("Error parsing {}".format(tarinfo.name)) from e

def insert_from_archives(db, archives_path, archive_glob, topic_message_identifier):
  archive_paths = glob.glob(os.path.join(archives_path, archive_glob))
  for archive_path in archive_paths:
    print("Processing {}...".format(archive_path))
    insert_tared_messages_in_db(db, archive_path, topic_message_identifier)

def main(argv):
  archive_glob = argv[1] if len(argv) > 1 else "*.tar.gz"
  topic_message_identifier = argv[2] if len(argv) > 2 else "topic"

  with MessageDB("data.db") as db:
    insert_from_archives(db, TESTDATA_PATH, archive_glob, topic_message_identifier)

if __name__ == "__main__":
  main(sys.argv)
