#!/usr/bin/env python3

import os
import sqlite3 as sql
import difflib as diff
import re
from Levenshtein import *

class DB(object):
    def __init__(self, dbFileName):
        dbExists = os.path.isfile(dbFileName)

        # Connect to the database
        self._conn = sql.connect("file:{}".format(dbFileName), uri=True)
        self._conn.row_factory = sql.Row

        # Initialize the database if it doesn't exist
        if not dbExists:
            try:
                self.create()
            except e:
                os.remove(dbFileName)
                raise IOError("Couldn't create the database") from e

    def __enter__(self):
        return self

    def __exit__(self, exec_type, exec_value, traceback):
        self.commit()
        self.close()

    def commit(self):
        self._conn.commit()

    def close(self):
        self._conn.close()

class LabelDB(DB):
    def create(self):
        curs = self._conn.cursor()
        curs.execute("""
            CREATE TABLE IF NOT EXISTS labels (
                oid INTEGER,
                hash TEXT,
                count INTEGER,
                dose REAL,
                unit TEXT,
                drug TEXT,
                category TEXT
            )""")
        self.commit()

    def insert(self, oid, hash_val, count, dose, unit, drug, category):
        curs = self._conn.cursor()
        curs.execute("INSERT INTO labels VALUES (?,?,?,?,?,?,?)",
                (oid, hash_val, count, dose, unit, drug, category))
        self.commit()

    def getNames(self):
        curs = self._conn.cursor()
        curs.execute("SELECT DISTINCT drug FROM labels")
        return curs.fetchall()

    def getUnits(self):
        curs = self._conn.cursor()
        curs.execute("SELECT DISTINCT unit FROM labels")
        return curs.fetchall()

    def getNamesInCategory(self, category):
        curs = self._conn.cursor()
        curs.execute("SELECT DISTINCT drug FROM labels WHERE category == (?)", (category,))
        return curs.fetchall()

    def get_training_data(self):
        # read in the data
        c = self._conn.cursor()
        c.execute("SELECT category, name, description from labels")
        result = c.fetchall()

        # concat the name and description for the data
        data = [name + " " + desc for (name, desc) in zip(result["name"], result["description"])]
        
        catagories = result["category"]
        labels     = [category.index(cat) for cat in catagories]

        return Bunch(data=data, labels=labels, catagories=catagories)

class MarketDB(DB):
    def randomListing(self, count=1):
        curs = self._conn.cursor()
        curs.execute("SELECT oid, hash, name, description FROM agora ORDER BY RANDOM() LIMIT {}".format(count))
        return curs.fetchall()

def random_listings(count=1):
    return MarketDB('market.db').randomListing(count)

class StringProcessor(object):
    re_alpha_dot_num = re.compile(r"[^A-Za-z0-9_.]")

    def remove_not_alpha_num_dot(self, a_string):
      return self.re_alpha_dot_num.sub(u" ", a_string)  

def suggest_name(candidates, description, threshold):
    words = set(StringProcessor().remove_not_alpha_num_dot(description).lower().strip().split())

    suggestion = ""
    best_distance = threshold
    for canidate in candidates:
        for word in words:
            current_distance = distance(str(word), str(canidate))
            if current_distance < best_distance:
                best_distance = current_distance
                suggestion = canidate

    return suggestion

class Labels(object):
    def __init__(self, db_filename):
        self._db = LabelDB(db_filename)

        self._categories = {
            "other"  : ["book", "hacking", "guide", "porn", "custom", "cash", "bitcoin", "id", "testkit", "watch", "pipe"],
            "rc"     : [],
            "mdma"   : ["mdma", "mda", "mde", "ecstasy"],
            "ped"    : ["ped", "steroid"],
            "psy"    : ["lsd", "25", "2c"],
            "benzo"  : ["alprazolam", "xanax", "clonazepam", "klonopin", "lorazepam", "ativan", "diazepam", "valium", "flurazepam", "dalmane"],
            "dissoc" : ["ketamine", "mxe"],
            "script" : ["viagra", "cialis", "tadalafil"],
            "opiate" : ["heroin", "tramadol", "oxycodone", "methadone", "opium", "fentanyl", "morphine"],
            "stim"   : ["meth", "amphetamine", "stimulant", "cocaine", "methylphenidate", "adderall", "modafinil", "provigil", "ritalin", "ephedrine"],
            "weed"   : ["weed", "oil", "wax", "hash", "cannabis", "spice"],
            "misc"   : ["ghb", "gho", "bdo", "cigarettes", "alcohol"]}

        for category in self._categories.keys():
            names = self._db.getNamesInCategory(category)
            self._categories[category] = list(set(self._categories[category]) | set(names))

    def insert(self, oid, hash_value, count, dose, unit, drug, category):
        if drug != "" and not (drug in self._categories[category]):
            self._categories[category].append(drug)

        self._db.insert(oid, hash_value, count, dose, unit, drug, category)

    def suggestCategory(self, text):
        # create a lookup tabel so we know which drug corresponds to which category
        drug_map = {}
        for (category, drugs) in self._categories.items():
            for drug in drugs:
                drug_map[drug] = category

        # check each word in the text with the drug list
        best_category = "other"
        best_distance = 5
        for word in text.lower().split(' '):
            for drug in drug_map.keys():
                dist = distance(str(drug), str(word))
                if dist < best_distance:
                    best_category = drug_map[drug]
                    best_distance = dist

        return best_category


    def suggestDrug(self, text, category):
        best_drug = ""
        best_distance = 5
        for word in text.lower().split(' '):
            for drug in self._categories[category]:
                dist = distance(str(drug), str(word))
                if dist < best_distance:
                    best_drug = drug
                    best_distance = dist

        return best_drug

    def suggestUnit(self, text, drug):
        return ""

def label_loop(count):
    labels = Labels('labels2.db')

    for listing in random_listings(count):
        # print the name and description 
        print("name: ", listing["name"])
        print("desc: ", listing["description"])

        # Compute a suggested name and unit
        text = listing["name"] + " " + listing["description"]

        # read the category, drug, count, dose, unit
        category_suggestion = labels.suggestCategory(text)
        category = input("category ({}) = ".format(category_suggestion)).lower().strip()
        if category == "":
            category = category_suggestion
        while not category in labels._categories.keys():
            category_suggestion = labels.suggestCategory(text)
            category  = input("bad category '" + category + "': correct to '" + category_suggestion + "' ?").lower().strip()
            if category == "":
                category = category_suggestion

        drug_suggestion = labels.suggestDrug(text, category)
        drug = input("drug ({}) = ".format(drug_suggestion)).lower().strip()
        if drug == "":
            drug = drug_suggestion

        # Read in the data, defaulting to the suggestions
        count = input("count  = ").lower().strip()
        dose  = input("amount = ").lower().strip()

        unit_suggestion = "" 
        unit  = input("unit   = ({}) ".format(unit_suggestion)).lower().strip()
        if unit == "":
            unit = unit_suggestion

        labels.insert(listing["oid"], listing["hash"], count, dose, unit, drug, category)

if __name__ == '__main__':
    label_loop(100)
