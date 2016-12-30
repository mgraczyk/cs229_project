class ClassifierNoLearning(object):
    def __init__(self):
        self._categories = {
                "other"  : ["book", "hacking", "guide", "porn", "custom", "cash", "bitcoin", "testkit", "watch", "pipe"],
                "rc"     : ["fa"],
                "mdma"   : ["mdma", "mda", "mde", "mdea"],
                "ped"    : ["ped", "steroid"],
                "psy"    : ["lsd", "25i", "25c", "2ci", "2cb", "mushrooms", "dmt"],
                "benzo"  : ["alprazolam", "xanax", "clonazepam", "klonopin", "lorazepam", "ativan", "diazepam", "valium", "flurazepam", "dalmane"],
                "dissoc" : ["ketamine", "mxe"],
                "script" : ["viagra", "cialis", "tadalafil", "prozac"],
                "opiate" : ["heroin", "tramadol", "oxycodone", "methadone", "opium", "fentanyl", "morphine"],
                "stim"   : ["meth", "amphetamine", "stimulant", "cocaine", "methylphenidate", "adderall", "modafinil", "provigil", "ritalin", "ephedrine", "ethylphenidate"],
                "weed"   : ["weed", "oil", "wax", "hash", "cannabis", "spice"],
                "misc"   : ["ghb", "gho", "bdo", "cigarettes", "alcohol"]}

        # Create a reverse lookup table
        self._drug_map = {}
        for (category, drugs) in self._categories.items():
            for drug in drugs:
                self._drug_map[drug] = category

    def classify(self, text):
        for word in text.lower().split(" "):
            if word in self._drug_map:
                return self._drug_map[word]

        return "other"
