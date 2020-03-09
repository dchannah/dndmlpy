"""
The goal here is to use this to build a classifier that ultimately
measures uniqueness. Imagine we build a classifier that predicts a
character class, then flag the characters that the classifier fails
most badly on.
"""

import numpy as np
import os
from enum import Enum


# Global variables
ATTRIBUTES_FIELD = "attributes"
STR_FIELD = "Str"
DEX_FIELD = "Dex"
CON_FIELD = "Con"
INT_FIELD = "Int"
WIS_FIELD = "Wis"
CHA_FIELD = "Cha"
CLASS_FIELD = "class"
LEVEL_FIELD = "level"
RACE_FIELD = "race"
RACE_SUBFIELD = "processedRace"
WEAPONS_FIELD = "weapons"
SKILLS_FIELD = "skills"


def charprint_list_rep(info_list: list) -> str:
    """ Converts a list into a bulleted list separated by newlines for pretty printing. """
    rep = ""
    for item in info_list:
        rep += f"    * {item}{os.linesep}"
    return rep


class Character:
    """ A Python representation of a D&D 5e character to facilitate data analysis.

    Notes:
        This character currently assumes a JSON-formatted document as its json_doc initialization
        argument; specifically, one of the JSON documents from oganm's excellent dnddata repo. In
        the future, this class could be subclassed to e.g. JSONCharacter, CSVCharacter,
        GSheetCharacter, and so forth.

    """

    def __init__(self, character_name: str, json_doc: dict) -> None:

        self.character_name = character_name

        # Attributes
        self.strength = json_doc[ATTRIBUTES_FIELD][STR_FIELD][0]
        self.dexterity = json_doc[ATTRIBUTES_FIELD][DEX_FIELD][0]
        self.constitution = json_doc[ATTRIBUTES_FIELD][CON_FIELD][0]
        self.intelligence = json_doc[ATTRIBUTES_FIELD][INT_FIELD][0]
        self.wisdom = json_doc[ATTRIBUTES_FIELD][WIS_FIELD][0]
        self.charisma = json_doc[ATTRIBUTES_FIELD][CHA_FIELD][0]

        # Classes
        self.classes = {
            class_name.strip("\n"): class_info[LEVEL_FIELD][0]
            for class_name, class_info in json_doc[CLASS_FIELD].items()
        }

        # Race
        self.race = json_doc[RACE_FIELD][RACE_SUBFIELD][0]

        # Level
        self.level = json_doc[LEVEL_FIELD][0]

        # Weapons
        self.weapons = set(weapon for weapon in json_doc[WEAPONS_FIELD])

        # Skills
        self.skills = json_doc[SKILLS_FIELD]

    def __str__(self):
        """ A pretty-print representation of the data that we store for characters. """
        return f"""
Name: {self.character_name}
Race: {self.race}
Level: {self.level}

Attributes:
    * Strength: {self.strength}
    * Dexterity: {self.dexterity}
    * Constitution: {self.constitution}
    * Intelligence: {self.intelligence}
    * Wisdom: {self.wisdom}
    * Charisma: {self.charisma}

Classes: 
{charprint_list_rep(["{}: {}".format(job, level) for job, level in self.classes.items()])}
Proficient Skills:
{charprint_list_rep(self.skills)}
Weapons:
{charprint_list_rep(list(self.weapons))}
        """

    @property
    def highest_class(self) -> str:
        """ 
        This implementation is hideous, but basically, we need to sort the dictionary of classes,
        then return the first tuple element (i.e. the class) of the last tuple (the tuple that 
        describes the class having the highest level).
        """
        return sorted(self.classes.items(), key=lambda class_level: class_level[1])[-1][
            0
        ]

    @property
    def attributes_vector(self) -> list:
        return [
            self.strength,
            self.dexterity,
            self.constitution,
            self.intelligence,
            self.wisdom,
            self.charisma,
        ]

    def popular_weapons(self, popular_weapons: list) -> set:
        return self.weapons.intersection(set(popular_weapons))

    def weapon_onehot(self, popular_weapons: list) -> list:
        """ Think of this list as [character_has_knife: 1, character_has_bow: 0, ..., etc.] """
        return [1 if weapon in self.weapons else 0 for weapon in popular_weapons]

    def skills_onehot(self, all_possible_skills: list) -> list:
        """ Think of this list as [is_proficient_in_nature: 1, is_proficient_in_insight: 0, ..] """
        return [1 if skill in self.skills else 0 for skill in all_possible_skills]

    def build_feature_vector(
        self,
        popular_weapons: list,
        all_possible_skills: list,
        include_attributes: bool,
        include_skills: bool,
        include_weapons: bool,
        include_char_names: bool = True,
    ) -> np.array:
        """ Returns a feature vector for use in ML algorithms.

        Notes:
            The popular_weapons and all_possible skills lists are needed because all featurized
            characters must have the same one-hot vector describing their skills and equipment;
            these lists essentially provide a consistent mask. Also note that the character name
            always rides along in this feature vector. This might seem like a strange choice in a
            vacuum, but is done so that name labels persist through a randomized test/train split.
            It's often useful to know which characters were, for example, outliers in a particular
            analysis.
        
        Args:
            popular_weapons: A list of sufficiently popular weapons.
            all_possible_skills: A list of skills a character *could* have.
            include_attributes: Should character attributes be included in the feature vector?
            include_skills: Should a one-hot of skill proficiencies be included?
            include_weapons: Should a one-hot of weapons be included?

        """
        feature_vector_as_list = []
        if include_char_names:
            feature_vector_as_list += [self.character_name]
        if include_attributes:
            feature_vector_as_list += self.attributes_vector
        if include_skills:
            feature_vector_as_list += self.skills_onehot(all_possible_skills)
        if include_weapons:
            feature_vector_as_list += self.weapon_onehot(popular_weapons)
        return np.array(feature_vector_as_list)
