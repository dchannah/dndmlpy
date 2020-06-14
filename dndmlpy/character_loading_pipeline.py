import json
import numpy as np
from collections import Counter
from dndmlpy.char_rep import Character


# For filtering out unusual classes with little data
NORMAL_CLASSES = [
    "Fighter",
    "Rogue",
    "Barbarian",
    "Cleric",
    "Paladin",
    "Bard",
    "Monk",
    "Druid",
    "Wizard",
    "Warlock",
    "Sorcerer",
]


def load_json(path_to_file: str) -> dict:
    with open(path_to_file, encoding="utf-8") as f:
        return json.load(f)


def json_to_list_of_characters(loaded_json: dict) -> list:
    return [Character(name, char) for name, char in loaded_json.items()]


def popular_weapons(charlist: list, popularity_fraction_threshold: float) -> list:
    all_weapons = []
    for character in charlist:
        all_weapons += list(character.weapons)
    weapon_counts = Counter(all_weapons)
    return [
        weapon
        for weapon, weapon_count in weapon_counts.items()
        if weapon_count > len(charlist) * popularity_fraction_threshold
    ]


def all_skills(charlist: list) -> list:
    all_skills = set()
    for character in charlist:
        all_skills.update(character.skills)
    return sorted(list(all_skills))


def build_input_data(
    charlist: list,
    possible_skills_list: list,
    possible_weapons_list: list,
    include_attributes: bool = True,
    include_skills: bool = True,
    include_weapons: bool = True,
    include_char_names: bool = True,
) -> tuple:

    feature_labels = []

    if include_attributes:
        feature_labels += ["str", "dex", "con", "int", "wis", "cha"]
    if include_skills:
        feature_labels += ["proficient in " + skill for skill in possible_skills_list]
    if include_weapons:
        feature_labels += ["has " + weapon for weapon in possible_weapons_list]

    return np.array(
        [
            character.build_feature_vector(
                possible_weapons_list,
                possible_skills_list,
                include_attributes,
                include_skills,
                include_weapons,
                include_char_names,
            )
            for character in charlist
        ]
    ), feature_labels


def build_output_data(charlist: list) -> np.array:
    return np.array([character.highest_class for character in charlist])


def filter_charlist(charlist: list) -> list:
    return [c for c in charlist if c.highest_class in NORMAL_CLASSES]

