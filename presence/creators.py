"""Registry of enrolled "creators" (the OpenSauce makers LT should recognize).

Canonical map of face-enrollment slug -> display name, used to (a) tag presence
entries as creators vs household on the booth "who's present" panel, and (b)
render proper display names (the slugs are lowercase/underscored for the on-disk
voiceprint/face-print convention). Mirrors ops/fetch_channel_faces.CHANNELS —
keep in sync when the recognize-list changes.
"""

# slug -> display name. Everyone here is a recognized creator (booth guest);
# anyone NOT here (dan, family, timmy, couples_therapist, unknown_*) is household.
CREATORS = {
    "william_osman": "William Osman",
    "nigel": "Nigel",
    "becky_stern": "Becky Stern",
    "ben": "Ben",
    "tomasz": "Tomasz",
    "colin_furze": "Colin Furze",
    "ruth_amos": "Ruth Amos",
    "estefannie": "Estefannie",
    "simone_giertz": "Simone Giertz",
    "michael_reeves": "Michael Reeves",
    "allen_pan": "Allen Pan",
    "kevin": "Kevin",
    "chroma": "Chroma",
    "keith": "Keith",
    "nate": "Nate",
}


def is_creator(name: str) -> bool:
    return (name or "").strip().lower() in CREATORS


def display_name(name: str) -> str:
    """Proper display name: the registry name for creators, else a title-cased
    de-slugged fallback (``william_osman`` -> ``William Osman``)."""
    n = (name or "").strip()
    if n.lower() in CREATORS:
        return CREATORS[n.lower()]
    return " ".join(w.capitalize() for w in n.replace("_", " ").split())
