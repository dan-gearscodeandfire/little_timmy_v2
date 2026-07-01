"""Fetch face-gallery source images for the makers LT should recognize.

The definitive record of WHO LT should spontaneously identify (Dan's list,
2026-06-30) + how their gallery source is fetched. Source is each creator's OWN
YouTube channel (avatar + recent video thumbnails), NOT web image search — the
channel's own content guarantees the right person (the recurring dominant face IS
the creator), avoiding the famous-namesake / cartoon contamination plain-name
search suffers.

    slug  ->  (display_name, channel_url)

display_name is what LT will call them (Dan's choice: channel name unless he
specified otherwise, e.g. NileRed -> "Nigel", Nighthawkinlight -> "Ben").

yt-dlp is heavy and pulls a CUDA torch on some resolvers, so it must NOT enter
the production venv. Run this with a throwaway venv's yt-dlp:

    python3 -m venv /tmp/ytv && /tmp/ytv/bin/pip install yt-dlp
    python ops/fetch_channel_faces.py --ytdlp /tmp/ytv/bin/yt-dlp \
        --out /tmp/faces          # writes /tmp/faces/{anchors,thumbs}/<slug>/

Then build galleries (production venv):
    python -m ops.build_maker_gallery --raw /tmp/faces/thumbs \
        --anchors /tmp/faces/anchors --out ops/calib/makers
"""

import argparse
import json
import subprocess
import urllib.request
from pathlib import Path

# slug -> (display_name, channel_url). Dan's list, 2026-06-30.
CHANNELS = {
    "william_osman": ("William Osman", "https://www.youtube.com/channel/UCfMJ2MchTSW2kWaT0kK94Yw"),
    "nigel": ("Nigel", "https://www.youtube.com/@NileRed"),
    "becky_stern": ("Becky Stern", "https://www.youtube.com/@BeckyStern"),
    "ben": ("Ben", "https://www.youtube.com/@Nighthawkinlight"),
    "tomasz": ("Tomasz", "https://www.youtube.com/@Mellow_labs"),  # Dan's booth co-host
    "colin_furze": ("Colin Furze", "https://www.youtube.com/@colinfurze"),
    "ruth_amos": ("Ruth Amos", "https://www.youtube.com/@Ruth_Amos"),
    "estefannie": ("Estefannie", "https://www.youtube.com/@Estefannie"),
    "simone_giertz": ("Simone Giertz", "https://www.youtube.com/@simonegiertz"),
    "michael_reeves": ("Michael Reeves", "https://www.youtube.com/@MichaelReeves"),
    "allen_pan": ("Allen Pan", "https://www.youtube.com/@allenpan"),
    "kevin": ("Kevin", "https://www.youtube.com/@TheBackyardScientist"),
    "chroma": ("Chroma", "https://www.youtube.com/@ChromaLock"),
    "keith": ("Keith", "https://www.youtube.com/@KeithSachs"),
    "nate": ("Nate", "https://www.youtube.com/@natescovill"),
}


def _ytdlp_json(ytdlp: str, url: str) -> dict:
    r = subprocess.run([ytdlp, "--dump-single-json", "--playlist-items", "0", url],
                       capture_output=True, text=True, timeout=120)
    return json.loads(r.stdout)


def _video_ids(ytdlp: str, url: str, n: int) -> list:
    r = subprocess.run([ytdlp, "--flat-playlist", "--get-id",
                        "--playlist-items", f"1-{n}", url],
                       capture_output=True, text=True, timeout=180)
    return [x for x in r.stdout.split() if x]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ytdlp", required=True, help="path to a throwaway-venv yt-dlp")
    ap.add_argument("--out", default="/tmp/faces")
    ap.add_argument("--thumbs", type=int, default=40)
    ap.add_argument("--only", nargs="*", help="subset of slugs")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "anchors").mkdir(parents=True, exist_ok=True)
    slugs = a.only or list(CHANNELS)
    for slug in slugs:
        name, url = CHANNELS[slug]
        try:
            d = _ytdlp_json(a.ytdlp, url)
            avs = [t for t in d.get("thumbnails", []) if "avatar" in t.get("id", "")]
            if avs:
                aurl = sorted(avs, key=lambda t: t.get("width", 0))[-1]["url"]
                urllib.request.urlretrieve(aurl, out / "anchors" / f"{slug}.jpg")
            tdir = out / "thumbs" / slug
            tdir.mkdir(parents=True, exist_ok=True)
            got = 0
            for vid in _video_ids(a.ytdlp, url, a.thumbs):
                for q in ("maxresdefault", "hqdefault"):
                    try:
                        urllib.request.urlretrieve(
                            f"https://i.ytimg.com/vi/{vid}/{q}.jpg", tdir / f"{vid}.jpg")
                        got += 1
                        break
                    except Exception:
                        continue
            print(f"{slug:16} {name:16} avatar={'y' if avs else 'n'} thumbs={got}")
        except Exception as e:
            print(f"{slug:16} ERROR {str(e)[:60]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
