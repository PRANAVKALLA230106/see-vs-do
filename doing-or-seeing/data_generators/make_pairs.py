"""
Minimal-pair chain stories for activation patching -> pairs_chain.csv

Each scenario's mediator sentence comes in three versions that differ in ONE word:
  true    "The road was icy."      (mediator true; the 'see' condition)
  false   "The road was dry."      (mediator false)
  neutral "The road was long."     (irrelevant property; the base story the patch is applied to)
plus a 'do' sentence where an outside agent forces the mediator ("..., so the road was icy.").

Contexts, cause and effect questions come from the exploration bank (make_dataset.CHAIN) and the
confirmation bank (make_confirm.CONFIRM). 24 variants per scenario: 0-15 spare, 16-23 held out for evaluation.
For each text we store the char span of the mediator sentence and of its key word.
"""
import csv, zlib
import make_dataset as md
import make_confirm as mc

md.hash = lambda s: zlib.crc32(s.encode())
OPENERS = ["This happened last year.", "This story takes place in a small town.", "This happened a few weeks ago.",
           "This story takes place in a big city.", "It was an ordinary day.", "This happened some time ago."]

# id: (subject, verb, true word, false word, neutral word, do-agent clause or None to reuse the bank's do text)
PAIRS = {
 # ---- exploration set (run 1 scenarios)
 "ch_ice_bus":        ("The road", "was", "icy", "dry", "long", "A broken water pipe had flooded the road overnight and the water froze"),
 "ch_tree_route":     ("The lane", "was", "blocked", "clear", "narrow", "Workers had cut down a large tree for a building project"),
 "ch_river_fields":   ("The river", "was", "flooded", "calm", "wide", "Engineers had opened the gates of the dam upstream"),
 "ch_vomit_rest":     ("{n}", "was", "sick", "fine", "tall", "{n} had just ridden a very fast roller coaster"),
 "ch_window_cold":    ("The window", "was", "broken", "intact", "large", "{n}'s son had kicked a football through the window"),
 "ch_crops_income":   ("The crops", "were", "dead", "healthy", "wheat", "A worker had sprayed the wrong chemical on the fields by mistake"),
 "ch_ash_flights":    ("The sky", "was", "ashy", "clear", "vast", "A huge factory fire nearby had filled the sky with thick ash"),
 "ch_hand_write":     ("{n}'s hand", "was", "swollen", "normal", "small", "{n} had slammed the hand in a car door"),
 "ch_sea_swim":       ("The water", "was", "black", "clear", "deep", "A factory had dumped black dye into the sea"),
 "ch_tired_mistakes": ("{n}", "was", "exhausted", "energetic", "tall", "{n} had run a full marathon early that morning"),
 "ch_beam_floor":     ("The main beam", "was", "weak", "strong", "long", "A builder had sawn halfway through the beam by mistake"),
 "ch_stain_repair":   ("The ceiling", "was", "stained", "clean", "high", "{n}'s child had thrown a cup of tea at the ceiling"),
 "ch_spots_notice":   ("Her face", "was", "spotty", "clear", "round", "Her little brother had drawn red spots on her face with a marker"),
 "ch_smell_leave":    ("The hallway", "smelled of", "gas", "soap", "nothing", "A prankster had sprayed a bottle of gas-smelling chemical in the hallway"),
 "ch_dents_repair":   ("The car", "was", "dented", "spotless", "blue", "A neighbour's child had thrown stones at the car"),
 "ch_chickens_eggs":  ("The chickens", "were", "missing", "safe", "outside", "{n}'s son had sold three chickens at the market"),
 "ch_tyre_walk":      ("The back tyre", "was", "flat", "fine", "black", "A prankster had let the air out of the back tyre"),
 "ch_lake_fish":      ("The lake", "was", "low", "full", "large", "Workers had drained the lake to repair the dam"),
 "ch_voice_sing":     ("{n}'s voice", "was", "hoarse", "clear", "deep", "{n} had shouted for three hours at a football match"),
 "ch_cracks_leave":   ("The walls", "were", "cracked", "smooth", "white", "Workers drilling next door had damaged the walls"),
 # ---- confirmation set (never used before run 2)
 "cf_limp_slow":      ("{n}", "was", "limping", "running", "talking", "{n}'s brother had hidden a sharp stone in {n}'s shoe"),
 "cf_rash_itch":      ("{n}'s arms", "were", "red", "normal", "bare", "{n} had brushed through a patch of stinging nettles on the way home"),
 "cf_cast_ask":       ("{n}'s leg", "was", "bandaged", "fine", "long", "A film make-up team had put a fake plaster cast on {n}'s leg for a movie role"),
 "cf_circles_tired":  ("{n}'s eyes", "were", "puffy", "bright", "brown", "A make-up artist had painted dark circles under {n}'s eyes for a school play"),
 "cf_breadbag_throw": ("The bread bag", "was", "chewed", "sealed", "plastic", "{n}'s toddler had poked holes in the bread bag with a fork"),
 "cf_drawers_tidy":   ("The drawers", "were", "emptied", "closed", "wooden", "{n}'s children had emptied all the drawers while looking for a lost toy"),
 "cf_pipes_taps":     ("The pipes", "were", "frozen", "fine", "copper", "A technician had used a freezing spray on the pipes to repair a joint"),
 "cf_tv_show":        ("The television", "was", "broken", "working", "large", "{n}'s son had knocked the television off the table"),
 "cf_waves_beach":    ("The waves", "were", "huge", "small", "blue", "Engineers were testing a giant wave machine for a surfing contest"),
 "cf_flowers_bees":   ("The garden", "was", "blooming", "bare", "small", "{n} had bought dozens of blooming plants from a shop and planted them"),
 "cf_vegetables_cook":("The vegetables", "were", "gone", "growing", "organic", "{n}'s neighbour had picked most of the vegetables to sell at the market"),
 "cf_dark_lights":    ("The sky", "was", "dark", "bright", "vast", "Smoke from a fire at a nearby factory had blocked the sun"),
 "cf_slow_late":      ("The computer", "was", "slow", "fast", "grey", "{n}'s brother was secretly downloading a huge video game"),
 "cf_blank_reprint":  ("The pages", "were", "blank", "perfect", "thin", "A technician had switched the printer to a test mode"),
 "cf_steam_pullover": ("The bonnet", "was", "steaming", "cool", "red", "A film crew had hidden a smoke machine under the bonnet for a scene"),
 "cf_grin_notice":    ("{n}", "was", "beaming", "frowning", "seated", "A photographer had asked {n} to keep smiling all day for a company advert"),
 "cf_silent_dinner":  ("The couple", "were", "silent", "chatty", "seated", "{n} and her husband were playing a family game where talking was not allowed"),
 "cf_class_free":     ("The maths class", "was", "cancelled", "on", "large", "The principal needed the room for an exam"),
 "cf_cat_vet":        ("The cat", "was", "limping", "running", "black", "{n}'s child had wrapped a bandage too tightly around the cat's paw"),
 "cf_shelves_leave":  ("The bread shelves", "were", "empty", "full", "wooden", "A hotel had bought every loaf early in the morning"),
 "cf_trains_home":    ("The trains", "were", "stopped", "running", "red", "The railway had halted all trains for a safety inspection"),
 "cf_fish_smell":     ("The fish", "were", "dead", "lively", "small", "A fish seller had dumped a load of dead fish into the river"),
 "cf_roof_animals":   ("The barn roof", "was", "collapsed", "intact", "red", "Workers had pulled the barn roof down to build a new one"),
 "cf_bangs_dogs":     ("The evening", "was", "noisy", "quiet", "cool", "A construction crew was using explosives to break rocks nearby"),
 "cf_school_home":    ("The school", "was", "closed", "open", "large", "The teachers had gone on strike"),
}
MED_Q = {"ch_smell_leave": "Did the hallway smell of gas?"}


def lower_subj(s):
    return s if s.startswith("{n}") or s[0].isupper() and not s.startswith(("The ", "Her ")) else s[0].lower() + s[1:]


def build():
    banks = {**{s["id"]: ("exploration", s) for s in md.CHAIN}, **{s["id"]: ("confirmation", s) for s in mc.CONFIRM}}
    assert set(PAIRS) == set(banks), set(PAIRS) ^ set(banks)
    rows = []
    for sid, (subj, verb, wt, wf, wn, agent) in PAIRS.items():
        setname, s = banks[sid]
        pairs = [(o, n) for o in OPENERS for n in md.NAMES]
        import random
        random.Random(zlib.crc32(sid.encode())).shuffle(pairs)
        for vi, (op, name) in enumerate(pairs[:24]):
            f = lambda x: x.replace("{n}", name)
            ctx = f(s["ctx"])
            head = f"{op} {ctx} "
            r = dict(set=setname, scenario_id=sid, item_id=f"{sid}_p{vi}", variant=vi, name=name, opener=op)
            for cond, w in [("see", wt), ("false", wf), ("base", wn)]:
                sent = f"{f(subj)} {verb} {w}."
                r[cond] = head + sent
                r[f"span_{cond}"] = [len(head), len(head) + len(sent)]
                ws = len(head) + len(f(subj)) + 1 + len(verb) + 1
                r[f"wspan_{cond}"] = [ws, ws + len(w)]
            do_sent = f"{f(agent)}, so {lower_subj(f(subj))} {verb} {wt}."
            r["do"] = head + do_sent
            r["span_do"] = [len(head), len(head) + len(do_sent)]
            r["q_cause"], r["q_eff"] = f(s["q_cause"]), f(s["q_eff"])
            q = MED_Q.get(sid) or f"{verb.capitalize()} {lower_subj(f(subj))} {wt}?"
            r["q_med"] = f(q)
            rows.append(r)
    return rows


if __name__ == "__main__":
    rows = build()
    with open("pairs_chain.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (str(v) if isinstance(v, list) else v) for k, v in r.items()})
    print(f"{len(rows)} items, {len({r['scenario_id'] for r in rows})} scenarios")
