"""
Minimal-pair FORK stories for activation patching -> pairs_fork.csv

Fork: common cause A -> observed fact B, and A -> sibling effect C.
  see   "The pond was frozen."   (B true: suggests cold weather A, hence thick coats C)
  false "The pond was rippling." (B false)
  base  "The pond was round."    (irrelevant property; the story the patch is applied to)
  do    "Park staff had used freezing machines for an ice show, so the pond was frozen."  (B forced: no info about A or C)
Contexts and questions come from make_dataset.FORK. q_eff is the sibling effect C.
24 variants per scenario; 16-23 are used for evaluation.
"""
import csv, zlib, random
import make_dataset as md

md.hash = lambda s: zlib.crc32(s.encode())
OPENERS = ["This happened last year.", "This story takes place in a small town.", "This happened a few weeks ago.",
           "This story takes place in a big city.", "It was an ordinary day.", "This happened some time ago."]

# id: (subject, verb, true word, false word, neutral word, do-agent clause, mediator question)
PAIRS = {
 "fk_branches_furniture": ("The lawn", "was", "littered", "tidy", "large",
                           "A gardener had dumped a pile of cut branches on the lawn", "Was the lawn littered with branches?"),
 "fk_music_cars":         ("The house next door", "was", "noisy", "silent", "old",
                           "The neighbour was testing new speakers alone", "Was the house next door noisy?"),
 "fk_bread_queue":        ("The bakery chimney", "was", "smoking", "cold", "tall",
                           "Workers were testing the empty bakery's chimney with a smoke machine", "Was the bakery chimney smoking?"),
 "fk_park_school":        ("The park", "was", "crowded", "empty", "green",
                           "A birthday party had brought many children to the park", "Was the park crowded with children?"),
 "fk_lift_lights":        ("The lift", "was", "dead", "working", "small",
                           "A technician had switched off the lift for maintenance", "Had the lift stopped working?"),
 "fk_pond_coats":         ("The pond", "was", "frozen", "rippling", "round",
                           "Park staff had used freezing machines for an ice show", "Was the pond frozen?"),
 "fk_stomach_guests":     ("{n}'s stomach", "was", "aching", "fine", "flat",
                           "{n} had done two hundred sit-ups after dinner", "Did {n} have a stomach ache?"),
 "fk_siren_glass":        ("The ambulance outside", "was", "speeding", "parked", "white",
                           "An ambulance crew was practising for a training drill", "Was an ambulance speeding past?"),
 "fk_pictures_news":      ("The pictures", "were", "scattered", "hanging", "colourful",
                           "{n}'s children had knocked the pictures off the walls while playing football indoors",
                           "Had the pictures fallen off the walls?"),
 "fk_car_quit":           ("{n}'s car", "was", "luxurious", "rusty", "blue",
                           "The company had given {n} a luxurious car as a work vehicle", "Did {n} have a luxurious new car?"),
 "fk_tar_pool":           ("The tar on the road", "was", "melting", "hard", "black",
                           "Road workers had heated the tar with machines", "Was the tar on the road soft?"),
 "fk_tired_babyclothes":  ("The couple", "were", "exhausted", "rested", "tall",
                           "The couple had been painting their house all night", "Did the couple look tired?"),
 "fk_fans_trophy":        ("The main street", "was", "jubilant", "deserted", "wide",
                           "A film crew had paid crowds of people to celebrate in the main street for a movie scene",
                           "Were fans celebrating in the street?"),
 "fk_call_neighbours":    ("The video call", "was", "disconnected", "smooth", "important",
                           "{n}'s little sister had unplugged the laptop", "Did the video call drop?"),
 "fk_busstop_roads":      ("The bus stop", "was", "crowded", "empty", "new",
                           "A free concert nearby had drawn a big crowd to the bus stop", "Was the bus stop crowded?"),
}


def lower_subj(s):
    return s if s.startswith("{n}") or (s[0].isupper() and not s.startswith(("The ", "Her "))) else s[0].lower() + s[1:]


def build():
    bank = {s["id"]: s for s in md.FORK}
    assert set(PAIRS) == set(bank), set(PAIRS) ^ set(bank)
    rows = []
    for sid, (subj, verb, wt, wf, wn, agent, qmed) in PAIRS.items():
        s = bank[sid]
        ops = [o for o in OPENERS if not (sid == "fk_fans_trophy" and "small town" in o)]   # story says "the city"
        pairs = [(o, n) for o in ops for n in md.NAMES]
        random.Random(zlib.crc32(sid.encode())).shuffle(pairs)
        for vi, (op, name) in enumerate(pairs[:24]):
            f = lambda x: x.replace("{n}", name)
            head = f"{op} {f(s['ctx'])} "
            r = dict(set="fork", scenario_id=sid, item_id=f"{sid}_p{vi}", variant=vi, name=name, opener=op)
            for cond, w in [("see", wt), ("false", wf), ("base", wn)]:
                sent = f"{f(subj)} {verb} {w}."
                r[cond] = head + sent
                r[f"span_{cond}"] = [len(head), len(head) + len(sent)]
                ws = len(head) + len(f(subj)) + 1 + len(verb) + 1
                r[f"wspan_{cond}"] = [ws, ws + len(w)]
                assert r[cond][ws:ws + len(w)] == w
            do_sent = f"{f(agent)}, so {lower_subj(f(subj))} {verb} {wt}."
            r["do"] = head + do_sent
            r["span_do"] = [len(head), len(head) + len(do_sent)]
            r["q_cause"], r["q_eff"], r["q_med"] = f(s["q_cause"]), f(s["q_sib"]), f(qmed)
            rows.append(r)
    return rows


if __name__ == "__main__":
    rows = build()
    with open("pairs_fork.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (str(v) if isinstance(v, list) else v) for k, v in r.items()})
    print(f"{len(rows)} items, {len({r['scenario_id'] for r in rows})} scenarios")
    for r in rows[16::24][:15]:
        print("|", r["see"], "\n|", r["base"], "\n|", r["do"], "\n|", r["q_cause"], "/", r["q_eff"], "/", r["q_med"], "\n")
