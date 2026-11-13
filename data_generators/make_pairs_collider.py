"""
Minimal-pair collider stories for activation patching -> pairs_collider.csv
Rule in every story: C happens if A or B. The C sentence comes in true / false / neutral versions that differ in one word.
Conditions: base (neutral C), see (C true), false (C false), do (C forced by an outside agent),
see_A (C true, then A stated), A_only (neutral C, then A stated). Spans point at the C sentence and its key word.
"""
import csv, random, zlib
import make_dataset as md

OPENERS = ["This happened last year.", "This story takes place in a small town.", "This happened a few weeks ago.",
           "This story takes place in a big city.", "It was an ordinary day.", "This happened some time ago."]

# id: (subject, verb, true, false, neutral, do-agent clause)
PAIRS = {
 "co_grass":    ("The grass", "was", "wet", "dry", "green", "A neighbour had poured buckets of water on the grass"),
 "co_alarm":    ("The house alarm", "was", "ringing", "silent", "new", "A security company was testing the alarm"),
 "co_flight":   ("The flight", "was", "delayed", "punctual", "full", "The airline was holding the flight for a VIP passenger"),
 "co_headache": ("{n}'s head", "was", "sore", "clear", "round", "{n} had been hit on the head by a falling box"),
 "co_plant":    ("The plant", "was", "wilted", "upright", "tall", "{n}'s child had bent the plant's stems"),
 "co_floor":    ("The kitchen floor", "was", "wet", "dry", "tiled", "A cleaner had just mopped the floor"),
 "co_train":    ("The train", "was", "late", "punctual", "long", "The driver had stopped the train to film a video"),
 "co_website":  ("The website", "was", "down", "up", "blue", "A worker had pulled out the network cable by accident"),
 "co_shop":     ("The shop", "was", "closed", "open", "small", "The police had closed the shop to film a TV show"),
 "co_road":     ("The main road", "was", "blocked", "open", "long", "Protesters had parked their tractors across the road"),
 "co_milk":     ("The milk", "was", "sour", "fresh", "white", "{n}'s brother had added lemon juice to the milk as a joke"),
 "co_screen":   ("The phone screen", "was", "cracked", "perfect", "large", "A repair shop had cracked the screen while testing it"),
 "co_exam":     ("{n}'s exam mark", "was", "low", "high", "printed", "The teacher had entered the wrong marks by mistake"),
 "co_toast":    ("The toast", "was", "burnt", "golden", "square", "{n}'s friend had burnt the toast with a lighter as a prank"),
 "co_match":    ("The match", "was", "cancelled", "on", "important", "The principal needed the ground for a ceremony"),
 "co_bill":     ("The electricity bill", "was", "huge", "normal", "printed", "A clerk had typed an extra zero by mistake"),
 "co_race":     ("{n}'s finishing place", "was", "last", "first", "announced", "The judges had disqualified {n} by mistake"),
 "co_crying":   ("The baby", "was", "crying", "calm", "small", "A loud firework outside had startled the baby"),
 "co_lights":   ("The street lights", "were", "off", "on", "tall", "The council had switched the street lights off to save money"),
}


def low(s):
    return s if s.startswith("{n}") else s[0].lower() + s[1:]


def build():
    bank = {s["id"]: s for s in md.COLLIDER}
    assert set(PAIRS) == set(bank), set(PAIRS) ^ set(bank)
    rows = []
    for sid, (subj, verb, wt, wf, wn, agent) in PAIRS.items():
        s = bank[sid]
        pairs = [(o, n) for o in OPENERS for n in md.NAMES]
        random.Random(zlib.crc32(sid.encode())).shuffle(pairs)
        for vi, (op, name) in enumerate(pairs[:24]):
            f = lambda x: x.replace("{n}", name)
            head = f"{op} {f(s['ctx'])} "
            r = dict(scenario_id=sid, item_id=f"{sid}_p{vi}", variant=vi, name=name, opener=op)
            def put(cond, word, after=""):
                sent = f"{f(subj)} {verb} {word}."
                r[cond] = head + sent + (" " + after if after else "")
                r[f"span_{cond}"] = [len(head), len(head) + len(sent)]
                ws = len(head) + len(f(subj)) + 1 + len(verb) + 1
                r[f"wspan_{cond}"] = [ws, ws + len(word)]
            put("see", wt); put("false", wf); put("base", wn)
            put("see_A", wt, f(s["a_t"])); put("A_only", wn, f(s["a_t"]))
            do_sent = f"{f(agent)}, so {f(low(subj))} {verb} {wt}."
            r["do"] = head + do_sent
            r["span_do"] = [len(head), len(head) + len(do_sent)]
            r["q_b"], r["q_a"] = f(s["q_b"]), f(s["q_a"])
            r["q_c"] = f(f"{verb.capitalize()} {low(subj)} {wt}?")
            rows.append(r)
    return rows


if __name__ == "__main__":
    rows = build()
    with open("pairs_collider.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (str(v) if isinstance(v, list) else v) for k, v in r.items()})
    print(f"{len(rows)} items, {len({r['scenario_id'] for r in rows})} scenarios")
