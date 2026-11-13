"""Collider stories with 24 variants per scenario (0-15 learn directions, 16-23 held out) -> stories_collider24.csv.
The first 11 variants are identical to stories_collider.csv used in the Stage 0 check."""
import zlib
import make_dataset as md
md.hash = lambda s: zlib.crc32(s.encode())
rows = md.build_collider(md.COLLIDER, per=24)
md.write(rows, "stories_collider24.csv")
print(f"collider24: {len(rows)} items / {len(md.COLLIDER)} scenarios")
