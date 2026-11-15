# Claude Chan, a 3D Arch figure

Claude Chan (Claude娘: orange hair and a chrysanthemum motif) as a static 3D figure, built from her
design sheet with nothing but parametric surfaces and a few curves.

| File | |
|---|---|
| `claude-chan-3d.txt` | The Arch save file. Open it with Arch's *Open file*. |
| `claude-chan-3d.arch` | The scene script (677 objects); the save file is built from it. |
| `generate.py` | Writes the scene script. `face.py`, `outfit.py`, `cape.py`, `arms.py`, `hair.py`, `boots.py` and `scene.py` are its parts. |
| `sheetfit.py` | Measures the design sheet and checks the model against it. |

Opening the file takes several minutes (it is about 11 MB): it is saved with `implicitPrecision=10`, so
each surface is meshed on a 125 x 125 grid instead of Arch's default 50 x 50, and the turned head's
surfaces are about three times as long to evaluate. Lower the *equation, inequality and parametric
surface precision* slider in Arch's settings for faster redraws while rotating.

## What is modelled

- **Proportions**: measured on the sheet's full-body front view (正面). Her head, crown to chin, is 1/7.9
  of her height; the elbows are at the waist, the cuffs at the hips and the fingertips at mid-thigh, with
  the forearms and hands angled forward and out, clear of the skirt. The skirt flares from the corset in a straight A-line, as
  wide as the sheet's at every height, and ends high enough to show the lace-up boots.
- **Head**: sculpted from parts, tilted toward her right shoulder and turned a little to her right, which
  gives her a lively pose. The tilt is 7 degrees at the eyes, between the sheet's calm portraits (7 to 9
  degrees) and its full-body front view (about 3). The face is a
  stack of horizontal arcs whose width, depth and roundness change with height, so it has a real profile:
  forehead, the dip between the eyes, a small pointed nose, lips, a receding chin, and a jawline that
  runs up to the ear. Seen from the front the jaw runs nearly straight from the cheekbones to a blunt,
  rounded chin: the portrait's V-shaped jaw, a little softer so it reads well in 3D. The round of each
  cheek, the brow and the eye sockets are shallow bumps and hollows on top. Under the jaw, the neck,
  and ears with helix, fold, concha, tragus and lobe.
- **Face**: the expression of the natural-smile portrait. Long, heavy-lidded almond eyes, each with its own
  shape as in the drawing: a bold lash line that starts thin at the inner corner, covers the top quarter of
  the iris and runs on past the outer corner into a long flat wing; a shallow lower lid drawn as one thin
  line from a small hook at the inner corner; both corners left open, with white showing between the lash
  line and the lower lid. A round iris resting on the lower lid, dark mauve-brown under the lid, warm
  brown in the middle and a peach-gold crescent below, a pupil tucked under the lash line and one small
  highlight on the viewer's right. Above each eye a pink-brown tint and the fold of the lid. Thin, long,
  nearly straight brows high above the eyes, carried over the bangs where a lock crosses them, as the
  sheet draws them. A thin L-shaped line down the side of the nose and under its tip, a highlight above
  the tip, a calm closed smile and a faint blush. Skin, iris and hair colours are sampled from the sheet.
- **Hair**: every lock starts at the crown, lies on the head, and falls from there, so the top of the
  hair and the long hair are the same surfaces. Down the back the locks fall straight in three layers;
  at the sides they fan out over the shoulders and arms; in front of the ears thinner locks fall over
  the chest. Locks wave more toward their ends and finish in a curled tip. The bangs follow the
  portrait: one wide lock sweeps down to the bridge of the nose, the forehead shows in a tall triangle
  above her left eye, a lock sweeps over her left temple, and a smaller gap above her right eye is
  framed by the locks at the side. The wide lock and the temple lock are made of narrow, rounded strands
  laid along the strand lines drawn on the sheet, stacked on each other, with staggered pointed tips and
  a lighter streak down the main ones. Locks fall from the temples to below the jaw: beside the face on
  her right, so the whole jaw line shows, and over the outer cheek on her left, where thin strands
  flick inward.
  A three-strand braid crown around the back of the head.
- **Accessories**: a chrysanthemum on a black bow with cream ribbons and a gold tassel above her left
  ear; gold tassel earrings.
- **Outfit**: high ruffled collar and jabot, black ascot with a chrysanthemum brooch, black corset with
  gold buttons, piping, chain, ring and tassel.
- **Capelet** (`cape.py`): measured on outlines traced over the sheet's three views, with the collar,
  lining, trim, folds and emblem colour from its detail panel (披肩). A short bell over the shoulders, open in
  a wide V at the front: each front edge runs from the collar down across the bust (x 0.6 at z 9.7) and
  round to an outer corner at the side (1.44, 9.12), with a turned-back facing marked by a gold fold line
  along its upper part. It reaches 1.58 out at the sides and stands 1.25 out at the back; its hem is highest
  at the back corners (9.28) and lowest at the centre back (9.09), where a lace frill hangs to 8.84. Double
  gold lines follow the front edges and the hem, 18 soft folds fan out from the shoulders, a beige lining
  shows at the edges, and a standing collar, open at the front, rises round the neck. Four embroidered
  chrysanthemums sit where the sheet circles them: two on the front panels (x ±1.10, z 9.67), two at the
  back (x ±1.23, z 9.66). Its sections are ovals whose half-width, front depth and back depth are splines
  in the level, fitted within 0.024 of the measured tables.
- **Sleeves and arms** (`arms.py`): measured on arm outlines traced over the sheet's front, side and back
  views, where the hair hides much of them. A balloon puff from under the capelet to the elbow (0.86 across
  and as deep, widest at z 8.47, its inner side straight), gathered into a small frilled band at 8.3-8.15; a
  fitted forearm sleeve (0.46 across) that runs 25 degrees forward and out to the wrist, crossed six times by
  brown lacing on its outer front; a frill cuff sewn on square to the forearm whose hem hangs nearly level at
  6.74, its inner side falling straight and its outer side flaring to 0.83 across. Each sleeve is one
  surface of horizontal sections whose centre and half-widths are splines in the height, fitted within 0.024
  of the measured table. The views disagree on some heights (the side view draws the forearm about 0.25
  longer than the front view), so each landmark is the mean over the five drawn arms.
- **Dress**: measured layer by layer on the sheet's three views (with `sheetfit.py`) and its skirt panel
  (三片式长裙与褶边衬裙), then refined on outlines traced over the three views. A cream skirt in 40 knife pleats
  (9 degrees apart, as traced; cut up to 5% of the radius deep, built in ten sectors because Arch meshes a
  surface with at most 60 steps a side, with a thin line drawn in each pleat at the front and back) with a frilled petticoat and a row of orange ferns along the hem (each a stem, a top leaf, three leaf pairs, two
  three-leaved twigs and a diamond, measured with Python on a close-up photo of one) where it
  shows at the front and back. Seen from the front it is a straight A-line from the corset; seen from the
  side the front hangs nearly flat and the back is full (the depth is 0.53 to 0.60 of the half-width at the
  front, measured on the side view, and 0.75 to 1 at the back). Over it, on each side, a black panel from
  29-31 to 152 degrees from the front, with gold bands and chrysanthemum emblems where the sheet circles
  them (x ±2.14 at 2.75 on the front view, ±1.84 at 2.84 on the back); its front edge is faced in orange, a
  thin edge on her right and wide on her left, where the facing carries the front tab's orange strip on
  down to the hem. Over that a cream overskirt piece lined in orange, from 34 to 141.5 degrees, caught up
  by a black bow at each end (their knots 5.55 high at the front and 5.5 at the back), each bow with a
  chrysanthemum medallion above it, tails and two gold tassels. The overskirt's hem, a spline through
  points on its traced lower edge in the front and back views, hangs from the front bow down its front edge
  to 4.3, stays high (3.9 at 60 degrees) before dropping steeply to 2.95 at the side, where the lining shows
  at the turn, and rises up the back to 4.5. Black bands run down the overskirt's edges from the waist to
  the bows. An orange tab hangs on her left of the front, its inner edge running in from 0.31 to 0.14 and
  its outer edge widening from 0.5 to 0.87, folded diagonally at the foot so that a lighter flap hangs to a
  point at (0.88, 4.5); a black strip edged in gold hangs over it from the corset's tassel to 6.6. A longer
  tab hangs at the centre back, 0.34 wide at the waist and widening to 0.9, folded from 4.76 to 5.48, its
  flap ending in a point at 3.85 on her left. On the sheet's
  side view the outline is a true profile (it fits the model seen from the side within 0.1), but the bows
  and facings are drawn as if turned about 30 degrees, so the layers were placed from the front and back
  views; all three views draw the hems as if seen from above, so they curve up at the sides there.
- **Boots**: cream lace-up ankle boots built from a three-view drawing measured on the sheet's front, side
  and back views and its props panel (米白系带短靴). Each is 1.44 long and 0.585 wide across the ball of the
  foot, with a domed toe cap, an oval shaft 0.455 wide at the rim (1.10 high, under the petticoat frill) and a
  heel counter that bulges behind. A dark brown block heel 0.33 high, 0.40 x 0.34 at the top and 0.32 x
  0.22 on the ground, with its front face upright and its back leaning in. The sole runs flat under the
  forefoot and rises through the arch to the heel. Seven pairs of gold eyelets with criss-cross laces and a
  bow with gold-tipped tails; a stitched facing, a toe-cap seam and a back seam; a padded collar with a tan
  lining; pale stockings above.
- **Hands**: palm, two-jointed fingers and thumb, 0.36 across in both the front and side views, hanging 0.1
  out from the forearm's line (the wrist bends a little), the back of the hand toward the front and out and
  the fingers together and a little curled, fingertips at 5.94. The book hangs against the back of her
  left hand.
- **Props**: a leather book with gold corners, a chrysanthemum emblem and a ribbon bookmark; a display base.

## Rebuilding

```sh
python3 examples/claude-chan/generate.py         # writes claude-chan-3d.arch (needs numpy)
node src/cli.js examples/claude-chan/claude-chan-3d.arch -o examples/claude-chan/claude-chan-3d.txt --timeout 3600
```

### Checking against the sheet

`sheetfit.py` reads a part's outline off the sheet's front, side and back views (row by row, in model units)
and compares it with the same projection of the generator's surfaces, evaluated in Python. For example, the
boots against the side view, lined up at the toe tip:

```sh
python3 sheetfit.py check --sheet <sheet.jpg> --parts boots --ref 'boot.*R' --view side --box 985,385,1035,407 --align min
```

prints the mean and largest edge errors, the overlap and the edges at a few heights; `--overlay out.png`
draws the same comparison over the sheet. For layered parts, `sheetfit.py layers` sorts the sheet's pixels
into light, black and orange and compares them with the model drawn with hidden parts removed: how much
each colour overlaps, and with `--map 3` both maps printed as text. On the dress (front view, below the
bows) 84% of the pixels agree. The arms are mostly hidden by hair on the sheet, so they were measured on
outlines traced over it by hand; against those the model's edges are off by 0.03 to 0.05 on average in the
front and side views (puff 0.01 to 0.02, forearm under 0.04, cuff and hand 0.03 to 0.08), while the back
view draws the arms 0.1 to 0.2 closer to the body. The capelet was measured the same way: its outline is within 0.03 of the
front and back views' average (the two traced sides differ by up to 0.12), its front edge within 0.015,
its back edge on the side view within 0.014, and its hem and lace heights within 0.03. The dress was refined the same way: the
model's borders were drawn as a surface-id map in each view and every traced stroke matched to its nearest
border; the tabs, the overskirt's hem, the pleat count, the facing on her right and the emblems were
changed where the two disagreed. The traced hem is now a median 0.09 from the
model's on the front view (0.12 before) and 0.08 on the back (0.09). `CLAUDE.md` has the working rules: measure and model
each part before writing it for Arch.

`generate.py --prec 4` writes a quicker version; `--parts head,hair` limits it to some parts and
`--view face` frames the face (`--parts boots --view boots` shows just the boots, `--view arms` the sleeves). The views and parts are
listed in `generate.py`.

## How it is built

- Every part is a parametric surface `(X,Y,Z,u,u0,u1,v,v0,v1)`. Shared shapes are Arch user functions.
- **Face.** For each height `h` the face is a superellipse arc from one side of the head around the
  front to the other: half-width `fwd(h)`, depth `fd(h)`, exponent `fq(h)` (boxy across the eyes,
  rounder over the cheeks and jaw). The side profile `fpl(h)` and the front outline `fws(h)` are
  natural cubic splines through points read off the sheet, written for Arch as
  `a + b d + sum c_i max(0, h - k_i)^3`. `fr(p,h)` adds the nose, lips, chin, cheeks, brow and eye
  sockets as Gaussian bumps. The same surface is drawn as nine patches (each cheek, the nose and mouth,
  each eye, the bridge of the nose, the brow) that share their edges exactly, so every part gets its
  own fine mesh and no seam shows.
- **Eyes, brows, mouth** are thin decals that follow the face. Arch expands user functions as text, so
  a decal that called the full face function with a long argument would grow to thousands of terms;
  `generate.py` fits the face over the eye and brow regions with small polynomials (within 0.002) and
  the decals use those.
- **Matching the portrait.** The eye, brow, nose and mouth outlines were traced over the natural-smile
  portrait, placed on the model by its pupils and projected onto this face from the portrait's angle, so
  they land where the drawing puts them with the head's tilt and turn accounted for. Each eye keeps the
  points measured that way (`EYES` in `face.py`): its upper lid, the top of its lash line and its lower
  lid, relative to its own iris, written for Arch as one short polynomial each. The bangs were placed the
  same way, against the portrait's border between hair and skin.
- **Bangs as strands.** A strand is a blade from the crown over the dome to the hairline, then through a
  middle point to a pointed tip just in front of the face; its section is narrow and round, so the light
  falls off across it like hair. Strands that overlap are lifted off the dome by different amounts so they
  lie on each other instead of cutting through, and a thin dark layer under the wide lock shows between
  its strands like the drawing's strand lines. The lighter streak is a thin ribbon on the front of a strand
  that tapers away at both ends.
- **Brows over the hair.** Each brow is a decal on the skin; where a lock of the bangs crosses it, the
  generator samples the front of that lock along the brow and adds a short strip just in front of it,
  so the brow shows over the hair from the front and three-quarter views.
- **Hair.** A lock is a tube around a flow line on one guiding surface: a dome around the head for the
  first part of its length `t`, then an ellipse whose half-axes `hrx(t)`, `hry(t)` are fitted so the
  hair clears the shoulders, capelet, arms, book and skirt everywhere (the generator prints the
  margin). Each lock has its own angle, drift, wave, layer, width and length. Its tip turns through
  three quarters of a circle with the cross-section turning along, either sideways (a hook) or outward
  (a scroll), and is a separate surface so the curl gets a fine mesh.
- **Head turn.** The generator draws the head's parts inside a turned block: each coordinate is
  replaced by a fixed combination of all three (a rotation about the top of the neck). The long locks
  use the same rotation with an angle that eases from the head's to zero down the lock, hb(t), so they
  turn with the head where they lie on it and hang straight below. The head is modelled large for its
  detail and scaled down (0.82) about the same point, by the same block; the long locks ease from that
  scale to none as they leave the head. Earrings and tassels hang straight from their turned attachment
  points and are scaled with the head.
- **Skirt.** Every layer is written around the same skirt: at height h and angle t from the front it lies at
  half-width `sa(s)` (s from the waist to the hem) times `sin(t)` across and `dk(s,t) cos(t)` deep, so the
  layers stack at small offsets. The overskirt pieces, black panels and tabs are each a strip of that
  surface between two angles, with a top and hem that vary across the strip (`ozh`, `bzh`). The front depth
  `fk(s)` and the overskirt's hem `ozh(q)` (a natural cubic spline) come from points measured on the sheet;
  `stp` is a smooth step. The tabs are
  laid out by x across the back or front (`asin(x/a)` gives the angle), so their edges are straight lines as
  drawn.
- **Boots.** The boot is first described in Python as the drawing: at each height z its front and back come
  from the side view, its width from the front and back views and its plan from a footprint, rounded at both
  ends and domed over the toe. For Arch, each level of the upper is an oval traced by an angle p,
  `s = boc(u)-boa(u)*cos(p)` back from the toe and `x = sin(p)*sum bok_k(u)*cos(k*p)` across, at height
  `boz(u)`; these are cubic splines in one parameter u up the boot, fitted to the drawing within 0.015.
  Low down a level stops where the sole rises under the arch (`p` up to `bop(u)`), and just above the heel
  seat the cut runs round the heel cup and the levels close, so the whole upper is one seamless surface; the
  closing happens inside the sole, whose top is fitted to the upper's bottom edge. The laces follow a fit of
  the front of the upper, and the seams are traced on the fitted surface.
- **Braid.** Three strands follow an oval ring on the dome and weave as `sin(tau)` across the ring and
  `sin(2 tau)` in height, a third of a turn apart.
- Arch lights a surface at half strength where its normal faces away from the camera, so the generator
  evaluates every surface in Python and swaps its two parameter ranges when the normals point inward.
