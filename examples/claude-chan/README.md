# Claude Chan, a 3D Arch figure

Claude Chan (Claude娘: orange hair and a chrysanthemum motif) as a static 3D figure, built from her
design sheet with nothing but parametric surfaces and a few curves.

| File | |
|---|---|
| `claude-chan-3d.txt` | The Arch save file. Open it with Arch's *Open file*. |
| `claude-chan-3d.arch` | The scene script (1925 objects); the save file is built from it. |
| `generate.py` | Writes the scene script. `face.py`, `outfit.py`, `motifs.py`, `cape.py`, `arms.py`, `hair.py`, `boots.py` and `scene.py` are its parts. |
| [haircontact.py](<haircontact.py>), [hairjoin.py](<hairjoin.py>) | Query the existing hair surfaces and fit tangent-continuous posterior roots while preserving the original curl entrances. |
| `hairflow.py` | Builds a long-hair strand in the back view from its sweep, widths, wave and curl (the back hair's `BK_FLOW`), and draws it in 2D for comparing with a picture. |
| `halfup.py` | The half-up hairstyle's long hair (the current style, `HAIR_STYLE` in `hair.py`): the locks out of the tie, the loose layer from under the braid and behind the ears, and wisps, hung as taut sheets over the capelet and fitted as polynomial strands. Pure numpy, so it can be designed outside Arch. |
| `sheetfit.py` | Measures the design sheet and checks the model against it. |
| `ornament_drape.py` | Lays the hair ornament's ribbons, tails and tassel on the hair, keeping where the hi-res portrait's view sees them; rewrites their positions in `generate.py`. |
| `expansion.py` | Estimates how large Arch's expanded formulas are (what it evaluates at every point); `generate.py` prints the total. |
| `compact.py` | Rewrites the surfaces that cost most to open (the hair's ring highlights, the face's decals, the short locks on the head) as short polynomials in their own parameters, within a tolerance of how they are drawn; `generate.py` runs it last (`--no-compact` leaves them as written). |
| `opentime.mjs` | Times opening a save file in the bundled Arch, and what each family of surfaces costs (`--drop`). |

Opening the file takes a while: about 27 s in the bundled Arch on an Apple M4 (`opentime.mjs`), 40 s before
`compact.py` rewrote its costliest surfaces. It is about 7.5 MB: it is built with `--lean`, and Arch rebuilds the
expanded formulas left out of it as it opens it. It is saved with `implicitPrecision=10`, so
each surface is meshed on a 32 x 32 grid instead of Arch's default 13 x 13 (and each point is evaluated five times,
for the lighting), and the turned head's surfaces are about three times as long to evaluate. That computing is the wait: rotating and zooming reuse the
meshes and redraw in about a second. The *low-precision buffer* (低精度缓冲) setting makes no difference in Arch
2.0.0, whose background drawing (异步绘图) skips it (measured). Lowering the *equation, inequality and
parametric surface precision* slider recomputes the scene faster and coarser.

## What is modelled

- **Proportions**: measured on the sheet's full-body front view (正面). Her head, crown to chin, is 1/7.9
  of her height; the elbows are at the waist, the cuffs at the hips and the fingertips at mid-thigh, with
  the forearms and hands angled forward and out, clear of the skirt. The skirt widens from the corset in one straight line to the hem, as
  wide as the sheet's at every height, and ends high enough to show the lace-up boots.
- **Head**: sculpted from parts, tilted toward her right shoulder and turned a little to her right, which
  gives her a lively pose. The tilt is 7 degrees at the eyes, between the sheet's calm portraits (7 to 9
  degrees) and its full-body front view (about 3). The face is a
  stack of horizontal arcs whose width, depth and roundness change with height, so it has a real profile:
  forehead, the dip between the eyes, a small pointed nose, lips, a receding chin, and a jawline that
  runs up to the ear. Seen from the front the jaw runs nearly straight from the cheekbones to a blunt,
  rounded chin: the portrait's V-shaped jaw, a little softer so it reads well in 3D. Across the cheeks the face is flatter than a real
  one and turns quickly at the outline, as anime faces are drawn, so the light stays even over them. The round of each
  cheek, the brow and the eye sockets are shallow bumps and hollows on top. Under the jaw, the neck,
  and ears with helix, fold, concha, tragus and lobe.
- **Face**: the expression of the natural-smile portrait. Long, heavy-lidded almond eyes, each with its own
  shape as in the drawing, re-traced on the hi-res portrait as she is seen from its viewpoint: the upper lid
  covers the top third of the iris and sweeps down at both ends to meet the lower lid in a point, so both
  corners are closed; the lower lid stays low and flat under the iris and rises steeply into the corners. A
  heavy lash line from corner to corner with a feathered top edge, two thin strokes running on past the inner
  corner of her left eye, and a wing of two strokes past each outer corner (level on her left, sloping down on
  her right; shorter than drawn, where the face turns away). The iris in the portrait's colours, stepping
  from dark mauve-brown at the top through red-brown to amber in the lower third, with a thin dark shade under
  the lash line, lavender-grey sheen streaks above and fine gold streaks below; as the anime eye guides paint
  it, warm amber-orange light gathers in its bottom and a faint ring of light circles the pupil part way out.
  A soft upright pupil with a lighter ring. The highlights are composed as the guides compose them: a crisp
  rounded catch-light at the top of the pupil, a small one opposite, a soft one low in the iris, a few
  sparkles in the light there, and two short white glints low on the rim. A light grey-mauve shade from the
  lid on the white. Lashes curve up and out of the lash line, gathered toward the outer corner and longest
  there; the thin reddish lower lid runs along the outer two-thirds only, with three faint lashes. Above each
  eye a pink-brown tint and the fold of the lid. Long, nearly straight brows in a soft orange-brown, sloping
  gently down toward the outside and fuller at the inner end, carried over the bangs where a lock crosses
  them, as the sheet draws them. A thin L-shaped line down the side
  of the nose and under its tip, a highlight above the tip, a calm closed smile in warm red-brown over a
  soft pink lower lip with a gloss highlight, and a wide blush under each eye that fades out toward the
  nose. These follow the sheet's high-resolution natural-smile portrait: its pupils, irises, lash line and
  brows were measured by unwarping it onto this face in its own pose (fitted to the pupils, nose, mouth and
  chin), and its skin, blush, lips and shades were sampled from it. The skin renders as the portrait's
  (#f5e3dd on average), with its shade on the neck; the face's colours are written as the colours to be
  seen through Arch's lighting (`unlit()` in
  `generate.py`), so the skin keeps its shading instead of turning white where it faces the camera.
  Arch casts no shadows, so the portrait's are painted on: each lock of the bangs lying on the forehead
  has a soft darker strip under it (its outline seen from the front, widened and moved down a little, in
  two steps), which shows as a rim along its edges and a point under its tip; and a band on the neck,
  whose lower edge follows the turned jaw, shades it under the chin in four steps.
- **Hairstyle: half-up** (`HAIR_STYLE = 'halfup'` in `hair.py`, after the user's back-view reference). The bangs,
  the locks framing the face, the two locks over the chest and the braid crown are as described below. The locks
  from the crown pass under the braid and close in to a small gold tie at the nape (z 11.45), so that below the
  braid they form a gathered V; the hair over her right temple is swept back as on her left. Out of the tie, eleven
  long wavy locks fan out over the capelet to about her hips, each ending in a point that flicks outward, with a
  zigzag band of shine across them just below the tie. Under them a looser, darker layer falls from under the braid
  and from behind both ears (three rounder locks a side, facing out to her side, that the ornament's ribbons lie
  on), and six thin wisps peel off the top locks. All of it hangs as a taut sheet over the capelet and then
  straight down, wrapping round her a little, and is painted as the bangs are (strand lines, and the shade under
  the tie). The shoulder locks, the long side locks and the curtain down her back described below belong to the
  earlier style, `HAIR_STYLE = 'curtain'`, which the generator still draws when it is selected.
- **Hair**: matched to the hi-res portrait from its viewpoint: the crown as low and the dome as wide as drawn;
  thin dark strand lines and light streaks on the strands around the
  face; the dark sheet under the hair turned so that it is lit where it shows between the locks; her left
  ear shows, with locks falling behind it. Every lock starts at the crown, lies on the head, and falls from
  there, so the top of the hair and the long hair are the same surfaces. The hair down the middle of her back
  is laid out as the user's back-view picture of her draws it, each strand made as the locks over the chest are
  (a sweep, a width, a wave and a flat curl): below the braid it falls straight and narrows down her back, parts
  above the corset into two groups beside the sash that wave and end in C-curls about the knees, and the rest
  runs down her sides: from the braid's ends over the sides of the head, the shoulders and the puffed sleeves, then
  down beside the arms into masses that fan out beyond the forearms, waving and curling at different heights. Short strands from under the braid
  cover the back of the head beside them; a few broad locks at her sides come over the shoulders and end over the
  upper arms, under the side hair. All the long hair is made as the bangs are: each clump hangs in one smooth curve,
  tapers to a sharp point or curls, and has the bangs' ridged section. The hair down the middle of her back hangs
  as a smooth sheet, straight out from under the braid over the shoulder blades and down, not tracing the body; the
  side hair lies on the sides of the head, shoulders and sleeves and hangs along the back edge of the arms. In
  depth the long hair down her back wraps round her, coming forward as it fans out, so that from above it curves round her instead of hanging in one flat
  plane; each strand waves gently in depth as well as across, and each curl turns back out of the sheet, so that
  from the side the hair has volume and its curls show as loops.
  It is painted as the bangs are, without outlines: thin dark strand lines down the lower part of each clump,
  closing to its centre at both ends, a light peach sheen in its middle, and a soft shade under the braid. Two locks fall from under the jaw over the capelet and the bust to curled tips at the top of
  the corset, beside the jabot, as traced on the sheet's portrait: on her right
  a band that fans into four strands, the inner one hooking in toward the jabot; on her left a narrower
  band that fans into three that curl inward. Outside them, as the hi-res portrait draws her, three more locks a side
  come from behind the jaw, fall forward over the collarbone and the shoulder and down the capelet, framing the neck. The hair is built as anime 3D figures build it, from round clumps about three times as wide as they are thick, with painted shading as anime hair is drawn: rings of zigzag-edged highlights across the hair, one on the crown and one where the long hair curves over the shoulders, and a deeper orange where the bangs pass under the crown. The bangs follow the
  portrait: one wide lock sweeps down to the bridge of the nose, the forehead shows in a tall triangle
  above her left eye, a lock sweeps over her left temple, and a smaller gap above her right eye is
  framed by the locks at the side. The wide lock and the temple lock are made of narrow, rounded strands
  laid along the strand lines drawn on the sheet, stacked on each other, with staggered pointed tips and
  a lighter streak down the main ones. Locks fall from the temples to below the jaw: beside the face on
  her right, so the whole jaw line shows, and over the outer cheek on her left, where thin strands
  flick inward; a wide lock from her right temple outside them, and on her left the hair over the temple swept
  back above the ear under the ornament.
  A three-strand braid crown around the back of the head.
- **Accessories**: as the hi-res portrait draws them from its viewpoint, a copper chrysanthemum at the side of her
  head level with her left temple, turned toward the viewer (two staggered rings of broad, round-tipped petals
  outlined in dark copper, a small inner ring and a domed button), with a black satin bow under it (two ribbon loops
  pinched at the knot, with a fold down each), two wide black tails edged in gold hanging to her jaw, two wide cream
  ribbons lying on the hair behind them and a tassel, all lying on the hair; the tassel and the earrings are strings
  of gold beads ending in a gold cap and cream silk threads. Her left ear shows, turned forward as drawn: the hair on
  that side falls behind it.
- **Outfit**: a blouse shaped over the bust with fine tucks, high collar with two gathered tiers of soft lace standing up to just under her chin, a jabot of four ruffled tiers showing beside the ascot,
  a black ascot of two fan-pleated tails with gold lines under a large chrysanthemum brooch (rebuilt from a
  close-up of the chest), drawn pleat lines down the blouse
  front and fold lines on the sleeves' puffs (soft grey-beige), black corset with
  gold buttons, piping, chain, ring and tassel.
- **Capelet** (`cape.py`): the user's Blender capelet (made after their reference drawing of it), fitted to
  her at about 4.6 units a metre, its front opened round the ascot and jabot (closed as drawn, it hid them).
  A short stand collar, open at the front in a V and edged in gold; shoulders that slope gently to a corner
  at |x| 0.81 and flare out in a bell to pointed tips over the arms (1.58 out at z 9.65); the back hanging
  longest (to z 9.06), so that from the front its beige lining shows under the side flares. Godet folds deepen
  toward the hem, the front ones running diagonally from beside the collar to a step in the hem; drawn fold
  lines run down the creases. Each front edge passes beside the brooch and round the jabot (0.05 clear of
  them), then cuts out to a point (|x| 0.93 at z 9.63, the drawing's notch) and back to a pointed tab at
  z 9.22. Double gold lines follow the front edges, the hem and the collar, mitred at the corners, and an
  orange sunburst of twenty pointed petals with a gold button sits on her left front panel. The shape is
  designed in numpy (`ck_surface`) and each of its 15 pieces (between the creases and sides; the front
  panels cut where the edge turns) is written as a polynomial in its own parameters, within 0.0015.
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
  front, measured on the side view, and 0.75 at the waist to 0.96 at knee height and 0.93 at the hem at the
  back). The dress is loose cloth hanging in a wind from the front, with nothing inside to hold it out: its
  side is one straight line from the corset to the hem (0.41 out per unit down; no bulge, no drawing in),
  ending at the sheet's widest half-width, 3.35, at the frill's foot. That makes it slimmer than the sheet's
  outline above (2.06 at z 4.7 where the sheet has 2.61): the user's choice between this and a hem a third
  wider than drawn. Every layer lies on the one beneath (panels 0.09 off the skirt with no flare, overskirt
  0.13 off with folds 0.07 deep). The frill hangs in two loose layers with deep folds (36 and 48 round, up to
  0.16 deep), built in twelve sectors each. The hem is level, where the sheet draws it as if seen from
  above. Over it, on each side, a black panel that really folds (measured on a close-up: the fold runs from
  under the front bow to the hem, 0.41 in front of the emblem; the cloth doubles back there, so the emblem's part
  laps over a narrow front strip whose hem ends at 2.07; shallow folds cross the rest), from
  29-31 to 152 degrees from the front, with a border (measured on close-ups: at the hem a pale gold line 0.05 wide on the edge, black for
  0.07, a second line and a faint thin line 0.25 up; up the front and back edges the edge line and the thin
  line, with a small sprig in each lower corner) and flower emblems (pointed petals in 16 places round a ringed
  centre, 0.23 long and the upper diagonal ones 0.27, a bud above reaching 0.35, a spray of three leaves below,
  a forked twig and a dot to each side: measured on the user's tracing of a close-up) where the sheet circles
  them (x ±2.14 at 2.75 on the front view, ±1.84 at 2.84 on the back); its front edge is faced in orange, a
  thin edge on her right and wide on her left, where the facing carries the front tab's orange strip on
  down to a diagonal foot (1.90 at its front edge, 0.16 higher where it meets the panel), bordered with a cream
  band on its front edge and foot and a thin line inside them, and embroidered with a leafy cross and a sprig. Over that a cream overskirt piece lined in orange, from 34 to 141.5 degrees, caught up
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
node src/cli.js examples/claude-chan/claude-chan-3d.arch -o examples/claude-chan/claude-chan-3d.txt --lean --timeout 3600
```

### Posterior hair-root regression tests

The lower back locks start on the actual upper-hair surface, with its tangent plane,
while keeping their lateral attachment positions (`BK_FAN`). Their sections open beneath
the braid and clear the actual under-layer rather than a fixed depth floor. A strand
given a ringlet keeps its original entrance when the hanging surfaces are refitted.

```sh
python3 -B -W error::RuntimeWarning -m unittest discover -s examples/claude-chan -p 'test_hair*.py'
python3 -B -W error::RuntimeWarning -m unittest discover -s examples/claude-chan -p 'test_compact.py'
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
view draws the arms 0.1 to 0.2 closer to the body. (The first capelet was measured the same way; it has since been
replaced by the user's own design.) The dress was refined the same way: the
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
  margin). Each lock has its own angle, drift, wave, layer, width and length. Its tip tapers to a point and flicks gently aside and out, and is a separate surface so the point gets a fine
  mesh.
- **The chest locks** (`chest_strands` in `hair.py`) are drawn from the user's tracing over the third
  expression portrait, registered on the sheet and put into model units by the first capelet's emblems
  (across: x ±1.10) and the chin, brooch and those emblems (down). Each strand's centre line and half-width follow the tracing in
  the front view; its depth is worked out so it hangs: it lies on the capelet, blouse, corset and collar
  (and on the strands laid before it) and, below the fullest point, falls straight. Above the tracing it
  rises from under the jaw into the head. Centre line, side and thickness vectors are one polynomial each
  in `t` (degree 20, within 0.005), and the lower half is a second surface so the curled tip gets a fine
  mesh.
- **The back locks** (`back_strands` in `hair.py`) are one curtain laid out as anime figures draw long back
  hair (`BK_FAN`, `_fan_strand`): each strand's centre line in the back view leaves the braid falling
  straight, eases out to its place across the curtain by z 9 (`BK_FAN_Z`, over the shoulder blades) and then
  falls nearly straight, closing in by 4% a unit of drop; over its last 0.55 it tapers to a point and turns in
  toward the middle. Neighbours never cross, and the tips are staggered into a soft V. There are seven broad
  clumps, about four times as wide as thick, with a smooth section (`lsm`). (It replaced the
  user's tracing of the sheet's back view, two locks that parted, bent in S-curves, twisted and wound into
  ringlets, which read as twisted and weird.) The strands hang on the hair above the nape and on her clothes,
  as close as they can lie: the generator runs itself for the clothes alone (about 3 s) and samples their
  surfaces into a grid of how far back the cloth reaches, gaps bridged across; a strand lies just off it,
  falls straight where the cloth falls away (under the capelet's hem), and lies only just above the strands
  beneath (on the head and nape they lie side by side on the hair as they come out of the braid). So that no
  strand is machine-smooth, each wanders across by up to 0.005 and varies in width, thickness and depth by
  a few percent (sums of three waves of incommensurate lengths, their phases a formula in the strand's
  number); more read as waves and lumps. Each strand is one clump (`BK_SUBS`), ending in a single point. Under the
  braid, four short wide strands (`_under_braid`)
  start right under the braid's lower edge at their own x (worked out from its ring, `braid_back`) and lie on
  the hair already drawn on the back of the head (the generator's own lines so far, sampled into a grid like
  the clothes, `head_back_grid`), the middle of their width resting on it so that their edges follow the head
  round; the dark under-layer that showed there (where the crown locks close in toward the middle) is now
  covered from behind. The section is turned square to the line smoothed over about 0.1, so that it does not
  swing at every small kink. A long strand is fitted in as many polynomial pieces as keep it within 0.006,
  joined exactly. A strand can still end in a ringlet (`BK_CURL`, `curl_xyz`): a helix about an upright axis
  written as it is, in its angle (fitting a fast-starting helix with polynomials times sines gave coefficients
  near 10^8 that cancel each other, and the surface came apart); none does now.
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
