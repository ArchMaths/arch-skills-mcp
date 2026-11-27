# Claude Chan: working rules

## Measure and model before Arch

Every part is measured on the design sheet and modelled in Python before any Arch surface is written for it.
Work in numbers, not screenshots: each image read is expensive, a table of edges is not.

1. **Measure.** `python3 sheetfit.py measure --view front|side|back --box x0,y0,x1,y1` prints the part's
   left and right edges, width and centre at each height, in model units, from the sheet's full-body views.
   Look at the sheet itself only to choose the boxes and to see what an outline cannot show (seams,
   colours, the props panels' 3/4 views), once per part.
2. **Model.** Describe the shape in Python first: a design model built from those numbers (as `_hw` in
   `boots.py`: front and back edges from the side view, widths from the front and back views, a plan shape),
   then fit it to Arch expressions and print the fit error. Small parts can go straight into the
   generator's surfaces.
3. **Check.** `python3 sheetfit.py check --parts <part> --ref <regex> --view ... --box ...` cuts the
   generator's surfaces as Arch meshes them and compares them with the sheet row by row. Iterate until the
   mean edge error is about 0.02 or less in each view, and explain any outliers (the sheet's views do not
   always agree with each other). For layered parts use `sheetfit.py layers` (colour classes on the sheet
   against the model drawn with hidden parts removed; `--map 3` prints both as text, `--runs n` the class
   runs in model units): measure features the same way on both, change the model, measure again.
4. **Review.** Send the user the `--overlay` pictures and a short table of the main dimensions. Draw a
   full three-view drawing only when asked for one.
5. **Arch.** One low-precision check of the part: `generate.py --prec 3 --parts <part> --view <view>`,
   then `node src/cli.js --check <file> --png <png> --brief`.

## Limits

- Never render the whole figure at high precision: it takes hours. Use `--prec 3` for every Arch check. The
  exception is the face, whose decals need a fine mesh to judge: the head alone (`--parts
  head,hair,ornament,earrings,body,cape`) renders at `--prec 10` in under a minute. Frame it with a custom
  view such as `--view '@view 3d azimuth=-90 elevation=3 unit=320 center=0,0,11.55'`; closer than about
  unit 400 Arch's clipping box (it scales with the zoom) cuts into the face, unless the centre moves
  forward with it (`center=0,-0.5,11.6` at unit 640 shows one eye and cheek).
- Do not build the full `.txt` save file without the user's approval; it is slow. Build it with `--lean`.
- `node src/cli.js --check` renders the file it checks: never run it on `claude-chan-3d.arch` itself (precision
  10), only on a `--prec 3` copy.
- Ask the user for a tracing or a close-up whenever a part's detail is unclear; it is quicker than guessing.

## How Arch draws

- A surface is meshed with floor(3.125 x precision) steps a side, 60 at most. Fine repeated detail (the skirt's
  40 pleats) needs the surface split into sectors of a few repeats each, plus drawn lines (curves keep their
  own resolution).
- Arch computes every mesh when the file opens (or an object changes); rotating and zooming in 3D reuse them. The
  low-precision buffer (低精度缓冲) setting does nothing in Arch 2.0.0, whose background drawing skips it
  (measured: zoom and rotate redraws took the same time with it on or off). What Arch computes is the formulas'
  expansion (see below), so that is what to keep small.
- Arch lights a surface from the camera: its colour times 0.3 + cos(angle between the normal and the camera)
  (its 3D fragment shader in `index.html`), so 1.3 face-on, 1.0 at 45 degrees, 0.8 at 60. Curves are not lit.
  Channels above about 196 clip to flat white or flat orange where a surface faces the camera and lose their
  shading, which makes cloth look like rubber and skin look pale: give folded cloth a colour 1/1.2 of the
  sheet's (`RUST`, `SKIRT_CREAM` in `outfit.py`) and real folds to shade, and write the face's colours as the
  colour to be seen, through `unlit()` in `generate.py` (1/1.22, the face's average over the front and 3/4
  views).
- Arch draws a surface in one colour. A soft fade (the blush, the lips) is nested decals that step between
  two colours in small steps, each a level of a Gaussian; keep each step under about 4 per channel or the
  rings show when zoomed in.

## The sheet

The design sheet is not in the repository; give its path with `--sheet` or `$CLAUDE_CHAN_SHEET`. Its
full-body views are calibrated in `sheetfit.py`: 10.96 model units from chin to sole on the front view, the
soles on row 409, centre lines at x 820 (front), 1010 (side) and 1225 (back). Known quirks, measured:
the side view's outline is a true profile (azimuth 0), but its bows and facings are drawn as if turned about
30 degrees, so place layers from the front and back views; all three views draw hems as if seen from above
(they curve up at the sides), so compare above the frills; the back view draws the skirt about 5% narrower
than the front view (the model follows the front); hair is the same orange as the fabric to the colour
classes, so keep the boxes below it.

The arms: the side view draws them turned about 30 degrees too (read at azimuth -30 the puff is centred
beside the shoulder; at 0 it would hang behind her back); it draws the forearm about 0.25 longer than the
front view, so `arms.py` takes each landmark height as the mean over the five drawn arms; the back view
draws the arms 0.1 to 0.2 closer to the body; the lacing is drawn facing whichever view it is in.

The capelet is no longer the sheet's: it is the user's own design (see "The capelet (current)" below). On the
sheet's back view the hair parts down the middle; the curtain style followed it (`back_strands`, the back locks
ending at the nape).

The dress, traced: the side view draws the overskirt's hem as if turned too (high at the front, low at the
back, where the front and back views put it low at the side), so the hem was fitted on the front and back
views; the frill's bottom rises from 1.13 at the centre to 2.1 at the sides in every view, the "seen from
above" effect (about 29 degrees here), so only its centre is compared. The overskirt's lower edge is one
wavy line that curls to show the orange lining; a second traced line at the same angle is a curl, not a
second hem.

The dress is loose hanging cloth: a wind from the front presses it to the legs and trails it behind, and
nothing inside holds it out. The user's rules: its side is one straight line from the corset to the hem (never
concave down, never vertical or drawn in at the bottom: that reads as a stuffed balloon); layers lie on the
layer beneath and never flare on their own; the frill's folds are deep; the hem is level; no wind effect beyond
the sheet's side view. The line ends at the sheet's widest half-width (3.35) at the frill's foot, so the model
is deliberately slimmer than the sheet's outline above (the sheet's outline spreads to 3.3 by z 3.3, stays
there to 2.1 and rounds inward below, which no straight-sided dress with a level hem can follow).

## When the outline is hidden

Where hair covers a part (the arms), automatic edge tracking drifts onto the hair. Ask the user to trace the
part's outline in black over an overlay picture, then read the strokes as numbers: register the marked-up
picture onto the overlay (`cv2.findTransformECC`, affine), keep pixels that are near-black, neutral and much
darker than the overlay at the same place, map them back to sheet pixels and then to model units, and take
each row's outermost strokes as the edges. A marked-up screenshot of the sheet at another scale registers
with SIFT features and a RANSAC similarity (`cv2.estimateAffinePartial2D`); enclosed holes in the ink find
drawn circles. If the strokes cross dark parts of the sheet and break up, ask for (or use) a picture
of the strokes alone and register it onto the extracted ink (ECC on the blurred masks). To compare traced
borders with a whole layered model, draw the model as a surface-id map (z-buffered, hidden parts removed),
take its borders where the visible surface changes, and match each stroke pixel to its nearest border
(`cv2.distanceTransformWithLabels`): the distance per border pair shows which part is off.

Strokes traced over an expression portrait (the chest locks): find the portrait on the sheet by template
matching (the marked-up crop was panel 3 at 4.2x), and put it into model units by landmarks the model already
has: the first capelet's emblems across (x +-1.10), the chin, brooch and emblems down (z 10.96, 10.5, 9.67, which
agree to 0.01). The portraits draw the chest about 9% wider than the full-body front view, so the scales
across and down differ; parts placed this way sit right against the emblems and the jabot. Hair drawn this
way only gives x and z: `chest_strands` in `hair.py` works out the depth so a strand hangs on what is under it.
The back view needs no landmarks (`back_strands`: it is calibrated in `sheetfit.py`). Read the colour runs of the
hair on the sheet against the model's hair (by height, in model units) before changing which locks cover what:
that is how the parting down her back was found (since closed into one curtain, below).

The user's rules for hair: strands are never machine-smooth (they wander and vary in width a little); the user
calls a whole lock a "strand" and its parts "sub-strands". The user later found the curls unnatural and asked for the
hair to be modelled as anime 3D figures model it, and then found the back and side hair twisted and weird and asked
for it neat, as the drawing guides draw long back hair (studied too on a VRoid figure from behind), then smoother, in
fewer and bigger clumps. So the hair down the back became one curtain of broad clumps from under the braid that follows the back of
the head, spreads to its widest over the shoulder blades, wide enough to hang behind the arms and meet the locks over
them (with gaps, the arms showed and the back read as three panels), the clumps under the others darker so the folds
between them read. The user then asked
for waves (one S-wave through the whole mass), line art (without a clear outline the clumps read as tubes), painted
strokes and streaks, hooked tips and a bell-shaped curtain. With all of them the back no longer looked anime, while
the bangs did (the user: "the bangs look really anime like which is perfect, reference to how they are modeled"). So
the long hair (back curtain, side locks, outer back locks, locks at the nape) is made as the bangs are (`blade`): each
clump hangs in one smooth curve (`HAIR_WAVE` amp 0 and `W_AM` 0; the wave code stays and still picks which way a tip
flicks), tapers to a sharp point, and has the bangs' ridged crescent `lcs` across it (`lts` uses it too).
It has no line art (`OUTLINE_W` 0; the code stays) and no zigzag highlight band at the shoulders, which read as a white
saw. It is painted as `blade` paints the bangs (`paint_plan`, `paint_hair`): two or three thin dark strand lines
(`HAIR_LINE`) down its lower half, each widest in its middle and closing to the centre at both ends; a light peach
sheen lens (`HAIR_HI`) in its middle above them; and on the curtain the translucent shade under the braid, full width
at the top and closing to a point (`HAIR_SHADE`, two steps). Each painted shape is sampled on its clump as drawn and
written as short polynomials: written as the clump's own formulas with the place across put in, they made the scene
take over 15 minutes to build, and Arch's check gave up. Seven clumps fanned evenly from left to right, the same
but for their lengths, then read as 'piled straight from left to right'; the user pointed to the locks over the chest
as natural, and then gave a large picture of her back view to model the hairstyle on (not in the repository; read
at 207 px a unit, her centre line at x 1022 and the head's top, z 12.95, at row 172 of the 2048 px image, which put
her left hand exactly on the model's). So the long hair is laid out as that picture draws it, each strand made as the
chest locks are (`BK_FLOW` in `hair.py`, built by `hairflow.flow`: a sweep through control points in the back view, a
half-width by height, a wave and a flat spiral curl, all in the sheet of the hair, none twisting about its own line).
Below the braid it falls straight and narrows down her back and parts above the corset into two groups beside the
sash (`BLa` ... `BRe`, a darker clump `BLu`, `BRu` under each), which swing out a little and back and end in C-curls
about the knees, her left group a little lower; two short clumps end in points above the parting (`BLm`, `BRm`). The
rest, the side hair (`BK_SIDE`, keys `B?o*`), runs down her sides (the user, marking it on a view from behind her
right: not behind her): from the braid's ends at the sides of the head (|x| 0.62, where the braid rises to z 12.1)
along the outline seen from behind of the head, the side locks over the shoulder and the puffed sleeve (|x| 1.04 at
z 11.1, 1.42 at 10.2, 1.95 at 9.4), then down beside the arm in masses that fan out beyond the forearms, waving (one
wave a group: `_WL`, `_WR`, `_WO`) and curling at different heights. Laid on the back-most depth of what is under it
(`body_back`, `locks_back`), as it first was, a strand just inside an outline sits on the back of it, so it hung 1
behind the arms whatever its course: the side hair is laid instead at a depth by height (`BK_SIDE_Y`, about the arm's
back edge, and `BK_SIDE_DY` more for each strand), out on how far the figure reaches to that side there (`SIDE_OUT`,
`side_out`: the head's hair, the long locks and the clothes, sampled into a grid of (y, z)) and on the side strands
laid before it (stacked only where it lies on her, above `BK_DRAPE_Z`, the sleeve's widest; below, it fans out as
drawn, what pushed it out easing off), turned to face about 65 degrees out to the side (`BK_SIDE_FACE`; the user:
about 90 degrees from facing back, at whatever angle looks best from the side too: at 50 it still read as edge-on
lines from the side, at 90 as tentacles from behind), 1.4 times as thick (body when seen edge-on), its curls in its
own sheet and 25 degrees out of it (`BK_SIDE_CURL_TILT`: C-curls from the side). It takes no wrap. The picture is only a back view, and laid out in it alone every strand hung in one plane, facing back (the user, seeing it
from the side: not natural; plotted from above, the side hair ran straight out to |x| 3.3 at one depth, 1 behind the
arms). So in depth (`back_strands`): the hair wraps round her back and arms, coming forward by `BK_WRAP` 0.1 (x^2 -
x0^2) as it fans out and facing out from that curve (psi, the section's turn about the upright, an extra column through
`attach_root` into `_back_sections`); "falls straight" means straight on that curve (the accumulations and `_taut`
work on the depth less the wrap; filling where nothing is under it must too, or the wrap is undone); each strand waves
in depth (`BK_DWAVE`, a phase a group, `BK_DPHASE`, and 0.7 more a strand), floored on the strands beneath and the
clothes; and each curl turns out of the sheet by `BK_CURL_TILT` 45 degrees, back and away from her, about the upright
through a point 0.3 above it (`BK_CURL_AT`), so it is a loop from the side and still a C from the back (the strands
laid after it lie on the sheet, not over the curl). Plot the strands from the side and from above (a dump of each
strand's line) before rendering. The long side locks round her sides now end over the upper arm
(`SIDE_TE`, in a point), under the side hair: seen from behind they were edge-on, as tentacles. Design a strand in 2D
first: draw `hairflow.flow_ribbon` over the picture at the same scale and compare. The fit needs smooth widths: a
power under 1 at a tip (`tip`, `_sub_at`) or two control points almost together gave pieces that missed 0.006. Waves of a strand's own phase
(`BK_WAVE`, 0), twists about a strand's line (`BK_TWIST`, empty) and an in-and-out wave (`lrc`) read as twisted; more
wander, width or depth noise than `BK_NOISE` and `_sub_at` give read as lumps. The long locks at the sides are a few big
ones in three layers (six a side on her right, three behind her left ear and two below it, where there were ten and
three). The back strands (not the side hair) hang as taut sheets (`_taut`): from under the braid straight out over the fullest point they
pass, then down, never tracing the body (the user: anime's long hair need not stick to the back).
The four strands that wound into corkscrew ringlets are gone (`BK_CURL` is empty; `curl_xyz` stays for a strand
given a curl). (Before, the back was two traced locks that parted, bent in S-curves and twisted about their lines.)
To fill gaps, count them: the outline's extent at each height against the
model's hair in the back view (drawn with hidden parts removed), then place strands through them and check that
none touches another's tip (nearest-point distance between the surfaces). To find bare patches, count the points of the
dark under-layer (`hairUnder`) that a z-buffered view can see, from several angles round her back; a colour map of
the hair by kind (back locks, crown locks, side locks, under-layer, braid) shows where. Grow a grid's cells with
padding, never `np.roll`: it wraps round the edges, and once carried the skirt's depth (z 4.8) into the grid's top
row (z 11.6), which pushed every strand starting there 0.7 off the head. A strand starts from the first thing it
lies on, not a floor of 0.96 (right only by the middle of the braid).

The back view is the only reference for the hair's depth (the user's choice: the side view draws the long hair
trailing far back, which the back view's corset between the locks rules out). So the hair lies as close to her
clothes as it can: the back locks on the clothes' own surfaces (`clothes_y`: the generator run for the clothes
alone inside `hair.py`, sampled into a grid) and falling straight where they fall away (under the capelet's hem:
the user saw the hair bend in toward the waist when it followed the clothes in, which hair cannot do), the
strands over others only just above them, and on the head and nape side by side on the hair as they come out of the
braid (`BK_STACK`: the user traced their start hugging the back of the head); the long locks at her sides on a shell that clears the arms' own outline
(`HAIR_ARM_R`) without flaring. Check every hair change from the side and in cross-sections as well (its gap to the
nearest cloth by height): laying each strand on its full thickness over the ones beneath once piled the eight
strands of a lock up to 1.6 behind the back, and a floor left in the easing near the braid held them 0.6 off the
corset, neither of which a back view shows.

Curls: write a helix as it is (cos and sin of its angle, `curl_xyz`), never fit one. A least-squares fit of
polynomials times cos and sin to a helix that starts steeply gave coefficients near 10^8 that cancel each other;
the error looked small in numpy and the surface came apart in the preview. In Arch, write what goes under a
fractional power so it cannot go negative by rounding (`(Th/2)*(1-t)`, not `Th-(Th/2)*(t+1)`): a negative base
gives NaN. `corset_ab` holds only on the corset (z 7.9 to 9.35); below it the flare term explodes.

All the modules run in one namespace (generate.py execs them), so a module-level name in one file replaces the
same name in another: `boots.py` once reused `_tz`, which broke `outfit.py`'s `torso_ab` for anything that
called it after the boots were built. Give module-level helpers names that are unique to their file.

## The half-up style (current)

The user then asked for the hairstyle of a back-view anime reference (a band round the back of the head, the hair
under it gathered into a small tie at the nape, long wavy locks out of the tie) in her orange, keeping the bangs, the
face-framing locks, the chest locks and the braid (in place of the reference's band), and removing the shoulder locks
(they show from behind), the long side locks and the curtain. `HAIR_STYLE` in `hair.py` switches between it and the
curtain, which is still drawn when selected. The long hair is `halfup.py`: pure numpy, called by `draw_halfup`.

- Design it outside Arch: dump the scene's surfaces once (`sheetfit.surfaces_from_lines` on the generator's lines,
  sampled into grids), then preview the new strands over them with a painter's-algorithm render in numpy and cv2,
  lit as Arch lights (seconds a view). Then one Arch check at `--prec 3`, and the head alone at `--prec 10`.
- The crown's back locks (`backU`, `back`, `backT`) are `tie_lock`s: they close in to angle 0 at the tie and end there
  in a point, which makes the V under the braid. The swept-back tucks over her left temple are mirrored on her right.
- Depth as for the curtain: a taut sheet (`taut_depth`, the upper hull of the root and the floor) and a wrap. Lift a
  strand over the floor smoothly (`lift_over`); clamped to it, it took the grid's steps and eight polynomial pieces
  still missed by 0.018 (smoothly: two pieces, within 0.003). End the tips along a quarter sine, not a power under 1.
- The loose layer must lie between the dark sheet under the hair (`hairUnder`) and the V: they were 0.03 apart, and
  laid on the sheet it covered the V. In this style the sheet is 0.12 deeper. Its locks start under the braid's lower
  edge (`braid_back`) so that the braid hides their tops; started below it, they stood beside the V like organ pipes.
- Behind the ears: three locks a side, narrow (half-width 0.16 to 0.2), rounder (thickness 0.42 of the width) and
  facing out to her side, about 60 degrees from facing back. Wide and facing back, they showed from the front as flat
  panels beside the neck; started outside the hair, they stood up beside the head like wings.
- The locks out of the tie have no sheen lenses (side by side at one height they read as rays from the tie): their
  light is the reference's zigzag band below the tie (`HALFUP_BAND`, `poly_band`).
- `ornament_drape.py` runs the generator with `--no-compact`: `sheetfit` cannot read back the face decals that
  `compact.py` rewrites. Rerun it (`--write`) after changing the locks behind her left ear.

## The capelet (current)

The user replaced the sheet's capelet with their own design: a reference drawing, which they had modelled in Blender
first (`clients/capelet/capelet.py`, in metres; the drawing's views and proportions are there). `cape.py` is that
model at about 4.6 units a metre (the side tips' reach, 0.344 m, to the old capelet's 1.58), fitted to her body,
with the front opened round the ascot, jabot and brooch (the user's choice: closed as drawn, it covered them).

- Design it in numpy and preview it in the scene before writing Arch (as for the half-up hair: dump the scene's
  surfaces, swap the capelet's for the design's grid, render with the painter's preview). Check its clearance as
  numbers: the front edge against the extent of the ascot, jabot and brooch by height (`CK_EDGE_XZ` keeps 0.05 clear).
- Each piece is fitted as a polynomial in its own u and v (`ck_fit`), so every kink of the shape must lie on a piece
  boundary: the creases, the sides (where the hem's superellipse changes and the side tips are pointed) and the rows
  where the front edge turns. Inside a piece the shape must be smooth: a fillet of arcs and lines, a clipped ramp, a
  power under 1 or a fold whose depth jumps at a crease (that last one also left a crack between pieces) would not
  fit below 0.0015 at degree 20. The profile is a cubic Bezier, the folds are -cos across each sector, the ramps are
  polynomials or Gaussians. The mesh's columns follow the creases (`ck_phi`), so each crease runs along a piece edge.
- Rows of the same s are at very different heights round the cloth, so the front edge is solved by height for the edge
  column itself (`_ck_edge_solve`), not as an |x| per row (that put it on the side of the cloth half-way down). The hem's
  drop belongs to the column (its angle at the hem), not to each point's own angle, or the front rows tilt.
- The lining is the cloth's own polynomials drawn 1.2% toward the axis and 0.006 lower: exact, no second fit. (An offset
  along the normals needed three times the terms.)
- The gold lines are offsets in the cloth's tangent plane, mitred where two edges meet (`ck_inset_path`), written as
  polynomial pieces joined by conditions in one curve (`(t<1?A:(t<2?B:C))`). Keep each coordinate under about 2000
  characters: the hem is two curves.
- `hair.py` hangs the hair on it (`cape_samples`, `cape_env`, and the clothes grid): after changing the capelet, rerun
  `ornament_drape.py --write` (the locks behind her left ear lie on it; the new capelet moved the ribbons up to 0.18).

## Patterns from close-up photos

Embroidery and woven borders are measured on the user's close-ups with Python: threshold the thread's colour,
split the mask into blobs (opening, then distance-transform peaks), fit an ellipse to each (centre, length,
width, angle) and scale by a known size. Draw them as curves (they keep their own resolution): one curve can
trace several leaves by taking the leaf's index as floor(t) (`leaf_off`, `fern_curves`, `sprig_xy` in
`motifs.py`). Start and end every leaf where it joins its stem, so the jump from one leaf to the next runs
along a stem and leaves no stray stroke. Draw the motif in 2D over its photo first (`python3 motifs.py --photo
<close-up>`) and tune it there. Lay it on the cloth through the same mapping as the surface (`on_skirt`,
`on_tab`) so it follows the pleats and folds. Arch silently drops a curve whose expression grows too long
(about 6000 characters a coordinate failed, 2000 worked): put the motif in functions, split long strokes, and
keep the mapping from repeating them.

The black panels' emblem went through several designs; the one in use is measured on the user's tracing of a
high-resolution close-up (`emblem_radii`, `emblem_fills`, `emblem_curves` in `motifs.py`): keep it. A flat
motif on a curved panel must stand off by more than its sagitta (0.03 for the emblem), or its tips sink into
the cloth.

The user has close-ups at about ten times the sheet's resolution. Match one to the sheet with multi-scale
template matching (`cv2.matchTemplate` over a range of scales) to learn which view it is from and its units per
pixel before measuring it. The side view draws the orange facing twice as wide as the front view; the model
keeps the front view's width and takes proportions from the side view's close-up.

## Matching the hi-res portrait

The natural-smile portrait (`hi-res asset.png`, beside this file but not in the repository) is matched from its own
viewpoint: Arch's camera at azimuth -92, elevation 12 shows the head as drawn (the jaw outline then fits within 4.5
px), with a further -3.4 degree turn of the image. To compare a part, render it from that camera, register the
render onto the portrait with small marker balls on known points (pupils, nose, mouth corners, chin; a scale, turn
and shift fitted to them), and blend the two at 50%. To measure a part in the face's own coordinates (the
(s, z) of the eye tables), render a grid of marker balls on the face around it, fit a cubic map from face
coordinates to pixels (0.8 px rms), anchor each eye on its iris centre, and unwarp the portrait through it; read
points there on a gridded crop and check them with column profiles. The drawn eye white and skin are equally light
but differ in tint: blue minus green is about 0 on the white and -4 to -8 on the skin. Features drawn past the side
of the face (the wings, s 0.31-0.34) are cut where the face turns away (0.27); everything else follows the drawing.
To place a part (the ornament), map head-frame points to portrait pixels: an affine 3D-to-2D map fitted to the same
landmark markers, composed with the registration; solve for the head point that lands on each drawn position. The
portrait fixes where a part is seen, not its depth: `ornament_drape.py` slides the ribbons, tails and tassel along
the portrait's line of sight onto the hair behind them (a depth map of the scene along that view), so they lie on the
hair from every side and land where drawn. Rerun it (`--write`) after any change to the hair under the ornament; the
hair branch's closer side hair left them hanging 0.2 to 0.3 in front of it. Check a hair change from behind too:
fuller sides below the dome (a bulge of 0.2 on the side shell) opened the gap between the side locks and the back
locks and showed the under-layer there 2 to 5 times as much (count it as above). Arch lights only one side of a
surface, so the under-layer is two sheets 0.005 apart, lit from behind and from the front. To score shapes, sort both images' pixels into hair, skin, black and other and map where they disagree (the face region
alone for the bangs). Arch colours a surface with one colour; a colour can be an expression of variables
(`color=hsb(h,s,v)`, h, s and v on 0..100) but it is evaluated once per object, not per point (checked in 2.0.05
and 2.0.07), so shading is evened out by the face's shape (`fq`), not by colour.

## Making it read well

The user prefers judging the look by eye, on renders beside the portrait, over tracking tools. What made parts look
cheap, and the fixes:

- Repeats round a ring alias on Arch's mesh (31 steps a side at precision 10): petals came out as spiky starbursts, the
  neck ruffle as a metal crown, the first capelet's frill and the jabot as grey shards. Give each repeat about eight
  steps: sectors of a few petals or flutes, each starting and ending in a notch (`copper_flower` in `generate.py`; the
  ruffles in `outfit.py`; the sunburst in `cape.py`).
- A funnel's outside faces down and away from Arch's light, so a frill flaring out under the chin drew grey: frills
  stand up and flare only a little, and their orientation point is chosen so that the side the viewer sees is the
  lit one (the ruffle's is above it).
- Skin goes grey where the face turns from the light: `SKIN` is written for a light of 1.17, a touch warmer than the
  drawing, so that face-on it is the drawing's and in 3/4 views it stays warm.
- The portrait draws more hair in front than the sheet: locks from behind the jaw over each shoulder (`RS*`, `LS*` in
  `CH_STRANDS`, rooted further back by `CH_ROOTS`), a lock at her right temple (`templeR`), hair swept back over her left
  temple (`sweptL*`), and a band of shine across the crown (`crown_shine`, the blades' `crown`). The long hair seen
  edgewise beside the face drew as dark slabs; front hair facing the viewer covers it.
- The cream ribbons lie on the hair (`RIBBON_FACE`, facing out and forward), not facing the portrait's camera: facing
  the camera they stood out from the side of her head like fins. When a ribbon's facing changes, reset its depth to a
  smooth line before rerunning `ornament_drape.py`, which otherwise starts from the old, wavy depth.
- Eyes come alive from softness, depth and moisture, not from more detail: a smooth iris gradient (20 steps; 7 read as
  stripes) brightening to a warm crescent at the bottom, the upper lid's shadow laid over white and iris alike
  (translucent decals blend well in Arch; kept light, or the whites turn grey), lashes in clumps of uneven length rooted
  in a soft-edged lash line, and the lower lid raised for the smile (`SMILE_LID`) with a faint fullness under it. Steps
  of the iris below the raised lid would turn over and show under it: `cap` skips or clamps them. Then, as the anime
  eye guides paint them (the user asked for them to be followed): light gathered in the bottom of the iris (`glow`,
  translucent amber-orange crescents: gold read lemon), a faint ring of light part way out round the pupil, a slightly
  larger pupil, the highlights composed (a main catch-light in the same place in both eyes, a small one opposite, a soft
  low one, a few sparkles), the lashes gathered toward the outer corner and longest there, and the lower lid line over
  the outer two-thirds only (drawn whole, it boxed the eye in).
- Design the face by eye, as an artist would, not by tracing or measuring the portrait (the user's direction): traced
  shapes that matched the drawing's projection looked gaunt or odd in 3D. So the cheeks are fuller and rounder than
  traced (`fws`, `fq`, the cheek's round in `fr`), the lower face longer (`lowstretch`: she looked tipped forward with a
  short chin), the eyes' outlines are designed on the usual anime pattern (`EYES`), and the irises are turned toward
  the viewer inside the lids (`GAZE_DX`, `GAZE_DZ`): painted eyes look along the face, which passed under the portrait's
  camera 12 degrees above.
- Shading is painted on, as anime 3D figures paint theirs (studied on a CC0 VRoid sample: its skin's shade colour is a
  light rose, its hair's a hue-shifted tone, and the hair's shine is painted, unlit). Translucent layers (`opacity`) blend
  cleanly in Arch and tint whatever lies under them, so shades are two or three translucent layers in rose or mauve on
  the face (`shadeFarR`, `flush*`, `shadeNose`, ...), never grey; on the hair a deeper orange (`HAIR_SHADE`) where the
  bangs pass under the crown, a band of peach shine across the crown (`HAIR_HI`), and one soft strand line a clump.
  The hair's three tones are kept close (`HAIR`, `HAIR_LT`, `HAIR_DK`): far apart, the locks read as stripes, not clumps.
- Hair is modelled as anime 3D figures model it: clumps about three times as wide as they are thick (four for the broad
  long clumps: three read as sausages), so Arch's light shades each one round (flat ribbons ten times as wide as thick
  read as one flat panel, and as dark slivers edge-on).
  The long side locks are a few broad smooth ones, in three layers (`side` in `run_hair`), and they and the outermost back locks
  grow to their full volume below the head (`lock(..., full=True)`: 0.55 of it on the head, so the crown keeps its
  size; their crown shine is placed for that). The bangs are 1.4 times as wide and thick as the traced strands (the
  wisps stay fine); the shoulder locks a third wider.
- Highlights follow the anime drawing guides: they run across the hair, not along it, as bands whose edges are zigzags
  of spikes pointing along the strands, in two crisp steps (a light peach `HAIR_HI` and a bright core `HAIR_GLOSS`).
  One ring crosses the crown on every clump at the same height (`RING_POLAR`; `crown_shine`, the blades' `crown`), and
  a second crosses the long hair where it curves over the shoulders (`BAND_Z`; `lock(band=True)`, `poly_band` on the
  back and chest strands), staggered a little from clump to clump: even, it read as a thread across the hair.
  `lock_parts` writes a lock for any expression in place of t; `poly_band` lays a short band on a polynomial strand's
  tangent plane (small expressions). Streaks down the strands were tried first and read wrong.
- Keep a local face fit (`qfit`) on the face: past its side the surface turns away too steeply to fit (the eye fit's
  error rose from 0.001 to 0.11 when its region reached s 0.32). Decals that run onto the side of the face, such as
  the lash wings, use the face itself (`q='fy'`).
- The ear is turned forward, as anime draws it, so the portrait's light reaches it; the earrings hang from the lobe's
  outer edge, clear of the cheek's hair.

## Keep Arch's expanded formulas small

Arch pastes every custom function's body in at each use, each argument as often as the body uses it, and
evaluates that expansion at every point of every object; it also saves it in the file (left out by `--lean`).
`max(a,b)` and `min(a,b)` become `0.5*((a)+(b)+-abs((a)-(b)))`, each argument twice, so nested ones double at
each level, and `a^b` becomes a function call. `generate.py` prints the total expansion
(`expansion.py`, within a few percent of Arch's); `--expansion 20` lists the worst objects. Cut it by:

- passing what a body uses several times as an argument computed once outside (the first capelet's section
  functions took the angle and the fold value as arguments, so its front edge was worked out once a point);
- writing fitted curves as polynomials in a variable scaled to [-1, 1], in Horner form, rather than splines of
  `max(0,u-k)^3` terms when a curve's parameter is pasted into them (the sleeve's `SL_PUFF`; the capelet is all
  polynomials, `ck_fit` in `cape.py`);
- smooth forms instead of `min`/`max` (`tanh` fades and ramps; `((1+cos x)/2)^400` for a peak at whole numbers);
- passing a sector's own numbers as arguments instead of a lookup over all of them (the emblem's `emi(w,kz,la,lb,lc)`);
- giving the layers inside a small part their own small fit of the face and short local versions of the curves
  that clip them, and dropping clipping that cannot happen there (the iris: `qi`, `eui`, `eli`; the eyes went from
  2.9 MB to 1.0 MB this way);
- fitting a decal's depth as a polynomial in its own parameters when its x and z are long expressions, instead
  of pasting them into a face fit (the bangs' shadows in `draw_hair_shadows`: 42 KB each through a face fit,
  6 KB this way);
- exact rewrites (the hair's `hdw`: one summed spline instead of `min` of two).

A curve whose expansion grows too long fails to compile ("compile error: Unexpected token (2:1)") even though
every definition is short. Write scattered values as a formula (`sin(2.39*k+0.7)`), not a table. Many lines drawn
as one stroke (up a line and back down, then along an edge that hides or suits the run to the next) keep the
object count down.

In generated expressions never write a minus before a negative number (`a--2` is a decrement to Arch and fails
with "Assigning to rvalue"): parenthesise it.

## Keep the file quick to open

Opening the save file is Arch meshing every surface (`getParametricSurfaceResolution`, `buildParametricSurfaceMesh`
in `index.html`): at precision 10 a 32 x 32 grid, each point evaluated five times (four more for its normal) and
each cell once more, about 6,100 evaluations a surface, in its drawing worker. Surfaces are nearly all of it (the
figure took 37.6 s to mesh, 4.8 s without its surfaces). Measure with `opentime.mjs` (the whole file, then each
`--drop` family left out: what it costs), not by formula size alone: `^` is a function call in Arch, so a formula
full of powers costs several times more per KB, and a very long one costs more again in the worker (the 22 ring
highlights, 4 MB, cost 5.7 s; the 407 back strands, polynomials, 1.6 s).

`compact.py` rewrites the heaviest surfaces, after everything is drawn (`COMPACT` in `generate.py`, and `hair.py`'s
ring highlights), as polynomials in their own parameters: the same grid and side lit, within a tolerance, and only
where Arch then evaluates at most 0.6 of what it did. Set each family's tolerance from what it lies on: a quarter
of the gap (the layered lip decals 0.0004 apart: 0.0001; the lashes, nose and smile decals at least 0.0014 above
the skin: 0.0003; the shades 0.0056 above it: 0.001; the locks on the head: 0.001, so seams between pieces stay
under 0.002). A surface whose formula creases or ends in a power under 1 does not fit whole: give it `cuts` where
it creases, or `split` it, or a form that keeps the crease exact (`sub`: a highlight band whose zigzag edges stay
in the formula, the lock fitted in w and the band's own t). Surfaces with holes, or running into the face's
vertical edge (the big skin patches, two lash wings), stay as written. Compare a precision-10 head render before
and after (`--no-compact`): pixel differences, not a look. It took the open from 39.9 s to 27.1 s and the
expanded formulas from 23.0 MB to 14.9 MB; `test_compact.py` covers it.

Rewrite exactly where the algebra allows, first: a placement `fw(p, q, 0, o, i, j, k)` pastes p and q into each
coordinate, so `o + r*(i*cos w + j*sin w)` halves a petal's formula (the side emblems).
