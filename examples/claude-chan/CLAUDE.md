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

- Never render at high precision: it takes hours. Use `--prec 3` for every Arch check.
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
- A surface facing the viewer is lit to about 1.2 times its colour. Colours above about 210 clip to flat white
  or flat orange and lose their shading, which makes cloth look like rubber: give folded cloth a colour 1/1.2
  of the sheet's (`RUST`, `SKIRT_CREAM` in `outfit.py`) and real folds to shade.

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

The capelet: unlike the arms, the side view draws it as a true profile (its front edge reaches the bust where
the front view's front edge, on a section hugging the chest, puts it), but it draws its upper front edge at
the side of the neck and cuts its lower front higher than the front view; its emblem there (y 0.8) fits no
single place on it, so the emblems follow the front and back views. On the back view the hair parts down the
middle and leaves the capelet's sides and back emblems visible; the model follows it (`back_strands`, the back
locks ending at the nape).

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
has: the capelet's emblems across (x +-1.10), the chin, brooch and emblems down (z 10.96, 10.5, 9.67, which
agree to 0.01). The portraits draw the chest about 9% wider than the full-body front view, so the scales
across and down differ; parts placed this way sit right against the emblems and the jabot. Hair drawn this
way only gives x and z: `chest_strands` in `hair.py` works out the depth so a strand hangs on what is under it.
The back view needs no landmarks (`back_strands`: it is calibrated in `sheetfit.py`). Read the colour runs of the
hair on the sheet against the model's hair (by height, in model units) before changing which locks cover what:
that is how the parting down her back was found.

The user's rules for hair: a curl at the end of a lock winds about an upright axis (its normal is the z-axis, even
where the sheet draws it as a loop facing the viewer) and drops as it turns; strands are never machine-smooth (they
wander, vary in width and twist gently about their own lines, about half a turn); the user calls a whole lock a
"strand" and its parts "sub-strands"; each back lock has the four traced strands and four more that fill its
gaps, in 18 sub-strands (`BK_SUBS`). To fill gaps, count them: the outline's extent at each height against the
model's hair in the back view, drawn with hidden parts removed), then place strands through them and check that none touches a
ringlet (nearest-point distance between the surfaces). Lay a strand on the strand beneath as that one really is
(its twisted section), rounded off, not with a running maximum: a running maximum carries every twist's edge all
the way down and piles the strands up.

Curls: write a helix as it is (cos and sin of its angle, `curl_xyz`), never fit one. A least-squares fit of
polynomials times cos and sin to a helix that starts steeply gave coefficients near 10^8 that cancel each other;
the error looked small in numpy and the surface came apart in the preview. In Arch, write what goes under a
fractional power so it cannot go negative by rounding (`(Th/2)*(1-t)`, not `Th-(Th/2)*(t+1)`): a negative base
gives NaN. `corset_ab` holds only on the corset (z 7.9 to 9.35); below it the flare term explodes.

All the modules run in one namespace (generate.py execs them), so a module-level name in one file replaces the
same name in another: `boots.py` once reused `_tz`, which broke `outfit.py`'s `torso_ab` for anything that
called it after the boots were built. Give module-level helpers names that are unique to their file.

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

## Keep Arch's expanded formulas small

Arch pastes every custom function's body in at each use, each argument as often as the body uses it, and
evaluates that expansion at every point of every object; it also saves it in the file (left out by `--lean`).
`max(a,b)` and `min(a,b)` become `0.5*((a)+(b)+-abs((a)-(b)))`, each argument twice, so nested ones double at
each level, and `a^b` becomes a function call. `generate.py` prints the total expansion
(`expansion.py`, within a few percent of Arch's); `--expansion 20` lists the worst objects. Cut it by:

- passing what a body uses several times as an argument computed once outside (the capelet's `kpx(u,p,w,s)` take
  the angle and the fold value; `kpp(u,q)` uses the front edge `kpe` once);
- writing fitted curves as polynomials in a variable scaled to [-1, 1], in Horner form, rather than splines of
  `max(0,u-k)^3` terms when a curve's parameter is pasted into them (`_xpoly` in `cape.py`, the sleeve's `SL_PUFF`);
- smooth forms instead of `min`/`max` (`tanh` fades and ramps; `((1+cos x)/2)^400` for a peak at whole numbers);
- passing a sector's own numbers as arguments instead of a lookup over all of them (the emblem's `emi(w,kz,la,lb,lc)`);
- exact rewrites (the hair's `hdw`: one summed spline instead of `min` of two).

A curve whose expansion grows too long fails to compile ("compile error: Unexpected token (2:1)") even though
every definition is short. Write scattered values as a formula (`sin(2.39*k+0.7)`), not a table. Many lines drawn
as one stroke (up a line and back down, then along an edge that hides or suits the run to the next) keep the
object count down.

In generated expressions never write a minus before a negative number (`a--2` is a decrement to Arch and fails
with "Assigning to rvalue"): parenthesise it.
