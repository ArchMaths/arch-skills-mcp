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
- Do not build the full `.txt` save file without the user's approval; it is slow.
- `node src/cli.js --check` renders the file it checks: never run it on `claude-chan-3d.arch` itself (precision
  10), only on a `--prec 3` copy.
- Ask the user for a tracing or a close-up whenever a part's detail is unclear; it is quicker than guessing.

## How Arch draws

- A surface is meshed with floor(3.125 x precision) steps a side, 60 at most. Fine repeated detail (the skirt's
  40 pleats) needs the surface split into sectors of a few repeats each, plus drawn lines (curves keep their
  own resolution).
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
single place on it, so the emblems follow the front and back views. On the back view the hair falls in a
narrower band than the model's hair, leaving the capelet's sides and back emblems visible.

The dress, traced: the side view draws the overskirt's hem as if turned too (high at the front, low at the
back, where the front and back views put it low at the side), so the hem was fitted on the front and back
views; the frill's bottom rises from 1.13 at the centre to 2.1 at the sides in every view, the "seen from
above" effect (about 29 degrees here), so only its centre is compared. The overskirt's lower edge is one
wavy line that curls to show the orange lining; a second traced line at the same angle is a curl, not a
second hem.

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
