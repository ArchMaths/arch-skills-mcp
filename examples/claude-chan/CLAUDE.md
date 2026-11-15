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

## When the outline is hidden

Where hair covers a part (the arms), automatic edge tracking drifts onto the hair. Ask the user to trace the
part's outline in black over an overlay picture, then read the strokes as numbers: register the marked-up
picture onto the overlay (`cv2.findTransformECC`, affine), keep pixels that are near-black, neutral and much
darker than the overlay at the same place, map them back to sheet pixels and then to model units, and take
each row's outermost strokes as the edges. A marked-up screenshot of the sheet at another scale registers
with SIFT features and a RANSAC similarity (`cv2.estimateAffinePartial2D`); enclosed holes in the ink find
drawn circles.
