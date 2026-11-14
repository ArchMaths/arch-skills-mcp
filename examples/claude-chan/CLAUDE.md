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
