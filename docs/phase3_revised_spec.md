# Phase 3 Revised — Vision Phase (OPTIONAL STRETCH)

Do not start this until the Phase 6 app is deployed and reachable
at a public URL.

## Checkpoint — Sunday Oct 18, evening
Proceed ONLY if all are true:
- App deployed, public URL loads, all three tabs work
- Video outline drafted
- At least 6 hours available Monday

If any is false: stop here. Finish the app, record the video,
submit. The vision phase does not happen and that is a correct
outcome, not a failure.

## HARD CUTOFF — Tuesday Oct 21, end of day
Whatever exists at that moment is what ships. Do not extend this.
Wednesday and Thursday are video and writeup, and they are not
compressible.

## Scope changes from the original spec

**The random 1,000 sample is CANCELLED.** It existed to measure the
gap-vs-risk relationship across all deciles. At 5.7% residential
coverage it would return almost nothing. Its loss means the
classifier is trained on arterials only, and that bias must be
stated explicitly wherever results appear.

**Target set:** the top 2,000 by risk, filtered to the ~70% with
Mapillary coverage. Roughly 1,400 intersections.

**Labeling target lowered to 400 images.** 800 was sized for a
multi-week phase. 400 is enough for a fine-tuned ResNet-18 to be
meaningful, and it is about 3 hours of honest clicking rather
than 6.

## Monday — collect and label

Morning: write `src/data/fetch_images.py`, start it running in the
background. Resumable, skip any image_id already in the manifest,
exponential backoff on 429, log failures without crashing.
Up to 2 images per intersection (not 4 — you are labeling, not
collecting).

While it runs, write the labeling rules down BEFORE labeling
anything:
- RAMP: a sloped transition from sidewalk to street is visible at
  the corner in frame
- NO_RAMP: the corner is clearly visible and shows a vertical curb
  face with no sloped transition
- SKIP: corner not in frame, obstructed, too dark, or ambiguous

Afternoon: label. Simple approach — a script that shows an image
and takes a keypress (r / n / s), moving the file into
`data/images/labeled/{ramp,no_ramp}/`. Do not install Label Studio;
setup time is the enemy.

Stop at 400 labeled or when Monday ends, whichever comes first.
Report the class balance. If it is worse than 85/15, say so — a
classifier on that split may not be worth reporting.

## Tuesday — train

- ResNet-18, ImageNet pretrained, replace the final layer
- 80/20 train/val split, stratified
- Augmentation: random horizontal flip, slight color jitter. Nothing
  exotic.
- `device="mps"`, batch 32, ~10 epochs, early stopping on val loss
- Report precision, recall, F1, and the confusion matrix. NOT accuracy.
- Save to `models/ramp_resnet18.pt`

If val F1 is below 0.6, the classifier is not good enough to show as
a result. Report it as an attempted phase with honest numbers rather
than hiding it — that is still a legitimate finding.

## If it works
Add a fourth map layer to the app marking covered intersections
with their predicted ramp status. One afternoon maximum, Wednesday
morning at the latest. If the app integration looks like it will
take longer, report the classifier in the writeup and leave the
app alone.