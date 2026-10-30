# Video evidence review — 29 October 2026

The four activity links were received in the local `videos.txt` file. This file
is ignored by Git, following the plan's decision to keep evidence links in private
preparation notes and the final report rather than the public README.

## Access and duration

All four YouTube oEmbed requests and watch pages returned HTTP 200 without login.
Each player response reported `playabilityStatus.status = OK`. Titles matched the
four supplied activity names. YouTube reports these total video durations:

| Activity | Total duration | Access result |
|---|---:|---|
| Still | 44 s | Publicly retrievable metadata and frame previews |
| Standing | 45 s | Publicly retrievable metadata and frame previews |
| Jumping | 14 s | Publicly retrievable metadata and frame previews |
| Walking | 19 s | Publicly retrievable metadata and frame previews |

These durations exceed five seconds, but total video duration is not proof of
five seconds of compliant activity demonstration. This check did not play the
complete videos with audio in a browser and is not a claim of an incognito
playback test. The user should perform that final playback check before submission.

## Visual findings from YouTube thumbnails and storyboard frames

Standing, walking and still offered one-second storyboard previews; jumping
offered a low-resolution overview plus a larger thumbnail. Findings are limited
to those images; no participant identity comparison was performed.

| Activity | Observed evidence | Requirement still not met or verified |
|---|---|---|
| Still | Face visible during introduction; phone is shown flat on a table later, with an app screen visible. | Full body is not shown. The camera points down at the table during the still demonstration, removing the face from frame. Active logging and an exact matching CSV are unverified. |
| Standing | Face and phone are visible while the participant stands. | Lower body/feet are cropped. App recording state and matching CSV are unverified. |
| Jumping | Thumbnail shows face and phone; overview frames show movement consistent with the labelled activity. | Lower body/feet are cropped, so take-off and landing cannot be verified. Low-resolution previews do not establish continuous jumping, adequate lighting throughout, or active logging. |
| Walking | Successive frames show movement across the room while holding the phone. | Lower legs/feet are cropped, and the participant moves partly out of frame. Motion blur limits face clarity during movement; active logging is unverified. |

The full-body requirement cannot be marked as passed for any of these videos from
the reviewed previews. The still video also does not keep the face visible during
the table demonstration. These are evidence/framing issues, not a reason to undo
the cleaned sensor dataset or to invent a link between a video and an existing CSV.

## Completing the evidence

For compliant replacement evidence, use a wider fixed camera view that includes
head, face, phone and feet during at least five seconds of each activity. Keep
front lighting and the full walking/jumping movement within that view. For still,
include both the table/phone and the participant standing beside it with face
visible. Show the logging app being started, then perform the activity in the
same continuous shot. Record the actual associated CSV name.

Existing training data can remain; replacing the evidence videos does not require
repeating the 52-clip dataset or the incomplete training sessions. The additional
short demonstrations should genuinely include phone logging, as the brief asks.
If no replacement footage is provided, the framing limitation must remain open;
the review cannot guarantee how the instructor will grade it.

No videos, frame previews, or private links were added to public documentation.
