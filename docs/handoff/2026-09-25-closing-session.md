# Closing session, 25 September 2026

The record of the fifth working session. It grilled the last fog items on the map in two rounds and closed them as five decision tickets. The only ticket left open is the spec-and-slice task. Read the four earlier records first, oldest first. GitHub issues are the source of truth for state; this file is the narrative and the reasoning.

The map: https://github.com/ShikhJohari/Ryuk/issues/1

## How it went

1. Shikhar asked for the latest handoff and the repo state to be read, then to be grilled on what remained. Everything was grilled at once with `/grilling`, since no remaining item blocked another.
2. Round one had 17 questions across live monitor behaviour, EDA, client information architecture, CI, and two loose ends. Shikhar accepted every recommendation except the EDA form: package plus committed JSON plus notebooks, "from the ground up", rather than package and JSON alone. Shikhar also asked why frames go to the service at 10 fps (below).
3. Round two had 9 questions: frame pacing, a revised confirmation rule, notebooks, figures, report format and structure, the demo script, and slicing. Shikhar accepted every recommendation but ruled viva preparation and the demo script out of the project, to be grilled separately once the build is done.
4. Shikhar confirmed with the instructor that a TypeScript client over a Python service is acceptable.
5. Tickets were created, resolved and closed with `/domain-modeling` applied to the glossary.

## Why 10 fps became result-paced

Shikhar asked why the camera "needs" 10 fps. The camera still records at its native rate, usually 30 fps; 10 fps was only the rate frames are sent for recognition. That figure came from the frame-streaming research, which assumed 40-80 ms of work per frame. The toolchain spike later measured 4-12 ms per face, so 10 fps was a leftover from before the measurement. Frames are now sent as soon as the previous result returns, capped at 30 fps.

That made "3 matches within 1 s" mean different things at different rates, so the confirmation rule became proportional: matched in at least half the frames processed over 500 ms, minimum 3.

## Decisions

Each lives in its ticket; not restated here.

- [Decide: live monitor behaviour](https://github.com/ShikhJohari/Ryuk/issues/16)
- [Decide: EDA scope, notebooks and the dataset report](https://github.com/ShikhJohari/Ryuk/issues/17)
- [Decide: client information architecture](https://github.com/ShikhJohari/Ryuk/issues/20)
- [Decide: repo layout, dependencies and CI](https://github.com/ShikhJohari/Ryuk/issues/18)
- [Decide: report structure](https://github.com/ShikhJohari/Ryuk/issues/19)

One change to an earlier decision: the charting session's build phases 6, 7 and 8 (API, client, live monitor as separate layers) are now vertical slices spanning all three: watchlist management, live monitor, sightings, evaluation page. Phases 1-5 keep their order, because only an evaluated model can be active. Recorded on the map's Notes and in the spec-and-slice ticket.

Glossary: new terms **Usable face** and **Confirmation**; **Sighting** gains its open and end rules; **Match** and **No match** now apply to usable faces only. No ADR: none of the decisions is hard to reverse.

## Housekeeping done

- The map's Notes no longer say "Compare SFace int8"; the fog list is empty and viva prep sits under Out of scope.
- `models.md`, `datasets.md` and `evaluation-protocol.md` each gained a dated corrections note at the top pointing to the tickets that overruled them. The bodies are kept as researched.

## A slip worth knowing

Tickets were first created by a shell loop that indexed an array from 0 under zsh, which indexes from 1. Three tickets briefly carried the next ticket's question and resolution, and one create failed. All were corrected by editing the bodies and comments in place, so GitHub's edit history and any notification emails show the wrong first versions. Numbers ended up out of order (client information architecture is 20, after report structure at 19).

## Resuming

The one open ticket is [Task: spec synthesis and slicing into build issues](https://github.com/ShikhJohari/Ryuk/issues/21). Run `/to-spec` (check the test seams with Shikhar first), then `/to-tickets` (quiz Shikhar on the breakdown before publishing). The `ready-for-agent` label those skills apply does not exist on the repo yet. Implementation starts only after that ticket closes. Role agents only, never bare agents, never Haiku.
