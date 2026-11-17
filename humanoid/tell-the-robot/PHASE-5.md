# Tell the robot, phase 5: the brain

Goal: say a sentence, the robot does it. Claude chooses the skills from phase 4 as tools, reads
each result, decides the next step, and answers in one or two spoken sentences. It never moves a
joint. All code in `humanoid/03-g1-pick-place/`.

```bash
humanoid/tell-the-robot/demo.sh g1-agent --stop-after 120s   # robot + windows, gently
cd humanoid/03-g1-pick-place && uv run --no-sync python nodes/chat.py   # second terminal: talk to it
```

Through Exxeta's LiteLLM proxy (allowed for this project) the environment needs
`ANTHROPIC_BASE_URL=https://ai.exxeta.info`, `ANTHROPIC_AUTH_TOKEN` (your proxy token, in
`~/.zshrc`, never in the repo) and `AGENT_MODEL=claude-opus-5.5` (the proxy writes model names
with a dot). Straight against Anthropic: only `ANTHROPIC_API_KEY`.

## What was added

| File | What it is |
| --- | --- |
| `g1/agent.py` | system prompt, 5 tools, `check_call` (the safety check), `Agent` (hear, think, give), `Conductor`, `converse` |
| `nodes/agent.py` | the dora node: model call in a worker thread, skills one at a time, "stop" straight to the robot |
| `nodes/chat.py` | dynamic dora node: type sentences in a terminal, read the replies |
| `agent.yml` | the skills graph with the agent and chat instead of the commander; `./run.sh g1-agent` |
| `nodes/dashboard.py` | a conversation panel: heard, tool calls, results, replies |
| `tests/test_agent.py` | 8 tests with a scripted fake model, no API calls |
| `spikes/agent_eval.py` | 12 sentences in 5 levels, judged against the simulator's truth |
| `demo.sh`, lower rates | live demos at 3 cores at most (see the end) |

## How a sentence becomes skills

1. The sentence goes to Claude with the system prompt and 5 tools: `look_around`, `get_objects`,
   `pick`, `place`, `go_to`
2. `get_objects` returns the memory as text, with a line per object: kind, color, table, position,
   and whether `pick` could reach it, and why not
3. Every tool call passes `check_call` before the robot moves: an id the memory doesn't have,
   picking with a full hand, placing with an empty one, placing next to itself: rejected with a
   reason the model reads
4. Robot skills run one at a time; their plain-language results go back to the model
5. When the model answers without a tool call, that answer is spoken

Settings: `claude-opus-5.5`, effort `medium`, adaptive thinking, refusal fallbacks on
(`AGENT_FALLBACKS=0` turns them off). The conversation only grows, so the cached system prompt
and tools stay valid.

## Results: 12 sentences, real model, judged against the truth

| # | Level | Sentence | Result |
| --- | --- | --- | --- |
| 1 | 1 direct | Put the blue bottle on the green mark. | PASS, 1.1 cm from the mark |
| 2 | 1 | Put the yellow can on table A. | PASS |
| 3 | 1 | Put the blue bottle on table B. | PASS |
| 4 | 2 attributes | Put the can on the mark. | PASS, 0.6 cm; "It was the only can on the tables." |
| 5 | 2 | Put the red bottle next to the blue bottle. | PASS, 13 cm apart |
| 6 | 2 | Move the blue bottle to the other table. | PASS |
| 7 | 3 several steps | Put the blue bottle and the yellow box on table B. | FAIL: said both done, the bottle fell over |
| 8 | 3 | Clear the blue bottle off table A, then put the yellow box next to it. | FAIL: the box tipped over; the model explained the failed check away |
| 9 | 4 ambiguous | Put the green box on the mark. (two green boxes) | PASS: asked which one, moved nothing |
| 10 | 4 vague | I'm thirsty. | PASS: brought the only bottle to the start |
| 11 | 5 things go wrong | Put the blue bottle on the mark. (knocked out of the hand) | PASS: "it fell out of my hand while I was carrying it" |
| 12 | 5 | Put the green box that's in the middle of table A on the mark. | PASS: "21 centimeters from the nearest edge, and my hand only reaches 17" |

**10 of 12.** 4 to 12 model calls per sentence, 15 to 77 s each.

## Things learned the hard way

- **Ground the model in ids, check every call.** The model only names ids from `get_objects`; the
  safety check catches the rest before anything moves. No case in the evaluation needed it, the
  offline tests show it works.
- **Tell the model what's possible, not just what's there.** `get_objects` says which objects can
  be picked and why not. Case 12's answer ("21 cm, my hand reaches 17") comes straight from that.
- **Results written for a reader pay off.** Phase 4 wrote skill results as sentences; the model
  quotes them in its replies (cases 11, 12).
- **The model can only be as honest as its inputs.** Case 7: the second placement knocked the
  first object over, no skill noticed, the model reported success. Case 8: the memory knows where
  an object is, not whether it stands, so the model concluded a tipped box was "next to it".
  Both are perception gaps, not model errors.
- **A proxy changes names, not the code.** Exxeta's LiteLLM proxy takes `claude-opus-5.5` (dot,
  not dash), passes tools, thinking, effort and even the fallback beta. One cheap probe request
  before the evaluation found the name.
- **A key in a chat is a leaked key.** Keys go into `~/.zshrc` by the user's own hand, and get
  regenerated if they ever appear in a conversation.
- **Live windows are the heaviest load.** The live demo with the MuJoCo window and Rerun used 3.6
  cores plus the GPU; the Mac got hot before macOS recorded any warning. Now: window 30 frames
  per second (`DRAW_FPS`), head camera 10 images per second (`CAMERA_HZ`), dashboard images 2 per
  second (`IMAGE_EVERY`), and `demo.sh` stops a demo above 3 cores for 20 s. Measured with
  `demo.sh g1-skills`, look around: median 1.8 cores (was 3.6), peak 2.8, still 100% of real
  time, and the Mac stayed cool. The Rerun window itself is the biggest part, 1.2 cores while
  drawing, then the physics, 1.1.

## Fix: standing or fallen over

Both evaluation failures came from one gap: the memory knew where an object was, not whether it
stood. Measured first (`spikes/fallen_check.py`): standing objects are 10 to 18 cm high, lying ones
6.2 to 6.8 cm, so a top below 8 cm means lying. Height says nothing about a lying object's kind
(all about 6 cm; before the fix every fallen bottle became a "box").

| Layer | Change |
| --- | --- |
| `g1/objects.py` | every sighting is standing, lying, or unknown (top cut off); lying sightings have no kind |
| `g1/world.py` | the memory stores `standing`; a lying sighting matches the same color up to 15 cm away, so a fallen bottle keeps its id and stays a bottle |
| `g1/agent.py` | `get_objects` shows "LYING ON ITS SIDE (fell over)"; `check_call` refuses to pick it |
| `g1/skills.py` | `pick` refuses lying objects; `place` succeeds only if the object is seen standing at the spot; neighbors that fell over, moved, or are not seen again are reported |
| system prompt | a FAILED result stays failed, even if the object is found nearby later |

| Case | Before | After |
| --- | --- | --- |
| 7: blue bottle and yellow box on table B | "I moved both onto table B" (the bottle was on the floor) | "after the box went down, the blue bottle was gone, and when I looked around again I couldn't find it anywhere" (it looked around again on its own) |
| 8: yellow box next to the bottle | "a fresh look shows the box … next to it" (it had tipped over) | "that failed: the box fell over and is lying on its side, so I can't pick it up again" |

The judges for these two now also pass an honest report of what went wrong; a false success
claim still fails.

**Full rerun of all 12 with the fix: 12 of 12.** The automatic judge said 11: case 7's reply
("the blue bottle I'd already put there is gone … can't see it on either table now") is true, the
bottle was on the floor, but the judge's list of honest words lacked "gone" and "can't see". The
list is fixed and checked against the recorded reply (passes) and against the old false claim
("I put the blue bottle and the yellow box on table B", still fails). The other 11 cases pass
unchanged; replies got slightly more precise ("it's standing there now"). 32 pure tests and the
4 failure scenarios pass.

## Fix: don't knock over what's already there

`spikes/knockover_check.py` replays case 7 without the model (room 6, bottle then box to one
table) and logs every contact between the robot and a still object. The box was not the problem:
the **wrist and middle finger** pushed the bottle 25 cm while the robot walked in carrying the box.
The place check allowed a neighbor 12 cm right of the spot; measured, the carrying hand reaches 24
cm right.

| Change | Where |
| --- | --- |
| `PLACE_SWEEP` is three measured rectangles, one per phase: carrying in (24 cm right), over the spot and down (14 cm right, 12 cm ahead), tucked and stepping back (33 cm right, off the table edge) | `g1/reach.py` |
| "next to" tries a second ring at 16 cm when the 13 cm ring has no free hand path | `g1/reach.py` |
| a test with case 7's exact positions | `tests/test_reach.py` |

| Check | Before | After |
| --- | --- | --- |
| case 7 replay, table B | bottle pushed 25 cm, on the floor | box at a spot 30 cm away, bottle moved under 1 cm, upright |
| same, table A | the box knocked a neighbor off | bottle upright, moved under 1 cm |
| `skills_eval.py`, 5 rooms | | 10 of 10 placed, 0.6 to 1.7 cm, upright; 5 honest refusals |
| failure scenarios, pure tests | | 4 of 4, 34 tests |

## Still open

- **Picking can still nudge a neighbor:** room 3 of the eval, the open hand pushed a bottle 5 cm
  on its way in (`open hand next to it`). `HAND_SWEEP` for picking is still the phase 4 guess;
  measure it the same way as `PLACE_SWEEP`.
- A wrist graze during `step back` moves a neighbor under 1 cm.

- A neighbor hidden behind another object can be reported as "can't see it anymore": a false
  alarm, less likely with the longer 2.5 s look after placing.

- **Live demo, room 2:** `look_around` missed the yellow can that the headless evaluation found.
  Claude answered honestly ("couldn't find a yellow can"). Detection while live needs more looks
  per view, or a third viewpoint.

## Exercises

1. Read `SYSTEM` in `g1/agent.py`. Which rule produced case 9's question? Remove it, run case 9
   (`agent_eval.py 9`, a few cents), compare.
2. Why does `check_call` exist when the system prompt already says "use ids from get_objects"?
3. In case 8 the model said a failed placement worked because a fresh look showed the box nearby.
   Write one sentence for `SYSTEM` that stops that, and the memory field that would make it right.
4. Why do the tool results go back as one message with all results, instead of one message each?
5. `nodes/agent.py` runs `think()` in a thread. What would go wrong if it called it directly in the
   event loop?

## Next: phase 6, voice

Push-to-talk, Whisper on the Mac for speech to text, `say` for the replies, and the dashboard laid
out for an audience.
