# Gesture Cat 🤫

Make a hand gesture at your webcam, get the matching cat. Fifteen gestures,
fifteen cats. Runs entirely in the browser — no build step, no backend, no
video ever leaves the device.

## The gestures

| | Gesture | How to make it | Cat |
|---|---|---|---|
| 🫶 | Heart | **both hands** — index tips touching on top, thumb tips touching below | cat with a heart on its side |
| 🤌 | Bunched | all five fingertips gathered to a point | italian cat |
| 👌 | OK | thumb + index touching, other three out | cat making an OK paw |
| 🤏 | Pinch | thumb + index touching, other three curled | side-eye cat |
| 🖖 | Vulcan | four fingers out, split down the middle | saluting cat |
| 🤞 | Crossed | index + middle out, tips crossed | Pretty Please Cat |
| ✌️ | Peace | index + middle out, apart | CAT RELAX |
| 🤟 | Love you | thumb + index + pinky | heart paw cat |
| 🤘 | Rock on | index + pinky, thumb tucked | ROCK STAR KITTY |
| 🤙 | Shaka | thumb + pinky only | cat in sunglasses |
| ☝️ | Shush | index only, pointing up | Cat Shush — paw over the mouth |
| ✋ | Palm | all four fingers out | High Five Kitty |
| 👍 | Thumbs up | fist, thumb pointing up | THUMBS UP CAT |
| 👎 | Thumbs down | fist, thumb pointing down | Pathetic Cat |
| ✊ | Fist | everything curled | no talk me am angy |

Tap any gesture in the strip at the bottom to preview its cat without using
the camera.

## How it works

- MediaPipe Hands tracks up to two hands (21 landmarks each) and draws the
  skeleton over the video.
- Finger extension is measured by **distance from the wrist** rather than raw
  vertical position, so a tilted or rotated hand still reads correctly.
  Everything else — pinch distance, finger spread, fingertip clustering — is
  normalised by palm length so it works at any distance from the camera.
- A gesture must hold for **300 ms** before it fires. Switching straight from
  one gesture to another is immediate, but re-firing the *same* gesture waits
  out a **1 s cooldown**. Losing tracking for under **250 ms** doesn't drop the
  meme, so it never flickers.
- The meme cross-fades in with a pop, over a blurred copy of itself so any
  image shape fills the frame.

## The images

Every cat is a real, unmodified meme photo from
[Imgflip](https://imgflip.com/memetemplates) — nothing drawn on top, nothing
AI-generated. Where a cat actually performs the gesture, that's the one used:
the thumbs-up cat really is giving a thumbs up, the OK cat really is making an
OK sign with its paw, the high-five cat has its paw up, the Italian cat has the
🤌 paw, and the salute cat is saluting. The rest are matched on meaning —
Pathetic Cat for thumbs down, "no talk me am angy" for a fist. Full source list
in [images/CREDITS.md](images/CREDITS.md).

One caveat: `ok.jpg` is only 216×233 — it's the one cat genuinely making an OK
sign, and no larger copy exists, so it looks soft at full size. Swap it for a
sharper but less literal cat if that bothers you.

## Files

```
index.html      markup
style.css       mobile-first layout; phone = big meme + camera PiP,
                tablet/desktop = two panels
script.js       gesture detection, debounce, display
images/         15 gesture memes + idle placeholder (~1 MB total)
```

## Run locally

Camera access needs a **secure context** — `https://` or `localhost`. Opening
`index.html` off the filesystem (`file://`) will not work.

```bash
python3 -m http.server 8000
```

Then open <http://localhost:8000>.

## Deploy

Fully static, so anywhere that serves over HTTPS works. HTTPS is not optional
here — browsers refuse camera access without it.

### GitHub Pages

Pages serves over HTTPS on `*.github.io`, so the camera works. Create an empty
repo on GitHub, then:

```bash
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

Then in the repo: **Settings → Pages → Source → Deploy from a branch**, pick
`main` and `/ (root)`, and save. The site appears at
`https://<you>.github.io/<repo>/` within a minute or two.

All asset paths are relative, so it works under the `/<repo>/` subpath without
changes. The `.nojekyll` file stops Pages running the files through Jekyll.

### Netlify Drop

Drag the folder onto <https://app.netlify.com/drop> for an instant HTTPS URL,
no repo needed.

## On phones and tablets

- Layout is mobile-first: the meme takes the screen and the camera rides along
  as a small picture-in-picture; landscape and tablet/desktop switch to two
  panels side by side.
- Phones automatically use MediaPipe's lighter model (`modelComplexity: 0`) at
  480×360 so the frame rate stays usable.
- A flip button appears in the camera corner when the device has more than one
  camera.
- **iOS needs Safari 14.3+**, and the page must be served over HTTPS.

## Troubleshooting

- **First load needs internet.** The images are local, but MediaPipe's library
  and WASM model files come from jsDelivr. The browser caches them afterward.
- **"Camera permission was denied"** — clear the site's camera permission in
  browser settings, reload, hit Retry.
- **"The camera is already in use"** — another tab or app has it. Close that,
  hit Retry.
- **A gesture isn't registering** — face your palm at the camera and keep the
  hand upright. Detection is geometric, so heavily foreshortened hands (fingers
  pointing at the lens) are the hard case.
- **Tuning** — the thresholds all live at the top of `classify()` and
  `features()` in [script.js](script.js), as named numbers with comments.
