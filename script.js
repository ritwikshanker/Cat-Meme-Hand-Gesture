/* =========================================================================
   Gesture Cat — 17 hand gestures, each with its own cat meme.
   Everything runs client-side; no frames leave the device.
   ========================================================================= */

'use strict';

/* ---------------- timing ---------------- */

const HOLD_MS     = 300;   // a gesture must be held this long before it fires
const COOLDOWN_MS = 1000;  // re-firing the SAME gesture waits this long
const RELEASE_MS  = 250;   // tolerate this much lost tracking before dropping

const MEDIAPIPE_VERSION = '0.4.1675469240';
const IDLE_SRC = 'images/idle.png';

/* =========================================================================
   Gesture table. Order matters — the first match wins, so the most
   specific shapes are listed before the looser ones.
   ========================================================================= */

const GESTURES = [
  { id:'heart',      emoji:'🫶',  name:'Heart',     hands:2,
    caption:'ok this one is for you 💘' },
  { id:'bunched',    emoji:'🤌',  name:'Bunched',
    caption:'mamma mia' },
  { id:'ok',         emoji:'👌',  name:'OK',
    caption:'cat approves' },
  { id:'pinch',      emoji:'🤏',  name:'Pinch',
    caption:'just a lil bit' },
  { id:'vulcan',     emoji:'🖖',  name:'Vulcan',
    caption:'live long and prosper' },
  { id:'crossed',    emoji:'🤞',  name:'Crossed',
    caption:'pretty pleeease' },
  { id:'peace',      emoji:'✌️',  name:'Peace',
    caption:'just vibing' },
  { id:'loveyou',    emoji:'🤟',  name:'Love you',
    caption:'love u 🩷' },
  { id:'rock',       emoji:'🤘',  name:'Rock on',
    caption:'🎸 ROCK ON' },
  { id:'shaka',      emoji:'🤙',  name:'Shaka',
    caption:'too cool for this' },
  { id:'shush',      emoji:'☝️',  name:'Shush',
    caption:'shhhhh 🤫' },
  { id:'palm',       emoji:'✋',  name:'Palm',
    caption:'talk to the paw' },
  { id:'thumbsup',   emoji:'👍',  name:'Thumbs up',
    caption:'certified good' },
  { id:'thumbsdown', emoji:'👎',  name:'Thumbs dn',
    caption:'pathetic.' },
  { id:'fist',       emoji:'✊',  name:'Fist',
    caption:'no talk me am angy' },
];

const BY_ID = Object.fromEntries(GESTURES.map((g) => [g.id, g]));
const srcFor = (g) => `images/${g.id}.jpg`;

/* =========================================================================
   Landmark helpers

   MediaPipe hands returns 21 normalised landmarks:
     0 wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky
   (…MCP, PIP, DIP, TIP for each finger).
   ========================================================================= */

const IDX = {
  index:  { mcp: 5,  pip: 6,  tip: 8  },
  middle: { mcp: 9,  pip: 10, tip: 12 },
  ring:   { mcp: 13, pip: 14, tip: 16 },
  pinky:  { mcp: 17, pip: 18, tip: 20 },
};

const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

/**
 * Per-hand feature extraction.
 *
 * Extension is measured by distance from the wrist rather than raw y, so it
 * survives a tilted or rotated hand: an extended fingertip is always further
 * from the wrist than its own PIP joint, whichever way the hand is facing.
 * Everything else is normalised by palm length so it works at any distance
 * from the camera.
 */
function features(lm) {
  const wrist = lm[0];
  const scale = Math.max(dist(wrist, lm[9]), 1e-6);   // wrist → middle MCP

  const ext = {};
  for (const f of Object.keys(IDX)) {
    ext[f] = dist(lm[IDX[f].tip], wrist) > dist(lm[IDX[f].pip], wrist) * 1.06;
  }
  // The thumb never folds toward the wrist, so measure how far its tip sits
  // from the opposite knuckle instead: tucked across the palm reads short
  // (~0.5 palms), splayed or raised reads long (~1.0-1.4 palms).
  ext.thumb = dist(lm[4], lm[17]) / scale > 0.85;

  const n = (a, b) => dist(lm[a], lm[b]) / scale;

  return {
    lm, scale, ext,
    count:      ['index', 'middle', 'ring', 'pinky'].filter((f) => ext[f]).length,
    thumbIndex: n(4, 8),     // pinch / OK distance
    indexMid:   n(8, 12),    // peace spread vs crossed fingers
    midRing:    n(12, 16),   // the Vulcan gap
    ringPinky:  n(16, 20),
    // How far the index tip reaches from the wrist, in palm lengths. A pinch
    // holds the tip out in front (~1.4+); a closed fist parks it on the palm
    // (~0.9), which otherwise looks identical to a pinch by tip distance alone.
    indexReach: dist(lm[8], wrist) / scale,
    // widest gap between any two fingertips (thumb included), in palm lengths
    tipSpread: (() => {
      const tips = [4, 8, 12, 16, 20];
      let max = 0;
      for (let i = 0; i < tips.length; i++)
        for (let j = i + 1; j < tips.length; j++)
          max = Math.max(max, dist(lm[tips[i]], lm[tips[j]]));
      return max / scale;
    })(),
    // shush should be an upright finger, not a sideways point
    indexUp:    lm[8].y < lm[6].y && lm[6].y < lm[5].y,
    thumbUp:    lm[4].y < lm[2].y - 0.04,
    thumbDown:  lm[4].y > lm[2].y + 0.04,
  };
}

/** Two-hand finger heart: index tips meet on top, thumb tips meet below. */
function isHeart(a, b) {
  const scale = (a.scale + b.scale) / 2;
  const tipsTogether  = dist(a.lm[8], b.lm[8]) / scale < 0.62;
  const thumbsTogether = dist(a.lm[4], b.lm[4]) / scale < 0.85;
  const indexOnTop =
    Math.min(a.lm[8].y, b.lm[8].y) < Math.min(a.lm[4].y, b.lm[4].y);
  return tipsTogether && thumbsTogether && indexOnTop;
}

/** Single-hand classifier. Returns a gesture id, or null. */
function classify(h) {
  const { ext, count, thumbIndex, indexMid, midRing, ringPinky } = h;
  const reaching = h.indexReach > 1.15;   // fingertips out in front of the palm

  // --- every fingertip bunched together -------------------------------
  // Checked first: a bunched hand also has the thumb against the index.
  if (reaching && h.tipSpread < 0.55) return 'bunched';

  // --- thumb and index brought together: OK or pinch --------------------
  // 0.95 palm-lengths, not "touching". People rarely close the OK loop
  // cleanly and MediaPipe rarely lands the tips on each other, so an OK with
  // a slightly open loop used to read as four fingers out and fire `palm`.
  // Measured separation: OK spans 0.1-0.85 even with a loose loop, while an
  // open palm, peace and vulcan all sit at 1.19 or above.
  if (thumbIndex < 0.95) {
    const openFingers = (ext.middle ? 1 : 0) + (ext.ring ? 1 : 0) + (ext.pinky ? 1 : 0);

    // OK: the other three stay out. Two of three is enough — the ring finger
    // is the least reliable landmark and one bad read shouldn't cost the
    // gesture. Those open fingers already rule out a fist, so this branch
    // deliberately skips the reach guard below: in a real OK the index curls
    // into a loop to meet the thumb, which pulls its tip back toward the
    // wrist and makes `reaching` fail.
    if (openFingers >= 2) return 'ok';

    // Pinch: the other three are curled, which looks just like a fist with
    // the thumb resting on it — so this one stays strict on both counts.
    if (thumbIndex < 0.45 && reaching) return 'pinch';
  }

  // --- four fingers out -------------------------------------------------
  // The thumb is the least reliable landmark, so it no longer splits these.
  if (count === 4) {
    const together = (indexMid + ringPinky) / 2;
    if (midRing > together * 1.7 && midRing > 0.35) return 'vulcan';
    return 'palm';
  }

  // --- index + middle ---------------------------------------------------
  if (ext.index && ext.middle && !ext.ring && !ext.pinky) {
    // crossed fingers overlap; even a tight peace sign stays wider than this
    return indexMid < 0.18 ? 'crossed' : 'peace';
  }

  // --- index + pinky (with or without thumb) ----------------------------
  if (ext.index && ext.pinky && !ext.middle && !ext.ring) {
    return ext.thumb ? 'loveyou' : 'rock';
  }

  // --- thumb + pinky ----------------------------------------------------
  if (ext.thumb && ext.pinky && count === 1) return 'shaka';

  // --- index only -------------------------------------------------------
  if (count === 1 && ext.index) return h.indexUp ? 'shush' : null;

  // --- nothing but the thumb -------------------------------------------
  if (count === 0) {
    if (ext.thumb && h.thumbUp)   return 'thumbsup';
    if (ext.thumb && h.thumbDown) return 'thumbsdown';
    if (!ext.thumb)               return 'fist';
    return null;
  }

  return null;
}

/** Whole-frame detection across every hand MediaPipe found. */
function detect(handsLandmarks) {
  if (!handsLandmarks || !handsLandmarks.length) return null;

  const feats = handsLandmarks
    .filter((lm) => lm && lm.length >= 21)
    .map(features);

  if (!feats.length) return null;
  if (feats.length >= 2 && isHeart(feats[0], feats[1])) return 'heart';
  return classify(feats[0]);
}

/* =========================================================================
   Elements
   ========================================================================= */

const video      = document.getElementById('video');
const overlay    = document.getElementById('overlay');
const ctx        = overlay.getContext('2d');
const bgLayer    = document.getElementById('bgLayer');
const layerA     = document.getElementById('layerA');
const layerB     = document.getElementById('layerB');
const captionEl  = document.getElementById('caption');
const legendEl   = document.getElementById('legend');
const camSpinner = document.getElementById('camSpinner');
const flipBtn    = document.getElementById('flipBtn');
const errorEl    = document.getElementById('camError');
const errorMsgEl = document.getElementById('camErrorMsg');
const retryBtn   = document.getElementById('retryBtn');
const statusDot  = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');

/* =========================================================================
   Meme display — two stacked layers that cross-fade
   ========================================================================= */

let frontLayer = layerA;   // currently visible
let backLayer  = layerB;
let shownId    = null;     // gesture id currently on screen, or null for idle

/** Warm the cache so the first trigger of each gesture is instant. */
function preload() {
  for (const g of GESTURES) {
    const img = new Image();
    img.src = srcFor(g);
  }
}

let paintToken = 0;

function paint(src, alt, pop) {
  // Two gestures can fire within a few hundred ms, so two decode() calls can
  // be in flight at once. Without this token the older one can resolve last
  // and leave the wrong meme on screen.
  const token    = ++paintToken;
  const incoming = backLayer;
  const outgoing = frontLayer;

  if (incoming.getAttribute('src') !== src) incoming.src = src;
  incoming.alt = alt;
  bgLayer.src = src;

  const swap = () => {
    if (token !== paintToken) return;          // a newer paint already won
    incoming.classList.add('is-visible');
    outgoing.classList.remove('is-visible', 'is-pop');
    if (pop) {
      incoming.classList.remove('is-pop');
      void incoming.offsetWidth;               // restart the animation
      incoming.classList.add('is-pop');
    }
    frontLayer = incoming;
    backLayer  = outgoing;
  };

  // Wait for the bitmap so the cross-fade never reveals a half-painted image —
  // but never block on it. A layer sitting at opacity:0 can have its decode
  // deferred indefinitely, so cap the wait and swap regardless.
  if (incoming.complete && incoming.naturalWidth > 0) { swap(); return; }

  let settled = false;
  const go = () => { if (!settled) { settled = true; swap(); } };
  incoming.addEventListener('load',  go, { once: true });
  incoming.addEventListener('error', go, { once: true });
  setTimeout(go, 150);
}

function showGesture(id) {
  if (shownId === id) return;
  shownId = id;

  const g = BY_ID[id];
  paint(srcFor(g), `${g.name} cat meme`, true);
  captionEl.textContent = g.caption;
  setStatus('active', `${g.emoji} ${g.name}`);
  highlight(id);
}

function showIdle() {
  if (shownId === null) return;
  shownId = null;

  paint(IDLE_SRC, '', false);
  captionEl.textContent = 'Waiting for a gesture…';
  highlight(null);
  if (errorEl.hidden) setStatus('ready', 'Tracking…');
}

function setStatus(kind, text) {
  statusDot.className = 'dot' + (kind ? ` is-${kind}` : '');
  statusText.textContent = text;
}

/* =========================================================================
   Legend — also a manual browser when the camera isn't available
   ========================================================================= */

const chips = new Map();

function buildLegend() {
  const frag = document.createDocumentFragment();
  for (const g of GESTURES) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'chip';
    b.dataset.gesture = g.id;
    b.innerHTML =
      `<span class="chip-emoji">${g.emoji}</span><span class="chip-name"></span>`;
    b.querySelector('.chip-name').textContent = g.name;
    b.addEventListener('click', () => {
      if (shownId === g.id) showIdle();
      else showGesture(g.id);
    });
    chips.set(g.id, b);
    frag.appendChild(b);
  }
  legendEl.appendChild(frag);
}

function highlight(id) {
  for (const [gid, el] of chips) el.classList.toggle('is-active', gid === id);
  const el = id && chips.get(id);
  if (el && legendEl.scrollWidth > legendEl.clientWidth) {
    el.scrollIntoView({ inline: 'center', block: 'nearest', behavior: 'smooth' });
  }
}

/* =========================================================================
   Debounced state machine — one candidate at a time
   ========================================================================= */

let active         = null;   // gesture currently held on screen
let candidate      = null;   // what we're seeing right now
let candidateSince = 0;
let lostSince      = null;
let lastReleased   = null;
let lastRelease    = 0;

function updateGesture(detected, now) {
  if (detected !== candidate) {
    candidate = detected;
    candidateSince = now;
  }

  const held = candidate !== null && now - candidateSince >= HOLD_MS;

  if (active === null) {
    lostSince = null;
    if (!held) return;
    // Only re-firing the *same* gesture has to wait out the cooldown;
    // moving on to a different one stays snappy.
    if (candidate === lastReleased && now - lastRelease < COOLDOWN_MS) return;
    active = candidate;
    showGesture(active);
    return;
  }

  if (detected === active) {
    lostSince = null;
    return;
  }

  if (detected === null) {
    // Hold the meme through a dropped frame or two instead of flickering.
    if (lostSince === null) lostSince = now;
    else if (now - lostSince >= RELEASE_MS) release(now);
    return;
  }

  // A different gesture — switch straight over once it has been held.
  lostSince = null;
  if (held) {
    active = candidate;
    showGesture(active);
  }
}

function release(now) {
  lastReleased = active;
  lastRelease  = now;
  active    = null;
  lostSince = null;
  showIdle();
}

/* =========================================================================
   Per-frame results
   ========================================================================= */

let rearFacing = false;

function onResults(results) {
  camSpinner.hidden = true;

  const w = results.image ? results.image.width  : video.videoWidth;
  const h = results.image ? results.image.height : video.videoHeight;
  if (w && h && (overlay.width !== w || overlay.height !== h)) {
    overlay.width  = w;
    overlay.height = h;
  }
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  const hands = results.multiHandLandmarks || [];
  updateGesture(detect(hands), performance.now());

  if (hands.length && window.drawConnectors) {
    const tint = active ? '#7c9cff' : '#4ade80';
    for (const lm of hands) {
      window.drawConnectors(ctx, lm, window.HAND_CONNECTIONS,
        { color: tint, lineWidth: 3 });
      window.drawLandmarks(ctx, lm,
        { color: '#ffffff', fillColor: tint, lineWidth: 1, radius: 2.5 });
    }
  }

  if (!active && errorEl.hidden) {
    setStatus('ready', hands.length ? 'Hold the gesture…' : 'Show me a hand');
  }
}

/* =========================================================================
   Camera
   ========================================================================= */

let hands  = null;
let camera = null;

function showError(message) {
  camSpinner.hidden = true;
  errorMsgEl.textContent = message;
  errorEl.hidden = false;
  setStatus('error', 'Camera off');
}

function describeCameraError(err) {
  if (!window.isSecureContext) {
    return 'This page is not in a secure context. Open it over https:// or from localhost.';
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    return 'This browser does not expose a webcam API (getUserMedia is missing).';
  }
  switch (err && err.name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return 'Camera permission was denied. Allow camera access for this site, then hit Retry.';
    case 'NotFoundError':
    case 'OverconstrainedError':
      return 'No usable camera was found on this device.';
    case 'NotReadableError':
      return 'The camera is already in use by another app or tab. Close it and hit Retry.';
    case 'AbortError':
      return 'The camera failed to start. Hit Retry.';
    default:
      return (err && err.message) || 'The camera could not be started.';
  }
}

/** Ask for the camera ourselves so permission failures surface cleanly. */
async function probeCamera() {
  if (!window.isSecureContext) {
    throw Object.assign(new Error('insecure context'), { name: 'SecurityError' });
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw Object.assign(new Error('getUserMedia unavailable'), { name: 'NotSupportedError' });
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: rearFacing ? 'environment' : 'user' },
    audio: false,
  });
  stream.getTracks().forEach((t) => t.stop());  // MediaPipe opens its own
}

/** Phones get the lighter model so the frame rate stays usable. */
function isLowPower() {
  const coarse = window.matchMedia('(pointer: coarse)').matches;
  const fewCores = (navigator.hardwareConcurrency || 4) <= 6;
  return coarse || fewCores;
}

function buildHands() {
  if (hands) return hands;

  if (typeof window.Hands !== 'function') {
    throw new Error('MediaPipe Hands failed to load. Check your connection and reload.');
  }
  hands = new window.Hands({
    locateFile: (file) =>
      `https://cdn.jsdelivr.net/npm/@mediapipe/hands@${MEDIAPIPE_VERSION}/${file}`,
  });
  hands.setOptions({
    maxNumHands: 2,                                // two hands, for 🫶
    modelComplexity: isLowPower() ? 0 : 1,
    minDetectionConfidence: 0.7,
    minTrackingConfidence: 0.5,
  });
  hands.onResults(onResults);
  return hands;
}

async function start() {
  errorEl.hidden = true;
  camSpinner.hidden = false;
  setStatus('', 'Starting…');

  try {
    await probeCamera();
  } catch (err) {
    showError(describeCameraError(err));
    return;
  }

  let detector;
  try {
    detector = buildHands();
  } catch (err) {
    showError(err.message);
    return;
  }

  if (typeof window.Camera !== 'function') {
    showError('MediaPipe camera utilities failed to load. Check your connection and reload.');
    return;
  }

  if (camera) {
    try { camera.stop(); } catch (_) { /* already stopped */ }
  }

  const short = isLowPower();
  camera = new window.Camera(video, {
    width:  short ? 480 : 640,
    height: short ? 360 : 480,
    facingMode: rearFacing ? 'environment' : 'user',
    onFrame: async () => {
      try { await detector.send({ image: video }); }
      catch (_) { /* a dropped frame is not worth tearing the app down for */ }
    },
  });

  try {
    await camera.start();
    setStatus('ready', 'Show me a hand');
    if (await hasMultipleCameras()) flipBtn.hidden = false;
  } catch (err) {
    showError(describeCameraError(err));
  }
}

async function hasMultipleCameras() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices.filter((d) => d.kind === 'videoinput').length > 1;
  } catch (_) {
    return false;
  }
}

flipBtn.addEventListener('click', () => {
  rearFacing = !rearFacing;
  video.classList.toggle('is-rear', rearFacing);
  overlay.classList.toggle('is-rear', rearFacing);
  start();
});

retryBtn.addEventListener('click', start);

// Release the camera when the tab is hidden; pick it back up on return.
document.addEventListener('visibilitychange', () => {
  if (!camera) return;
  if (document.hidden) {
    try { camera.stop(); } catch (_) { /* noop */ }
  } else if (errorEl.hidden) {
    camera.start().catch((err) => showError(describeCameraError(err)));
  }
});

/* ---------------- go ---------------- */

buildLegend();
preload();
start();
