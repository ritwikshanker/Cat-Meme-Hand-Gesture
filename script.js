/* =========================================================================
   Gesture Cat — 17 hand gestures, each with its own cat meme.
   Everything runs client-side; no frames leave the device.
   ========================================================================= */

'use strict';

/* ---------------- timing ---------------- */

const HOLD_MS     = 300;   // a gesture must be held this long before it fires
const COOLDOWN_MS = 1000;  // re-firing the SAME gesture waits this long
const RELEASE_MS  = 250;   // tolerate this much lost tracking before dropping

const MODEL_TIMEOUT_MS = 45000;  // WASM runtime + model download from the CDN
const SEND_TIMEOUT_MS  = 10000;  // a single frame should never take this long
const STALL_MS         = 12000;  // frames going out with nothing coming back

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
 * MediaPipe normalises x by the frame's WIDTH and y by its HEIGHT, so on any
 * non-square frame the two axes have different scales and a plain distance
 * between two landmarks is stretched. Every threshold below compares a mostly
 * horizontal distance against a mostly vertical one, so that stretch changes
 * the verdict: the same hand measures ~0.87 on a 640x480 laptop frame and
 * ~2.05 on a 720x1280 portrait phone frame.
 *
 * Rescaling x by the aspect ratio puts both axes in the same units (frame
 * heights), which makes every measurement below shape-independent.
 */
function toIsotropic(lm, aspect) {
  return lm.map((p) => ({ x: p.x * aspect, y: p.y, z: p.z }));
}

/**
 * Per-hand feature extraction.
 *
 * Extension is measured by distance from the wrist rather than raw y, so it
 * survives a tilted or rotated hand: an extended fingertip is always further
 * from the wrist than its own PIP joint, whichever way the hand is facing.
 * Everything else is normalised by palm length so it works at any distance
 * from the camera.
 */
function features(raw, aspect) {
  const lm = toIsotropic(raw, aspect);
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
    // In palm lengths, like every other measurement here. A raw 0.04 of the
    // frame height meant a different amount of thumb depending on how large
    // the hand landed in frame, and a hand at arm's length from a phone lands
    // much smaller than one in front of a laptop.
    thumbUp:    (lm[2].y - lm[4].y) / scale > 0.15,
    thumbDown:  (lm[4].y - lm[2].y) / scale > 0.15,
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

/**
 * Whole-frame detection across every hand MediaPipe found.
 * `aspect` is the frame's width/height; see toIsotropic above.
 */
function detect(handsLandmarks, aspect = 1) {
  if (!handsLandmarks || !handsLandmarks.length) return null;

  const feats = handsLandmarks
    .filter((lm) => lm && lm.length >= 21)
    .map((lm) => features(lm, aspect));

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
const errorTitle = document.getElementById('camErrorTitle');
const retryBtn   = document.getElementById('retryBtn');
const statusDot  = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');

/* =========================================================================
   Diagnostics — tap the status badge (or add ?debug=1) to open.
   The camera can only be tested on a real device, so this reports what
   actually happened rather than leaving a silent failure.
   ========================================================================= */

const diag = {
  enabled: /[?&]debug=1/.test(location.search),
  sent: 0, results: 0, hands: 0, sendErrors: 0,
  lastError: null, lastGesture: null, notes: {}, fps: 0,
  _t0: 0, _f0: 0,

  reset() {
    this.sent = this.results = this.hands = this.sendErrors = 0;
    this.lastError = null; this.notes = {};
    this._t0 = performance.now(); this._f0 = 0;
  },
  note(k, v) { this.notes[k] = v; },
  fail(k, err) { this.notes[k] = 'FAILED: ' + ((err && (err.name + ' ' + err.message)) || err); },

  render() {
    const el = document.getElementById('diagPanel');
    if (!el) return;
    el.hidden = !this.enabled;
    if (!this.enabled) return;
    const now = performance.now();
    if (now - this._t0 > 1000) {
      this.fps = Math.round((this.sent - this._f0) * 1000 / (now - this._t0));
      this._t0 = now; this._f0 = this.sent;
    }
    const rows = [
      ['secure ctx', String(window.isSecureContext)],
      ['stream', this.notes.stream || '-'],
      ['video', this.notes.video || '-'],
      ['frame aspect', this.notes.aspect || '-'],
      ['MediaPipe', typeof window.Hands === 'function' ? 'loaded' : 'MISSING'],
      ['model', this.notes.model || '-'],
      ['frames sent', String(this.sent)],
      ['results in', String(this.results)],
      ['fps', String(this.fps)],
      ['hands seen', String(this.hands)],
      ['gesture', this.lastGesture || 'none'],
      ['send errors', String(this.sendErrors)],
    ];
    if (this.notes.play) rows.push(['play', this.notes.play]);
    if (this.notes.getUserMedia) rows.push(['getUserMedia', this.notes.getUserMedia]);
    if (this.lastError) rows.push(['last error', this.lastError]);
    rows.push(['ua', navigator.userAgent.slice(0, 60)]);
    el.innerHTML = rows.map(([k, v]) =>
      `<span class="diag-k"></span><span class="diag-v"></span>`).join('');
    [...el.querySelectorAll('.diag-k')].forEach((n, i) => { n.textContent = rows[i][0]; });
    [...el.querySelectorAll('.diag-v')].forEach((n, i) => { n.textContent = rows[i][1]; });
  },
};

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
  const aspect = (w && h) ? w / h : 1;
  const raw = detect(hands, aspect);

  diag.results++;
  diag.hands = hands.length;
  diag.lastGesture = raw;
  diag.note('aspect', `${w}x${h} = ${aspect.toFixed(3)}`);
  diag.render();

  updateGesture(raw, performance.now());

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
let stream = null;
let rafId  = null;

function showError(message, label = 'Camera off', title = 'Camera unavailable') {
  camSpinner.hidden = true;
  errorTitle.textContent = title;
  errorMsgEl.textContent = message;
  errorEl.hidden = false;
  setStatus('error', label);
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

/** Open the camera. Acquired exactly once — see start(). */
async function openStream() {
  if (!window.isSecureContext) {
    throw Object.assign(new Error('insecure context'), { name: 'SecurityError' });
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw Object.assign(new Error('getUserMedia unavailable'), { name: 'NotSupportedError' });
  }
  return navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: rearFacing ? 'environment' : 'user',
      width:  { ideal: isLowPower() ? 480 : 640 },
      height: { ideal: isLowPower() ? 360 : 480 },
    },
    audio: false,
  });
}

function stopStream() {
  if (rafId !== null) { cancelAnimationFrame(rafId); rafId = null; }
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  video.srcObject = null;
}

/** Resolve once the video actually has pixels, or reject if it never does. */
function waitForVideo(timeoutMs = 8000) {
  return new Promise((resolve, reject) => {
    const t0 = Date.now();
    const check = () => {
      if (video.videoWidth > 0 && video.videoHeight > 0 && video.readyState >= 2) return resolve();
      if (Date.now() - t0 > timeoutMs) {
        return reject(Object.assign(
          new Error('camera opened but never produced a frame'), { name: 'AbortError' }));
      }
      setTimeout(check, 100);
    };
    check();
  });
}

/** Phones get the lighter model so the frame rate stays usable. */
function isLowPower() {
  const coarse = window.matchMedia('(pointer: coarse)').matches;
  const fewCores = (navigator.hardwareConcurrency || 4) <= 6;
  return coarse || fewCores;
}

/** Reject if `promise` has not settled in `ms`, so nothing can hang forever. */
function withTimeout(promise, ms, what) {
  let timer;
  const bell = new Promise((_, reject) => {
    timer = setTimeout(
      () => reject(Object.assign(new Error(what + ' timed out after ' + ms + 'ms'),
                                 { name: 'TimeoutError' })),
      ms);
  });
  return Promise.race([promise, bell]).finally(() => clearTimeout(timer));
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
    maxNumHands: 2,                                // two hands, for the heart
    modelComplexity: isLowPower() ? 0 : 1,
    minDetectionConfidence: 0.7,
    minTrackingConfidence: 0.5,
  });
  hands.onResults(onResults);
  return hands;
}

// Bumped on every start(). Startup is a chain of awaits and a phone can be
// backgrounded partway through any of them, which tears the stream down from
// visibilitychange; without this the half-finished run kept going against a
// stream that no longer existed and parked the app on an error nothing cleared.
let startGen = 0;

async function start() {
  const gen = ++startGen;

  errorEl.hidden = true;
  camSpinner.hidden = false;
  setStatus('', 'Starting…');
  diag.reset();

  // The camera is opened ONCE and handed straight to the video element.
  // An earlier version probed with getUserMedia, stopped the tracks, then let
  // MediaPipe's Camera helper open its own stream. Desktops reacquire
  // instantly; phones often hand back a stream that never emits a frame, which
  // looked like "nothing works" with no error to show for it.
  let opened;
  try {
    stopStream();
    opened = await openStream();
    if (gen !== startGen) { opened.getTracks().forEach((t) => t.stop()); return; }
    stream = opened;
    diag.note('stream', stream.getVideoTracks().map((t) => {
      const st = t.getSettings ? t.getSettings() : {};
      return `${st.width || '?'}x${st.height || '?'} ${st.facingMode || ''}`;
    }).join(', '));
  } catch (err) {
    if (gen !== startGen) return;
    diag.fail('getUserMedia', err);
    showError(describeCameraError(err));
    return;
  }

  video.srcObject = stream;
  video.setAttribute('playsinline', '');   // iOS refuses to play inline without it
  video.muted = true;
  try {
    await video.play();
  } catch (err) {
    diag.note('play', 'rejected: ' + (err && err.name));   // often plays anyway
  }
  if (gen !== startGen) return;

  try {
    await waitForVideo();
    diag.note('video', `${video.videoWidth}x${video.videoHeight}`);
  } catch (err) {
    if (gen !== startGen) return;
    diag.fail('video', err);
    showError('The camera opened but never sent a frame. Close any other app using '
            + 'the camera, then hit Retry.');
    return;
  }

  let detector;
  try {
    detector = buildHands();
  } catch (err) {
    diag.fail('hands', err);
    showError(err.message);
    return;
  }

  // Pull the WASM runtime and the landmark model down explicitly. They are
  // ~12 MB from the CDN and are otherwise fetched lazily inside the first
  // send(), where a slow or blocked mobile connection produced no error at
  // all: the status line said "Show me a hand", the camera spinner never
  // cleared, and no gesture could ever fire.
  setStatus('', 'Loading model…');
  diag.note('model', 'loading…');
  try {
    await withTimeout(detector.initialize(), MODEL_TIMEOUT_MS, 'model download');
    diag.note('model', 'ready');
  } catch (err) {
    diag.fail('model', err);
    showError('The hand-tracking model could not be downloaded (' + err.name + '). '
            + 'It is a ~12 MB one-time download from a CDN — check the connection, '
            + 'or any blocker that might be stopping cdn.jsdelivr.net, then hit Retry.',
              'Model failed', 'Hand tracking unavailable');
    return;
  }
  if (gen !== startGen) return;   // a newer start() took over while we waited

  // Drive frames ourselves rather than via MediaPipe's Camera helper, so a
  // stalled or throwing send() is visible instead of silently swallowed.
  let busy = false;
  let lastResults = diag.results;
  let lastProgress = performance.now();

  const pump = () => {
    rafId = requestAnimationFrame(pump);
    if (video.readyState < 2) return;

    // A send() that never settles used to wedge `busy` on forever and kill
    // the loop in silence. Time it out so one bad frame costs one frame.
    if (busy) return;

    const now = performance.now();
    if (diag.results !== lastResults) {
      lastResults = diag.results;
      lastProgress = now;
    } else if (now - lastProgress > STALL_MS) {
      cancelAnimationFrame(rafId);
      rafId = null;
      diag.lastError = 'no results for ' + Math.round((now - lastProgress) / 1000) + 's';
      showError('Hand tracking started but stopped responding. This usually means the '
              + 'browser could not keep the tracker running. Hit Retry, and close other '
              + 'tabs or apps using the camera.',
                'Tracking stalled', 'Hand tracking stopped');
      return;
    }

    busy = true;
    diag.sent++;
    withTimeout(Promise.resolve().then(() => detector.send({ image: video })),
                SEND_TIMEOUT_MS, 'frame')
      .catch((err) => { diag.sendErrors++; diag.lastError = String((err && err.message) || err); })
      .finally(() => { busy = false; });
  };
  rafId = requestAnimationFrame(pump);

  setStatus('ready', 'Show me a hand');
  hasMultipleCameras().then((multi) => { if (multi) flipBtn.hidden = false; });
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
// Phones background the page constantly — permission sheets, the notification
// shade, a lock — so the resume path has to be the one that always works.
// Restarting on "no live stream OR no running pump" rather than on `!stream`
// matters: a run torn down mid-startup could leave a non-null `stream` with
// nothing driving it, and the old guard then refused to restart it forever.
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    stopStream();
    return;
  }
  if (!errorEl.hidden) return;   // wait for Retry; the error is still on screen
  const live = stream && stream.getVideoTracks().some((t) => t.readyState === 'live');
  if (!live || rafId === null) start();
});

/* ---------------- go ---------------- */

window.__diag = diag;   // for debugging from the console

buildLegend();
preload();

// Tap the status badge to show the diagnostics readout.
document.querySelector('.stage-badge').addEventListener('click', () => {
  diag.enabled = !diag.enabled;
  diag.render();
});
diag.render();

start();
