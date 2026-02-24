/**
 * main.js — PiSlider Web Command Engine v2.4
 *
 * New in v2.4:
 * - Session persistence: reconnect restores full state from server
 * - Stop/Reset dual-mode button
 * - HG locks exposure controls when enabled
 * - Trigger mode: picam_motion_only / picam_motion_hybrid + ROI settings
 * - Sequence Progress: estimated end time, remaining time, live interval
 * - Removed Generate Plan button (runs automatically)
 * - init packet from server restores all UI fields on reconnect
 */

"use strict";

// ─── CONFIG ──────────────────────────────────────────────────────────────────
const WS_URL          = `ws://${window.location.host}/ws`;
const RECONNECT_DELAY = 2000;

// Quarter-stop shutter table (seconds)
const SHUTTER_VALS = buildQuarterStops([
    1/8000, 1/4000, 1/2000, 1/1000, 1/500, 1/250, 1/125,
    1/60,   1/30,   1/15,   1/8,    1/4,   1/2,
    1,      2,      4,      8,      15,    30
]);

// Quarter-stop ISO table
const ISO_VALS = buildQuarterStops([
    100, 200, 400, 800, 1600, 3200, 6400, 12800
]);

// ─── STATE ───────────────────────────────────────────────────────────────────
let socket               = null;
let joystickActive       = false;
let isRunning            = false;          // tracks sequence state — kept in sync via run_state msgs
let latestFrameInterval  = null;           // setInterval handle for latest-frame polling during a run
let currentMode          = 'timelapse';
let _loupeUserVisible    = true;           // controlled by 🔍 toggle button
let relay1On             = false;
let relay2On             = false;
let motionScripts        = [];
let folderBrowserCurrentPath = '/home/tim/Pictures';

// ─── DOM CACHE ───────────────────────────────────────────────────────────────
const els = {
    mjpegFeed:        document.getElementById('mjpegFeed'),
    latestFrame:      document.getElementById('latestFrame'),
    feedContainer:    document.getElementById('feedContainer'),
    statusLine:       document.getElementById('statusLine'),
    debugOverlay:     document.getElementById('debugOverlay'),
    shutterIndicator: document.getElementById('shutter_indicator'),

    joystickPad:  document.getElementById('joystickPad'),
    joystickKnob: document.getElementById('joystickKnob'),

    cameraSelect: document.getElementById('camera_select'),
    sonyGuide:    document.getElementById('sony_guide'),
    savePath:     document.getElementById('save_path'),

    // Exposure
    aeToggle:     document.getElementById('ae_toggle'),
    awbToggle:    document.getElementById('awb_toggle'),
    shutterSlider: document.getElementById('shutter_slider'),
    shutterLabel:  document.getElementById('shutter_label'),
    isoSlider:    document.getElementById('iso_slider'),
    isoLabel:     document.getElementById('iso_label'),
    wbSlider:     document.getElementById('wb_slider'),
    wbLabel:      document.getElementById('wb_label'),

    // Sequence
    totalFrames:  document.getElementById('total_frames'),
    vibeDelay:    document.getElementById('vibe_delay'),
    expMargin:    document.getElementById('exp_margin'),

    // Telemetry
    nodeReadout:  document.getElementById('nodeReadout'),
    curFrame:     document.getElementById('cur_f'),
    totFrame:     document.getElementById('tot_f'),
    progressMsg:  document.getElementById('progress_msg'),
    valS:         document.getElementById('val_s'),
    valP:         document.getElementById('val_p'),
    valT:         document.getElementById('val_t'),

    // HG telemetry
    hgPhase:   document.getElementById('hg_phase'),
    hgSunAlt:  document.getElementById('hg_sun_alt'),
    hgEV:      document.getElementById('hg_ev'),
    hgISO:     document.getElementById('hg_iso'),
    hgShutter: document.getElementById('hg_shutter'),
    hgKelvin:  document.getElementById('hg_kelvin'),

    // Motion script
    scriptSelect: document.getElementById('motion_script_select'),

    // Fan — auto-managed, no slider in UI
    fanSlider: null,
    fanPct:    null,

    // Macro panel
    macroPanel: document.getElementById('macro_panel'),
};

// ─── INIT ────────────────────────────────────────────────────────────────────
window.onload = () => {
    log("Interface bootstrapped. Initialising hardware link…");
    initSliders();
    connectWS();
    setupJoystick();
    setupSliderStrip();
    setupCanvasOverlay();
    setupMotionAxisListener();
    setupSeqMode();
    onCameraChange(els.cameraSelect?.value || 'picam');
    loadMotionScripts();
    updateHGExposureLock();
    onTriggerModeChange('normal', false);
    document.body.classList.add('idle');
    refreshDiskInfo();
    startLoupePolling();
    scanDrives();
    macroCalc();
    sendCmd('macro_load_lens_profiles');
    // Silently attempt GPS on load — updates HG lat/lon if browser permits
    grabGPS(true);
};

// ─── QUARTER-STOP MATH ───────────────────────────────────────────────────────
function buildQuarterStops(stops) {
    const out = [];
    for (let i = 0; i < stops.length - 1; i++) {
        const s = stops[i], e = stops[i + 1];
        for (let q = 0; q < 4; q++) {
            out.push(s * Math.pow(e / s, q / 4));
        }
    }
    out.push(stops[stops.length - 1]);
    return out;
}

function prettyShutter(s) {
    if (s >= 1) return `${Math.round(s * 10) / 10}s`;
    return `1/${Math.round(1 / s)}`;
}

function initSliders() {
    // Set initial slider positions to sensible defaults
    const shutterDefault = SHUTTER_VALS.findIndex(v => Math.abs(v - 1/125) < 0.0001);
    els.shutterSlider.max = SHUTTER_VALS.length - 1;
    els.shutterSlider.value = shutterDefault >= 0 ? shutterDefault : Math.floor(SHUTTER_VALS.length / 2);
    els.shutterLabel.innerText = prettyShutter(SHUTTER_VALS[els.shutterSlider.value]);

    const isoDefault = ISO_VALS.findIndex(v => Math.abs(v - 400) < 1);
    els.isoSlider.max = ISO_VALS.length - 1;
    els.isoSlider.value = isoDefault >= 0 ? isoDefault : Math.floor(ISO_VALS.length / 2);
    els.isoLabel.innerText = Math.round(ISO_VALS[els.isoSlider.value]);
}

// ─── SLIDER CALLBACKS ────────────────────────────────────────────────────────
function onShutterSlider(val) {
    const s = SHUTTER_VALS[parseInt(val)];
    els.shutterLabel.innerText = prettyShutter(s);
    // Touching shutter disables AE
    if (els.aeToggle.checked) {
        els.aeToggle.checked = false;
    }
    sendPicamSettings();
}

function onISOSlider(val) {
    const iso = Math.round(ISO_VALS[parseInt(val)]);
    els.isoLabel.innerText = iso;
    if (els.aeToggle.checked) {
        els.aeToggle.checked = false;
    }
    sendPicamSettings();
}

function onWBSlider(val) {
    els.wbLabel.innerText = `${val}K`;
    if (els.awbToggle.checked) {
        els.awbToggle.checked = false;
    }
    sendPicamSettings();
}

function sendPicamSettings() {
    const s = SHUTTER_VALS[parseInt(els.shutterSlider.value)];
    const iso = Math.round(ISO_VALS[parseInt(els.isoSlider.value)]);
    const kelvin = parseInt(els.wbSlider.value);
    sendCmd('set_picam_settings', {
        ae:     els.aeToggle.checked,
        awb:    els.awbToggle.checked,
        shutter_s: s,
        iso:    iso,
        kelvin: kelvin,
    });
}

// ─── HOLY GRAIL SETTINGS ─────────────────────────────────────────────────────
function sendHGSettings() {
    const basic = {
        enabled:   document.getElementById('hg_enabled').checked,
        lat:       parseFloat(document.getElementById('hg_lat').value),
        lon:       parseFloat(document.getElementById('hg_lon').value),
        tz:        document.getElementById('hg_tz').value.trim(),
        cam_az:    parseFloat(document.getElementById('hg_cam_az').value),
        cam_alt:   parseFloat(document.getElementById('hg_cam_alt').value),
        hfov:      parseFloat(document.getElementById('hg_hfov').value),
        vfov:      parseFloat(document.getElementById('hg_vfov').value),
        // Advanced
        ev_day:      parseFloat(document.getElementById('hg_ev_day').value),
        ev_golden:   parseFloat(document.getElementById('hg_ev_golden').value),
        ev_twilight: parseFloat(document.getElementById('hg_ev_twilight').value),
        ev_night:    parseFloat(document.getElementById('hg_ev_night').value),
        kelvin_day:      parseInt(document.getElementById('hg_k_day').value),
        kelvin_golden:   parseInt(document.getElementById('hg_k_golden').value),
        kelvin_twilight: parseInt(document.getElementById('hg_k_twilight').value),
        kelvin_night:    parseInt(document.getElementById('hg_k_night').value),
        interval_day:      parseFloat(document.getElementById('hg_int_day').value),
        interval_golden:   parseFloat(document.getElementById('hg_int_golden').value),
        interval_twilight: parseFloat(document.getElementById('hg_int_twilight').value),
        interval_night:    parseFloat(document.getElementById('hg_int_night').value),
        iso_min:     parseInt(document.getElementById('hg_iso_min').value),
        iso_max:     parseInt(document.getElementById('hg_iso_max').value),
        aperture_day:   parseFloat(document.getElementById('hg_ap_day').value),
        aperture_night: parseFloat(document.getElementById('hg_ap_night').value),
    };
    sendCmd('set_hg_settings', basic);
    log(`HG Settings applied. Enabled=${basic.enabled}, Lat=${basic.lat}`);
}

// ─── GPS AUTO-LOCATION ────────────────────────────────────────────────────────
async function grabGPS(silent = false) {
    // Use server-side location lookup (/api/gps) instead of browser geolocation.
    // navigator.geolocation is blocked by Chrome on non-HTTPS origins (http://pislider.local).
    // The Pi fetches its approximate location via ip-api.com — no browser HTTPS needed.
    const btn = document.getElementById('gpsBtn');
    if (btn) { btn.innerText = '⏳'; btn.style.color = 'var(--accent-gold)'; }
    try {
        const res  = await fetch('/api/gps');
        const data = await res.json();
        if (data.error && !data.lat) {
            if (!silent) log(`⚠ GPS lookup failed: ${data.error} — enter coordinates manually.`);
            if (btn) { btn.innerText = '📍'; btn.style.color = '#555'; }
            return;
        }
        const latEl = document.getElementById('hg_lat');
        const lonEl = document.getElementById('hg_lon');
        if (latEl) latEl.value = data.lat;
        if (lonEl) lonEl.value = data.lon;
        // Also update timezone if returned
        const tzEl = document.getElementById('hg_timezone');
        if (tzEl && data.timezone) tzEl.value = data.timezone;
        sendHGSettings();
        if (btn) { btn.innerText = '📍'; btn.style.color = 'var(--accent-green)'; }
        const city = data.city ? ` (${data.city})` : '';
        log(`📍 Location updated: ${data.lat}, ${data.lon}${city}`);
        setTimeout(() => { if (btn) btn.style.color = '#888'; }, 3000);
    } catch(e) {
        if (!silent) log(`⚠ GPS: ${e.message} — enter coordinates manually.`);
        if (btn) { btn.innerText = '📍'; btn.style.color = '#555'; }
    }
}
function setupMotionAxisListener() {
    document.querySelectorAll('input[name="motion_axis"]').forEach(radio => {
        radio.addEventListener('change', () => {
            const units = radio.value === 'slider' ? 'mm' : '°';
            document.getElementById('travel_units').innerText = units;
        });
    });
}

function getSelectedAxis() {
    const r = document.querySelector('input[name="motion_axis"]:checked');
    return r ? r.value : 'slider';
}

function runMotionTest() {
    const axis      = getSelectedAxis();
    const curve     = document.getElementById('motion_curve').value;
    const total     = parseFloat(document.getElementById('motion_total').value);
    const intervals = parseInt(document.getElementById('motion_intervals').value);
    if (isNaN(total) || isNaN(intervals) || intervals <= 0) {
        log("Motion: invalid parameters.");
        return;
    }
    sendCmd('run_motion_test', { axis, curve, total, intervals });
    log(`Motion test: ${axis}, ${curve}, ${total}${axis==='slider'?'mm':'°'}, ${intervals} steps`);
}

function homeAxis() {
    const axis = getSelectedAxis();
    sendCmd('home_axis', { axis });
    log(`Homing axis: ${axis}`);
}

// ─── RELAY CONTROL ───────────────────────────────────────────────────────────
function toggleRelay(n) {
    if (n === 1) {
        relay1On = !relay1On;
        document.getElementById('relay1_state').innerText = relay1On ? 'ON' : 'OFF';
        document.getElementById('relay1_btn').classList.toggle('relay-active', relay1On);
        sendCmd('set_relay', { relay: 1, on: relay1On });
    } else {
        relay2On = !relay2On;
        document.getElementById('relay2_state').innerText = relay2On ? 'ON' : 'OFF';
        document.getElementById('relay2_btn').classList.toggle('relay-active', relay2On);
        sendCmd('set_relay', { relay: 2, on: relay2On });
    }
}

// ─── FAN CONTROL ─────────────────────────────────────────────────────────────
function setFan(val) {
    if (els.fanPct) els.fanPct.innerText = val;
    sendCmd('set_fan', parseInt(val));
}

// ─── MOTION SCRIPTS ──────────────────────────────────────────────────────────
async function loadMotionScripts() {
    try {
        const resp = await fetch('/static/motion_scripts.json');
        const data = await resp.json();
        motionScripts = data.scripts || [];
        const sel = els.scriptSelect;
        motionScripts.forEach(s => {
            const opt = document.createElement('option');
            opt.value = s.id;
            opt.textContent = `${s.label} (${s.duration_s}s)`;
            sel.appendChild(opt);
        });
    } catch (e) {
        log(`Motion scripts not loaded: ${e}`);
    }
}

function loadMotionScript() {
    const id = els.scriptSelect.value;
    if (!id) return;
    sendCmd('load_motion_script', { script_id: id });
    log(`Motion script loaded: ${id}`);
}

// ─── COLLAPSIBLE PANELS ──────────────────────────────────────────────────────
function toggleAdvanced(id) {
    const panel = document.getElementById(id);
    const btn   = panel.previousElementSibling;
    const open  = panel.style.display !== 'none';
    panel.style.display = open ? 'none' : 'block';
    btn.textContent = open ? 'Advanced Settings ▼' : 'Advanced Settings ▲';
}

// ─── MODE SWITCHING ──────────────────────────────────────────────────────────
function setMode(mode) {
    currentMode = mode;
    ['timelapse','cinematic','macro'].forEach(m => {
        document.getElementById(`mode${m.charAt(0).toUpperCase()+m.slice(1)}`)
                .classList.toggle('active', m === mode);
    });

    const isMacro     = mode === 'macro';
    const isCinematic = mode === 'cinematic';

    // Panel visibility
    els.macroPanel.style.display = isMacro ? 'block' : 'none';
    const cp = document.getElementById('cinematic_panel');
    if (cp) cp.style.display = isCinematic ? 'block' : 'none';

    // Hide HG and trigger sections in macro and cinematic mode
    const hgSection      = document.getElementById('hg_section');
    const triggerSection = document.getElementById('trigger_section');
    if (hgSection)      hgSection.style.display      = (isMacro || isCinematic) ? 'none' : '';
    if (triggerSection) triggerSection.style.display  = (isMacro || isCinematic) ? 'none' : '';

    // Hide timelapse Start button in non-timelapse modes
    const startBtn = document.getElementById('startBtn');
    if (startBtn) startBtn.style.display = (isMacro || isCinematic) ? 'none' : '';

    // Enforce safe state when entering macro or cinematic mode
    if (isMacro || isCinematic) {
        const hgCb = document.getElementById('hg_enabled');
        if (hgCb && hgCb.checked) {
            hgCb.checked = false;
            sendHGSettings();
        }
        updateHGExposureLock();
        const normalRadio = document.querySelector('input[name="trigger_mode"][value="normal"]');
        if (isMacro && normalRadio && !normalRadio.checked) {
            normalRadio.checked = true;
            onTriggerModeChange('normal', true);
        }
    }

    // Switch feed container aspect ratio to match stream
    const container = document.getElementById('feedContainer');
    if (container) {
        container.style.aspectRatio = isCinematic ? '16 / 9' : '4 / 3';
    }

    // Always hide motion ROI box in cinematic and macro modes
    if (isCinematic || isMacro) {
        setRoiVisible(false);
    }

    // Load cinematic state on first switch
    if (isCinematic) {
        sendCmd('cinematic_get_state');
        sendCmd('cinematic_list_moves');
        // Stop inertia engine if running when switching away
    } else if (_cineInertiaRunning) {
        sendCmd('cinematic_live_stop');
        _cineInertiaRunning = false;
    }

    sendCmd('set_mode', { value: mode });
    log(`Mode: ${mode.toUpperCase()}`);
}

// ─── SEQUENCE MODE (frames vs start/end time) ─────────────────────────────────
function setupSeqMode() {
    // Seed datetime-local inputs with "now" and "now + 1hr"
    const now = new Date();
    const plus1 = new Date(now.getTime() + 3600000);
    const fmt = d => d.toISOString().slice(0,16);
    const st = document.getElementById('seq_start_time');
    const et = document.getElementById('seq_end_time');
    if (st) st.value = fmt(now);
    if (et) et.value = fmt(plus1);

    // Wire duration calc
    ['seq_start_time','seq_end_time','hg_int_dur'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.addEventListener('change', calcDurationFrames);
    });
    calcDurationFrames();
}

function onSeqModeChange(mode) {
    document.getElementById('seq_frames_row').style.display   = (mode === 'frames')   ? 'flex'  : 'none';
    document.getElementById('seq_duration_row').style.display = (mode === 'duration') ? 'block' : 'none';
    if (mode === 'duration') calcDurationFrames();
}

function calcDurationFrames() {
    const st  = document.getElementById('seq_start_time')?.value;
    const et  = document.getElementById('seq_end_time')?.value;
    const iv  = parseFloat(document.getElementById('hg_int_dur')?.value) || 5;
    const out = document.getElementById('seq_duration_calc');
    if (!st || !et || !out) return;
    const secs = (new Date(et) - new Date(st)) / 1000;
    if (secs <= 0) { out.innerText = 'End must be after Start'; return; }
    const frames = Math.floor(secs / iv);
    out.innerText = `≈ ${frames} frames  (${(secs/3600).toFixed(1)} hrs)`;
}

function getSeqConfig() {
    const seqMode = document.querySelector('input[name="seq_mode"]:checked')?.value || 'frames';
    const triggerMode = document.querySelector('input[name="trigger_mode"]:checked')?.value || 'normal';

    let total_frames, interval, save_path;
    save_path = document.getElementById('save_path')?.value || '/home/tim/Pictures/PiSlider';

    if (seqMode === 'frames') {
        // In frames mode, interval comes from HG (adaptive) or HG day interval as baseline
        interval     = parseFloat(document.getElementById('hg_int_day')?.value) || 5;
        total_frames = parseInt(document.getElementById('total_frames')?.value) || 300;
    } else {
        interval     = parseFloat(document.getElementById('hg_int_dur')?.value) || 5;
        const st  = new Date(document.getElementById('seq_start_time')?.value);
        const et  = new Date(document.getElementById('seq_end_time')?.value);
        const secs = (et - st) / 1000;
        total_frames = Math.max(1, Math.floor(secs / interval));
    }

    // For time-based mode, pass the schedule start time as ISO string
    let schedule_start = null;
    if (seqMode === 'duration') {
        const stVal = document.getElementById('seq_start_time')?.value;
        if (stVal) schedule_start = new Date(stVal).toISOString();
    }

    return {
        interval,
        total_frames,
        vibe_delay:     parseFloat(document.getElementById('vibe_delay')?.value) || 1.0,
        exp_margin:     parseFloat(document.getElementById('exp_margin')?.value) || 0.2,
        save_path,
        trigger_mode:   triggerMode,
        mode:           currentMode,
        schedule_start,   // null in frames mode, ISO string in time mode
    };
}

// ─── RUN STATE ────────────────────────────────────────────────────────────────
function setRunState(running) {
    isRunning = running;
    document.body.classList.toggle('running', running);
    document.body.classList.toggle('idle',    !running);

    const startBtn = document.getElementById('startBtn');
    const stopBtn  = document.getElementById('stopBtn');

    if (startBtn) {
        startBtn.innerText = running ? 'RUNNING…' : 'Start Sequence';
        startBtn.style.opacity      = running ? '0.4' : '1';
        startBtn.style.pointerEvents = running ? 'none' : 'auto';
    }

    // Stop button becomes E-Stop when running, Reset when idle
    if (stopBtn) {
        if (running) {
            stopBtn.innerText = 'E-Stop';
            stopBtn.classList.add('stop-style');
            stopBtn.classList.remove('reset-style');
        } else {
            stopBtn.innerText = 'Reset Session';
            stopBtn.classList.remove('stop-style');
            stopBtn.classList.add('reset-style');
        }
    }

    // Cinematic mode: keep the live feed running always (continuous video)
    const isCinematic = currentMode === 'cinematic';

    if (isCinematic) {
        // Always show live stream in cinematic — never switch to still frame
        if (els.mjpegFeed)   els.mjpegFeed.style.display  = 'block';
        if (els.latestFrame) els.latestFrame.style.display = 'none';
        const feedLabel = document.getElementById('feedLabel');
        if (feedLabel) feedLabel.innerText =
            'Optical Feed — Live (640×360) + Focus Loupe';
    } else {
        // Timelapse / macro: pause feed during sequence, show last frame
        if (els.mjpegFeed)   els.mjpegFeed.style.display   = running ? 'none'  : 'block';
        if (els.latestFrame) els.latestFrame.style.display  = running ? 'block' : 'none';
        const feedLabel = document.getElementById('feedLabel');
        if (feedLabel) feedLabel.innerText = running
            ? 'Last Captured Frame (live feed paused during sequence)'
            : 'Optical Feed — Framing (640×360) + Focus Loupe';
    }

    // Only hide loupe during non-cinematic running sequences
    if (!isCinematic) {
        setLoupeVisible(!running);
    }

    // Progress estimates panel
    const pe = document.getElementById('progress_estimates');
    if (pe) pe.style.display = running ? 'block' : 'none';

    if (els.progressMsg) {
        els.progressMsg.innerText = running ? 'Sequence running…' : 'Idle';
    }

    if (running && !isCinematic) {
        if (!latestFrameInterval) {
            latestFrameInterval = setInterval(() => {
                if (els.latestFrame) els.latestFrame.src = `/latest_frame?t=${Date.now()}`;
            }, 2000);
        }
        stopLoupePolling();
    } else {
        if (latestFrameInterval) { clearInterval(latestFrameInterval); latestFrameInterval = null; }
        if (!running || isCinematic) startLoupePolling();
    }

    // Status line
    if (running) {
        if (els.statusLine) {
            els.statusLine.innerText = 'STATUS: SEQUENCE RUNNING';
            els.statusLine.style.color = 'var(--accent-gold)';
        }
        updateHGExposureLock();
    } else {
        updateDiskSpace();
        if (els.statusLine) {
            els.statusLine.innerText = 'STATUS: IDLE';
            els.statusLine.style.color = 'var(--accent-green)';
        }
        if (els.mjpegFeed) els.mjpegFeed.src = `/video_feed?t=${Date.now()}`;
        updateHGExposureLock();
    }
}

// ─── STOP / RESET DUAL BUTTON ─────────────────────────────────────────────────
function handleStopReset() {
    if (isRunning) {
        // E-Stop — give immediate feedback, server confirms with run_state:false
        sendCmd('stop');
        log('E-Stop sent — sequence halting…');
        const stopBtn = document.getElementById('stopBtn');
        if (stopBtn) { stopBtn.innerText = 'STOPPING…'; stopBtn.style.opacity = '0.5'; }
    } else {
        // Full server restart
        if (confirm('Restart the server?\n\nThis will:\n• Stop all motors & release relays\n• Fully restart the server process\n• Restore default settings\n• Clear calibration, HG settings and keyframes\n\nThe page will reconnect automatically.')) {
            sendCmd('reset_session');
            log('Server restart requested — reconnecting…');
            // Immediately reset frame counter in UI — server will confirm on reconnect
            const cf = document.getElementById('curFrame');
            const tf = document.getElementById('totalFrames');
            if (cf) cf.innerText = '000';
            if (tf) tf.innerText = '000';
        }
    }
}

// ─── HG EXPOSURE LOCK ─────────────────────────────────────────────────────────
function updateHGExposureLock() {
    const hgEnabled = document.getElementById('hg_enabled')?.checked ?? false;
    const notice    = document.getElementById('hg_override_notice');
    const controls  = document.getElementById('remote_controls');

    if (hgEnabled) {
        if (notice)   notice.style.display   = 'block';
        if (controls) { controls.style.opacity = '0.35'; controls.style.pointerEvents = 'none'; }
    } else {
        if (notice)   notice.style.display   = 'none';
        if (controls) { controls.style.opacity = '1';    controls.style.pointerEvents = 'auto';  }
    }
}

// ─── TRIGGER MODE CHANGE ─────────────────────────────────────────────────────
function onTriggerModeChange(mode, sendToServer = true) {
    const motionSettings = document.getElementById('motion_settings');
    const auxFireBtn     = document.getElementById('auxFireBtn');
    const isMotion  = mode.startsWith('picam_motion');
    const isAux     = mode.startsWith('aux');

    if (motionSettings) motionSettings.style.display = isMotion ? 'block' : 'none';
    if (auxFireBtn)     auxFireBtn.style.display      = isAux    ? 'block' : 'none';

    // Show/hide the canvas ROI box
    setRoiVisible(isMotion);

    // If motion mode active and camera is not picam, show a note
    const noteEl = document.getElementById('motion_camera_note');
    if (noteEl) {
        const cam = document.getElementById('camera_select')?.value;
        noteEl.style.display = (isMotion && cam !== 'picam') ? 'block' : 'none';
    }

    if (sendToServer) sendCmd('set_trigger_mode', { mode });
}

// ─── MOTION DETECTION SETTINGS ───────────────────────────────────────────────
function sendMotionSettings() {
    const roi = overlay.roi;
    sendCmd('set_motion_roi', {
        roi:       [roi.x1, roi.y1, roi.x2, roi.y2],
        threshold: parseInt(document.getElementById('motion_threshold')?.value ?? 5000),
        warmup:    parseInt(document.getElementById('motion_warmup')?.value    ?? 10),
    });
}

// ─── PROGRESS ESTIMATES ───────────────────────────────────────────────────────
function updateProgressEstimates(data) {
    const cur = parseInt(els.curFrame.innerText) || 0;
    const tot = parseInt(els.totFrame.innerText) || 1;
    const pct = tot > 0 ? Math.round(cur / tot * 100) : 0;

    if (els.progressMsg) {
        els.progressMsg.innerText = isRunning
            ? `${pct}% — ${cur} of ${tot} frames`
            : 'Idle';
    }

    // Live interval + estimated end
    if (data.current_interval !== undefined) {
        const ci = document.getElementById('cur_interval');
        if (ci) ci.innerText = data.current_interval;

        const hn = document.getElementById('hg_interval_note');
        if (hn) {
            const hgOn = document.getElementById('hg_enabled')?.checked;
            hn.style.display = hgOn ? 'inline' : 'none';
        }
    }
    if (data.estimated_end) {
        const ee = document.getElementById('est_end');
        if (ee) ee.innerText = data.estimated_end;
    }
    if (data.secs_remaining !== undefined) {
        const er = document.getElementById('est_remaining');
        if (er) {
            const m = Math.floor(data.secs_remaining / 60);
            const s = data.secs_remaining % 60;
            er.innerText = `${m}m ${s}s`;
        }
    }
}

// ─── HG FIELD RESTORE (from init packet) ─────────────────────────────────────
function restoreHGFields(hg) {
    const set = (id, val) => { const el = document.getElementById(id); if (el && val !== undefined) el.value = val; };
    const check = (id, val) => { const el = document.getElementById(id); if (el && val !== undefined) el.checked = val; };

    check('hg_enabled',     hg.enabled);
    set('hg_lat',           hg.lat);
    set('hg_lon',           hg.lon);
    set('hg_tz',            hg.tz);
    set('hg_cam_az',        hg.cam_az);
    set('hg_cam_alt',       hg.cam_alt);
    set('hg_hfov',          hg.hfov);
    set('hg_vfov',          hg.vfov);
    set('hg_ev_day',        hg.ev_day);
    set('hg_ev_golden',     hg.ev_golden);
    set('hg_ev_twilight',   hg.ev_twilight);
    set('hg_ev_night',      hg.ev_night);
    set('hg_kelvin_day',    hg.kelvin_day);
    set('hg_kelvin_golden', hg.kelvin_golden);
    set('hg_kelvin_twilight', hg.kelvin_twilight);
    set('hg_kelvin_night',  hg.kelvin_night);
    set('hg_int_day',       hg.interval_day);
    set('hg_int_golden',    hg.interval_golden);
    set('hg_int_twilight',  hg.interval_twilight);
    set('hg_int_night',     hg.interval_night);
    set('hg_iso_min',       hg.iso_min);
    set('hg_iso_max',       hg.iso_max);
    set('hg_aperture_day',  hg.aperture_day);
    set('hg_aperture_night',hg.aperture_night);

    updateHGExposureLock();
}

// ─── SEQUENCE START ──────────────────────────────────────────────────────────
function startRun() {
    if (isRunning) return;

    const config = getSeqConfig();

    // If WebSocket isn't open yet, wait up to 5 s for it to connect then fire.
    if (!socket || socket.readyState !== WebSocket.OPEN) {
        log("⏳ Waiting for WebSocket link before starting…");
        const deadline = Date.now() + 5000;
        const poll = setInterval(() => {
            if (socket && socket.readyState === WebSocket.OPEN) {
                clearInterval(poll);
                _doStartRun(config);
            } else if (Date.now() > deadline) {
                clearInterval(poll);
                log("⚠ Start failed — server link could not be established. Check Pi connection.");
            }
        }, 100);
        return;
    }

    _doStartRun(config);
}

function _doStartRun(config) {
    if (isRunning) return;   // guard against double-fire if somehow called twice
    log(`Start: ${config.mode.toUpperCase()} — ${config.total_frames} frames @ ${config.interval}s [${config.trigger_mode}]`);
    // Push latest HG settings to backend before starting — no manual Apply button needed
    if (document.getElementById('hg_enabled')?.checked) {
        sendHGSettings();
    }
    // Optimistically update UI so the button doesn't feel dead.
    setRunState(true);
    sendCmd('start_run', config);
}

// ─── SOFT LIMITS ─────────────────────────────────────────────────────────────
function captureLimitNow(axis, which) {
    // Tell backend to record current motor position as limit
    sendCmd('set_limits', { axis, which, value: null });
    log(`Limit captured: ${axis} ${which} = current position`);
}

function setLimitFromField(axis, which) {
    const id  = `${axis}_${which}_val`;
    const val = parseFloat(document.getElementById(id)?.value);
    if (isNaN(val)) return;
    sendCmd('set_limits', { axis, which, value: val });
    log(`Limit set: ${axis} ${which} = ${val}°`);
}

function updateLimitsReadout(data) {
    const el = document.getElementById('limits_readout');
    if (!el) return;
    const pMin = data.pan_min  ?? '—';
    const pMax = data.pan_max  ?? '—';
    const tMin = data.tilt_min ?? '—';
    const tMax = data.tilt_max ?? '—';
    el.innerText = `Pan: ${pMin}° → ${pMax}°   |   Tilt: ${tMin}° → ${tMax}°`;
    // Sync fields
    if (data.pan_min  !== undefined) { const f = document.getElementById('pan_min_val');  if(f) f.value = data.pan_min; }
    if (data.pan_max  !== undefined) { const f = document.getElementById('pan_max_val');  if(f) f.value = data.pan_max; }
    if (data.tilt_min !== undefined) { const f = document.getElementById('tilt_min_val'); if(f) f.value = data.tilt_min; }
    if (data.tilt_max !== undefined) { const f = document.getElementById('tilt_max_val'); if(f) f.value = data.tilt_max; }
}


// ─── DISK INFO + ALERTS ──────────────────────────────────────────────────────
async function refreshDiskInfo() {
    try {
        const resp = await fetch('/disk_info');
        const data = await resp.json();
        if (data.error) return;

        const freeGB  = data.free  / 1073741824;
        const totalGB = data.total / 1073741824;
        const pct     = data.free  / data.total;   // fraction free

        // Estimated frames remaining
        const cam = document.getElementById('camera_select')?.value || 'picam';
        const mbPerFrame = cam === 'sony' ? 30 : 25;
        const framesLeft = Math.floor(data.free / (mbPerFrame * 1048576));

        const label = document.getElementById('disk_free_label');
        if (label) {
            label.className = pct < 0.05 ? 'disk-crit' : pct < 0.15 ? 'disk-warn' : 'disk-ok';
            label.innerHTML = `${freeGB.toFixed(1)} GB free &nbsp;(~<b>${framesLeft.toLocaleString()}</b> frames)`;
        }

        // Pre-flight inline warning below frame count
        const total  = parseInt(document.getElementById('total_frames')?.value) || 0;
        const warnEl = document.getElementById('disk_preflight_warn');
        if (warnEl) {
            const show = total > 0 && framesLeft < total;
            warnEl.style.display = show ? 'block' : 'none';
            if (show) warnEl.textContent =
                `⚠ Only ~${framesLeft} frames fit — ${total}-frame sequence may fail mid-run.`;
        }
    } catch(_) {}
}

// Auto-refresh disk info every 15 s
setInterval(refreshDiskInfo, 15000);

function showDiskFullAlert(msg) {
    document.getElementById('diskAlertBanner')?.remove();
    document.getElementById('diskWarnBanner')?.remove();
    const banner = document.createElement('div');
    banner.id = 'diskAlertBanner';
    banner.className = 'disk-alert-banner';
    banner.innerHTML = `<strong>⛔ DISK FULL — SEQUENCE HALTED</strong><br>
        <span style="font-size:0.8rem">${msg}</span>
        <button onclick="this.parentElement.remove()">Dismiss</button>`;
    document.body.appendChild(banner);
    log(msg);
    setRunState(false);
    refreshDiskInfo();
}

function showDiskWarnAlert(msg) {
    if (document.getElementById('diskAlertBanner')) return;
    document.getElementById('diskWarnBanner')?.remove();
    const banner = document.createElement('div');
    banner.id = 'diskWarnBanner';
    banner.className = 'disk-alert-banner';
    banner.style.cssText = banner.style.cssText +
        ';border-color:var(--accent-gold);color:var(--accent-gold);background:#1a1200;';
    banner.innerHTML = `<strong>⚠ LOW DISK SPACE</strong><br>
        <span style="font-size:0.8rem">${msg}</span>
        <button onclick="this.parentElement.remove()" style="border-color:var(--accent-gold)">Dismiss</button>`;
    document.body.appendChild(banner);
    log(msg);
    setTimeout(() => document.getElementById('diskWarnBanner')?.remove(), 30000);
}

// ─── HIGH-RES LOUPE POLLING ───────────────────────────────────────────────────
// Polls /loupe_crop at 2fps. The returned JPEG is used by drawOverlay()
// instead of sampling from the low-res MJPEG stream.
const loupeCropImage = new Image();  // shared Image object reused each poll
let _loupePollTimer  = null;

function startLoupePolling() {
    if (_loupePollTimer) return;
    _loupePollTimer = setInterval(async () => {
        if (!overlay?.loupe?.visible) return;
        const l  = overlay.loupe;
        const cw = overlay.cw || overlay.canvas?.clientWidth || 640;
        const ch = overlay.ch || overlay.canvas?.clientHeight || 480;

        const cx = l.x.toFixed(3);
        const cy = l.y.toFixed(3);

        // Container is exactly 4:3 — image fills full width with no letterbox bars.
        // Crop radius in frame-fraction = loupe_radius_px / canvas_width / zoom
        // e.g. r=180px, cw=960px, zoom=4 → rFrac = 0.047 → tight 4× zoom crop
        const rFrac = (l.r / cw / l.zoom).toFixed(4);

        try {
            const url = `/loupe_crop?cx=${cx}&cy=${cy}&r=${rFrac}&t=${Date.now()}`;
            const resp = await fetch(url);
            if (!resp.ok) return;
            const blob = await resp.blob();
            const objUrl = URL.createObjectURL(blob);
            loupeCropImage.onload = () => URL.revokeObjectURL(loupeCropImage._prevUrl);
            loupeCropImage._prevUrl = objUrl;
            loupeCropImage.src = objUrl;
        } catch (_) {}
    }, 500);   // 2fps — enough for focus checking
}

function stopLoupePolling() {
    clearInterval(_loupePollTimer);
    _loupePollTimer = null;
}

// ─── CAMERA SWITCHING ────────────────────────────────────────────────────────
function onCameraChange(val) {
    els.sonyGuide.style.display = (val === 'sony') ? 'block' : 'none';
    updatePreviewToggleVisibility();
    // Cache-bust the MJPEG stream
    els.mjpegFeed.src = `/video_feed?t=${Date.now()}`;
    if (val === 'sony') {
        log("wlan1: Triggering headless handshake…");
        sendCmd('connect_camera_wifi');
    } else if (val === 's2') {
        log("S2 mode: manual control on camera body.");
    } else {
        log("PiCam mode: full remote control active.");
    }
    sendCmd('set_camera', val);
}

// ─── WEBSOCKET ENGINE ────────────────────────────────────────────────────────
function connectWS() {
    // Close any existing broken socket cleanly before creating a new one
    if (socket && socket.readyState !== WebSocket.CLOSED) {
        socket.onclose = null;   // prevent recursive reconnect from old socket
        socket.close();
    }

    socket = new WebSocket(WS_URL);

    socket.onopen = () => {
        log("WebSocket: High-speed link ACTIVE.");
        els.statusLine.innerText = "STATUS: CONNECTED";
        els.statusLine.style.color = "var(--accent-green)";
    };

    socket.onmessage = (event) => {
        try {
            handleIncomingData(JSON.parse(event.data));
        } catch (e) {
            console.warn("WS parse error:", e);
        }
    };

    socket.onclose = () => {
        els.statusLine.innerText = "STATUS: LINK SEVERED — RECONNECTING…";
        els.statusLine.style.color = "var(--stop-red)";
        if (isRunning) {
            // Do NOT call setRunState(false) — the server sequence keeps running.
            // The init packet on reconnect will restore the correct state.
            log("⚠ Connection lost. Server sequence likely still running. Reconnecting…");
        }
        setTimeout(connectWS, RECONNECT_DELAY);
    };

    socket.onerror = (err) => {
        console.error("WebSocket error:", err);
        // onerror always fires before onclose — onclose will handle reconnect
    };
}

function sendCmd(command, value = null) {
    if (socket && socket.readyState === WebSocket.OPEN) {
        const payload = (value !== null && typeof value === 'object')
            ? { command, ...value }
            : { command, value };
        socket.send(JSON.stringify(payload));
    }
}

// ─── INCOMING DATA ROUTER ────────────────────────────────────────────────────
function handleIncomingData(data) {

    // ── Kicked by newer client ────────────────────────────────────────────────
    if (data.type === "kicked") {
        log("⚠ " + data.msg);
        // Show a prominent overlay so the user knows this tab is dead
        _showKickedBanner(data.msg);
        // Stop all polling — this tab is no longer in control
        stopLoupePolling();
        if (latestFrameInterval) { clearInterval(latestFrameInterval); latestFrameInterval = null; }
        // Prevent the onclose reconnect loop from spinning up again
        if (socket) { socket.onclose = null; socket.onerror = null; }
        return;
    }

    // ── Full state restore on connect / reset ─────────────────────────────────
    if (data.type === "init") {
        setRunState(data.running || false);

        // Frame counters
        if (data.current_frame !== undefined) els.curFrame.innerText = String(data.current_frame).padStart(3,'0');
        if (data.total_frames  !== undefined) {
            els.totFrame.innerText = data.total_frames;
            const tf = document.getElementById('total_frames');
            if (tf) tf.value = data.total_frames;
        }

        // Axis positions
        if (data.pan_deg    !== undefined) { els.valP.innerText = data.pan_deg.toFixed(1);    calibState.pan_deg  = data.pan_deg;  }
        if (data.tilt_deg   !== undefined) { els.valT.innerText = data.tilt_deg.toFixed(1);   calibState.tilt_deg = data.tilt_deg; }
        if (data.slider_mm  !== undefined)   els.valS.innerText = data.slider_mm.toFixed(1);

        // Save path
        if (data.save_path) {
            const sp = document.getElementById('save_path');
            if (sp) sp.value = data.save_path;
        }

        // Trigger mode
        if (data.trigger_mode) {
            const r = document.querySelector(`input[name="trigger_mode"][value="${data.trigger_mode}"]`);
            if (r) { r.checked = true; onTriggerModeChange(data.trigger_mode, false); }
        }

        // Motion ROI
        if (data.motion_roi) {
            setRoiFromData(data.motion_roi);
        }
        if (data.motion_threshold !== undefined) {
            const el = document.getElementById('motion_threshold');
            if (el) el.value = data.motion_threshold;
        }

        // HG settings
        if (data.hg_settings) restoreHGFields(data.hg_settings);

        // Active camera
        if (data.active_camera) {
            const cs = document.getElementById('camera_select');
            if (cs) { cs.value = data.active_camera; onCameraChange(data.active_camera); }
        }

        // Camera orientation
        if (data.camera_orientation) {
            _applyOrientationUI(data.camera_orientation);
        }

        // Cinematic fps
        if (data.cine_fps) {
            _applyCineFpsUI(data.cine_fps);
        }

        // Vibe / exp margin
        if (data.vibe_delay !== undefined) { const el=document.getElementById('vibe_delay'); if(el) el.value=data.vibe_delay; }
        if (data.exp_margin !== undefined) { const el=document.getElementById('exp_margin'); if(el) el.value=data.exp_margin; }

        // Limits
        updateLimitsReadout(data);
        updateCalibReadout();

        if (data.running) {
            log(`Reconnected — sequence IN PROGRESS (frame ${data.current_frame} of ${data.total_frames})`);
            // Status line stays green/running — setRunState above handles it
        } else if (data.interrupted) {
            // Server was restarted mid-sequence (process kill, power blip, etc.)
            // Show in log only — don't overwrite status line with alarming text
            log(`⚠ Server was restarted. Previous sequence stopped at frame ${data.current_frame}. Settings restored. Ready to start a new run.`);
            const statusEl = document.getElementById('statusLine');
            if (statusEl) {
                statusEl.innerText = `STATUS: READY (restarted at frame ${data.current_frame})`;
                statusEl.style.color = 'var(--accent-gold)';
            }
        } else {
            log('Connected — system idle.');
        }
    }

    // ── Run state change ──────────────────────────────────────────────────────
    if (data.type === "run_state") {
        setRunState(data.running);
    }

    // ── Status / telemetry ────────────────────────────────────────────────────
    if (data.type === "status") {
        if (data.nodes !== undefined)
            els.nodeReadout?.innerText && (els.nodeReadout.innerText = `Nodes: ${data.nodes}`);

        if (data.frame !== undefined)
            els.curFrame.innerText = String(data.frame).padStart(3, '0');
        if (data.total !== undefined)
            els.totFrame.innerText = data.total;

        if (data.pos_s !== undefined) els.valS.innerText = data.pos_s.toFixed(1);
        if (data.pos_p !== undefined) {
            els.valP.innerText  = data.pos_p.toFixed(1);
            calibState.pan_deg  = data.pos_p;
            updateCalibReadout();
        }
        if (data.pos_t !== undefined) {
            els.valT.innerText  = data.pos_t.toFixed(1);
            calibState.tilt_deg = data.pos_t;
        }

        // Update cinematic limit position bars with live position
        if (data.pos_s !== undefined && data.pos_p !== undefined && data.pos_t !== undefined) {
            _updateLimitBars(data.pos_s, data.pos_p, data.pos_t);
        }

        // HG telemetry
        if (data.hg_phase   !== undefined) els.hgPhase.innerText   = data.hg_phase;
        if (data.hg_sun_alt !== undefined) els.hgSunAlt.innerText  = parseFloat(data.hg_sun_alt).toFixed(1);
        if (data.hg_ev      !== undefined) els.hgEV.innerText      = parseFloat(data.hg_ev).toFixed(2);
        if (data.hg_iso     !== undefined) els.hgISO.innerText     = data.hg_iso;
        if (data.hg_shutter !== undefined) els.hgShutter.innerText = data.hg_shutter;
        if (data.hg_kelvin  !== undefined) els.hgKelvin.innerText  = data.hg_kelvin;

        // Sequence progress estimates
        updateProgressEstimates(data);
    }

    // ── Shutter flash + latest frame ─────────────────────────────────────────
    if (data.type === "shutter_event") {
        if (els.shutterIndicator) {
            els.shutterIndicator.innerText = "FIRING…";
            els.shutterIndicator.style.color = "var(--accent-red)";
        }
        if (isRunning && els.latestFrame)
            els.latestFrame.src = `/latest_frame?t=${Date.now()}`;
        setTimeout(() => {
            if (els.shutterIndicator) {
                els.shutterIndicator.innerText = "READY";
                els.shutterIndicator.style.color = "var(--text-dim)";
            }
        }, 600);
    }

    if (data.type === "sony_status")    updateSonyStatus(data);
    if (data.type === "limits_updated") updateLimitsReadout(data);
    if (data.type === "log")            log(data.msg);
    if (data.type === "disk_full")      showDiskFullAlert(data.msg);
    if (data.type === "disk_warn")      showDiskWarnAlert(data.msg);
    if (data.type === "folder_created") { browseTo(data.path); log(`Folder created: ${data.path}`); }
    if (data.type === "preview_camera_changed") {
        const btn = document.getElementById('previewToggleBtn');
        if (btn) btn.textContent = data.camera === 'sony' ? '🔁 Sony' : '📷 PiCam';
    }

    if (data.type === "disk_full") {
        handleDiskFull(data);
        setRunState(false);
    }

    if (data.type === "disk_warn")       handleDiskWarn(data);
    if (data.type === "folder_created")  handleFolderCreated(data);

    // ── Macro mode messages ───────────────────────────────────────────────────
    if (data.type === "macro_rail_mark")       { handleMacroRailMark(data);      return; }
    if (data.type === "macro_rotation_mark")   { handleMacroRotMark(data);       return; }
    if (data.type === "macro_aux_mark")        { handleMacroAuxMark(data);       return; }
    if (data.type === "macro_progress")        { handleMacroProgress(data);      return; }
    if (data.type === "macro_stack_complete")  { handleMacroStackComplete(data); return; }
    if (data.type === "macro_done")            { handleMacroDone(data);          return; }
    if (data.type === "macro_lens_profiles")   { handleMacroLensProfiles(data);  return; }

    // ── Cinematic ──────────────────────────────────────────────────────────
    if (data.type === "camera_orientation")    { handleCameraOrientation(data);    return; }
    if (data.type === "cinematic_limits")      { handleCineLimits(data);         return; }
    if (data.type === "cinematic_keyframes")   { handleCineKeyframes(data);      return; }
    if (data.type === "cinematic_keyframe_added") {
        _cineKeyframes.push(data); _renderKeyframeList();                        return; }
    if (data.type === "cinematic_progress")    { handleCineProgress(data);       return; }
    if (data.type === "cinematic_play_done")   { handleCinePlayDone();           return; }
    if (data.type === "cinematic_origin_set")  { handleCineOriginSet(data);      return; }
    if (data.type === "cinematic_moves")       { handleCineMoves(data);          return; }
    if (data.type === "cinematic_state")       { handleCineState(data);          return; }
    if (data.type === "cinematic_inertia")     {
        const m = document.getElementById('cine_mass');
        const d = document.getElementById('cine_drag');
        if (m) { m.value = data.mass; updateInertiaLabel('mass'); }
        if (d) { d.value = data.drag; updateInertiaLabel('drag'); }
                                                                                 return; }
    if (data.type === "arctan_status")         { handleArctanStatus(data);       return; }
    if (data.type === "arctan_enabled")        { handleArctanEnabled(data);      return; }
    if (data.type === "record_state")          { handleRecordState(data);        return; }
    if (data.type === "gamepad_btn")           { handleGamepadBtn(data);         return; }
    if (data.type === "gamepad_status")        { handleGamepadStatus(data);      return; }
    if (data.type === "cinematic_status")      { log(data.msg);                  return; }

    if (data.type === "preview_camera_changed") {
        const btn = document.getElementById('previewToggleBtn');
        if (btn) btn.innerText = data.camera === 'sony' ? '📷 Sony' : '📷 PiCam';
        if (els.mjpegFeed) els.mjpegFeed.src = `/video_feed?t=${Date.now()}`;
    }

    if (data.type === "calibration_done") {
        calibState.calibrated = true;
        calibState.origin_az  = data.origin_az;
        updateCalibReadout();
        const f1 = document.getElementById('hg_cam_az');
        const f2 = document.getElementById('hg_cam_alt');
        if (f1) f1.value = data.cam_az.toFixed(1);
        if (f2) f2.value = data.cam_alt.toFixed(1);
    }
}

// ─── JOYSTICK ENGINE ────────────────────────────────────────────────────────
let sliderStripActive = false;
let sliderVz = 0;

function setupJoystick() {
    const handleMove = (e) => {
        if (!joystickActive) return;
        const rect    = els.joystickPad.getBoundingClientRect();
        const cx      = rect.left + rect.width  / 2;
        const cy      = rect.top  + rect.height / 2;
        const maxR    = rect.width / 2;
        let dx = e.clientX - cx;
        let dy = e.clientY - cy;
        const dist    = Math.sqrt(dx*dx + dy*dy);
        if (dist > maxR) { dx *= maxR/dist; dy *= maxR/dist; }
        const vx = dx / maxR;
        const vy = -(dy / maxR);
        els.joystickKnob.style.left = `calc(50% + ${dx}px)`;
        els.joystickKnob.style.top  = `calc(50% + ${dy}px)`;
        sendCmd('joystick', { vx, vy, vz: sliderVz });
    };

    els.joystickPad.addEventListener('pointerdown', (e) => {
        joystickActive = true;
        els.joystickPad.setPointerCapture(e.pointerId);
        handleMove(e);
    });

    window.addEventListener('pointermove', handleMove);

    window.addEventListener('pointerup', () => {
        joystickActive = false;
        els.joystickKnob.style.left = '50%';
        els.joystickKnob.style.top  = '50%';
        if (!sliderStripActive) sendCmd('joystick', { vx: 0, vy: 0, vz: 0 });
    });
}

// ─── SLIDER STRIP — see full implementation below ────────────────────────────

// ─── SONY WIFI CONNECTION ────────────────────────────────────────────────────
function connectSony() {
    const ssid     = document.getElementById('sony_ssid').value.trim();
    const password = document.getElementById('sony_password').value;
    const ip       = document.getElementById('sony_ip').value.trim();
    const statusEl = document.getElementById('sony_status');

    if (!ssid || !password) {
        statusEl.innerText = "ERROR — SSID and password required.";
        statusEl.style.color = "var(--accent-red)";
        return;
    }

    statusEl.innerText = "CONNECTING… joining " + ssid;
    statusEl.style.color = "var(--accent-gold)";

    sendCmd('connect_sony_wifi', { ssid, password, ip });
    log(`Sony: initiating connection to ${ssid} → ${ip}`);
}

function disconnectSony() {
    sendCmd('disconnect_sony_wifi');
    const statusEl = document.getElementById('sony_status');
    statusEl.innerText = "DISCONNECTED";
    statusEl.style.color = "var(--text-dim)";
    log("Sony: WiFi dropped.");
}

// Called from WS handler when backend reports Sony status
function updateSonyStatus(status) {
    const statusEl = document.getElementById('sony_status');
    if (!statusEl) return;
    if (status.connected) {
        statusEl.innerText = `CONNECTED — ${status.ip}  Model: ${status.model || '?'}`;
        statusEl.style.color = "var(--accent-green)";
    } else if (status.error) {
        statusEl.innerText = `FAILED — ${status.error}`;
        statusEl.style.color = "var(--accent-red)";
    } else {
        statusEl.innerText = status.msg || "—";
        statusEl.style.color = "var(--text-dim)";
    }
}

// ─── COMPASS CALIBRATION ────────────────────────────────────────────────────
// Track current rig position in motor-odometer space
const calibState = {
    pan_deg:  0.0,  // current pan in degrees (relative to motor zero)
    tilt_deg: 0.0,  // current tilt in degrees
    origin_az: null, // real-world bearing of motor-zero after calibration
    calibrated: false,
};

// Show/hide custom bearing input
document.addEventListener('DOMContentLoaded', () => {
    const bearingSelect = document.getElementById('calib_bearing');
    if (bearingSelect) {
        bearingSelect.addEventListener('change', () => {
            const wrap = document.getElementById('calib_custom_wrap');
            if (wrap) wrap.style.display = bearingSelect.value === 'custom' ? 'block' : 'none';
        });
    }
});

function getSelectedBearing() {
    const sel = document.getElementById('calib_bearing');
    if (!sel) return 90;
    if (sel.value === 'custom') {
        return parseFloat(document.getElementById('calib_custom_deg').value) || 0;
    }
    return parseFloat(sel.value);
}

function nudgeAxis(axis, deg) {
    // Send incremental move to backend — backend executes it and returns new odometer position
    sendCmd('nudge_axis', { axis, deg });
    // Optimistically update local display
    if (axis === 'pan') {
        calibState.pan_deg  = Math.round((calibState.pan_deg  + deg) * 10) / 10;
    } else {
        calibState.tilt_deg = Math.round((calibState.tilt_deg + deg) * 10) / 10;
    }
    updateCalibReadout();
}

function calibrateOrigin() {
    const bearing = getSelectedBearing();
    // Tell backend: "current motor position = this real-world bearing"
    sendCmd('calibrate_origin', {
        bearing_deg: bearing,
        current_pan_deg:  calibState.pan_deg,
        current_tilt_deg: calibState.tilt_deg,
    });
    calibState.origin_az = bearing;
    calibState.calibrated = true;

    // Update HG cam_az field to match
    const camAzEl = document.getElementById('hg_cam_az');
    if (camAzEl) camAzEl.value = bearing;

    updateCalibReadout();
    log(`Calibration set: motor-zero → ${bearing}° (${getBearingName(bearing)}). HG az updated.`);
}

function updateCalibReadout() {
    const el = document.getElementById('calib_readout');
    if (!el) return;
    if (calibState.calibrated) {
        const worldPan = ((calibState.origin_az + calibState.pan_deg) % 360 + 360) % 360;
        el.innerText = `CALIBRATED ✓  |  Pan: ${calibState.pan_deg.toFixed(1)}°  Tilt: ${calibState.tilt_deg.toFixed(1)}°  |  World Az: ${worldPan.toFixed(1)}°`;
        el.style.color = "var(--accent-green)";
    } else {
        el.innerText = `NOT CALIBRATED  |  Pan: ${calibState.pan_deg.toFixed(1)}°  Tilt: ${calibState.tilt_deg.toFixed(1)}°`;
        el.style.color = "var(--text-dim)";
    }
}

function getBearingName(deg) {
    const names = { 0:'N', 45:'NE', 90:'E', 135:'SE', 180:'S', 225:'SW', 270:'W', 315:'NW' };
    return names[deg] || `${deg}°`;
}


function log(msg) {
    const ts = new Date().toLocaleTimeString();
    els.debugOverlay.innerText = `[${ts}] ${msg}`;
    console.log(`[PiSlider] ${msg}`);
}


// ─── CANVAS OVERLAY ENGINE ──────────────────────────────────────────────────
// Handles two overlays drawn on a single canvas:
//   1. LOUPE: draggable circle with 4× magnified crop from the live feed
//   2. MOTION ROI: draggable bounding box for motion detection zone

const overlay = {
    canvas:  null,
    ctx:     null,
    feed:    null,       // the <img> element we sample pixels from
    cw: 0, ch: 0,        // canvas pixel dimensions

    // Loupe state
    loupe: {
        x: 0.5, y: 0.5, // centre as fraction of canvas (0–1)
        r: 100,          // radius in canvas px
        zoom: 4,         // magnification factor
        dragging: false,
        visible: true,
    },

    // Motion ROI state (fractions 0–1)
    roi: {
        x1: 0.25, y1: 0.25,
        x2: 0.75, y2: 0.75,
        visible: false,
        dragging: null,  // null | 'x1y1' | 'x2y2' | 'x1y2' | 'x2y1' | 'body'
        handleR: 10,     // handle hit radius px
        dragStartMx: 0, dragStartMy: 0,
        dragStartRoi: null,
    },
};

function setupCanvasOverlay() {
    overlay.canvas = document.getElementById('overlayCanvas');
    overlay.feed   = document.getElementById('mjpegFeed');
    if (!overlay.canvas) return;
    overlay.ctx = overlay.canvas.getContext('2d');

    const container = document.getElementById('feedContainer');

    function resize() {
        const r = container.getBoundingClientRect();
        overlay.canvas.width  = r.width;
        overlay.canvas.height = r.height;
        overlay.cw = r.width;
        overlay.ch = r.height;
        // Loupe radius: 18% of the shorter dimension (height in 4:3 landscape)
        overlay.loupe.r = Math.round(Math.min(r.width, r.height) * 0.18);
    }
    resize();
    new ResizeObserver(resize).observe(container);

    // Enable pointer events on canvas for drag handling
    overlay.canvas.style.pointerEvents = 'auto';
    overlay.canvas.style.cursor = 'crosshair';

    overlay.canvas.addEventListener('pointerdown', onOverlayDown);
    overlay.canvas.addEventListener('pointermove', onOverlayMove);
    overlay.canvas.addEventListener('pointerup',   onOverlayUp);
    overlay.canvas.addEventListener('pointercancel', onOverlayUp);
    overlay.canvas.addEventListener('dblclick', onOverlayDblClick);

    // Kick off render loop
    requestAnimationFrame(drawOverlay);

    log("Canvas overlay ready — loupe + motion ROI active.");
}

function onOverlayDown(e) {
    const {mx, my} = getMouseFrac(e);
    const l = overlay.loupe;
    const roi = overlay.roi;

    overlay.canvas.setPointerCapture(e.pointerId);

    // Check if inside loupe circle
    const dx = (mx - l.x) * overlay.cw;
    const dy = (my - l.y) * overlay.ch;
    if (Math.sqrt(dx*dx + dy*dy) < l.r) {
        l.dragging = true;
        return;
    }

    // Check ROI handles if visible
    if (roi.visible) {
        const handle = getRoiHandle(mx, my);
        if (handle) {
            roi.dragging = handle;
            roi.dragStartMx = mx; roi.dragStartMy = my;
            roi.dragStartRoi = {...roi};
            return;
        }
        // Click inside ROI body → move whole box
        if (mx > roi.x1 && mx < roi.x2 && my > roi.y1 && my < roi.y2) {
            roi.dragging = 'body';
            roi.dragStartMx = mx; roi.dragStartMy = my;
            roi.dragStartRoi = {...roi};
        }
    }
}

function onOverlayMove(e) {
    const {mx, my} = getMouseFrac(e);
    const l   = overlay.loupe;
    const roi = overlay.roi;

    if (l.dragging) {
        l.x = Math.max(0, Math.min(1, mx));
        l.y = Math.max(0, Math.min(1, my));
        return;
    }

    if (roi.dragging) {
        const ddx = mx - roi.dragStartMx;
        const ddy = my - roi.dragStartMy;
        const sr  = roi.dragStartRoi;
        const MIN = 0.05;

        if (roi.dragging === 'body') {
            const w = sr.x2 - sr.x1, h = sr.y2 - sr.y1;
            roi.x1 = Math.max(0,     Math.min(1 - w, sr.x1 + ddx));
            roi.y1 = Math.max(0,     Math.min(1 - h, sr.y1 + ddy));
            roi.x2 = roi.x1 + w;
            roi.y2 = roi.y1 + h;
        } else {
            if (roi.dragging.includes('x1')) roi.x1 = Math.max(0,     Math.min(roi.x2 - MIN, sr.x1 + ddx));
            if (roi.dragging.includes('x2')) roi.x2 = Math.min(1,     Math.max(roi.x1 + MIN, sr.x2 + ddx));
            if (roi.dragging.includes('y1')) roi.y1 = Math.max(0,     Math.min(roi.y2 - MIN, sr.y1 + ddy));
            if (roi.dragging.includes('y2')) roi.y2 = Math.min(1,     Math.max(roi.y1 + MIN, sr.y2 + ddy));
        }

        // Update cursor
        overlay.canvas.style.cursor = getCursorForHandle(roi.dragging);
        sendMotionSettings();
        updateMotionRoiReadout();
        return;
    }

    // Update cursor based on hover
    if (roi.visible) {
        const h = getRoiHandle(mx, my);
        if (h) { overlay.canvas.style.cursor = getCursorForHandle(h); return; }
        if (mx > roi.x1 && mx < roi.x2 && my > roi.y1 && my < roi.y2) {
            overlay.canvas.style.cursor = 'move'; return;
        }
    }
    const l2 = overlay.loupe;
    const dx = (mx - l2.x) * overlay.cw, dy = (my - l2.y) * overlay.ch;
    overlay.canvas.style.cursor = Math.sqrt(dx*dx+dy*dy) < l2.r ? 'grab' : 'crosshair';
}

function onOverlayUp(e) {
    overlay.loupe.dragging = false;
    overlay.roi.dragging   = null;
    overlay.canvas.style.cursor = 'crosshair';
    overlay.canvas.releasePointerCapture(e.pointerId);
}

function onOverlayDblClick(e) {
    // Double-click recenters loupe
    overlay.loupe.x = 0.5;
    overlay.loupe.y = 0.5;
    log("Loupe: recentered.");
}

function getMouseFrac(e) {
    const r = overlay.canvas.getBoundingClientRect();
    return {
        mx: (e.clientX - r.left) / r.width,
        my: (e.clientY - r.top)  / r.height,
    };
}

function getRoiHandle(mx, my) {
    const roi = overlay.roi;
    const hr  = overlay.roi.handleR / overlay.cw;
    const corners = [
        { name: 'x1y1', x: roi.x1, y: roi.y1 },
        { name: 'x2y1', x: roi.x2, y: roi.y1 },
        { name: 'x1y2', x: roi.x1, y: roi.y2 },
        { name: 'x2y2', x: roi.x2, y: roi.y2 },
    ];
    for (const c of corners) {
        const dx = (mx - c.x) * overlay.cw;
        const dy = (my - c.y) * overlay.ch;
        if (Math.sqrt(dx*dx + dy*dy) < overlay.roi.handleR) return c.name;
    }
    return null;
}

function getCursorForHandle(h) {
    const map = {
        x1y1: 'nw-resize', x2y2: 'se-resize',
        x2y1: 'ne-resize', x1y2: 'sw-resize',
        body: 'move',
    };
    return map[h] || 'crosshair';
}

function drawOverlay() {
    requestAnimationFrame(drawOverlay);
    const ctx = overlay.ctx;
    const cw  = overlay.cw, ch = overlay.ch;
    if (!ctx || cw === 0) return;

    ctx.clearRect(0, 0, cw, ch);

    // ── Draw motion ROI ──────────────────────────────────────────────────────
    const roi = overlay.roi;
    if (roi.visible) {
        const rx1 = roi.x1 * cw, ry1 = roi.y1 * ch;
        const rx2 = roi.x2 * cw, ry2 = roi.y2 * ch;

        // Shaded outside
        ctx.fillStyle = 'rgba(0,0,0,0.35)';
        ctx.fillRect(0, 0, cw, ry1);
        ctx.fillRect(0, ry2, cw, ch - ry2);
        ctx.fillRect(0, ry1, rx1, ry2 - ry1);
        ctx.fillRect(rx2, ry1, cw - rx2, ry2 - ry1);

        // Box outline
        ctx.strokeStyle = '#00AAFF';
        ctx.lineWidth = 2;
        ctx.strokeRect(rx1, ry1, rx2 - rx1, ry2 - ry1);

        // Corner handles
        const hr = overlay.roi.handleR;
        ctx.fillStyle = '#00AAFF';
        [[rx1,ry1],[rx2,ry1],[rx1,ry2],[rx2,ry2]].forEach(([hx,hy]) => {
            ctx.beginPath();
            ctx.arc(hx, hy, hr, 0, Math.PI * 2);
            ctx.fill();
        });

        // Label
        ctx.fillStyle = '#00AAFF';
        ctx.font = '11px monospace';
        ctx.fillText('MOTION ZONE', rx1 + 6, ry1 + 16);
    }

    // ── Draw loupe ───────────────────────────────────────────────────────────
    const l = overlay.loupe;
    if (!l.visible) return;

    const lx = l.x * cw;
    const ly = l.y * ch;
    const r  = l.r;

    ctx.save();

    // Always fill the loupe background first so it's never transparent
    ctx.beginPath();
    ctx.arc(lx, ly, r, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(0,0,0,0.75)';
    ctx.fill();

    // Clip to circle
    ctx.beginPath();
    ctx.arc(lx, ly, r, 0, Math.PI * 2);
    ctx.clip();

    // Draw the high-res crop if available (blob URL — never taints canvas)
    if (loupeCropImage && loupeCropImage.complete && loupeCropImage.naturalWidth > 0) {
        ctx.drawImage(loupeCropImage, lx - r, ly - r, r * 2, r * 2);
    } else {
        // Waiting for first crop — show a dim "loading" indicator
        ctx.fillStyle = 'rgba(0,170,255,0.12)';
        ctx.fillRect(lx - r, ly - r, r * 2, r * 2);
        ctx.fillStyle = 'rgba(0,170,255,0.5)';
        ctx.font = `${Math.round(r * 0.22)}px monospace`;
        ctx.textAlign = 'center';
        ctx.fillText('LOADING…', lx, ly + 6);
        ctx.textAlign = 'left';
    }

    // Crosshair
    ctx.strokeStyle = 'rgba(255,255,255,0.7)';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(lx, ly - r); ctx.lineTo(lx, ly + r); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(lx - r, ly); ctx.lineTo(lx + r, ly); ctx.stroke();

    ctx.restore();

    // Loupe border ring (drawn outside clip so it's crisp)
    ctx.beginPath();
    ctx.arc(lx, ly, r, 0, Math.PI * 2);
    ctx.strokeStyle = overlay.roi.visible ? 'rgba(0,170,255,0.5)' : 'rgba(0,170,255,0.9)';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Loupe label
    ctx.fillStyle = 'rgba(0,170,255,0.85)';
    ctx.font = '10px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(`${l.zoom}× FOCUS`, lx, ly + r - 10);
    ctx.textAlign = 'left';
}

// Called from setRunState — hide loupe while a non-cinematic sequence runs
function setLoupeVisible(v) {
    // Always respect user toggle — if user hid the loupe, don't show it
    overlay.loupe.visible = v && _loupeUserVisible;
}

// ── Frame rate selector ───────────────────────────────────────────────────────
function setCineFps(fps) {
    sendCmd('set_cine_fps', { value: fps });
    _applyCineFpsUI(fps);
}

function _applyCineFpsUI(fps) {
    [24, 25, 30, 60].forEach(f => {
        const btn = document.getElementById(`fps${f}`);
        if (!btn) return;
        const active = f === fps;
        btn.style.background  = active ? 'var(--accent-teal)' : 'none';
        btn.style.borderColor = active ? 'var(--accent-teal)' : '#333';
        btn.style.color       = active ? '#000' : '#555';
    });
}
const _ORIENT_ICONS = {
    landscape:    '⬜',
    portrait_cw:  '↻',
    portrait_ccw: '↺',
    inverted:     '⬛',
};

function setCameraOrientation(orient) {
    sendCmd('set_camera_orientation', { value: orient });
    _applyOrientationUI(orient);
}

function _applyOrientationUI(orient) {
    ['landscape','portrait_cw','portrait_ccw','inverted'].forEach(o => {
        const btn = document.getElementById('orient' + {
            landscape: 'Land', portrait_cw: 'CW',
            portrait_ccw: 'CCW', inverted: 'Flip'
        }[o]);
        if (!btn) return;
        const active = o === orient;
        btn.style.background   = active ? 'var(--accent-teal)' : 'none';
        btn.style.borderColor  = active ? 'var(--accent-teal)' : '#444';
        btn.style.color        = active ? '#000' : '#666';
    });
}

function handleCameraOrientation(data) {
    _applyOrientationUI(data.value);
}
function toggleLoupeVisibility() {
    _loupeUserVisible = !_loupeUserVisible;
    overlay.loupe.visible = _loupeUserVisible;
    const btn = document.getElementById('loupeToggleBtn');
    if (btn) {
        btn.style.opacity     = _loupeUserVisible ? '1' : '0.4';
        btn.style.borderColor = _loupeUserVisible ? 'var(--accent-teal)' : '#444';
        btn.style.color       = _loupeUserVisible ? 'var(--accent-teal)' : '#555';
        btn.title = _loupeUserVisible ? 'Hide focus loupe' : 'Show focus loupe';
    }
}

// Called from onTriggerModeChange
function setRoiVisible(v) {
    overlay.roi.visible = v;
    // When motion mode enabled and non-PiCam active, show a note
    const note = document.getElementById('motion_camera_note');
    if (note) note.style.display = v ? 'block' : 'none';
}

function updateMotionRoiReadout() {
    const roi = overlay.roi;
    const el  = document.getElementById('motion_roi_readout');
    if (el) el.innerText = `Zone: x ${(roi.x1*100).toFixed(0)}%–${(roi.x2*100).toFixed(0)}%,  y ${(roi.y1*100).toFixed(0)}%–${(roi.y2*100).toFixed(0)}%`;
}

// Restore ROI from server data
function setRoiFromData(roi) {
    if (!roi || roi.length < 4) return;
    overlay.roi.x1 = roi[0]; overlay.roi.y1 = roi[1];
    overlay.roi.x2 = roi[2]; overlay.roi.y2 = roi[3];
    updateMotionRoiReadout();
}


// ═══════════════════════════════════════════════════════════════════════════════
// DISK SPACE DISPLAY + ALERTS
// ═══════════════════════════════════════════════════════════════════════════════

async function updateDiskSpace() {
    try {
        const resp = await fetch('/disk_info');
        const d    = await resp.json();
        if (d.error) return;
        const freeGB  = (d.free  / 1e9).toFixed(1);
        const totalGB = (d.total / 1e9).toFixed(1);
        const pct     = Math.round(d.used / d.total * 100);
        const cam     = document.getElementById('camera_select')?.value || 'picam';
        const framesEst = Math.floor(d.free / ((cam === 'sony' ? 30 : 25) * 1e6));

        const el = document.getElementById('disk_free_label');
        if (!el) return;
        el.innerHTML = `${freeGB} GB free of ${totalGB} GB (${pct}% used) — ~<b>${framesEst.toLocaleString()}</b> frames of space`;

        const row = document.getElementById('disk_space_row');
        if (row) {
            row.classList.remove('disk-ok','disk-warn','disk-crit');
            if (pct > 90)      row.classList.add('disk-crit');
            else if (pct > 75) row.classList.add('disk-warn');
            else               row.classList.add('disk-ok');
        }

        // Pre-flight warning
        const total   = parseInt(document.getElementById('total_frames')?.value) || 0;
        const warnEl  = document.getElementById('disk_preflight_warn');
        if (warnEl) {
            const show = total > 0 && framesEst < total;
            warnEl.style.display = show ? 'block' : 'none';
            if (show) warnEl.textContent =
                `⚠ Only ~${framesEst.toLocaleString()} frames fit on disk — sequence of ${total} may fail.`;
        }
    } catch (_) {}
}

let _diskPollTimer = null;
function startDiskPolling() {
    updateDiskSpace();
    if (!_diskPollTimer) _diskPollTimer = setInterval(updateDiskSpace, 15000);
}

function showDiskAlert(msg, isFull) {
    document.getElementById('diskAlertBanner')?.remove();
    const banner = document.createElement('div');
    banner.id = 'diskAlertBanner';
    banner.className = 'disk-alert-banner';
    banner.innerHTML = `
        <div style="font-size:1.1rem; margin-bottom:4px;">${isFull ? '⛔ DISK FULL' : '⚠ LOW DISK SPACE'}</div>
        <div style="font-size:0.8rem; opacity:0.9;">${msg}</div>
        <button onclick="document.getElementById('diskAlertBanner').remove()">Dismiss</button>
    `;
    document.body.appendChild(banner);
    log(msg);
    if (!isFull) setTimeout(() => { banner?.remove(); }, 30000);
}

function _showKickedBanner(msg) {
    // Full-screen overlay — makes it impossible to accidentally use a displaced tab
    const existing = document.getElementById('kickedOverlay');
    if (existing) existing.remove();
    const overlay = document.createElement('div');
    overlay.id = 'kickedOverlay';
    overlay.style.cssText = `
        position: fixed; inset: 0; z-index: 9999;
        background: rgba(0,0,0,0.88);
        display: flex; flex-direction: column;
        align-items: center; justify-content: center;
        color: var(--accent-gold, #f0a500);
        font-family: monospace; text-align: center; gap: 16px;
        backdrop-filter: blur(4px);
    `;
    overlay.innerHTML = `
        <div style="font-size:2rem;">⚠</div>
        <div style="font-size:1.1rem; font-weight:bold;">TAB REPLACED</div>
        <div style="font-size:0.85rem; max-width:360px; color:#ccc; line-height:1.6;">${msg}</div>
        <button onclick="window.location.reload()"
                style="margin-top:8px; padding:10px 28px; background:var(--accent-gold,#f0a500);
                       color:#000; border:none; border-radius:6px; font-weight:bold;
                       cursor:pointer; font-size:0.9rem;">
            Take Control Back
        </button>
    `;
    document.body.appendChild(overlay);
}

function handleDiskFull(data)  { showDiskAlert(data.msg, true);  updateDiskSpace(); }
function handleDiskWarn(data)  { showDiskAlert(data.msg, false); updateDiskSpace(); }


// ═══════════════════════════════════════════════════════════════════════════════
// MACRO MODE
// ═══════════════════════════════════════════════════════════════════════════════

let _macroMode        = 'scan';   // 'scan' | 'art'
let _macroRotMode     = 'full';   // 'full' | 'range'
let _macroRailStart   = null;
let _macroRailEnd     = null;
let _macroRotStart    = null;
let _macroRotEnd      = null;
let _macroAuxStart    = null;
let _macroAuxEnd      = null;
let _macroLensProfiles = {};

// ── Sub-mode toggle ──────────────────────────────────────────────────────────
function setMacroMode(mode) {
    _macroMode = mode;
    document.getElementById('macroModeScan').classList.toggle('active', mode === 'scan');
    document.getElementById('macroModeArt').classList.toggle('active',  mode === 'art');
    // In Art mode surface the easing curve; in Scan it defaults to 'even'
    const easingRow = document.getElementById('macro_rotation_easing');
    if (easingRow) {
        if (mode === 'scan') easingRow.value = 'even';
    }
    macroCalc();
}

// ── Rotation mode (full 360 vs range) ────────────────────────────────────────
function setRotationMode(mode) {
    _macroRotMode = mode;
    document.getElementById('macroRotFull').classList.toggle('active',  mode === 'full');
    document.getElementById('macroRotRange').classList.toggle('active', mode === 'range');
    const rc = document.getElementById('macro_rotation_range_controls');
    if (rc) rc.style.display = mode === 'range' ? '' : 'none';
    macroCalc();
}

// ── Aux axis toggle ───────────────────────────────────────────────────────────
function toggleAuxAxis(enabled) {
    const ctrl = document.getElementById('macro_aux_controls');
    if (ctrl) ctrl.style.display = enabled ? '' : 'none';
}

// ── Soft limits ───────────────────────────────────────────────────────────────
function sendMacroSoftLimits() {
    sendCmd('macro_set_soft_limits', {
        rail_min:  parseFloat(document.getElementById('macro_rail_soft_min')?.value  ?? -999),
        rail_max:  parseFloat(document.getElementById('macro_rail_soft_max')?.value  ??  999),
        pan_min:   -360, pan_max: 360,   // rotation stage — wide range
        tilt_min:  parseFloat(document.getElementById('macro_aux_soft_min')?.value   ?? -90),
        tilt_max:  parseFloat(document.getElementById('macro_aux_soft_max')?.value   ??  90),
    });
}

// ── Live calculation ──────────────────────────────────────────────────────────
function macroCalc() {
    const stepUm    = parseFloat(document.getElementById('macro_step_um')?.value   || 100);
    const numStacks = parseInt(  document.getElementById('macro_num_stacks')?.value || 36);
    const slotA     = document.getElementById('macro_slot_a_enabled')?.checked ? 1 : 0;
    const slotB     = document.getElementById('macro_slot_b_enabled')?.checked ? 1 : 0;
    const slots     = slotA + slotB;

    let frames = 0;
    if (_macroRailStart !== null && _macroRailEnd !== null && stepUm > 0) {
        const travelUm = Math.abs(_macroRailEnd - _macroRailStart) * 1000;
        frames = Math.max(1, Math.ceil(travelUm / stepUm) + 1);
    }

    const totalImages = frames * numStacks * Math.max(1, slots);
    const storageGb   = (totalImages * 25 / 1024).toFixed(2);

    const fps  = document.getElementById('macro_frames_per_stack');
    if (fps)  fps.value = frames || '—';

    const sf = document.getElementById('macro_sum_frames');
    const ss = document.getElementById('macro_sum_stacks');
    const si = document.getElementById('macro_sum_images');
    const sg = document.getElementById('macro_sum_storage');
    if (sf) sf.innerText = frames   || '—';
    if (ss) ss.innerText = numStacks;
    if (si) si.innerText = frames ? totalImages : '—';
    if (sg) sg.innerText = frames ? storageGb  : '—';
}

// ── Position markers ─────────────────────────────────────────────────────────
function handleMacroRailMark(data) {
    if (data.which === 'start') {
        _macroRailStart = data.mm;
        const el = document.getElementById('macro_rail_start_disp');
        if (el) el.innerText = data.mm.toFixed(3);
    } else {
        _macroRailEnd = data.mm;
        const el = document.getElementById('macro_rail_end_disp');
        if (el) el.innerText = data.mm.toFixed(3);
    }
    // Update travel display
    if (_macroRailStart !== null && _macroRailEnd !== null) {
        const travel = Math.abs(_macroRailEnd - _macroRailStart);
        const el = document.getElementById('macro_rail_travel_disp');
        if (el) el.innerText = travel.toFixed(3);
    }
    macroCalc();
    log(`Rail ${data.which} set: ${data.mm.toFixed(3)} mm`);
}

function handleMacroRotMark(data) {
    if (data.which === 'start') {
        _macroRotStart = data.deg;
        const el = document.getElementById('macro_rot_start_disp');
        if (el) el.innerText = data.deg.toFixed(1);
    } else {
        _macroRotEnd = data.deg;
        const el = document.getElementById('macro_rot_end_disp');
        if (el) el.innerText = data.deg.toFixed(1);
    }
    log(`Rotation ${data.which} set: ${data.deg.toFixed(1)}°`);
}

function handleMacroAuxMark(data) {
    if (data.which === 'start') {
        _macroAuxStart = data.deg;
        const el = document.getElementById('macro_aux_start_disp');
        if (el) el.innerText = data.deg.toFixed(1);
    } else {
        _macroAuxEnd = data.deg;
        const el = document.getElementById('macro_aux_end_disp');
        if (el) el.innerText = data.deg.toFixed(1);
    }
    log(`Aux ${data.which} set: ${data.deg.toFixed(1)}°`);
}

// ── Progress updates ─────────────────────────────────────────────────────────
function handleMacroProgress(data) {
    const panel = document.getElementById('macro_progress_panel');
    if (panel) panel.style.display = '';

    const set = (id, val) => { const el = document.getElementById(id); if (el) el.innerText = val; };
    set('macro_prog_stack',        data.stack        ?? '—');
    set('macro_prog_total_stacks', data.total_stacks ?? '—');
    set('macro_prog_frame',        data.frame        ?? '—');
    set('macro_prog_total_frames', data.total_frames ?? '—');
    if (data.rotation_deg !== undefined) set('macro_prog_rot',  data.rotation_deg.toFixed(1));
    if (data.rail_mm      !== undefined) set('macro_prog_rail', data.rail_mm.toFixed(3));
    if (data.msg)                        set('macro_prog_msg',  data.msg);

    // Progress bar: based on (stack-1)*frames + frame / total_stacks*total_frames
    if (data.total_stacks && data.total_frames) {
        const done  = ((data.stack - 1) * data.total_frames + (data.frame || 0));
        const total = data.total_stacks * data.total_frames;
        const pct   = Math.min(100, Math.round(done / total * 100));
        const bar   = document.getElementById('macro_prog_bar');
        if (bar) bar.style.width = pct + '%';
    }
}

function handleMacroStackComplete(data) {
    log(`✓ Stack ${data.stack}/${data.total_stacks} complete — rot ${data.rotation_deg?.toFixed(1)}°`);
}

function handleMacroDone(data) {
    const panel = document.getElementById('macro_progress_panel');
    if (panel) panel.style.display = 'none';
    const bar = document.getElementById('macro_prog_bar');
    if (bar) bar.style.width = '0%';
    log(data.msg || (data.interrupted ? '⚠ Macro stopped.' : '✓ Macro sequence complete.'));
}

// ── Lens profiles ─────────────────────────────────────────────────────────────
function macroStoreLensProfile() {
    const name = document.getElementById('macro_lens_name')?.value?.trim();
    if (!name || name === 'unknown') { log('Enter a lens name before saving.'); return; }
    const profile = _readLensProfile();
    sendCmd('macro_store_lens_profile', { name, profile });
    log(`Lens profile '${name}' saved.`);
}

function _readLensProfile() {
    return {
        name:                document.getElementById('macro_lens_name')?.value  || 'unknown',
        lens_type:           document.getElementById('macro_lens_type')?.value  || 'macro',
        magnification:       parseFloat(document.getElementById('macro_lens_mag')?.value  || 1),
        working_distance_mm: parseFloat(document.getElementById('macro_lens_wd')?.value   || 0),
        notes: '',
    };
}

function macroLoadLensProfile(name) {
    if (!name || !_macroLensProfiles[name]) return;
    const p = _macroLensProfiles[name];
    const set = (id, val) => { const el = document.getElementById(id); if (el) el.value = val; };
    set('macro_lens_name', p.name            || '');
    set('macro_lens_type', p.lens_type       || 'macro');
    set('macro_lens_mag',  p.magnification   || 1);
    set('macro_lens_wd',   p.working_distance_mm || 0);
    log(`Lens profile '${name}' loaded.`);
}

function handleMacroLensProfiles(data) {
    _macroLensProfiles = data.profiles || {};
    const sel = document.getElementById('macro_lens_profile_select');
    if (!sel) return;
    sel.innerHTML = '<option value="">— Load saved —</option>';
    Object.keys(_macroLensProfiles).forEach(name => {
        const opt = document.createElement('option');
        opt.value = name; opt.innerText = name;
        sel.appendChild(opt);
    });
}

// ── Start sequence ────────────────────────────────────────────────────────────
function macroStart() {
    if (isRunning) { log('Already running.'); return; }
    if (_macroRailStart === null || _macroRailEnd === null) {
        log('⚠ Set rail start and end positions before starting.');
        return;
    }

    const auxEnabled = document.getElementById('macro_aux_enabled')?.checked || false;

    const payload = {
        project_name:      document.getElementById('macro_project_name')?.value || 'macro_project',
        orbit_label:       document.getElementById('macro_orbit_label')?.value  || 'orbit_001',
        session_mode:      _macroMode,
        rail_start_mm:     _macroRailStart,
        rail_end_mm:       _macroRailEnd,
        rail_step_um:      parseFloat(document.getElementById('macro_step_um')?.value || 100),
        rail_soft_min:     parseFloat(document.getElementById('macro_rail_soft_min')?.value ?? -999),
        rail_soft_max:     parseFloat(document.getElementById('macro_rail_soft_max')?.value ??  999),
        rotation_mode:     _macroRotMode,
        rotation_start_deg: _macroRotStart ?? 0,
        rotation_end_deg:   _macroRotEnd   ?? 360,
        num_stacks:        parseInt(document.getElementById('macro_num_stacks')?.value || 36),
        rotation_easing:   document.getElementById('macro_rotation_easing')?.value || 'even',
        rotation_axis_angle_deg: parseFloat(document.getElementById('macro_rot_axis_angle')?.value || 90),
        rotation_axis_description: document.getElementById('macro_rot_axis_desc')?.value || 'vertical',
        aux_enabled:       auxEnabled,
        aux_label:         document.getElementById('macro_aux_label')?.value || 'aux',
        aux_start_deg:     _macroAuxStart ?? 0,
        aux_end_deg:       _macroAuxEnd   ?? 0,
        aux_easing:        document.getElementById('macro_aux_easing')?.value || 'even',
        vibe_delay_s:      parseFloat(document.getElementById('macro_vibe_delay')?.value || 0.5),
        exp_margin_s:      parseFloat(document.getElementById('macro_exp_margin')?.value || 0.2),
        lens_profile:      _readLensProfile(),
        slots: [
            {
                id:               'slot_A',
                label:            document.getElementById('macro_slot_a_label')?.value  || 'diffuse',
                enabled:          document.getElementById('macro_slot_a_enabled')?.checked ?? true,
                relay1:           document.getElementById('macro_slot_a_relay1')?.checked ?? false,
                relay2:           document.getElementById('macro_slot_a_relay2')?.checked ?? false,
                relay_settle_ms:  parseInt(document.getElementById('macro_slot_a_settle')?.value  || 0),
                relay_release_ms: parseInt(document.getElementById('macro_slot_a_release')?.value || 0),
                iso:              parseInt(document.getElementById('macro_slot_a_iso')?.value    || 400),
                shutter_s:        parseFloat(document.getElementById('macro_slot_a_shutter')?.value || 0.008),
                kelvin:           parseInt(document.getElementById('macro_slot_a_kelvin')?.value || 5500),
                ae:               document.getElementById('macro_slot_a_ae')?.checked ?? false,
                awb:              false,
            },
            {
                id:               'slot_B',
                label:            document.getElementById('macro_slot_b_label')?.value  || 'laser',
                enabled:          document.getElementById('macro_slot_b_enabled')?.checked ?? false,
                relay1:           document.getElementById('macro_slot_b_relay1')?.checked ?? false,
                relay2:           document.getElementById('macro_slot_b_relay2')?.checked ?? true,
                relay_settle_ms:  parseInt(document.getElementById('macro_slot_b_settle')?.value  || 250),
                relay_release_ms: parseInt(document.getElementById('macro_slot_b_release')?.value || 0),
                iso:              parseInt(document.getElementById('macro_slot_b_iso')?.value    || 200),
                shutter_s:        parseFloat(document.getElementById('macro_slot_b_shutter')?.value || 0.033),
                kelvin:           parseInt(document.getElementById('macro_slot_b_kelvin')?.value || 5500),
                ae:               document.getElementById('macro_slot_b_ae')?.checked ?? false,
                awb:              false,
            }
        ],
    };

    log(`Starting macro: ${payload.num_stacks} stacks × ${payload.rail_step_um}µm steps`);
    setRunState(true);
    sendCmd('macro_start', payload);
}


// ═══════════════════════════════════════════════════════════════════════════════
// CINEMATIC MODE
// ═══════════════════════════════════════════════════════════════════════════════

let _cineSubMode        = 'live';
let _cineInertiaRunning = false;
let _cineRecording      = false;
let _cineRecordTimer    = null;
let _cineRecordStart    = null;
let _cineKeyframes      = [];
let _cineLimits         = {};
let _cineOrigin         = null;

// ── Sub-mode toggle ──────────────────────────────────────────────────────────
function setCineSubMode(mode) {
    _cineSubMode = mode;
    document.getElementById('cineModeLive').classList.toggle('active', mode === 'live');
    document.getElementById('cineModeProg').classList.toggle('active', mode === 'programmed');
    document.getElementById('cine_live_panel').style.display  = mode === 'live' ? '' : 'none';
    document.getElementById('cine_prog_panel').style.display  = mode === 'programmed' ? '' : 'none';
    sendCmd('cinematic_set_mode', { value: mode });
}

// ── Soft limits ──────────────────────────────────────────────────────────────
function calibrateLimit(axis, end) {
    sendCmd('cinematic_calibrate_limit', { axis, end });
    log(`Soft limit: ${axis} ${end} set at current position.`);
}

function handleCineLimits(data) {
    _cineLimits = data.limits || {};
    _updateLimitDisplay();
}

function _updateLimitDisplay() {
    const CAL_LABELS = ['uncalibrated — crawl only', 'one end set — half speed', '✓ both ends — full speed'];
    const SPEED_LABELS = ['CRAWL ONLY', 'HALF SPEED', 'FULL SPEED'];
    const SPEED_COLORS = ['#cc4400', '#ccaa00', '#00cc66'];

    let maxCal = 2;
    ['slider', 'pan', 'tilt'].forEach(axis => {
        const ax = _cineLimits[axis];
        if (!ax) return;
        const cal = ax.cal_state ?? 0;
        if (cal < maxCal) maxCal = cal;
        const labelEl = document.getElementById(`cine_${axis}_cal`);
        if (labelEl) {
            labelEl.innerText = CAL_LABELS[cal] || '';
            labelEl.style.color = SPEED_COLORS[cal];
        }
    });

    const badge = document.getElementById('cine_speed_badge');
    if (badge) {
        badge.innerText = SPEED_LABELS[maxCal] || 'CRAWL ONLY';
        badge.style.color       = SPEED_COLORS[maxCal];
        badge.style.borderColor = SPEED_COLORS[maxCal];
    }
}

function _updateLimitBars(posS, posP, posT) {
    // Update position indicators on limit bars
    function _setBar(barId, posId, pos, min, max) {
        const bar = document.getElementById(barId);
        const dot = document.getElementById(posId);
        if (!bar || !dot || min == null || max == null || max <= min) return;
        const pct = Math.max(0, Math.min(100, (pos - min) / (max - min) * 100));
        dot.style.left = pct + '%';
    }
    const sl = _cineLimits.slider || {};
    const pa = _cineLimits.pan    || {};
    const ti = _cineLimits.tilt   || {};
    _setBar('cine_slider_bar', 'cine_slider_pos', posS, sl.min, sl.max);
    _setBar('cine_pan_bar',    'cine_pan_pos',    posP, pa.min, pa.max);
    _setBar('cine_tilt_bar',   'cine_tilt_pos',   posT, ti.min, ti.max);
}

// ── Arctan tracker ───────────────────────────────────────────────────────────
function arctanMarkPoint() {
    sendCmd('arctan_add_point');
}

function arctanClear() {
    sendCmd('arctan_clear');
    const cb = document.getElementById('arctan_enable_cb');
    if (cb) { cb.checked = false; cb.disabled = true; }
    const badge = document.getElementById('arctan_lock_badge');
    if (badge) { badge.innerText = 'OFF'; badge.style.color = 'var(--text-muted)'; }
    document.getElementById('arctan_status_row').style.display = 'none';
    document.getElementById('arctan_point_count').innerText = '(0)';
}

function arctanEnable(enabled) {
    sendCmd('arctan_enable', { enabled });
}

function handleArctanStatus(data) {
    const countEl = document.getElementById('arctan_point_count');
    if (countEl) countEl.innerText = `(${data.points})`;

    const statusRow = document.getElementById('arctan_status_row');
    const cb        = document.getElementById('arctan_enable_cb');
    const badge     = document.getElementById('arctan_lock_badge');

    if (data.solved) {
        if (statusRow) statusRow.style.display = '';
        const res = document.getElementById('arctan_residual');
        if (res) res.innerText = (data.residual || 0).toFixed(2);
        const warn = document.getElementById('arctan_warning');
        if (warn) warn.innerText = data.warning || '';
        if (cb) cb.disabled = false;
    } else {
        if (statusRow) statusRow.style.display = 'none';
        if (cb) cb.disabled = true;
    }
}

function handleArctanEnabled(data) {
    const badge = document.getElementById('arctan_lock_badge');
    const cb    = document.getElementById('arctan_enable_cb');
    if (badge) {
        badge.innerText = data.enabled ? 'LOCKED' : 'OFF';
        badge.style.color       = data.enabled ? 'var(--accent-teal)' : 'var(--text-muted)';
        badge.style.borderColor = data.enabled ? 'var(--accent-teal)' : '#333';
    }
    if (cb) cb.checked = !!data.enabled;
}

// ── Inertia / live mode ──────────────────────────────────────────────────────
function updateInertiaLabel(param) {
    const el  = document.getElementById(`cine_${param}`);
    const lbl = document.getElementById(`cine_${param}_val`);
    if (!el || !lbl) return;
    lbl.innerText = param === 'mass' ? el.value + 's' : el.value;
}

function sendInertia() {
    sendCmd('cinematic_set_inertia', {
        mass: parseFloat(document.getElementById('cine_mass')?.value || 0.4),
        drag: parseFloat(document.getElementById('cine_drag')?.value || 0.55),
    });
}

function setRigPreset(name) {
    const presets = {
        light:    { mass: 0.15, drag: 0.80 },
        standard: { mass: 0.40, drag: 0.55 },
        heavy:    { mass: 0.90, drag: 0.30 },
    };
    const p = presets[name];
    if (!p) return;
    document.getElementById('cine_mass').value = p.mass;
    document.getElementById('cine_drag').value = p.drag;
    updateInertiaLabel('mass');
    updateInertiaLabel('drag');
    sendCmd('cinematic_set_inertia', { preset: name });
    // Update preset button highlight
    ['light','standard','heavy'].forEach(n => {
        document.getElementById(`preset${n.charAt(0).toUpperCase()+n.slice(1)}`)
            ?.classList.toggle('active', n === name);
    });
}

function cinematicLiveStart() {
    if (isRunning) return;
    sendCmd('cinematic_live_start', {
        mass: parseFloat(document.getElementById('cine_mass')?.value || 0.4),
        drag: parseFloat(document.getElementById('cine_drag')?.value || 0.55),
    });
    _cineInertiaRunning = true;
    const btn = document.getElementById('liveStartBtn');
    if (btn) {
        btn.innerText = '■ Stop Live Control';
        btn.onclick   = cinematicLiveStop;
    }
}

function cinematicLiveStop() {
    sendCmd('cinematic_live_stop');
    _cineInertiaRunning = false;
    const btn = document.getElementById('liveStartBtn');
    if (btn) {
        btn.innerText = '▶ Start Live Control';
        btn.onclick   = cinematicLiveStart;
    }
}

// ── Keyframes ────────────────────────────────────────────────────────────────
function addKeyframeAtCurrent() {
    sendCmd('cinematic_add_keyframe', {
        duration_s: 3.0,
        easing:     'gaussian',
    });
}

function handleCineKeyframes(data) {
    _cineKeyframes = data.keyframes || [];
    _renderKeyframeList();
}

function _renderKeyframeList() {
    const container = document.getElementById('cine_kf_list');
    if (!container) return;

    if (_cineKeyframes.length === 0) {
        container.innerHTML = `<div style="color:var(--text-muted); font-size:0.72rem;
            text-align:center; padding:12px 0;">No keyframes. Jog to position and click Add.</div>`;
        return;
    }

    const EASINGS = ['gaussian','parabolic','linear','even','catenary','lame'];

    container.innerHTML = _cineKeyframes.map((kf, i) => `
        <div style="border-bottom:1px solid #222; padding:6px 8px; font-size:0.7rem;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                <span style="color:var(--accent-teal); font-weight:600;">KF ${i+1}</span>
                <span style="color:var(--text-dim); font-family:monospace;">
                    s:${kf.slider_mm.toFixed(1)}mm
                    p:${kf.pan_deg.toFixed(1)}°
                    t:${kf.tilt_deg.toFixed(1)}°
                </span>
                <button onclick="sendCmd('cinematic_remove_keyframe',{index:${i}})"
                        style="background:none; border:1px solid #440000; color:#cc2200;
                               border-radius:3px; padding:1px 6px; cursor:pointer; font-size:0.65rem;">✕</button>
            </div>
            ${i < _cineKeyframes.length - 1 ? `
            <div style="display:flex; gap:6px; align-items:center;">
                <label style="color:var(--text-muted); white-space:nowrap;">→ next in</label>
                <input type="number" value="${kf.duration_s}" step="0.5" min="0.1"
                       style="width:56px; font-size:0.75rem; padding:2px 4px;"
                       onchange="sendCmd('cinematic_update_keyframe',
                                 {index:${i}, duration_s:parseFloat(this.value)})">
                <label style="color:var(--text-muted);">s</label>
                <select style="font-size:0.72rem; flex:1;"
                        onchange="sendCmd('cinematic_update_keyframe',
                                  {index:${i}, easing:this.value})">
                    ${EASINGS.map(e =>
                        `<option value="${e}" ${e === kf.easing ? 'selected' : ''}>${e}</option>`
                    ).join('')}
                </select>
            </div>` : `<div style="color:var(--text-muted);">End point</div>`}
        </div>
    `).join('');
}

// ── Playback ─────────────────────────────────────────────────────────────────
function cinematicPlay() {
    if (isRunning) { log('Already running.'); return; }
    if (_cineKeyframes.length < 2) {
        log('⚠ Need at least 2 keyframes to play a move.');
        return;
    }
    if (!_cineOrigin) {
        log('⚠ Set origin before playing — rig needs a reference position.');
        return;
    }
    sendCmd('cinematic_play');
}

function handleCineProgress(data) {
    const wrap = document.getElementById('cine_prog_bar_wrap');
    const bar  = document.getElementById('cine_prog_bar');
    const msg  = document.getElementById('cine_prog_msg');
    if (wrap) wrap.style.display = '';
    if (msg)  msg.style.display  = '';
    if (bar)  bar.style.width    = Math.round((data.progress || 0) * 100) + '%';
    if (msg)  msg.innerText = `Segment ${data.segment}/${data.segments} — `
        + `s:${data.pos_s?.toFixed(1)}mm p:${data.pos_p?.toFixed(1)}° t:${data.pos_t?.toFixed(1)}°`;
}

function handleCinePlayDone() {
    const wrap = document.getElementById('cine_prog_bar_wrap');
    const msg  = document.getElementById('cine_prog_msg');
    if (wrap) wrap.style.display = 'none';
    if (msg)  msg.style.display  = 'none';
}

// ── Origin ───────────────────────────────────────────────────────────────────
function handleCineOriginSet(data) {
    _cineOrigin = data;
    const el = document.getElementById('cine_origin_disp');
    if (el) el.innerText = `s:${data.slider_mm?.toFixed(1)}mm `
        + `p:${data.pan_deg?.toFixed(1)}° t:${data.tilt_deg?.toFixed(1)}°`;
    log(`Origin set — s:${data.slider_mm?.toFixed(1)}mm p:${data.pan_deg?.toFixed(1)}° t:${data.tilt_deg?.toFixed(1)}°`);
}

// ── Move library ─────────────────────────────────────────────────────────────
function saveCurrentMove() {
    const name = document.getElementById('cine_move_name')?.value?.trim();
    if (!name) { log('Enter a move name first.'); return; }
    sendCmd('cinematic_save_move', { name });
    log(`Saving move '${name}'…`);
}

function handleCineMoves(data) {
    const moves = data.moves || [];
    const container = document.getElementById('cine_move_library');
    if (!container) return;

    if (moves.length === 0) {
        container.innerHTML = `<div style="color:var(--text-muted); font-size:0.72rem;
            text-align:center; padding:12px;">No saved moves.</div>`;
        return;
    }

    container.innerHTML = moves.map(m => `
        <div style="display:flex; gap:6px; align-items:center; padding:6px 8px;
                    border-bottom:1px solid #1a1a1a;">
            <div style="flex:1; min-width:0;">
                <div style="font-size:0.78rem; color:var(--accent-teal);
                            white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                    ${m.name}</div>
                <div style="font-size:0.62rem; color:var(--text-muted);">
                    ${m.keyframes} kf · ${m.duration?.toFixed(1)}s ·
                    ${new Date(m.created).toLocaleDateString()}</div>
            </div>
            <button onclick="sendCmd('cinematic_load_move',{name:'${m.name}'})"
                    style="font-size:0.65rem; padding:3px 8px; background:none;
                           border:1px solid var(--accent-teal); color:var(--accent-teal);
                           border-radius:4px; cursor:pointer; white-space:nowrap;">Load</button>
            <button onclick="if(confirm('Delete \\'${m.name}\\'?')) sendCmd('cinematic_delete_move',{name:'${m.name}'})"
                    style="font-size:0.65rem; padding:3px 8px; background:none;
                           border:1px solid #440000; color:#cc2200;
                           border-radius:4px; cursor:pointer;">✕</button>
        </div>
    `).join('');
}

// ── Recording ────────────────────────────────────────────────────────────────
function toggleRecord() {
    if (_cineRecording) {
        sendCmd('record_stop');
    } else {
        sendCmd('record_start');
    }
}

function handleRecordState(data) {
    _cineRecording = data.recording;
    const btn  = document.getElementById('recordBtn');
    const ind  = document.getElementById('recIndicator');
    const tc   = document.getElementById('timecodeDisplay');

    if (data.recording) {
        _cineRecordStart = Date.now();
        if (btn) {
            btn.innerText          = '■ STOP REC';
            btn.style.background   = '#2a0000';
            btn.style.borderColor  = '#ff2200';
            btn.style.color        = '#ff3300';
        }
        if (ind) { ind.style.background = '#ff2200'; ind.style.boxShadow = '0 0 8px #ff2200'; }
        // Start timecode ticker
        if (_cineRecordTimer) clearInterval(_cineRecordTimer);
        _cineRecordTimer = setInterval(() => {
            const elapsed = (Date.now() - _cineRecordStart) / 1000;
            const h = Math.floor(elapsed / 3600);
            const m = Math.floor((elapsed % 3600) / 60);
            const s = Math.floor(elapsed % 60);
            const f = Math.floor((elapsed % 1) * 30);
            if (tc) tc.innerText =
                `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:`+
                `${String(s).padStart(2,'0')}:${String(f).padStart(2,'0')}`;
        }, 33);
    } else {
        if (_cineRecordTimer) { clearInterval(_cineRecordTimer); _cineRecordTimer = null; }
        if (btn) {
            btn.innerText          = '● REC';
            btn.style.background   = '#1a0000';
            btn.style.borderColor  = '#cc2200';
            btn.style.color        = '#ff4433';
        }
        if (ind) { ind.style.background = '#333'; ind.style.boxShadow = 'none'; }
        if (tc)  tc.innerText = '00:00:00:00';
    }
}

// ── Full state restore ────────────────────────────────────────────────────────
function handleCineState(data) {
    if (data.limits)    handleCineLimits({ limits: data.limits });
    if (data.keyframes) handleCineKeyframes({ keyframes: data.keyframes });
    if (data.moves)     handleCineMoves({ moves: data.moves });
    if (data.arctan) {
        const ct = document.getElementById('arctan_point_count');
        if (ct) ct.innerText = `(${data.arctan.points})`;
        if (data.arctan.solved) {
            document.getElementById('arctan_status_row').style.display = '';
            const res = document.getElementById('arctan_residual');
            if (res) res.innerText = (data.arctan.residual || 0).toFixed(2);
            const cb = document.getElementById('arctan_enable_cb');
            if (cb) cb.disabled = false;
        }
    }
    if (data.inertia) {
        const massEl = document.getElementById('cine_mass');
        const dragEl = document.getElementById('cine_drag');
        if (massEl) { massEl.value = data.inertia.mass; updateInertiaLabel('mass'); }
        if (dragEl) { dragEl.value = data.inertia.drag; updateInertiaLabel('drag'); }
    }
    if (data.recording) handleRecordState({ recording: data.recording });
    if (data.rail_tilt !== undefined) {
        const el = document.getElementById('rail_tilt_deg');
        if (el) el.value = data.rail_tilt;
    }
    if (data.high_power !== undefined) {
        const el = document.getElementById('high_power_mode');
        if (el) el.checked = data.high_power;
    }
    if (data.origin && data.origin.slider_mm !== undefined) {
        handleCineOriginSet(data.origin);
    }
}

// ── Gamepad button events from server ────────────────────────────────────────
function handleGamepadBtn(data) {
    const btn = data.btn;
    if (btn === 'record')       toggleRecord();
    if (btn === 'play')         cinematicPlay();
    if (btn === 'stop')         sendCmd('cinematic_stop');
    if (btn === 'arctan_toggle') {
        const cb = document.getElementById('arctan_enable_cb');
        if (cb && !cb.disabled) { cb.checked = !cb.checked; arctanEnable(cb.checked); }
    }
}

function handleGamepadStatus(data) {
    const connected = data.connected;
    log(connected ? '🎮 Controller connected.' : '⚠ Controller disconnected.');
    // Future: show persistent indicator in header
}


// ═══════════════════════════════════════════════════════════════════════════════
// PREVIEW CAMERA TOGGLE (Sony framing ↔ PiCam motion zone)
// ═══════════════════════════════════════════════════════════════════════════════

let _HAS_PICAM_FEED = true;  // false when Sony is the preview source (loupe polling pauses)

function togglePreviewCamera() {
    const btn = document.getElementById('previewToggleBtn');
    const cur = btn?.dataset.cam || 'picam';
    setPreviewCamera(cur === 'picam' ? 'sony' : 'picam');
}

function setPreviewCamera(cam) {
    const btn = document.getElementById('previewToggleBtn');
    if (btn) { btn.dataset.cam = cam; btn.innerText = cam === 'sony' ? '📷 Sony' : '📷 PiCam'; }
    sendCmd('set_preview_camera', { camera: cam });
    if (els.mjpegFeed) els.mjpegFeed.src = `/video_feed?t=${Date.now()}`;
    _HAS_PICAM_FEED = (cam === 'picam');
    log(`Preview: ${cam === 'sony' ? 'Sony (framing)' : 'PiCam (motion zone)'}`);
}

function updatePreviewToggleVisibility() {
    const cam = document.getElementById('camera_select')?.value;
    const btn = document.getElementById('previewToggleBtn');
    if (!btn) return;
    btn.style.display = (cam === 'sony') ? 'inline-block' : 'none';
    if (cam !== 'sony') setPreviewCamera('picam');
}


// ═══════════════════════════════════════════════════════════════════════════════
// SLIDER STRIP (linear rail axis — horizontal single-axis joystick)
// ═══════════════════════════════════════════════════════════════════════════════

function setupSliderStrip() {
    const track = document.getElementById('sliderStrip');
    const knob  = document.getElementById('sliderStripKnob');
    if (!track || !knob) return;

    let dragging = false;
    let lastVel  = 0;

    function getVelFromX(clientX) {
        const rect = track.getBoundingClientRect();
        const frac = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
        const vel  = (frac - 0.5) * 2;   // -1 to +1
        return Math.abs(vel) < 0.10 ? 0 : vel;   // ±10% dead zone
    }

    function updateKnob(clientX) {
        const rect = track.getBoundingClientRect();
        const frac = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
        knob.style.left = `calc(${frac * 100}% - 14px)`;
    }

    function sendVel(vel) {
        if (vel === lastVel) return;
        lastVel = vel;
        sendCmd('joystick', { vx: 0, vy: 0, vz: vel });
    }

    function stopSlider() {
        dragging = false;
        track.classList.remove('active');
        knob.style.left = 'calc(50% - 14px)';
        sendVel(0);
    }

    track.addEventListener('pointerdown', (e) => {
        dragging = true;
        track.classList.add('active');
        track.setPointerCapture(e.pointerId);
        updateKnob(e.clientX);
        sendVel(getVelFromX(e.clientX));
        e.preventDefault();
    });
    track.addEventListener('pointermove',  (e) => { if (!dragging) return; updateKnob(e.clientX); sendVel(getVelFromX(e.clientX)); });
    track.addEventListener('pointerup',     stopSlider);
    track.addEventListener('pointercancel', stopSlider);
}


// ═══════════════════════════════════════════════════════════════════════════════
// FOLDER BROWSER (complete implementation)
// ═══════════════════════════════════════════════════════════════════════════════

async function openFolderBrowser() {
    folderBrowserCurrentPath = document.getElementById('save_path')?.value || '/home/tim/Pictures';
    document.getElementById('folderModal').style.display = 'flex';
    cancelCreateFolder();
    await browseTo(folderBrowserCurrentPath);
}

function closeFolderBrowser(e) {
    if (!e || e.target === document.getElementById('folderModal')) {
        document.getElementById('folderModal').style.display = 'none';
        cancelCreateFolder();
    }
}

async function browseTo(path) {
    folderBrowserCurrentPath = path;
    document.getElementById('folderModalPath').innerText = path;
    document.getElementById('folderSelectedPath').innerText = path;
    cancelCreateFolder();

    // Highlight matching drive chip
    document.querySelectorAll('.drive-chip').forEach(c => c.classList.remove('active'));
    const chipMap = {
        'chip_home':  '/home/tim',
        'chip_pics':  '/home/tim/Pictures',
        'chip_media': '/media',
        'chip_mnt':   '/mnt',
    };
    for (const [id, prefix] of Object.entries(chipMap)) {
        if (path.startsWith(prefix)) {
            document.getElementById(id)?.classList.add('active'); break;
        }
    }

    try {
        const resp = await fetch(`/browse?path=${encodeURIComponent(path)}`);
        const data = await resp.json();
        if (data.error) { log(`Browse error: ${data.error}`); return; }

        if (data.disk_free !== undefined) {
            const freeGB  = (data.disk_free  / 1e9).toFixed(1);
            const totalGB = (data.disk_total / 1e9).toFixed(1);
            document.getElementById('folderDiskInfo').innerText =
                `${freeGB} GB free of ${totalGB} GB on this volume`;
        }

        const list = document.getElementById('folderList');
        list.innerHTML = '';
        data.entries.forEach(entry => {
            const row = document.createElement('div');
            row.className = 'folder-entry' + (entry.type === 'file' ? ' is-file' : '');
            row.innerHTML = `<span class="icon">${entry.type === 'dir' ? '📁' : '📄'}</span><span>${entry.name}</span>`;
            if (entry.type === 'dir') row.addEventListener('click', () => browseTo(entry.path));
            list.appendChild(row);
        });
    } catch(e) {
        log(`Browse error: ${e}`);
    }
}

function confirmFolderSelection() {
    const path = document.getElementById('folderSelectedPath')?.innerText;
    if (!path) return;
    const sp = document.getElementById('save_path');
    if (sp) sp.value = path;
    sendCmd('set_save_path', { value: path });
    document.getElementById('folderModal').style.display = 'none';
    updateDiskSpace();
    log(`Save folder: ${path}`);
}

function createFolderPrompt() {
    const row = document.getElementById('newFolderRow');
    if (row) { row.style.display = 'flex'; document.getElementById('newFolderName')?.focus(); }
}

function cancelCreateFolder() {
    const row = document.getElementById('newFolderRow');
    if (row) row.style.display = 'none';
    const inp = document.getElementById('newFolderName');
    if (inp) inp.value = '';
}

function confirmCreateFolder() {
    const name = document.getElementById('newFolderName')?.value.trim();
    if (!name) { log("New folder: name is empty."); return; }
    sendCmd('create_folder', { path: folderBrowserCurrentPath, name });
    cancelCreateFolder();
}

function handleFolderCreated(data) {
    if (data.path) browseTo(data.path);
}

async function scanDrives() {
    const bar = document.getElementById('drivesBar');
    if (!bar) return;
    bar.querySelectorAll('.drive-chip.dynamic').forEach(c => c.remove());

    for (const base of ['/media/tim', '/media', '/mnt']) {
        try {
            const resp = await fetch(`/browse?path=${encodeURIComponent(base)}`);
            const data = await resp.json();
            if (data.error) continue;
            data.entries.filter(e => e.type === 'dir' && e.name !== '..').forEach(e => {
                const chip = document.createElement('button');
                chip.className = 'drive-chip dynamic';
                chip.innerText = `💾 ${e.name}`;
                chip.onclick = () => browseTo(e.path);
                bar.insertBefore(chip, bar.lastElementChild);
            });
        } catch (_) {}
    }
    log("Drives: rescanned external mounts.");
}
