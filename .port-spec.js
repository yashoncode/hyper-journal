export const meta = {
  name: 'hyperjournal-port-spec',
  description: 'Read Hyperjournal Python source and produce a verified Rust/Tauri v2 port specification',
  phases: [
    { title: 'Read', detail: 'one agent per subsystem: journal, llm, web routes, emotion, stt, frontend' },
    { title: 'Design', detail: '3 independent Rust crate architectures' },
    { title: 'Judge', detail: 'score each architecture on portability, diff size, Android risk' },
    { title: 'Spec', detail: 'synthesize the winning design into a concrete port plan' },
    { title: 'Critic', detail: 'adversarial completeness pass' },
  ],
}

const ROOT = 'C:/Users/Yashwanth/hyperjournal'

const SPEC_SCHEMA = {
  type: 'object',
  properties: {
    subsystem: { type: 'string' },
    summary: { type: 'string', description: 'What this subsystem does, 3-5 sentences' },
    public_surface: {
      type: 'array',
      description: 'Every function/route/handler another subsystem or the frontend calls',
      items: {
        type: 'object',
        properties: {
          name: { type: 'string' },
          signature: { type: 'string' },
          behavior: { type: 'string' },
          location: { type: 'string', description: 'file:line' },
        },
        required: ['name', 'signature', 'behavior', 'location'],
      },
    },
    state: { type: 'string', description: 'Mutable/global state held, and its lifetime' },
    data_shapes: { type: 'string', description: 'Exact JSON / file / frontmatter shapes produced or consumed, verbatim keys' },
    external_deps: { type: 'array', items: { type: 'string' }, description: 'Python packages used and what for' },
    port_hazards: {
      type: 'array',
      description: 'Things that will NOT translate cleanly to Rust or to Android',
      items: {
        type: 'object',
        properties: {
          hazard: { type: 'string' },
          why: { type: 'string' },
          suggested_rust_equivalent: { type: 'string' },
        },
        required: ['hazard', 'why', 'suggested_rust_equivalent'],
      },
    },
    subtle_behavior: { type: 'array', items: { type: 'string' }, description: 'Non-obvious logic a naive port would silently break. Quote the code.' },
  },
  required: ['subsystem', 'summary', 'public_surface', 'state', 'data_shapes', 'external_deps', 'port_hazards', 'subtle_behavior'],
}

const READERS = [
  { key: 'journal', files: 'journal.py, and the journal/ directory layout on disk, and calib.json',
    focus: 'Storage. Path scheme, the day_start_hour rollover rule, frontmatter keys, the dedupe -N suffix, atomic write via tmp file, trash/discard, entries() ordering and _order(), summaries(), normalize(). Read the _selftest() function carefully - it encodes the exact expected behavior. Also state what a new `video:` frontmatter key would need.' },
  { key: 'llm', files: 'llm.py',
    focus: 'The conversation engine. Every prompt template verbatim, the NVIDIA API request/response shape, the exchange state machine, MAX_EXCHANGES, CONFIDENCE_FLOOR, themes/summary/song extraction, settings() and api_key() and how .env is read and written, the canned/no-key fallback path, and the model listing call.' },
  { key: 'web', files: 'web.py',
    focus: 'The HTTP contract that the frontend depends on. For EVERY route (/config GET+POST, /models, /entries, /entries/all DELETE, /analyze, /greet, /listen, /say, /delete, /discard, /save, /baseline, /baseline/clear, and the static file routes) give: method, exact request body shape, exact response JSON keys, and status codes. Also: what server-side session state lives between requests (the in-progress conversation, turns, the loaded models) and how load() initializes it.' },
  { key: 'emotion', files: 'emotion_cam.py and calib.json',
    focus: 'The vision pipeline. cv2.FaceDetectorYN.create params (input size 320x320, score threshold 0.8, nms 0.3, topK 5000), how the detection result array is indexed, the crop fed to HSEmotionRecognizer, the emotion label ordering, the baseline calibration math (how calib.json is produced and subtracted), and the confidence floor logic. Be exact about array shapes and index order - a Rust port has no cv2 to do postprocessing.' },
  { key: 'stt', files: 'stt.py',
    focus: 'Speech to text. WhisperModel params (base.en, cpu, int8), the MIN_SECONDS and MIN_LOGPROB gates, the HALLUCINATIONS set and how it is matched, what audio container arrives from the frontend MediaRecorder, and exactly what transcribe() returns on silence vs speech.' },
  { key: 'frontend', files: 'index.html, settings.html, app.css',
    focus: 'The client. Every fetch() call with its method/body/expected-response. The getUserMedia + canvas + toBlob frame loop and its cadence. The MediaRecorder audio flow. All DOM ids and the render() function. The routing between / and /settings. What CSS assumptions exist. Crucially: enumerate every place the code assumes a same-origin HTTP server, because those are exactly the lines a Tauri port must change.' },
]

phase('Read')
const specs = (await parallel(READERS.map(r => () =>
  agent(
    `Read the Hyperjournal codebase at ${ROOT}. It is a local journaling app: webcam emotion detection + whisper speech-to-text + an LLM conversation, storing markdown entries.\n\n` +
    `YOUR SUBSYSTEM: ${r.key}\nFILES: ${r.files}\n\nFOCUS: ${r.focus}\n\n` +
    `Read the actual files in full - do not guess, do not summarize from filenames. Quote real code for anything subtle. ` +
    `Your output is a port specification: a Rust engineer who has never seen this Python must be able to reimplement this subsystem byte-compatibly from your spec alone. ` +
    `Be exhaustive about exact string keys, exact numeric constants, and exact ordering - those are what a port silently breaks.`,
    { label: `read:${r.key}`, phase: 'Read', schema: SPEC_SCHEMA }
  )
))).filter(Boolean)

log(`read ${specs.length}/${READERS.length} subsystems; ${specs.reduce((n, s) => n + s.port_hazards.length, 0)} port hazards found`)

const digest = specs.map(s =>
  `## ${s.subsystem}\n${s.summary}\n\nPUBLIC SURFACE:\n${s.public_surface.map(p => `- ${p.name} ${p.signature} @ ${p.location}\n  ${p.behavior}`).join('\n')}\n\n` +
  `STATE: ${s.state}\n\nDATA SHAPES:\n${s.data_shapes}\n\nDEPS: ${s.external_deps.join(', ')}\n\n` +
  `PORT HAZARDS:\n${s.port_hazards.map(h => `- ${h.hazard}\n  why: ${h.why}\n  rust: ${h.suggested_rust_equivalent}`).join('\n')}\n\n` +
  `SUBTLE:\n${s.subtle_behavior.map(b => `- ${b}`).join('\n')}`
).join('\n\n---\n\n')

const CONSTRAINTS =
  `HARD CONSTRAINTS (the user already chose these - do not relitigate):\n` +
  `- Tauri v2 shell, Rust backend. Windows x64 portable single .exe, no installer.\n` +
  `- Depend on the SYSTEM WebView2 runtime (do NOT bundle a fixed runtime).\n` +
  `- ML runs RUST-NATIVE, not WASM: the \`ort\` crate for the two ONNX models (YuNet face detect, hsemotion enet_b0_8_best_vgaf), and \`whisper-rs\` (whisper.cpp) for speech-to-text.\n` +
  `- Camera and microphone capture stay in the webview via getUserMedia/MediaRecorder; frames and audio cross into Rust over Tauri IPC.\n` +
  `- Journal data stays LOCAL: markdown + YAML frontmatter, same on-disk scheme as today.\n` +
  `- NEW FEATURE: video journal. MediaRecorder records webm, bytes go to Rust, saved next to the markdown entry, referenced by a new frontmatter key.\n` +
  `- Android APK is a REQUIRED future target, full-featured (same capabilities as desktop), built via \`cargo tauri android\`. Every decision must keep that cheap. No desktop-only crate (no OpenCV, no nokhwa) anywhere on the critical path.\n` +
  `- The existing index.html/settings.html/app.css should be reused with the SMALLEST possible diff.\n` +
  `ENVIRONMENT (already verified present): Node 24.14.1, MSVC BuildTools 2022, Windows SDK 10.0.26100, WebView2 152.0.4191.66. Rust is being installed now.\n`

const ARCH_SCHEMA = {
  type: 'object',
  properties: {
    approach_name: { type: 'string' },
    thesis: { type: 'string', description: 'The one idea that makes this design different, 2 sentences' },
    ipc_strategy: { type: 'string', description: 'How the existing frontend fetch() calls reach Rust. Be concrete and show the code.' },
    file_tree: { type: 'string', description: 'Full proposed repo tree after the port, with a one-line purpose per file' },
    tauri_commands: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          name: { type: 'string' },
          rust_signature: { type: 'string' },
          replaces_route: { type: 'string' },
          notes: { type: 'string' },
        },
        required: ['name', 'rust_signature', 'replaces_route', 'notes'],
      },
    },
    crates: {
      type: 'array',
      description: 'Every cargo dependency with a real, currently-published version and why it is needed',
      items: {
        type: 'object',
        properties: {
          name: { type: 'string' },
          version: { type: 'string' },
          purpose: { type: 'string' },
          android_ok: { type: 'string', description: 'Does this build for aarch64-linux-android? Say how you know, or say UNVERIFIED.' },
        },
        required: ['name', 'version', 'purpose', 'android_ok'],
      },
    },
    frame_pipeline: { type: 'string', description: 'Exact path a camera frame takes from getUserMedia to an emotion label, including encoding and IPC mechanism' },
    audio_pipeline: { type: 'string', description: 'Exact path audio takes from MediaRecorder to whisper-rs, including how webm/opus becomes f32 16kHz PCM' },
    yunet_postprocess: { type: 'string', description: 'Concrete plan for replacing cv2.FaceDetectorYN: tensor shapes out of ort, priors/anchors, decode, NMS' },
    model_bundling: { type: 'string', description: 'How ONNX + ggml model files ship inside a single portable exe, and where they land at runtime on Windows vs Android' },
    frontend_diff: { type: 'string', description: 'Exactly which lines of index.html/settings.html change, and how many' },
    android_delta: { type: 'string', description: 'What additionally must be done for the APK, and what this design already solved for it' },
    weaknesses: { type: 'array', items: { type: 'string' }, description: 'Honest self-criticism: where this design is worst' },
  },
  required: ['approach_name', 'thesis', 'ipc_strategy', 'file_tree', 'tauri_commands', 'crates', 'frame_pipeline', 'audio_pipeline', 'yunet_postprocess', 'model_bundling', 'frontend_diff', 'android_delta', 'weaknesses'],
}

const ANGLES = [
  { key: 'minimal-diff', push: 'Optimize ruthlessly for the SMALLEST total diff and the least new code. The existing frontend is already written against fetch() - find the cheapest possible way to keep it working unchanged. Ask at every step: does this code need to exist at all? Prefer one shim over twenty rewrites.' },
  { key: 'android-first', push: 'Design as if the Android APK ships in week two, not someday. Every choice is judged by what it costs on aarch64-linux-android with the NDK. Assume desktop is the easy target and mobile is the one that will break; pre-solve mobile.' },
  { key: 'testability', push: 'Optimize for a port that can be PROVEN correct against the Python it replaces. The Python has a _selftest() in journal.py and a --selftest in emotion_cam.py. Design so every ported subsystem has a runnable Rust check that fails loudly if behavior drifts from the Python, including byte-identical markdown output.' },
]

phase('Design')
const archs = (await parallel(ANGLES.map(a => () =>
  agent(
    `You are a Rust + Tauri v2 architect. Design the port of this Python app to a Tauri v2 desktop app.\n\n${CONSTRAINTS}\n\n` +
    `YOUR ANGLE: ${a.key}\n${a.push}\n\n` +
    `Here are verified specifications of every existing subsystem, produced by engineers who read the real source:\n\n${digest}\n\n` +
    `You may read files under ${ROOT} yourself to check any detail. You may use WebSearch/WebFetch to confirm current crate versions and Tauri v2 APIs - do NOT invent version numbers or API names, and mark anything you could not confirm as UNVERIFIED.\n\n` +
    `Produce a complete, buildable architecture. Concrete code over prose. Real crate versions. Name the parts you are least sure about.`,
    { label: `design:${a.key}`, phase: 'Design', schema: ARCH_SCHEMA, effort: 'high' }
  )
))).filter(Boolean)

log(`${archs.length} architectures: ${archs.map(a => a.approach_name).join(', ')}`)

const archDigest = archs.map((a, i) =>
  `### CANDIDATE ${i}: ${a.approach_name}\nTHESIS: ${a.thesis}\n\nIPC: ${a.ipc_strategy}\n\nTREE:\n${a.file_tree}\n\n` +
  `COMMANDS:\n${a.tauri_commands.map(c => `- ${c.name}: ${c.rust_signature}  (was ${c.replaces_route}) ${c.notes}`).join('\n')}\n\n` +
  `CRATES:\n${a.crates.map(c => `- ${c.name} ${c.version} :: ${c.purpose} :: android=${c.android_ok}`).join('\n')}\n\n` +
  `FRAME: ${a.frame_pipeline}\n\nAUDIO: ${a.audio_pipeline}\n\nYUNET: ${a.yunet_postprocess}\n\n` +
  `MODELS: ${a.model_bundling}\n\nFRONTEND DIFF: ${a.frontend_diff}\n\nANDROID: ${a.android_delta}\n\n` +
  `SELF-DECLARED WEAKNESSES:\n${a.weaknesses.map(w => `- ${w}`).join('\n')}`
).join('\n\n---\n\n')

const JUDGE_SCHEMA = {
  type: 'object',
  properties: {
    lens: { type: 'string' },
    scores: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          candidate: { type: 'string' },
          score: { type: 'number', description: '0-10' },
          reasoning: { type: 'string' },
          fatal_flaw: { type: 'string', description: 'The single thing most likely to make this design fail in practice, or NONE' },
        },
        required: ['candidate', 'score', 'reasoning', 'fatal_flaw'],
      },
    },
    winner: { type: 'string' },
    ideas_to_graft: { type: 'array', items: { type: 'string' }, description: 'Specific good ideas from the LOSING candidates that the winner should absorb' },
    factual_errors: { type: 'array', items: { type: 'string' }, description: 'Any claim in any candidate you believe is factually wrong (wrong crate version, nonexistent API, wrong tensor shape). Check these.' },
  },
  required: ['lens', 'scores', 'winner', 'ideas_to_graft', 'factual_errors'],
}

const LENSES = [
  { key: 'will-it-build', push: 'You are a skeptical build engineer. Your only question: will this actually compile and run on Windows x64 today, and later on aarch64-linux-android? Attack invented APIs, wrong crate versions, crates with no Android support, C++ deps that need an NDK toolchain nobody configured, and hand-waving around linking. Verify crate names and versions against crates.io. Assume every UNVERIFIED claim is false until checked.' },
  { key: 'correctness', push: 'You are the engineer who will be paged when a user loses a journal entry. Your question: does this port preserve the Python behavior exactly? Attack anywhere the design glosses over the storage semantics, the calibration math, the whisper hallucination filters, the confidence floor, or the conversation state machine. Silent behavior drift is worse than a crash.' },
  { key: 'laziness', push: 'You are a senior engineer who has been paged at 3am for an over-engineered codebase. Your question: how much of this design does not need to exist? Attack speculative abstraction, layers that wrap one thing, new dependencies that replace ten lines, and any rewrite of frontend code that already works. The best code is the code never written. Reward the design with the smallest honest diff - but do NOT reward a design that is small because it skipped something necessary.' },
]

phase('Judge')
const judgments = (await parallel(LENSES.map(l => () =>
  agent(
    `Three Rust/Tauri architectures for the same port are below. Score each 0-10 through YOUR lens, pick a winner, and list good ideas worth stealing from the losers.\n\n` +
    `${CONSTRAINTS}\n\nYOUR LENS: ${l.key}\n${l.push}\n\n${archDigest}\n\n` +
    `Use WebSearch/WebFetch and read files under ${ROOT} to CHECK claims rather than trusting them. Be harsh and specific; a vague critique is worthless. Name exact lines or exact claims.`,
    { label: `judge:${l.key}`, phase: 'Judge', schema: JUDGE_SCHEMA, effort: 'high' }
  )
))).filter(Boolean)

const tally = {}
for (const j of judgments) for (const s of j.scores) tally[s.candidate] = (tally[s.candidate] || 0) + s.score
log(`judge tally: ${Object.entries(tally).map(([k, v]) => `${k}=${v}`).join('  ')}`)

const judgeDigest = judgments.map(j =>
  `### LENS: ${j.lens}  (winner: ${j.winner})\n${j.scores.map(s => `- ${s.candidate}: ${s.score}/10 — ${s.reasoning}\n  FATAL: ${s.fatal_flaw}`).join('\n')}\n` +
  `GRAFT:\n${j.ideas_to_graft.map(i => `- ${i}`).join('\n')}\nFACTUAL ERRORS FOUND:\n${j.factual_errors.map(e => `- ${e}`).join('\n')}`
).join('\n\n')

const PLAN_SCHEMA = {
  type: 'object',
  properties: {
    chosen_approach: { type: 'string' },
    rationale: { type: 'string' },
    file_tree: { type: 'string', description: 'Final repo tree after the port, one line of purpose per file' },
    cargo_toml: { type: 'string', description: 'The complete, literal src-tauri/Cargo.toml contents, ready to paste' },
    tauri_conf_json: { type: 'string', description: 'The complete, literal src-tauri/tauri.conf.json contents, ready to paste' },
    ipc_shim: { type: 'string', description: 'The complete, literal JS shim that lets the existing frontend fetch() calls reach Tauri commands unchanged. Ready to paste.' },
    steps: {
      type: 'array',
      description: 'Ordered, independently-shippable build steps. Each must end in something runnable and checkable.',
      items: {
        type: 'object',
        properties: {
          n: { type: 'number' },
          title: { type: 'string' },
          files_touched: { type: 'array', items: { type: 'string' } },
          what: { type: 'string', description: 'Concretely what gets written, in enough detail to implement without re-deriving' },
          verification: { type: 'string', description: 'The exact command to run and the exact output that proves this step works' },
          depends_on: { type: 'array', items: { type: 'number' } },
          risk: { type: 'string', enum: ['low', 'medium', 'high'] },
        },
        required: ['n', 'title', 'files_touched', 'what', 'verification', 'depends_on', 'risk'],
      },
    },
    behavior_parity_checks: { type: 'array', items: { type: 'string' }, description: 'Specific assertions that prove the Rust matches the Python, especially for journal storage and calibration math' },
    open_risks: {
      type: 'array',
      items: {
        type: 'object',
        properties: { risk: { type: 'string' }, mitigation: { type: 'string' }, when_it_bites: { type: 'string' } },
        required: ['risk', 'mitigation', 'when_it_bites'],
      },
    },
    corrected_claims: { type: 'array', items: { type: 'string' }, description: 'Factual errors the judges caught, and the corrected fact' },
  },
  required: ['chosen_approach', 'rationale', 'file_tree', 'cargo_toml', 'tauri_conf_json', 'ipc_shim', 'steps', 'behavior_parity_checks', 'open_risks', 'corrected_claims'],
}

phase('Spec')
const plan = await agent(
  `Synthesize the final port plan. Three architectures were designed and judged by three independent lenses.\n\n${CONSTRAINTS}\n\n` +
  `SUBSYSTEM SPECS:\n${digest}\n\n===\n\nARCHITECTURES:\n${archDigest}\n\n===\n\nJUDGMENTS:\n${judgeDigest}\n\n` +
  `Take the winning architecture as the base, graft in the good ideas from the losers, and FIX every factual error the judges caught - verify the corrections yourself with WebSearch/WebFetch against crates.io and the Tauri v2 docs. Do not carry forward any invented API or unpublished crate version.\n\n` +
  `The output is an implementation plan a different engineer executes without asking questions. Literal file contents where the schema asks for them. Every step must end in a runnable verification command. Order steps so Windows works end-to-end before any Android work starts.`,
  { label: 'synthesize-plan', phase: 'Spec', schema: PLAN_SCHEMA, effort: 'high' }
)

phase('Critic')
const CRITIC_SCHEMA = {
  type: 'object',
  properties: {
    gaps: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          gap: { type: 'string' },
          severity: { type: 'string', enum: ['blocker', 'major', 'minor'] },
          fix: { type: 'string' },
        },
        required: ['gap', 'severity', 'fix'],
      },
    },
    unverified_claims: { type: 'array', items: { type: 'string' } },
    missing_from_python: { type: 'array', items: { type: 'string' }, description: 'Behavior present in the Python that the plan never accounts for' },
    verdict: { type: 'string', enum: ['ready', 'ready-with-fixes', 'not-ready'] },
  },
  required: ['gaps', 'unverified_claims', 'missing_from_python', 'verdict'],
}

const critics = (await parallel([
  `Hunt for behavior in the Python at ${ROOT} that this plan silently drops. Read every .py file yourself and diff it mentally against the plan's steps. Anything in the Python with no home in the plan is a gap.`,
  `Hunt for unverified or invented technical claims in this plan: crate versions that do not exist on crates.io, Tauri v2 APIs that are not real, ONNX tensor shapes asserted without evidence, Android support asserted without evidence. Check them with WebSearch/WebFetch. Assume the plan is lying until you confirm each claim.`,
  `Hunt for what makes this plan unexecutable: steps whose verification command would not actually prove the step works, steps that depend on something not yet built, missing toolchain setup, and anything requiring a decision the plan never makes.`,
].map((push, i) => () =>
  agent(
    `Adversarially critique this Rust/Tauri port plan. Your job is to find what is WRONG or MISSING, not to praise it.\n\n${push}\n\n` +
    `${CONSTRAINTS}\n\nTHE PLAN:\n${JSON.stringify(plan, null, 2)}\n\n` +
    `Default to reporting a gap when uncertain. A plan that looks complete and is not is the expensive failure here.`,
    { label: `critic:${i}`, phase: 'Critic', schema: CRITIC_SCHEMA, effort: 'high' }
  )
))).filter(Boolean)

const allGaps = critics.flatMap(c => c.gaps)
const blockers = allGaps.filter(g => g.severity === 'blocker')
log(`critics: ${allGaps.length} gaps (${blockers.length} blockers); verdicts ${critics.map(c => c.verdict).join(', ')}`)

return {
  plan,
  blockers,
  major_gaps: allGaps.filter(g => g.severity === 'major'),
  minor_gaps: allGaps.filter(g => g.severity === 'minor'),
  unverified_claims: critics.flatMap(c => c.unverified_claims),
  missing_from_python: critics.flatMap(c => c.missing_from_python),
  verdicts: critics.map(c => c.verdict),
  judge_tally: tally,
  subsystem_hazards: specs.flatMap(s => s.port_hazards.map(h => `[${s.subsystem}] ${h.hazard} -> ${h.suggested_rust_equivalent}`)),
}
