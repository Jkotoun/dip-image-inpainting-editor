# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project

Browser-based image inpainting editor ("Smart Object Remover"). A SvelteKit SPA that
removes objects from photos: click the object → MobileSAM segments it → optional mask
dilation → MI-GAN fills the hole. **Everything runs fully client-side** — the three
ONNX models execute in the browser via ONNX Runtime Web (WebGPU where available, WASM
fallback). No image leaves the device and there is no backend. Deployed to GitHub Pages.

Live demo: https://jkotoun.github.io/dip-image-inpainting-editor/

## Tech stack

- **Framework:** SvelteKit 1.x, **Svelte 4** (stores with `$`, `export let`, `on:` — not Svelte 5 runes)
- **Build:** Vite 4; `@sveltejs/adapter-static` (`fallback: index.html`) → static SPA
- **UI:** Skeleton 2 (`wintry` theme) + Tailwind CSS 3, `lucide-svelte` icons
- **ML runtime:** ONNX Runtime Web `1.18.0`, loaded at runtime from the jsDelivr CDN (not an npm dep)
- **Canvas:** `@panzoom/panzoom`; `svelte-compare-image` (homepage before/after)
- **Package manager:** **pnpm** (`engine-strict=true`, Node 20 in CI)

## Commands

Use **pnpm**. There are **no tests**; `check` and `lint` are the only verification gates.

| Command | What it does |
|---|---|
| `pnpm dev` | Vite dev server (COOP/COEP headers set by a Vite plugin) |
| `pnpm build` | Static SPA build to `build/` |
| `pnpm preview` | Preview the production build |
| `pnpm check` | `svelte-kit sync && svelte-check` — Svelte + TypeScript checking |
| `pnpm lint` | `prettier --check .` then `eslint .` |
| `pnpm format` | `prettier --write .` |

**Style (Prettier-enforced):** tabs for indentation, single quotes, no trailing
commas, 100-column width. `.svelte` files use the svelte parser.

## Layout

```
src/
  app.html              HTML shell; loads /coi-serviceworker.min.js (cross-origin isolation)
  routes/
    +layout.js          prerender = true (static generation)
    +page.svelte        Homepage: upload/dropzone, examples, high-res resize modal;
                        CREATES the worker on mount and posts INIT
    editor/
      +page.svelte      Editor orchestrator: canvases, panzoom, brush, SAM clicks,
                        undo/redo/reset, worker message handling
      editorModule.ts   Editor logic: encoder/decoder runs, mask handling, inpainting, rendering
  lib/
    editorHelpers.ts    dilateMaskByPixels (BFS), mask helpers, downloadImage, clearCanvas
    onnxHelpers.ts      resize to long-side 1024, RGB/NCHW/CHW<->HWC conversions
  stores/
    imgStore.ts         uploaded image (base64), filename, target resolution
    workerStore.ts      the single Worker instance
  workers/
    mainworker.worker.js  Web worker: owns all ONNX sessions + inference (plain JS, //@ts-nocheck)
    messageTypes.js       MESSAGE_TYPES constants shared main-thread <-> worker
  components/           Navbar, EditorToolSelection, PanzoomCanvasControls, EditStepCard
  types/editorTypes.d.ts   tool / brushMode / SAMMode types

static/                mobile_sam.encoder.onnx, sam_onnx_decoder_*_quantized.onnx,
                       migan_pipeline_v2.onnx, coi-serviceworker.min.js, example_photos/, img/
```

## Architecture — two threads

The main thread (Svelte components) owns UI and canvases; a **single web worker**
(`src/workers/mainworker.worker.js`) owns all ONNX sessions and inference. They talk
via `postMessage` using the string constants in `src/workers/messageTypes.js`. The
worker reference lives in `stores/workerStore.ts`.

- **Worker lifecycle:** created ONCE on the homepage `onMount` and sent `INIT` with
  `{ env, appBasePath: base }`. It dynamically imports ONNX Runtime from the CDN, loads
  the encoder (WebGPU, or WASM on mobile/failure) + decoder (always WASM), then the
  editor triggers `LOAD_INPAINTER` (MI-GAN). **Mobile is forced to WASM** (WebGPU is
  unstable there).
- **Pipeline:** upload → resize (long side 1024) → **MobileSAM encoder** (once per
  image, produces the embedding) → **per-click decoder** against the cached embedding →
  threshold mask → **dilation** (slider, scaled by resolution, debounced) → combine
  with brush mask → **MI-GAN inpaint** → result becomes the new working image and the
  encoder re-runs on it. Encoder/decoder/inpaint calls live in `editorModule.ts`; the
  model execution lives in the worker.
- **Editor state:** `{ maskBrush, maskSAM, maskSAMDilated, clickedPositions, imgData,
  currentImgEmbedding }`; undo/redo are arrays of immutable snapshots. Two stacked
  canvases (image+overlay, brush) plus a hidden `<img>` for hold-to-compare.

## Rules for agents

- **Keep the worker message contract in sync.** Any change to what the main thread
  posts or the worker returns must update both sides and the `MESSAGE_TYPES` constants.
  The worker is plain JS with `//@ts-nocheck`.
- **Route every asset/URL through `base`** (`$app/paths`). In production the app is
  served under `/dip-image-inpainting-editor`, so hardcoded root paths (`/foo.onnx`,
  `/editor`) break on GitHub Pages. Model paths, example images, and `goto()` links all
  prepend `base`.
- **Only ONNX inference belongs in the worker.** Heavy array work (dilation, mask
  building) currently runs on the main thread — follow the existing split.
- **Path aliases:** only the default `$lib` → `src/lib` exists. `stores/`, `workers/`,
  `components/` are imported via relative paths. `$app/paths` (`base`) and
  `$app/navigation` (`goto`) are the SvelteKit built-ins in use.
- TypeScript is `strict: true`; JS files are type-checked (`allowJs`/`checkJs`) unless
  `//@ts-nocheck`.

## Deployment & cross-origin isolation

- `.github/workflows/deploy.yaml` builds on every push to `main` (pnpm, Node 20) with
  `BASE_PATH=/dip-image-inpainting-editor` and publishes `build/` to GitHub Pages.
  `svelte.config.js` feeds `BASE_PATH` into `paths.base` (empty in dev).
- ONNX Runtime Web needs a **cross-origin-isolated** context (`SharedArrayBuffer` for
  threaded WASM, plus WebGPU). This requires COOP/COEP headers:
  - **Dev:** `vite-plugin-cross-origin-isolation` sets the headers.
  - **Production:** GitHub Pages can't set headers, so `static/coi-serviceworker.min.js`
    (loaded from `app.html`) injects them client-side. Don't remove it when hosting on Pages.
- When hosting elsewhere, set `BASE_PATH` accordingly (empty for a domain root) and
  ensure the host sends COOP/COEP headers or keep the COI service worker in place.

## Gotchas

- Model files and the ONNX runtime are large and CDN-hosted; without cross-origin
  isolation, WebGPU / threaded WASM inference fails.
- Inference is heavy: large images on a WASM-only browser can take tens of seconds and
  a lot of memory. Images with a side > 2000 px prompt a downscale on upload.
- Editor globals in `editor/+page.svelte` are prefixed with `g`.
