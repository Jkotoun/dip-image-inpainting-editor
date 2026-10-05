import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/kit/vite';
const dev = process.argv.includes('dev')
/** @type {import('@sveltejs/kit').Config} */
const config = {
	// Consult https://kit.svelte.dev/docs/integrations#preprocessors
	// for more information about preprocessors
	preprocess: vitePreprocess(),

	kit: {
		// adapter-auto only supports some environments, see https://kit.svelte.dev/docs/adapter-auto for a list.
		// If your environment is not supported or you settled on a specific environment, switch out the adapter.
		// See https://kit.svelte.dev/docs/adapters for more information about adapters.
		adapter: adapter({
			// Use 404.html (not index.html) for the SPA fallback: adapter-static writes the
			// fallback to this exact filename, and naming it "index.html" clobbered the real
			// prerendered home page with an empty, unstyled CSR shell (no inlined stylesheet,
			// no rendered markup), causing a flash of unstyled content on every first load.
			fallback: '404.html'
		}),
		paths:{
			base: dev? '' : process.env.BASE_PATH,
		}
	}
};

export default config;
