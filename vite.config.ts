import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';
import { viteStaticCopy } from 'vite-plugin-static-copy';
//@ts-ignore
import crossOriginIsolation from 'vite-plugin-cross-origin-isolation'

export default defineConfig({
	plugins: [
		crossOriginIsolation(),
		sveltekit(),

	],
});
